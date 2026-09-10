use ruda_kernel::dsl as kernel_dsl;
use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceProblem,
    ReduceVectorSettings,
};
use crate::reduce::{
    BoundChecks, IdleMode, ReduceError, VectorizationMode,
    launch::{calculate_plane_count_per_ruda, support_plane},
    routines::{BlueprintStrategy, RudaBlueprint, Routine},
};
use ruda_kernel::dsl::RudaCount;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::ir::features::Plane;
use ruda_kernel::tiling::ruda_count::ruda_count_spread_with_total;

#[derive(Debug, Clone)]
pub struct RudaRoutine;

#[derive(Debug, Clone)]
pub struct RudaStrategy {
    /// If we use plane to aggregate accumulators.
    pub use_planes: bool,
}

impl Routine for RudaRoutine {
    type Strategy = RudaStrategy;
    type Blueprint = RudaBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let address_type = problem.address_type;
        let (blueprint, ruda_dim, num_rudas) = match strategy {
            BlueprintStrategy::Forced(blueprint, ruda_dim) => {
                super::validate_ruda_dim(client, ruda_dim)?;
                // One accumulator per plane.
                if blueprint.use_planes {
                    if !support_plane(client) {
                        return Err(ReduceError::PlanesUnavailable);
                    }
                    if ruda_dim.x != client.properties().hardware.plane_size_max {
                        return Err(ReduceError::Validation {
                            details: "`ruda_dim.x` must match `plane_size_max`",
                        });
                    }
                    let required_accumulators = client.properties().max_plane_count(ruda_dim.num_elems()) as usize;
                    if blueprint.num_shared_accumulators != required_accumulators {
                        return Err(ReduceError::Validation {
                            details: "Num accumulators must cover the device's possible subgroup count",
                        });
                    }
                // One accumulator per unit.
                } else if blueprint.num_shared_accumulators != ruda_dim.num_elems() as usize {
                    return Err(ReduceError::Validation {
                        details: "Num accumulators should match ruda_dim.num_elems()",
                    });
                }

                let work_size = match settings.vectorization_mode {
                    VectorizationMode::Parallel => problem.reduce_len / settings.vector_size_input,
                    VectorizationMode::Perpendicular => problem.reduce_len,
                };
                if blueprint.bound_checks == BoundChecks::None
                    && blueprint.ruda_idle != IdleMode::Mask
                    && !work_size.is_multiple_of(ruda_dim.num_elems() as usize)
                {
                    return Err(ReduceError::Validation {
                        details: "Partial ruda reads require bound checks",
                    });
                }

                let working_rudas = working_rudas(&settings, &problem);
                let (ruda_count, launched_rudas) =
                    ruda_count_spread_with_total(client, working_rudas);

                if working_rudas != launched_rudas && !blueprint.ruda_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many rudas launched for the problem causing OOD, but `ruda_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    vectorization_mode: settings.vectorization_mode,
                    global: GlobalReduceBlueprint::Ruda(blueprint),
                };

                (blueprint, ruda_dim, ruda_count)
            }
            BlueprintStrategy::Inferred(strategy) => {
                let (blueprint, ruda_dim, ruda_count) =
                    generate_blueprint::<R>(client, problem, &settings, strategy)?;
                (blueprint, ruda_dim, ruda_count)
            }
        };

        let launch = ReduceLaunchSettings {
            ruda_dim,
            ruda_count: num_rudas,
            address_type,
            vector: settings,
        };

        Ok((blueprint, launch))
    }
}

fn generate_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    settings: &ReduceVectorSettings,
    strategy: RudaStrategy,
) -> Result<(ReduceBlueprint, RudaDim, RudaCount), ReduceError> {
    if strategy.use_planes && !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let hardware_properties = &client.properties().hardware;
    let plane_size = hardware_properties.plane_size_max;

    let use_planes = strategy.use_planes;

    let working_rudas = working_rudas(settings, &problem);
    let working_units = working_rudas * problem.reduce_len.div_ceil(settings.vector_size_input);
    let plane_count =
        calculate_plane_count_per_ruda(working_units, plane_size, hardware_properties);
    let ruda_dim = RudaDim::new_2d(plane_size, plane_count);
    let ruda_size = ruda_dim.num_elems();

    let work_size = match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_len / settings.vector_size_input,
        VectorizationMode::Perpendicular => problem.reduce_len,
    };
    let bound_checks = match work_size.is_multiple_of(ruda_size as usize) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let num_shared_accumulators = match use_planes {
        true => client.properties().max_plane_count(ruda_size) as usize,
        false => ruda_size as usize,
    };

    let (ruda_count, launched_rudas) = ruda_count_spread_with_total(client, working_rudas);

    let ruda_idle = match working_rudas != launched_rudas {
        true => match strategy.use_planes
            && !client
                .properties()
                .features
                .plane
                .contains(Plane::NonUniformControlFlow)
        {
            true => IdleMode::Mask,
            false => IdleMode::Terminate,
        },
        false => IdleMode::None,
    };
    let blueprint = ReduceBlueprint {
        vectorization_mode: settings.vectorization_mode,
        global: GlobalReduceBlueprint::Ruda(RudaBlueprint {
            ruda_idle,
            bound_checks,
            num_shared_accumulators,
            use_planes,
        }),
    };

    Ok((blueprint, ruda_dim, ruda_count))
}

fn working_rudas(settings: &ReduceVectorSettings, problem: &ReduceProblem) -> usize {
    match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_count / settings.vector_size_output,
        VectorizationMode::Perpendicular => problem.reduce_count / settings.vector_size_input,
    }
}
