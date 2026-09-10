use ruda_kernel::dsl as kernel_dsl;
use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceProblem,
    ReduceVectorSettings,
};
use crate::reduce::{
    BoundChecks, IdleMode, ReduceError, VectorizationMode,
    launch::{calculate_plane_count_per_ruda, support_plane},
    routines::{BlueprintStrategy, PlaneMergeStrategy, PlaneReduceBlueprint, Routine},
};
use ruda_kernel::dsl::RudaCount;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::ir::features::Plane;
use ruda_kernel::dsl::prelude::ComputeClient;
use ruda_kernel::tiling::ruda_count::ruda_count_spread_with_total;

#[derive(Debug, Clone)]
pub struct PlaneRoutine;

#[derive(Debug, Clone)]
pub struct PlaneStrategy {
    /// How the accumulators are handled in a plane.
    pub independent: bool,
}

impl Routine for PlaneRoutine {
    type Strategy = PlaneStrategy;
    type Blueprint = PlaneReduceBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let address_type = problem.address_type;
        let (blueprint, ruda_dim, ruda_count) = match strategy {
            BlueprintStrategy::Forced(blueprint, ruda_dim) => {
                super::validate_ruda_dim(client, ruda_dim)?;
                if !support_plane(client) {
                    return Err(ReduceError::PlanesUnavailable);
                }

                let properties = &client.properties().hardware;
                if ruda_dim.x != properties.plane_size_max {
                    return Err(ReduceError::Validation {
                        details: "`ruda_dim.x` must match `plane_size_max`",
                    });
                }
                let working_planes = working_planes(&settings, &problem);

                let planes_per_ruda = ruda_dim.y as usize * ruda_dim.z as usize;
                let working_rudas = working_planes.div_ceil(planes_per_ruda);
                let (ruda_count, launched_rudas) =
                    ruda_count_spread_with_total(client, working_rudas);
                let plane_idle = launched_rudas * planes_per_ruda != working_planes;

                if plane_idle && !blueprint.plane_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many planes launched for the problem causing OOD, but `plane_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    vectorization_mode: settings.vectorization_mode,
                    global: GlobalReduceBlueprint::Plane(blueprint),
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
            ruda_count,
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
    strategy: PlaneStrategy,
) -> Result<(ReduceBlueprint, RudaDim, RudaCount), ReduceError> {
    if !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_planes = working_planes(settings, &problem);
    let working_units = working_planes * plane_size as usize;
    let plane_count = calculate_plane_count_per_ruda(working_units, plane_size, properties);
    let working_rudas = working_planes.div_ceil(plane_count as usize);

    let ruda_dim = RudaDim::new_2d(plane_size, plane_count);
    let (ruda_count, ruda_launched) = ruda_count_spread_with_total(client, working_rudas);

    let plane_idle = ruda_launched * ruda_dim.num_elems() as usize != working_units;
    let work_size = match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_len / settings.vector_size_input,
        VectorizationMode::Perpendicular => problem.reduce_len,
    };
    let bound_checks = match work_size.is_multiple_of(plane_size as usize) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let plane_idle = match plane_idle {
        true => match client
            .properties()
            .features
            .plane
            .contains(Plane::NonUniformControlFlow)
        {
            true => IdleMode::Terminate,
            false => IdleMode::Mask,
        },
        false => IdleMode::None,
    };

    let strategy_enum = if strategy.independent {
        PlaneMergeStrategy::Lazy
    } else {
        PlaneMergeStrategy::Eager
    };

    let blueprint = ReduceBlueprint {
        vectorization_mode: settings.vectorization_mode,
        global: GlobalReduceBlueprint::Plane(PlaneReduceBlueprint {
            plane_idle,
            bound_checks,
            plane_merge_strategy: strategy_enum,
            plane_dim_ceil: properties.plane_size_max != properties.plane_size_min,
        }),
    };

    Ok((blueprint, ruda_dim, ruda_count))
}

fn working_planes(settings: &ReduceVectorSettings, problem: &ReduceProblem) -> usize {
    match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_count / settings.vector_size_output,
        VectorizationMode::Perpendicular => problem.reduce_count / settings.vector_size_input,
    }
}
