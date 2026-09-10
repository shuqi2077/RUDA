use ruda_kernel::dsl as kernel_dsl;
use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceProblem,
    ReduceVectorSettings,
};
use crate::reduce::{
    IdleMode, ReduceError, VectorizationMode,
    launch::calculate_plane_count_per_ruda,
    routines::{BlueprintStrategy, Routine, UnitReduceBlueprint},
};
use ruda_kernel::dsl::RudaCount;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::tiling::ruda_count::ruda_count_spread_with_total;

#[derive(Debug, Clone)]
pub struct UnitRoutine;

#[derive(Debug, Clone)]
pub struct UnitStrategy;

impl Routine for UnitRoutine {
    type Strategy = UnitStrategy;
    type Blueprint = UnitReduceBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ruda_kernel::dsl::prelude::ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let address_type = problem.address_type;
        let (blueprint, ruda_dim, ruda_count) = match strategy {
            BlueprintStrategy::Forced(blueprint, ruda_dim) => {
                super::validate_ruda_dim(client, ruda_dim)?;
                let working_units = working_units(&settings, &problem);
                let num_units_in_ruda = ruda_dim.num_elems();
                let working_rudas = working_units.div_ceil(num_units_in_ruda as usize);

                let (ruda_count, launched_rudas) =
                    ruda_count_spread_with_total(client, working_rudas);

                let unit_idle = !working_units.is_multiple_of(num_units_in_ruda as usize)
                    || working_rudas != launched_rudas;
                if unit_idle && !blueprint.unit_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many units launched for the problem causing OOD, but `unit_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    vectorization_mode: settings.vectorization_mode,
                    global: GlobalReduceBlueprint::Unit(blueprint),
                };

                (blueprint, ruda_dim, ruda_count)
            }
            BlueprintStrategy::Inferred(_) => {
                let (blueprint, ruda_dim, ruda_count) =
                    generate_blueprint::<R>(client, problem, &settings)?;
                (blueprint, ruda_dim, ruda_count)
            }
        };

        let launch = ReduceLaunchSettings {
            ruda_dim,
            ruda_count,
            vector: settings,
            address_type,
        };

        Ok((blueprint, launch))
    }
}

fn generate_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    settings: &ReduceVectorSettings,
) -> Result<(ReduceBlueprint, RudaDim, RudaCount), ReduceError> {
    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_units = working_units(settings, &problem);
    let plane_count = calculate_plane_count_per_ruda(working_units, plane_size, properties);

    let ruda_dim = RudaDim::new_2d(plane_size, plane_count);
    let num_units_in_ruda = ruda_dim.num_elems();

    let working_rudas = working_units.div_ceil(num_units_in_ruda as usize);
    let (ruda_count, ruda_launched) = ruda_count_spread_with_total(client, working_rudas);
    let unit_idle =
        !working_units.is_multiple_of(num_units_in_ruda as usize) || ruda_launched != working_rudas;

    let unit_idle = match unit_idle {
        true => IdleMode::Terminate,
        false => IdleMode::None,
    };
    let blueprint = ReduceBlueprint {
        vectorization_mode: settings.vectorization_mode,
        global: GlobalReduceBlueprint::Unit(UnitReduceBlueprint { unit_idle }),
    };

    Ok((blueprint, ruda_dim, ruda_count))
}

fn working_units(settings: &ReduceVectorSettings, problem: &ReduceProblem) -> usize {
    match settings.vectorization_mode {
        VectorizationMode::Parallel => problem.reduce_count / settings.vector_size_output,
        VectorizationMode::Perpendicular => problem.reduce_count / settings.vector_size_input,
    }
}
