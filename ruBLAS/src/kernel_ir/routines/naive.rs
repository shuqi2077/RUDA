use ruda_kernel::dsl as kernel_dsl;
use std::fmt::Display;

use ruda_kernel::tiling::ruda_count::RudaCountPlan;

use crate::kernel_ir::{
    components::batch::{
        BatchMatmulFamily,
        naive::{NaiveBatchMatmulFamily, NaiveBlueprint},
    },
    definition::{MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine},
};

pub struct NaiveRoutine {}

#[derive(Default, Clone)]
pub struct NaiveStrategy {}

impl Display for NaiveStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for NaiveStrategy {
    fn from(_value: ()) -> Self {
        Self {}
    }
}

impl Routine<()> for NaiveRoutine {
    type Strategy = NaiveStrategy;
    type BatchMatmul = NaiveBatchMatmulFamily;
    type Blueprint = <Self::BatchMatmul as BatchMatmulFamily<()>>::Blueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<()>>::Config;

    fn expand_blueprint<R: ruda_kernel::dsl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        _strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let blueprint = NaiveBlueprint {
            vector_size_out: device_settings.vector_sizes.out as u32,
            dtypes: dtypes.clone(),
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: ruda_kernel::dsl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let ruda_dim = Self::BatchMatmul::rudadim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?
        .to_ruda_dim(device_settings.plane_dim)?;

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            ruda_dim,
            ruda_count_plan: simple_ruda_count(
                &problem.lhs_shape,
                &problem.rhs_shape,
                &problem.out_shape,
                ruda_dim.x,
                ruda_dim.y,
            )?,
            address_type: problem.address_type,
            vector_sizes: device_settings.vector_sizes,
        })
    }
}

#[allow(clippy::result_large_err)]
fn simple_ruda_count(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    ruda_dim_x: u32,
    ruda_dim_y: u32,
) -> Result<RudaCountPlan, MatmulSetupError> {
    let ndims = lhs_shape.len();
    let m = lhs_shape[ndims - 2];
    let n = rhs_shape[ndims - 1];

    let m_rudas = f32::ceil(m as f32 / ruda_dim_x as f32) as u32;
    let n_rudas = f32::ceil(n as f32 / ruda_dim_y as f32) as u32;
    let mut batch_rudas = 1u32;

    #[allow(clippy::needless_range_loop)]
    for i in 0..ndims - 2 {
        batch_rudas *= output_shape[i] as u32;
    }

    let ruda_count_plan = RudaCountPlan::new_from_problem((m_rudas, n_rudas, batch_rudas).into());
    let max_ruda_count = u16::MAX as u32;

    if m_rudas > max_ruda_count || n_rudas > max_ruda_count || batch_rudas > max_ruda_count {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::RudaCountTooBig(ruda_count_plan.resolve()),
        ));
    }

    Ok(ruda_count_plan)
}
