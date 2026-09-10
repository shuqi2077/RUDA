use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;

use crate::{
    RandomFamily, lcg_step, taus_step_0, taus_step_1, taus_step_2, to_unit_interval_closed_open,
};

use super::{PrngArgs, PrngRuntime, random};

#[derive(RudaLaunch, RudaType)]
pub(crate) struct Uniform {
    lower_bound: f32,
    upper_bound: f32,
}

#[derive(Debug)]
struct UniformFamily;

impl RandomFamily for UniformFamily {
    type Runtime = Uniform;
}

impl PrngArgs for Uniform {
    type Args = Self;

    fn args<R: Runtime>(self) -> UniformLaunch<R> {
        UniformLaunch::new(self.lower_bound, self.upper_bound)
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: Runtime>(
    client: &ComputeClient<R>,
    lower_bound: f32,
    upper_bound: f32,
    out: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    random::<UniformFamily, R>(
        client,
        Uniform {
            lower_bound,
            upper_bound,
        },
        out,
        dtype,
    )
}

mod kernel;
