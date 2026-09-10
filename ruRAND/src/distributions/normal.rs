use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use std::f32::consts::PI;

use super::{PrngArgs, PrngRuntime, random};

use crate::{RandomFamily, lcg_step, taus_step_0, taus_step_1, taus_step_2, to_unit_interval_open};

#[derive(RudaLaunch, RudaType)]
pub(crate) struct Normal {
    mean: f32,
    std: f32,
}

#[derive(Debug)]
struct NormalFamily;

impl RandomFamily for NormalFamily {
    type Runtime = Normal;
}

impl PrngArgs for Normal {
    type Args = Self;

    fn args<R: Runtime>(self) -> NormalLaunch<R> {
        NormalLaunch::new(self.mean, self.std)
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: Runtime>(
    client: &ComputeClient<R>,
    mean: f32,
    std: f32,
    out: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    random::<NormalFamily, R>(client, Normal { mean, std }, out, dtype)
}

mod kernel;
