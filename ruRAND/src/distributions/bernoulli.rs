use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::dsl::RudaType;
use ruda_kernel::dsl::Runtime;

use crate::RandomFamily;

use super::{
    PrngArgs, PrngRuntime, lcg_step, random, taus_step_0, taus_step_1, taus_step_2,
    to_unit_interval_closed_open,
};

#[derive(RudaLaunch, RudaType)]
pub(crate) struct Bernoulli {
    probability: f32,
}

#[derive(Debug)]
struct BernoulliFamily;

impl RandomFamily for BernoulliFamily {
    type Runtime = Bernoulli;
}

impl PrngArgs for Bernoulli {
    type Args = Self;

    fn args<R: Runtime>(self) -> BernoulliLaunch<R> {
        BernoulliLaunch::new(self.probability)
    }
}

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: Runtime>(
    client: &ComputeClient<R>,
    probability: f32,
    out: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    random::<BernoulliFamily, R>(client, Bernoulli { probability }, out, dtype)
}

mod kernel;
