use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::prelude::TensorBinding;
use ruda_kernel::tiling::InputBinding;

use crate::kernel_ir::{
    definition::{MatmulElems, MatmulSetupError},
    launch::Strategy,
};

#[allow(clippy::result_large_err)]
/// Launches a matrix multiplication kernel..
///
/// # Notes
///
/// The matmul elements may get changed during selection for improved performance when
/// the hardware supports it.
/// Only the inner element types may change such as the stage or register element types.
pub fn launch_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    strategy.launch_ref(client, lhs, rhs, out, dtypes)
}
