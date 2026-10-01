use gpu::prelude::*;

#[ruda(launch)]
pub fn add(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

pub fn launch<R: Runtime>(
    client: &gpu::runtime::client::ComputeClient<R>,
    a: gpu::runtime::server::Handle,
    b: gpu::runtime::server::Handle,
    output: gpu::runtime::server::Handle,
) {
    // SAFETY: The buffers are checked to hold four f32 values before dispatch.
    assert!(a.size_in_used() >= 16 && b.size_in_used() >= 16 && output.size_in_used() >= 16);
    unsafe {
        add::launch::<R>(
            client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(32),
            ArrayArg::from_raw_parts(a, 4),
            ArrayArg::from_raw_parts(b, 4),
            ArrayArg::from_raw_parts(output, 4),
        );
    }
}

#[cfg(feature = "cuda")]
pub use gpu::cuda::{CudaDevice, CudaRuntime};
#[cfg(feature = "hip")]
pub use gpu::hip::{AmdDevice, HipRuntime};
#[cfg(feature = "wgpu")]
pub use gpu::wgpu::{WgpuDevice, WgpuRuntime};
