//! Storage-aware pointwise dispatch. No tensor-wide promotion allocations.
use super::{bf16, f16, client, finish_dispatch, kernels, CudaRuntime, View,
    LAUNCHES, DIRECT_POINTWISE_CALLS, Ordering};
use ruda_kernel::dsl::prelude::*;

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View, scalar: f32) {
    if out.len == 0 { return; }
    let client = client();
    // ABSOLUTE_POS is a u32 in these compatibility kernels.
    assert!(out.len <= u32::MAX as usize, "native pointwise/scalar grid exceeds 32-bit indexing");
    let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
    macro_rules! run {
        ($a:ty, $b:ty, $o:ty) => {
            kernels::pointwise::launch::<$a, $b, $o, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                a.arg(), b.arg(), out.arg(), scalar, op)
        };
    }
    unsafe {
        match (a.dtype, b.dtype, out.dtype) {
            (0, 0, 0) => run!(f32, f32, f32),
            (0, 0, 1) => run!(f32, f32, f16),
            (0, 0, 2) => run!(f32, f32, bf16),
            (0, 1, 0) => run!(f32, f16, f32),
            (0, 1, 1) => run!(f32, f16, f16),
            (0, 1, 2) => run!(f32, f16, bf16),
            (0, 2, 0) => run!(f32, bf16, f32),
            (0, 2, 1) => run!(f32, bf16, f16),
            (0, 2, 2) => run!(f32, bf16, bf16),
            (1, 0, 0) => run!(f16, f32, f32),
            (1, 0, 1) => run!(f16, f32, f16),
            (1, 0, 2) => run!(f16, f32, bf16),
            (1, 1, 0) => run!(f16, f16, f32),
            (1, 1, 1) => run!(f16, f16, f16),
            (1, 1, 2) => run!(f16, f16, bf16),
            (1, 2, 0) => run!(f16, bf16, f32),
            (1, 2, 1) => run!(f16, bf16, f16),
            (1, 2, 2) => run!(f16, bf16, bf16),
            (2, 0, 0) => run!(bf16, f32, f32),
            (2, 0, 1) => run!(bf16, f32, f16),
            (2, 0, 2) => run!(bf16, f32, bf16),
            (2, 1, 0) => run!(bf16, f16, f32),
            (2, 1, 1) => run!(bf16, f16, f16),
            (2, 1, 2) => run!(bf16, f16, bf16),
            (2, 2, 0) => run!(bf16, bf16, f32),
            (2, 2, 1) => run!(bf16, bf16, f16),
            (2, 2, 2) => run!(bf16, bf16, bf16),
            _ => panic!("unsupported RUDA pointwise dtype combination"),
        }
    }
    // Keep the existing synchronous bridge contract until streams and
    // allocation lifetime tracking are connected end-to-end.
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    DIRECT_POINTWISE_CALLS.fetch_add(1, Ordering::Relaxed);
}
