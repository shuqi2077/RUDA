//! Softmax and log-softmax with storage-aware F32 accumulation.
use super::{bf16, f16, client, finish_dispatch, kernels, CudaRuntime, View, LAUNCHES,
    WARP_SOFTMAX_CALLS, SCALAR_SOFTMAX_CALLS, Ordering};
use ruda_kernel::dsl::prelude::*;

fn use_warp(width: usize) -> bool {
    match std::env::var("RUDA_TORCH_SOFTMAX") {
        Ok(value) => match value.as_str() {
            "auto" => width >= 32,
            "warp" => true,
            "scalar" => false,
            _ => panic!("RUDA_TORCH_SOFTMAX must be auto, warp, or scalar"),
        },
        Err(std::env::VarError::NotPresent) => width >= 32,
        Err(_) => panic!("RUDA_TORCH_SOFTMAX must be Unicode"),
    }
}

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View, axis: usize) {
    assert!((31..=34).contains(&op));
    assert_eq!(a.dtype, b.dtype);
    assert_eq!(a.shape, out.shape);
    assert_eq!(a.shape, b.shape);
    assert!(axis < a.shape.len());
    if out.len == 0 { return; }
    let rows = out.len / a.shape[axis];
    let warp = use_warp(a.shape[axis]);
    let limit = if warp { u32::MAX as usize / 32 } else { u32::MAX as usize };
    assert!(rows <= limit, "softmax row grid exceeds 32-bit indexing");
    let blocks = if warp { rows.div_ceil(4) } else { rows.div_ceil(128) };
    let count = u32::try_from(blocks).expect("softmax launch grid overflow");
    let client = client();
    macro_rules! run {
        ($i:ty, $o:ty) => {
            if warp {
                kernels::softmax_warp::launch::<$i, $o, CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), axis, op == 32 || op == 34, op >= 33)
            } else {
                kernels::softmax::launch::<$i, $o, CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), axis, op == 32 || op == 34, op >= 33)
            }
        };
    }
    unsafe {
        match (a.dtype, out.dtype) {
            (0, 0) => run!(f32, f32),
            (0, 1) => run!(f32, f16),
            (0, 2) => run!(f32, bf16),
            (1, 0) => run!(f16, f32),
            (1, 1) => run!(f16, f16),
            (1, 2) => run!(f16, bf16),
            (2, 0) => run!(bf16, f32),
            (2, 1) => run!(bf16, f16),
            (2, 2) => run!(bf16, bf16),
            _ => panic!("unsupported RUDA softmax dtype combination"),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    if warp { WARP_SOFTMAX_CALLS.fetch_add(1, Ordering::Relaxed); }
    else { SCALAR_SOFTMAX_CALLS.fetch_add(1, Ordering::Relaxed); }
}
