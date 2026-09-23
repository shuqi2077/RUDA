//! Fused native normalization paths for PyTorch inference.
use super::{bf16, f16, client, finish_dispatch, kernels, CudaRuntime, View, LAUNCHES,
    FUSED_LAYER_NORM_CALLS, FUSED_RMS_NORM_CALLS, Ordering};
use ruda_kernel::dsl::prelude::*;

fn packed(view: &View) -> bool {
    let mut expected = 1usize;
    for (&dim, &stride) in view.shape.iter().zip(&view.strides).rev() {
        if dim > 1 && stride != expected { return false; }
        expected = expected.checked_mul(dim).expect("normalization stride overflow");
    }
    true
}

pub(super) fn layer_norm(
    input: &View, weight: Option<&View>, bias: Option<&View>,
    out: &View, mean: &View, rstd: &View, epsilon: f32,
) {
    assert!(input.dtype <= 2 && input.dtype == out.dtype);
    assert_eq!(input.shape, out.shape);
    assert!(epsilon.is_finite() && epsilon > 0.0);
    assert!(!input.shape.is_empty());
    let width = *input.shape.last().unwrap();
    assert!(width > 0);
    assert!(packed(input) && packed(out) && packed(mean) && packed(rstd),
        "native LayerNorm requires contiguous tensors");
    let rows = input.len / width;
    assert_eq!(mean.len, rows);
    assert_eq!(rstd.len, rows);
    assert_eq!(mean.dtype, input.dtype);
    assert_eq!(rstd.dtype, input.dtype);
    for affine in [weight, bias].into_iter().flatten() {
        assert_eq!(affine.dtype, input.dtype);
        assert_eq!(affine.shape.as_slice(), &[width]);
        assert_eq!(affine.strides.as_slice(), &[1]);
    }
    if rows == 0 { return; }
    let client = client();
    let count = u32::try_from(rows.checked_mul(32).expect("LayerNorm launch overflow").div_ceil(128))
        .expect("LayerNorm launch grid overflow");
    let dummy = weight.or(bias).unwrap_or(input);
    macro_rules! run {
        ($dtype:ty) => { kernels::layer_norm_warp::launch::<$dtype, CudaRuntime>(
            &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
            input.arg(), weight.unwrap_or(dummy).arg(), bias.unwrap_or(dummy).arg(),
            out.arg(), mean.arg(), rstd.arg(), epsilon, weight.is_some(), bias.is_some()) };
    }
    unsafe {
        match input.dtype {
            0 => run!(f32), 1 => run!(f16), 2 => run!(bf16),
            _ => unreachable!(),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    FUSED_LAYER_NORM_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub(super) fn rms_norm(input: &View, weight: Option<&View>, out: &View, epsilon: f32) {
    assert!(input.dtype <= 2 && input.dtype == out.dtype);
    assert_eq!(input.shape, out.shape);
    assert!(epsilon.is_finite() && epsilon > 0.0);
    assert!(!input.shape.is_empty());
    let width = *input.shape.last().unwrap();
    assert!(width > 0);
    assert!(packed(input) && packed(out), "native RMSNorm requires contiguous tensors");
    if let Some(weight) = weight {
        assert_eq!(weight.dtype, input.dtype);
        assert_eq!(weight.shape.as_slice(), &[width]);
        assert_eq!(weight.strides.as_slice(), &[1]);
    }
    let rows = input.len / width;
    if rows == 0 { return; }
    let client = client();
    let count = u32::try_from(rows.checked_mul(32).expect("RMSNorm launch overflow").div_ceil(128))
        .expect("RMSNorm launch grid overflow");
    let dummy = weight.unwrap_or(input);
    macro_rules! run {
        ($dtype:ty) => { kernels::rms_norm_warp::launch::<$dtype, CudaRuntime>(
            &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
            input.arg(), weight.unwrap_or(dummy).arg(), out.arg(), epsilon, weight.is_some()) };
    }
    unsafe {
        match input.dtype {
            0 => run!(f32), 1 => run!(f16), 2 => run!(bf16),
            _ => unreachable!(),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    FUSED_RMS_NORM_CALLS.fetch_add(1, Ordering::Relaxed);
}
