//! Native PyTorch -> existing ruBLAS bridge. No CPU or CUDA-C++ fallback.
use super::{bf16, f16, client, finish_dispatch, kernels, primitives, CudaRuntime, View,
    LAUNCHES, RUBLAS_CALLS, SCALAR_MATMUL_CALLS, ADDMM_EPILOGUES,
    ADDMM_WORKSPACE_BYTES, Ordering};
use ruda_kernel::dsl::prelude::*;
use rublas::kernel_ir::definition::MatmulSetupError;
use rublas::tensor_matmul::{matmul_with_precision, F32MathMode, MatmulStrategy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy { Auto, Rublas, Scalar }

fn parse_policy(value: &str) -> Result<Policy, &'static str> {
    match value {
        "auto" => Ok(Policy::Auto),
        "rublas" => Ok(Policy::Rublas),
        "scalar" => Ok(Policy::Scalar),
        _ => Err("RUDA_TORCH_MATMUL must be auto, rublas, or scalar"),
    }
}

fn policy() -> Policy {
    match std::env::var("RUDA_TORCH_MATMUL") {
        Ok(value) => parse_policy(&value).expect("invalid RUDA matmul policy"),
        Err(std::env::VarError::NotPresent) => Policy::Auto,
        Err(_) => panic!("RUDA_TORCH_MATMUL must be Unicode"),
    }
}

fn validate(op: u32, a: &View, b: &View, out: &View) {
    assert!(op == 7 || op == 30);
    assert!(a.dtype <= 2 && a.dtype == b.dtype, "matmul inputs must have matching floating dtypes");
    assert!(out.dtype == a.dtype || out.dtype == 0, "unsupported matmul output dtype");
    let rank = if op == 30 { 3 } else { 2 };
    assert_eq!(a.shape.len(), rank);
    assert_eq!(b.shape.len(), rank);
    assert_eq!(out.shape.len(), rank);
    assert_eq!(a.shape[rank - 1], b.shape[rank - 2]);
    assert_eq!(out.shape[rank - 2], a.shape[rank - 2]);
    assert_eq!(out.shape[rank - 1], b.shape[rank - 1]);
    if op == 30 {
        assert_eq!(a.shape[0], b.shape[0]);
        assert_eq!(out.shape[0], a.shape[0]);
    }
}

pub(super) fn launch(op: u32, a: &View, b: &View, out: &View) {
    validate(op, a, b, out);
    if out.len == 0 { return; }
    let selected = policy();
    // Empty K has a well-defined zero result. Bypass optimized tiling code
    // that requires a non-empty reduction, even in strict-rublas mode.
    if a.shape[a.shape.len() - 1] == 0 || selected == Policy::Scalar {
        scalar(op, a, b, out);
        return;
    }
    let output = primitives::tensor(out);
    let dtype = output.dtype;
    let result = matmul_with_precision(
        primitives::tensor(a), primitives::tensor(b), Some(output),
        MatmulStrategy::Ruda, dtype, F32MathMode::Strict,
    );
    match result {
        Ok(result) => {
            // An explicit output binding lets ruBLAS write straight into
            // PyTorch storage; do not call primitives::store (extra copy).
            finish_dispatch(&client());
            drop(result);
            RUBLAS_CALLS.fetch_add(1, Ordering::Relaxed);
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            // Only setup rejection can select a compatibility GPU kernel.
            // PTX compilation, launch, and synchronization failures surface.
            let setup_only = matches!(&error,
                MatmulSetupError::Unavailable(_) |
                MatmulSetupError::InvalidConfig(_) |
                MatmulSetupError::Vectorization(_));
            if selected == Policy::Auto && setup_only {
                scalar(op, a, b, out);
            } else {
                panic!("RUDA ruBLAS matmul failed: {error}");
            }
        }
    }
}

fn scalar(op: u32, a: &View, b: &View, out: &View) {
    let client = client();
    // ABSOLUTE_POS is a u32 in these compatibility kernels.
    assert!(out.len <= u32::MAX as usize, "native pointwise/scalar grid exceeds 32-bit indexing");
    let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
    macro_rules! run {
        ($i:ty, $o:ty) => { kernels::matmul_storage::launch::<$i, $o, CudaRuntime>(
            &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
            a.arg(), b.arg(), out.arg(), op == 30) };
    }
    unsafe {
        match (a.dtype, out.dtype) {
            (0, 0) => run!(f32, f32),
            (1, 1) => run!(f16, f16), (1, 0) => run!(f16, f32),
            (2, 2) => run!(bf16, bf16), (2, 0) => run!(bf16, f32),
            _ => panic!("unsupported RUDA scalar matmul dtype combination"),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    SCALAR_MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub(super) fn addmm(bias: &View, a: &View, b: &View, out: &View, alpha: f32, beta: f32) {
    validate(7, a, b, out);
    assert_eq!(a.dtype, out.dtype);
    assert_eq!(bias.dtype, out.dtype);
    assert_eq!(bias.shape, out.shape);
    if out.len == 0 { return; }
    if alpha == 0.0 || a.shape[1] == 0 {
        bias_only(bias, out, beta);
        return;
    }
    if alpha == 1.0 && beta == 0.0 {
        launch(7, a, b, out);
        return;
    }
    let client = client();
    // F32 can use its own output for accumulation. For low precision, only
    // M*N values are held in F32; inputs/weights/bias are never promoted.
    let scratch = if out.dtype != 0 {
        let bytes = out.len.checked_mul(4).expect("addmm workspace overflow");
        ADDMM_WORKSPACE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        Some(View::packed(client.empty(bytes), out.shape.clone(), 0))
    } else { None };
    let accumulator = scratch.as_ref().unwrap_or(out);
    launch(7, a, b, accumulator);
    // ABSOLUTE_POS is a u32 in these compatibility kernels.
    assert!(out.len <= u32::MAX as usize, "native pointwise/scalar grid exceeds 32-bit indexing");
    let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
    macro_rules! run {
        ($dtype:ty) => { kernels::addmm_epilogue::launch::<$dtype, CudaRuntime>(
            &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
            accumulator.arg(), bias.arg(), out.arg(), alpha, beta, true, beta != 0.0) };
    }
    unsafe {
        match out.dtype {
            0 => run!(f32), 1 => run!(f16), 2 => run!(bf16),
            _ => panic!("unsupported RUDA addmm dtype"),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    ADDMM_EPILOGUES.fetch_add(1, Ordering::Relaxed);
}

// No product is formed when alpha=0 or K=0. In particular a NaN/Inf in an
// ignored matrix cannot leak into the result, and there is no scratch buffer.
fn bias_only(bias: &View, out: &View, beta: f32) {
    let client = client();
    // ABSOLUTE_POS is a u32 in these compatibility kernels.
    assert!(out.len <= u32::MAX as usize, "native pointwise/scalar grid exceeds 32-bit indexing");
    let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
    macro_rules! run {
        ($dtype:ty) => { kernels::addmm_bias::launch::<$dtype, CudaRuntime>(
            &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
            bias.arg(), out.arg(), beta, beta != 0.0) };
    }
    unsafe {
        match out.dtype {
            0 => run!(f32), 1 => run!(f16), 2 => run!(bf16),
            _ => panic!("unsupported RUDA addmm dtype"),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    ADDMM_EPILOGUES.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_is_explicit() {
        assert_eq!(parse_policy("auto"), Ok(Policy::Auto));
        assert_eq!(parse_policy("rublas"), Ok(Policy::Rublas));
        assert_eq!(parse_policy("scalar"), Ok(Policy::Scalar));
        for invalid in ["", "cpu", "nvrtc", "AUTO", "auto "] {
            assert!(parse_policy(invalid).is_err());
        }
    }
}
