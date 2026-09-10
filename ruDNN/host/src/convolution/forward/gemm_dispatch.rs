use super::*;

// ============================================================================
// gemm implementations
// ============================================================================
//
// One thin wrapper per element type around `gemm::gemm` with the strides and
// parallelism heuristic fixed to what conv3d_impl needs. The wrappers share
// every line of logic aside from the numeric type, zero, and one literals,
// so they're generated via a macro.

macro_rules! gemm_typed {
    ($fn_name:ident, $T:ty, $zero:expr, $one:expr) => {
        pub(super) fn $fn_name(a: &[$T], b: &[$T], m: usize, k: usize, n: usize) -> Vec<$T> {
            let mut c = vec![$zero; m * n];
            #[cfg(feature = "rayon")]
            let parallelism = if m * n * k >= 192 * 192 * 192 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            #[cfg(not(feature = "rayon"))]
            let parallelism = gemm::Parallelism::None;
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    1,
                    n as isize,
                    false,
                    a.as_ptr(),
                    1,
                    k as isize,
                    b.as_ptr(),
                    k as isize,
                    1,
                    $zero,
                    $one,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
            c
        }
    };
}

gemm_typed!(gemm_f32, f32, 0.0f32, 1.0f32);
gemm_typed!(gemm_f64, f64, 0.0f64, 1.0f64);
gemm_typed!(gemm_f16, f16, f16::from_f32(0.0), f16::from_f32(1.0));

