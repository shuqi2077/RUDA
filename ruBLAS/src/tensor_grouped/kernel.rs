use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn grouped_nt<F: Float>(
    input: &Array<F>,
    weights: &Array<F>,
    row_experts: &Array<u32>,
    output: &mut Array<F>,
    experts: u32,
    columns: u32,
    inner: u32,
    #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= output.len() {
        terminate!();
    }
    let row = position / columns as usize;
    let column = position % columns as usize;
    let expert = row_experts[row];
    let mut sum = 0.0f32;
    if expert < experts {
        let mut i = 0usize;
        while i < inner as usize {
            let a = f32::cast_from(input[row * inner as usize + i]);
            let b = f32::cast_from(
                weights[(expert as usize * columns as usize + column) * inner as usize + i],
            );
            sum += a * b;
            i += 1;
        }
    }
    output[position] = F::cast_from(sum);
}

/// dInput for Y[m,n] = sum_k X[m,k] * W[expert(m),n,k].
#[ruda(launch)]
pub(super) fn grouped_nt_dinput<F: Float>(
    grad: &Array<F>, weights: &Array<F>, row_experts: &Array<u32>, dinput: &mut Array<F>,
    experts: u32, columns: u32, inner: u32, #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= dinput.len() { terminate!(); }
    let row = position / inner as usize;
    let k = position % inner as usize;
    let expert = row_experts[row];
    let mut sum = 0.0f32;
    if expert < experts {
        let mut n = 0usize;
        while n < columns as usize {
            sum = fma(f32::cast_from(grad[row * columns as usize + n]),
                      f32::cast_from(weights[(expert as usize * columns as usize + n) * inner as usize + k]), sum);
            n += 1;
        }
    }
    dinput[position] = F::cast_from(sum);
}

/// Deterministic FP32 dWeight for segmented grouped GEMM. One output element
/// owns one (expert,n,k) and reduces only that expert's contiguous token range,
/// avoiding atomics and preserving FP32 accumulation for low-precision weights.
#[ruda(launch)]
pub(super) fn grouped_nt_dweight<F: Float>(
    input: &Array<F>, grad: &Array<F>, offsets: &Array<u32>, dweight: &mut Array<f32>,
    experts: u32, columns: u32, inner: u32, #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= dweight.len() { terminate!(); }
    let k = position % inner as usize;
    let q = position / inner as usize;
    let n = q % columns as usize;
    let expert = q / columns as usize;
    if expert >= experts as usize { terminate!(); }
    let begin = offsets[expert] as usize;
    let end = offsets[expert + 1] as usize;
    let mut sum = 0.0f32;
    let mut row = begin;
    while row < end {
        sum = fma(f32::cast_from(grad[row * columns as usize + n]),
                  f32::cast_from(input[row * inner as usize + k]), sum);
        row += 1;
    }
    dweight[position] = sum;
}
