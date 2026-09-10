use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn awq_gemm<F: Float>(
    input: &Array<F>,
    qweight: &Array<i32>,
    qzeros: &Array<i32>,
    scales: &Array<F>,
    bias: &Array<F>,
    output: &mut Array<F>,
    input_features: u32,
    output_features: u32,
    group_size: u32,
    has_bias: u32,
    #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= output.len() {
        terminate!();
    }
    let k = input_features as usize;
    let n = output_features as usize;
    let row = position / n;
    let column = position % n;
    let packed_columns = n / 8;
    let lane = column % 8;
    let shift = ((lane % 2) * 4 + lane / 2) as u32 * 4;
    let mut sum = 0.0f32;
    let mut inner = 0usize;
    while inner < k {
        let group = inner / group_size as usize;
        let word = u32::cast_from(qweight[inner * packed_columns + column / 8]);
        let zero_word = u32::cast_from(qzeros[group * packed_columns + column / 8]);
        let quant = (word >> shift) & 15u32;
        let zero = (zero_word >> shift) & 15u32;
        let weight = F::cast_from(
            (f32::cast_from(quant) - f32::cast_from(zero))
                * f32::cast_from(scales[group * n + column]),
        );
        sum += f32::cast_from(input[row * k + inner]) * f32::cast_from(weight);
        inner += 1;
    }
    let mut value = F::cast_from(sum);
    if has_bias != 0 {
        value += bias[column];
    }
    output[position] = value;
}
