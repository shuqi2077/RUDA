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
