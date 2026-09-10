use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn swiglu<F: Float>(
    gate: &mut Array<F>,
    up: &Array<F>,
    #[define(F)] _dtype: StorageType,
) {
    let position = ABSOLUTE_POS;
    if position >= gate.len() {
        terminate!();
    }
    let value = f32::cast_from(gate[position]);
    let silu = F::cast_from(value / (1.0f32 + f32::exp(-value)));
    gate[position] = F::cast_from(f32::cast_from(silu) * f32::cast_from(up[position]));
}
