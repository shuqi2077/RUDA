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

/// Training forward must retain gate/up for backward, but need not copy gate.
#[ruda(launch)]
pub(crate) fn swiglu_out<F: Float>(
    gate: &Array<F>, up: &Array<F>, out: &mut Array<F>,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let g = f32::cast_from(gate[i]);
        let silu = F::cast_from(g / (1.0f32 + f32::exp(-g)));
        out[i] = F::cast_from(f32::cast_from(silu) * f32::cast_from(up[i]));
    }
}

#[ruda(launch)]
pub(crate) fn swiglu_backward<F: Float>(
    gate: &Array<F>, up: &Array<F>, grad: &Array<F>,
    dgate: &mut Array<F>, dup: &mut Array<F>,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS;
    if i >= grad.len() { terminate!(); }
    let g = f32::cast_from(gate[i]);
    let u = f32::cast_from(up[i]);
    let dy = f32::cast_from(grad[i]);
    let sigmoid = 1.0f32 / (1.0f32 + f32::exp(-g));
    // Match forward's stored SiLU and mul-backward storage rounding. Keep
    // this contract aligned with ruda-torch's native silu_mul training kernel.
    let silu = F::cast_from(g / (1.0f32 + f32::exp(-g)));
    let intermediate = F::cast_from(dy * u);
    dgate[i] = F::cast_from(f32::cast_from(intermediate) * sigmoid * (1.0f32 + g * (1.0f32 - sigmoid)));
    dup[i] = F::cast_from(dy * f32::cast_from(silu));
}

#[ruda(launch)]
pub(crate) fn add<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>, #[define(F)] _dtype: StorageType) {
    let i = ABSOLUTE_POS;
    if i < out.len() { out[i] = F::cast_from(f32::cast_from(a[i]) + f32::cast_from(b[i])); }
}
