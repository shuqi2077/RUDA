//! First-order training kernels. All tensor arithmetic stays on the RUDA device.
//! Inputs/outputs are contiguous; host validation enforces shapes and aliases.
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub fn rms_forward<F: Float + RudaElement, W: Float + RudaElement>(
    x: &Tensor<F>, weight: &Tensor<W>, y: &mut Tensor<F>, rstd: &mut Tensor<f32>,
    epsilon: f32, #[comptime] weighted: bool,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let width = x.shape(x.rank() - 1);
    if row < rstd.len() {
        let mut local = 0.0f32;
        let mut col = lane;
        while col < width {
            let a = f32::cast_from(x[row * width + col]);
            local += a * a;
            col += 32;
        }
        let inv = (plane_sum(local) / width as f32 + epsilon).inverse_sqrt();
        if lane == 0 { rstd[row] = inv; }
        let mut col = lane;
        while col < width {
            let mut value = f32::cast_from(x[row * width + col]) * inv;
            if comptime!(weighted) { value *= f32::cast_from(weight[col]); }
            y[row * width + col] = F::cast_from(value);
            col += 32;
        }
    }
}

#[ruda(launch)]
pub fn rms_dx<F: Float + RudaElement, W: Float + RudaElement>(
    x: &Tensor<F>, weight: &Tensor<W>, dy: &Tensor<F>, rstd: &Tensor<f32>,
    dx: &mut Tensor<F>, #[comptime] weighted: bool,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let width = x.shape(x.rank() - 1);
    if row < rstd.len() {
        let inv = rstd[row];
        let mut dot = 0.0f32;
        let mut col = lane;
        while col < width {
            let pos = row * width + col;
            let mut g = f32::cast_from(dy[pos]);
            if comptime!(weighted) { g *= f32::cast_from(weight[col]); }
            dot += g * f32::cast_from(x[pos]) * inv;
            col += 32;
        }
        let correction = plane_sum(dot) / width as f32;
        let mut col = lane;
        while col < width {
            let pos = row * width + col;
            let mut g = f32::cast_from(dy[pos]);
            if comptime!(weighted) { g *= f32::cast_from(weight[col]); }
            dx[pos] = F::cast_from(inv * (g - f32::cast_from(x[pos]) * inv * correction));
            col += 32;
        }
    }
}

// Independent FP32 partials, followed by a deterministic merge. No floating
// atomic additions and no full-size normalized activation/gradient temporary.
#[ruda(launch)]
pub fn rms_dw_partial<F: Float + RudaElement>(
    x: &Tensor<F>, dy: &Tensor<F>, rstd: &Tensor<f32>, partial: &mut Tensor<f32>,
) {
    let pos = ABSOLUTE_POS as usize;
    let width = partial.shape(1);
    let parts = partial.shape(0);
    if pos < partial.len() {
        let col = pos % width;
        let mut row = pos / width;
        let mut sum = 0.0f32;
        while row < rstd.len() {
            sum += f32::cast_from(dy[row * width + col])
                * f32::cast_from(x[row * width + col]) * rstd[row];
            row += parts;
        }
        partial[pos] = sum;
    }
}

#[ruda(launch)]
pub fn rms_dw_merge<W: Float + RudaElement>(partial: &Tensor<f32>, dw: &mut Tensor<W>) {
    let col = ABSOLUTE_POS as usize;
    if col < dw.len() {
        let mut sum = 0.0f32;
        for part in 0..partial.shape(0) { sum += partial[part * dw.len() + col]; }
        dw[col] = W::cast_from(sum);
    }
}

#[ruda(launch)]
pub fn silu_mul_forward<F: Float + RudaElement>(
    gate: &Tensor<F>, up: &Tensor<F>, y: &mut Tensor<F>,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < y.len() {
        let x = f32::cast_from(gate[pos]);
        let activation = F::cast_from(x / (1.0f32 + (-x).exp()));
        y[pos] = F::cast_from(f32::cast_from(activation) * f32::cast_from(up[pos]));
    }
}

#[ruda(launch)]
pub fn silu_mul_backward<F: Float + RudaElement>(
    gate: &Tensor<F>, up: &Tensor<F>, dy: &Tensor<F>, dg: &mut Tensor<F>, du: &mut Tensor<F>,
    #[comptime] need_gate: bool, #[comptime] need_up: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < gate.len() {
        let x = f32::cast_from(gate[pos]);
        let sigmoid = 1.0f32 / (1.0f32 + (-x).exp());
        let grad = f32::cast_from(dy[pos]);
        if comptime!(need_gate) {
            // Preserve the mul-backward storage rounding before SiLU backward.
            let intermediate = F::cast_from(grad * f32::cast_from(up[pos]));
            dg[pos] = F::cast_from(f32::cast_from(intermediate) * sigmoid * (1.0f32 + x * (1.0f32 - sigmoid)));
        }
        if comptime!(need_up) {
            let activation = F::cast_from(x / (1.0f32 + (-x).exp()));
            du[pos] = F::cast_from(grad * f32::cast_from(activation));
        }
    }
}



// Fused last-axis LayerNorm training path. Statistics remain FP32 even when
// activations/affine parameters use FP16/BF16 storage. One warp owns one row.
#[ruda(launch)]
pub fn layer_forward<F: Float + RudaElement, W: Float + RudaElement, B: Float + RudaElement>(
    x: &Tensor<F>, weight: &Tensor<W>, bias: &Tensor<B>, y: &mut Tensor<F>,
    mean: &mut Tensor<f32>, rstd: &mut Tensor<f32>, epsilon: f32,
    #[comptime] weighted: bool, #[comptime] biased: bool,
) {
    let row=(ABSOLUTE_POS/32) as usize;
    let lane=(ABSOLUTE_POS%32) as usize;
    let width=x.shape(x.rank()-1);
    if row < mean.len() {
        let base=row*width;
        let mut local=0.0f32;
        let mut col=lane;
        while col<width { local += f32::cast_from(x[base+col]); col+=32; }
        let mu=plane_sum(local)/width as f32;
        let mut var=0.0f32; col=lane;
        while col<width { let d=f32::cast_from(x[base+col])-mu; var += d*d; col+=32; }
        let inv=(plane_sum(var)/width as f32 + epsilon).inverse_sqrt();
        if lane==0 { mean[row]=mu; rstd[row]=inv; }
        col=lane;
        while col<width {
            let mut value=(f32::cast_from(x[base+col])-mu)*inv;
            if comptime!(weighted) { value *= f32::cast_from(weight[col]); }
            if comptime!(biased) { value += f32::cast_from(bias[col]); }
            y[base+col]=F::cast_from(value); col+=32;
        }
    }
}

#[ruda(launch)]
pub fn layer_dx<F: Float + RudaElement, W: Float + RudaElement>(
    x:&Tensor<F>, weight:&Tensor<W>, dy:&Tensor<F>, mean:&Tensor<f32>, rstd:&Tensor<f32>,
    dx:&mut Tensor<F>, #[comptime] weighted: bool,
) {
    let row=(ABSOLUTE_POS/32) as usize;
    let lane=(ABSOLUTE_POS%32) as usize;
    let width=x.shape(x.rank()-1);
    if row < mean.len() {
        let base=row*width; let mu=mean[row]; let inv=rstd[row];
        let mut sum_g=0.0f32; let mut sum_gx=0.0f32; let mut col=lane;
        while col<width {
            let pos=base+col; let xhat=(f32::cast_from(x[pos])-mu)*inv;
            let mut g=f32::cast_from(dy[pos]);
            if comptime!(weighted) { g *= f32::cast_from(weight[col]); }
            sum_g += g; sum_gx += g*xhat; col+=32;
        }
        let mean_g=plane_sum(sum_g)/width as f32;
        let mean_gx=plane_sum(sum_gx)/width as f32;
        col=lane;
        while col<width {
            let pos=base+col; let xhat=(f32::cast_from(x[pos])-mu)*inv;
            let mut g=f32::cast_from(dy[pos]);
            if comptime!(weighted) { g *= f32::cast_from(weight[col]); }
            dx[pos]=F::cast_from(inv*(g-mean_g-xhat*mean_gx)); col+=32;
        }
    }
}

// FP32 partials avoid atomics and avoid materializing normalized activations.
// Layout is [2, parts, width]: plane 0=dWeight, plane 1=dBias.
#[ruda(launch)]
pub fn layer_affine_partial<F: Float + RudaElement>(
    x:&Tensor<F>, dy:&Tensor<F>, mean:&Tensor<f32>, rstd:&Tensor<f32>,
    partial:&mut Tensor<f32>, #[comptime] need_weight: bool, #[comptime] need_bias: bool,
) {
    let pos=ABSOLUTE_POS as usize;
    let parts=partial.shape(1); let width=partial.shape(2);
    if pos < parts*width {
        let part=pos/width; let col=pos%width; let mut row=part;
        let mut dw=0.0f32; let mut db=0.0f32;
        while row<mean.len() {
            let i=row*width+col; let grad=f32::cast_from(dy[i]);
            if comptime!(need_weight) { dw += grad*(f32::cast_from(x[i])-mean[row])*rstd[row]; }
            if comptime!(need_bias) { db += grad; }
            row += parts;
        }
        if comptime!(need_weight) { partial[pos]=dw; }
        if comptime!(need_bias) { partial[parts*width+pos]=db; }
    }
}

#[ruda(launch)]
pub fn layer_affine_merge<W: Float + RudaElement, B: Float + RudaElement>(
    partial:&Tensor<f32>, dw:&mut Tensor<W>, db:&mut Tensor<B>,
    #[comptime] need_weight: bool, #[comptime] need_bias: bool,
) {
    let col=ABSOLUTE_POS as usize; let parts=partial.shape(1); let width=partial.shape(2);
    if col<width {
        let mut sw=0.0f32; let mut sb=0.0f32;
        for part in 0..parts {
            if comptime!(need_weight) { sw += partial[part*width+col]; }
            if comptime!(need_bias) { sb += partial[parts*width+part*width+col]; }
        }
        if comptime!(need_weight) { dw[col]=W::cast_from(sw); }
        if comptime!(need_bias) { db[col]=B::cast_from(sb); }
    }
}
