use ruda_kernel::dsl::prelude::*;

#[ruda]
fn offset<F: Numeric>(tensor: &Tensor<F>, position: usize) -> usize {
    let mut remaining = position;
    let mut offset = 0usize;
    let mut dim = tensor.rank();
    while dim > 0 {
        dim -= 1;
        offset += (remaining % tensor.shape(dim)) * tensor.stride(dim);
        remaining /= tensor.shape(dim);
    }
    offset
}

#[ruda(launch)]
pub fn convert<I: Numeric + RudaElement, O: Numeric + RudaElement>(a: &Tensor<I>, out: &mut Tensor<O>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        out[offset(out, pos)] = O::cast_from(a[offset(a, pos)]);
    }
}

#[ruda(launch)]
pub fn copy_bytes(a: &Tensor<u8>, out: &mut Tensor<u8>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() { out[offset(out, pos)] = a[offset(a, pos)]; }
}

#[ruda(launch)]
pub fn fill_bytes(out: &mut Tensor<u8>, low: u32, high: u32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let byte = pos % out.shape(out.rank() - 1);
        let mut word = low;
        if byte >= 4 { word = high; }
        out[offset(out, pos)] = u8::cast_from((word >> u32::cast_from((byte % 4) * 8)) & 255u32);
    }
}

#[ruda(launch)]
pub fn convert_integer(a: &Tensor<u8>, out: &mut Tensor<u8>,
    #[comptime] signed: bool, #[comptime] input_bool: bool, #[comptime] output_bool: bool) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let input_bytes = a.shape(a.rank() - 1);
        let output_bytes = out.shape(out.rank() - 1);
        let element = pos / output_bytes;
        let byte = pos % output_bytes;
        let mut value = 0u32;
        if comptime!(output_bool) {
            for i in 0..input_bytes {
                if u32::cast_from(a[offset(a, element * input_bytes + i)]) != 0u32 { value = 1; }
            }
        } else if comptime!(input_bool) {
            if byte == 0 && u32::cast_from(a[offset(a, element * input_bytes)]) != 0u32 { value = 1; }
        } else {
            if byte < input_bytes {
                value = u32::cast_from(a[offset(a, element * input_bytes + byte)]);
            } else if comptime!(signed) {
                let last = u32::cast_from(a[offset(a, element * input_bytes + input_bytes - 1)]);
                if (last & 128u32) != 0 { value = 255; }
            }
        }
        out[offset(out, pos)] = u8::cast_from(value);
    }
}

#[ruda(launch)]
pub fn float_to_bool<F: Float + RudaElement>(a: &Tensor<F>, out: &mut Tensor<u8>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut value: u32 = 0;
        if a[offset(a, pos)] != F::cast_from(0.0f32) { value = 1; }
        out[offset(out, pos)] = u8::cast_from(value);
    }
}

#[ruda(launch)]
pub fn bool_to_float<F: Float + RudaElement>(a: &Tensor<u8>, out: &mut Tensor<F>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut value: u32 = 0;
        if u32::cast_from(a[offset(a, pos)]) != 0u32 { value = 1; }
        out[offset(out, pos)] = F::cast_from(value);
    }
}

#[ruda(launch)]
pub fn compare_float<F: Float + RudaElement>(
    a: &Tensor<F>, b: &Tensor<F>, out: &mut Tensor<u8>, #[comptime] op: u32,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let x = a[offset(a, pos)];
        let y = b[offset(b, pos)];
        let mut matches = false;
        if comptime!(op == 78) { matches = x == y; }
        else if comptime!(op == 79) { matches = x != y; }
        else if comptime!(op == 80) { matches = x < y; }
        else if comptime!(op == 81) { matches = x < y || x == y; }
        else if comptime!(op == 82) { matches = x > y; }
        else if comptime!(op == 83) { matches = x > y || x == y; }
        let mut value: u32 = 0;
        if matches { value = 1; }
        out[offset(out, pos)] = u8::cast_from(value);
    }
}

#[ruda(launch)]
pub fn compare_integer(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>,
    #[comptime] signed: bool, #[comptime] boolean: bool, #[comptime] op: u32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let bytes = a.shape(a.rank() - 1);
        let mut equal = true;
        let mut less = false;
        let mut byte = bytes;
        while byte > 0 {
            byte -= 1;
            let mut x = u32::cast_from(a[offset(a, pos * bytes + byte)]);
            let mut y = u32::cast_from(b[offset(b, pos * bytes + byte)]);
            if comptime!(boolean) {
                if x != 0 { x = 1; }
                if y != 0 { y = 1; }
            }
            if comptime!(signed) {
                if byte == bytes - 1 { x = x ^ 128u32; y = y ^ 128u32; }
            }
            if equal && x != y { less = x < y; equal = false; }
        }
        let mut matches = false;
        if comptime!(op == 78) { matches = equal; }
        else if comptime!(op == 79) { matches = !equal; }
        else if comptime!(op == 80) { matches = less; }
        else if comptime!(op == 81) { matches = less || equal; }
        else if comptime!(op == 82) { matches = !less && !equal; }
        else if comptime!(op == 83) { matches = !less; }
        let mut value: u32 = 0;
        if matches { value = 1; }
        out[offset(out, pos)] = u8::cast_from(value);
    }
}

#[ruda(launch)]
pub fn select_bytes(condition: &Tensor<u8>, a: &Tensor<u8>, out: &mut Tensor<u8>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let element = pos / out.shape(out.rank() - 1);
        if u32::cast_from(condition[offset(condition, element)]) != 0u32 {
            out[offset(out, pos)] = a[offset(a, pos)];
        }
    }
}

#[ruda(launch)]
pub fn logical(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>, #[comptime] op: u32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let x = u32::cast_from(a[offset(a, pos)]) != 0u32;
        let y = u32::cast_from(b[offset(b, pos)]) != 0u32;
        let mut matches = false;
        if comptime!(op == 85) { matches = x && y; }
        else if comptime!(op == 86) { matches = x || y; }
        else if comptime!(op == 87) { matches = x != y; }
        else if comptime!(op == 88) { matches = !x; }
        let mut value: u32 = 0;
        if matches { value = 1; }
        out[offset(out, pos)] = u8::cast_from(value);
    }
}

#[ruda(launch)]
pub fn pointwise<A: Float + RudaElement, B: Float + RudaElement, O: Float + RudaElement>(
    a: &Tensor<A>, b: &Tensor<B>, out: &mut Tensor<O>, scalar: f32, #[comptime] op: u32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let target = offset(out, pos);
        if comptime!(op == 5) { out[target] = O::cast_from(scalar); }
        else {
            let x = f32::cast_from(a[offset(a, pos)]);
            if comptime!(op == 0) { out[target] = O::cast_from(x); }
            else if comptime!(op == 1) { out[target] = O::cast_from(x + scalar * f32::cast_from(b[offset(b, pos)])); }
            else if comptime!(op == 2) { out[target] = O::cast_from(x * f32::cast_from(b[offset(b, pos)])); }
            else if comptime!(op == 3) {
                let mut result = x;
                if x < 0.0 { result = 0.0; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 4) {
                let mut result = x;
                if f32::cast_from(b[offset(b, pos)]) <= scalar { result = 0.0; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 8) { out[target] = O::cast_from(x / f32::cast_from(b[offset(b, pos)])); }
            else if comptime!(op == 9) { out[target] = O::cast_from(x.exp()); }
            else if comptime!(op == 10) { out[target] = O::cast_from(x.ln()); }
            else if comptime!(op == 11) { out[target] = O::cast_from(x.sqrt()); }
            else if comptime!(op == 12) { out[target] = O::cast_from(1.0 / x.sqrt()); }
            else if comptime!(op == 13) { out[target] = O::cast_from(1.0 / (1.0 + (-x).exp())); }
            else if comptime!(op == 14) { out[target] = O::cast_from(x / (1.0 + (-x).exp())); }
            else if comptime!(op == 15) {
                let input = f32::cast_from(b[offset(b, pos)]);
                let sigmoid = 1.0f32 / (1.0f32 + (-input).exp());
                out[target] = O::cast_from(x * (sigmoid * (1.0f32 + input * (1.0f32 - sigmoid))));
            }
            else if comptime!(op == 16) {
                let y = f32::cast_from(b[offset(b, pos)]);
                out[target] = O::cast_from(x * y * (1.0f32 - y));
            }
            else if comptime!(op == 17) { out[target] = O::cast_from(x.tanh()); }
            else if comptime!(op == 18) {
                let y = f32::cast_from(b[offset(b, pos)]);
                out[target] = O::cast_from(x * (1.0f32 - y * y));
            }
            else if comptime!(op == 19) { out[target] = O::cast_from(x.sin()); }
            else if comptime!(op == 20) { out[target] = O::cast_from(x.cos()); }
            else if comptime!(op == 21) { out[target] = O::cast_from(x.abs()); }
            else if comptime!(op == 22) {
                let mut result = 0.0f32;
                if x > 0.0 { result = 1.0; }
                else if x < 0.0 { result = -1.0; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 23) { out[target] = O::cast_from(x.floor()); }
            else if comptime!(op == 24) { out[target] = O::cast_from(x.ceil()); }
            else if comptime!(op == 25) { out[target] = O::cast_from(x.trunc()); }
            else if comptime!(op == 26) { out[target] = O::cast_from(x.round()); }
            else if comptime!(op == 27) { out[target] = O::cast_from(x.recip()); }
            else if comptime!(op == 28) { out[target] = O::cast_from(f32::cast_from(out[target]) + scalar * x * f32::cast_from(b[offset(b, pos)])); }
            else if comptime!(op == 29) { out[target] = O::cast_from(f32::cast_from(out[target]) + scalar * x / f32::cast_from(b[offset(b, pos)])); }
            else if comptime!(op == 35) { out[target] = O::cast_from(x.log1p()); }
            else if comptime!(op == 36) { out[target] = O::cast_from(x.sinh()); }
            else if comptime!(op == 37) { out[target] = O::cast_from(x.cosh()); }
            else if comptime!(op == 38) { out[target] = O::cast_from(x.asinh()); }
            else if comptime!(op == 39) { out[target] = O::cast_from(x.acosh()); }
            else if comptime!(op == 40) { out[target] = O::cast_from(x.atanh()); }
            else if comptime!(op == 41) {
                let mut result = x * scalar;
                if x > 0.0 { result = x; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 42) {
                let input = f32::cast_from(b[offset(b, pos)]);
                let mut result = x * scalar;
                if input > 0.0 { result = x; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 43 || op == 45) {
                let mut clipped = x + 3.0f32;
                if clipped < 0.0 { clipped = 0.0; }
                if clipped > 6.0 { clipped = 6.0; }
                if comptime!(op == 43) { out[target] = O::cast_from(clipped / 6.0f32); }
                else { out[target] = O::cast_from(x * clipped / 6.0f32); }
            }
            else if comptime!(op == 44) {
                let input = f32::cast_from(b[offset(b, pos)]);
                let mut result = 0.0f32;
                if input > -3.0 && input < 3.0 { result = x * (1.0f32 / 6.0f32); }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 46) {
                let input = f32::cast_from(b[offset(b, pos)]);
                let mut result = x;
                if input <= -3.0 { result = 0.0; }
                else if input < 3.0 { result = x * (input / 3.0f32 + 0.5f32); }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 47) {
                let mut minimum = x;
                if x > 0.0 { minimum = 0.0; }
                out[target] = O::cast_from(minimum - (-x.abs()).exp().log1p());
            }
            else if comptime!(op == 48) {
                let input = f32::cast_from(b[offset(b, pos)]);
                let z = (-input.abs()).exp();
                let fraction = z / (1.0f32 + z);
                let mut derivative = fraction;
                if input < 0.0 { derivative = 1.0f32 - fraction; }
                out[target] = O::cast_from(x * derivative);
            }
            else if comptime!(op == 49) {
                let start = f32::cast_from(out[target]);
                let weight = f32::cast_from(b[offset(b, pos)]);
                let difference = x - start;
                let mut coefficient = weight;
                let mut base = start;
                if weight.abs() >= 0.5 {
                    coefficient = weight - 1.0f32;
                    base = x;
                }
                out[target] = O::cast_from(coefficient * difference + base);
            }
            else if comptime!(op == 50) { out[target] = O::cast_from(f32::cast_from(b[offset(b, pos)]) - scalar * x); }
            else if comptime!(op == 51) {
                let maximum = f32::cast_from(b[offset(b, pos)]);
                let mut result = x;
                if result < scalar || scalar != scalar { result = scalar; }
                if result > maximum || maximum != maximum { result = maximum; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 52) {
                let mut result = x;
                if f32::cast_from(b[offset(b, pos)]) >= scalar { result = 0.0; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 53) {
                let scaled = x * scalar;
                let threshold = f32::cast_from(b[offset(b, pos)]);
                let mut result = x;
                if !(scaled > threshold) { result = scaled.exp().log1p() / scalar; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 54) {
                let scaled = x * scalar;
                let threshold = f32::cast_from(b[offset(b, pos)]);
                let gradient = f32::cast_from(out[target]);
                let mut result = gradient;
                if !(scaled > threshold) {
                    let z = scaled.exp();
                    result = gradient * z / (z + 1.0f32);
                }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 55) {
                let difference = x - f32::cast_from(b[offset(b, pos)]);
                out[target] = O::cast_from(difference * difference);
            }
            else if comptime!(op == 56) {
                let difference = x - f32::cast_from(b[offset(b, pos)]);
                out[target] = O::cast_from(scalar * difference * f32::cast_from(out[target]));
            }
            else if comptime!(op == 57) { out[target] = O::cast_from((x / f32::cast_from(b[offset(b, pos)])).trunc()); }
            else if comptime!(op == 62) {
                let mut result = f32::cast_from(b[offset(b, pos)]) * x;
                if x > 0.0 { result = x; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 63) {
                let gradient = f32::cast_from(out[target]);
                let mut result = f32::cast_from(b[offset(b, pos)]) * gradient;
                if x > 0.0 { result = gradient; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 64) {
                let mut result = x * f32::cast_from(b[offset(b, pos)]);
                if x > 0.0 { result = 0.0; }
                out[target] = O::cast_from(result);
            }
            else if comptime!(op == 69) { out[target] = O::cast_from(x * x.exp().log1p().tanh()); }
            else if comptime!(op == 70) {
                let input = f32::cast_from(b[offset(b, pos)]);
                let sigmoid = 1.0f32 / (1.0f32 + (-input).exp());
                let activated = input.exp().log1p().tanh();
                out[target] = O::cast_from(x * (activated + input * sigmoid * (1.0f32 - activated * activated)));
            }
            else if comptime!(op == 71 || op == 72) {
                let gate = f32::cast_from(b[offset(b, pos)]);
                let sigmoid = 1.0f32 / (1.0f32 + (-gate).exp());
                if comptime!(op == 71) { out[target] = O::cast_from(x * sigmoid); }
                else { out[target] = O::cast_from((1.0f32 - sigmoid) * sigmoid * f32::cast_from(out[target]) * x); }
            }
        }
    }
}

#[ruda(launch)]
pub fn storage_pointwise<F: Float + RudaElement>(
    a: &Tensor<F>, b: &Tensor<F>, out: &mut Tensor<F>, scalar: f32, #[comptime] op: u32,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let target = offset(out, pos);
        let x = a[offset(a, pos)];
        let y = b[offset(b, pos)];
        let zero = F::cast_from(0.0f32);
        if comptime!(op == 58 || op == 60) {
            let mut result = x;
            if comptime!(op == 60) { result = out[target]; }
            if x >= -y && x <= y { result = zero; }
            out[target] = result;
        }
        else if comptime!(op == 59) {
            let mut result = zero;
            if x != x { result = x; }
            else if x > y { result = x - y; }
            else if x < -y { result = x + y; }
            out[target] = result;
        }
        else if comptime!(op == 61) {
            let mut result = x;
            if x <= y { result = out[target]; }
            out[target] = result;
        }
        else if comptime!(op == 73) {
            let mut result = out[target];
            if x <= y { result = zero; }
            out[target] = result;
        }
        else if comptime!(op == 65 || op == 66) {
            let z = f32::cast_from(x - y).abs();
            let boundary = f32::cast_from(F::cast_from(scalar));
            let half_boundary = f32::cast_from(F::cast_from(0.5f32) * F::cast_from(scalar));
            let mut result = 0.0f32;
            if comptime!(op == 65) {
                result = z - half_boundary;
                if z < boundary { result = 0.5f32 * z * z / boundary; }
            } else {
                result = boundary * (z - half_boundary);
                if z < boundary { result = 0.5f32 * z * z; }
            }
            out[target] = F::cast_from(result);
        }
        else if comptime!(op == 67 || op == 68) {
            let boundary = F::cast_from(scalar);
            let norm = out[target];
            let mut result = norm * x * y;
            if comptime!(op == 67) {
                if x < -boundary { result = (F::cast_from(-1.0f32) * norm) * y; }
                else if x > boundary { result = norm * y; }
                else { result = result / boundary; }
            } else {
                if x < -boundary { result = (F::cast_from(-1.0f32) * norm) * y * boundary; }
                else if x > boundary { result = norm * y * boundary; }
            }
            out[target] = result;
        }
    }
}

#[ruda(launch)]
pub fn adaptive_avg_pool<F: Float + RudaElement>(
    a: &Tensor<F>, out: &mut Tensor<F>, #[comptime] spatial_dims: usize, #[comptime] backward: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let rank = out.rank();
        let source_h = a.shape(rank - 2);
        let source_w = a.shape(rank - 1);
        let output_h = out.shape(rank - 2);
        let output_w = out.shape(rank - 1);
        let mut source_d = 1usize;
        let mut output_d = 1usize;
        if comptime!(spatial_dims == 3) {
            source_d = a.shape(rank - 3);
            output_d = out.shape(rank - 3);
        }
        let w = pos % output_w;
        let h = (pos / output_w) % output_h;
        let d = (pos / (output_w * output_h)) % output_d;
        let plane = pos / (output_w * output_h * output_d);
        let start_d = pool_start(d, output_d, source_d);
        let end_d = pool_end(d, output_d, source_d);
        let start_h = pool_start(h, output_h, source_h);
        let end_h = pool_end(h, output_h, source_h);
        let start_w = pool_start(w, output_w, source_w);
        let end_w = pool_end(w, output_w, source_w);
        let mut total = 0.0f32;
        let mut gradient = F::cast_from(0.0f32);
        for sd in start_d..end_d {
            for sh in start_h..end_h {
                for sw in start_w..end_w {
                    let index = ((plane * source_d + sd) * source_h + sh) * source_w + sw;
                    let value = a[offset(a, index)];
                    if comptime!(backward) {
                        let kd = pool_end(sd, source_d, output_d) - pool_start(sd, source_d, output_d);
                        let kh = pool_end(sh, source_h, output_h) - pool_start(sh, source_h, output_h);
                        let kw = pool_end(sw, source_w, output_w) - pool_start(sw, source_w, output_w);
                        let mut contribution = value;
                        if comptime!(spatial_dims == 2) {
                            contribution = value / F::cast_from(kw) / F::cast_from(kh);
                        } else {
                            contribution = F::cast_from(f32::cast_from(value) / f32::cast_from(kd * kh * kw));
                            if output_d % source_d != 0 || output_h % source_h != 0 || output_w % source_w != 0 {
                                contribution = value / F::cast_from(kd) / F::cast_from(kh) / F::cast_from(kw);
                            }
                        }
                        gradient += contribution;
                    } else {
                        total += f32::cast_from(value);
                    }
                }
            }
        }
        if comptime!(backward) {
            out[offset(out, pos)] = gradient;
        } else {
            if comptime!(spatial_dims == 2) {
                total = total / f32::cast_from(end_h - start_h) / f32::cast_from(end_w - start_w);
            } else {
                total = total / f32::cast_from((end_d - start_d) * (end_h - start_h) * (end_w - start_w));
            }
            out[offset(out, pos)] = F::cast_from(total);
        }
    }
}

#[ruda]
fn pool_start(position: usize, output_size: usize, input_size: usize) -> usize {
    (position / output_size) * input_size + ((position % output_size) * input_size) / output_size
}

#[ruda]
fn pool_end(position: usize, output_size: usize, input_size: usize) -> usize {
    1 + ((position + 1) * input_size - 1) / output_size
}

#[ruda(launch)]
pub fn bmm(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let col = pos % out.shape(2);
        let row = (pos / out.shape(2)) % out.shape(1);
        let batch = pos / (out.shape(2) * out.shape(1));
        let mut value = 0.0f32;
        for k in 0..a.shape(2) {
            value += a[batch * a.stride(0) + row * a.stride(1) + k * a.stride(2)]
                * b[batch * b.stride(0) + k * b.stride(1) + col * b.stride(2)];
        }
        out[batch * out.stride(0) + row * out.stride(1) + col * out.stride(2)] = value;
    }
}

#[ruda]
fn row_offset<F: Numeric>(tensor: &Tensor<F>, row: usize, #[comptime] axis: usize) -> usize {
    let mut remaining = row;
    let mut base = 0usize;
    let mut dim = tensor.rank();
    while dim > 0 {
        dim -= 1;
        if dim != axis {
            base += (remaining % tensor.shape(dim)) * tensor.stride(dim);
            remaining /= tensor.shape(dim);
        }
    }
    base
}

#[ruda(launch)]
pub fn softmax<F: Float + RudaElement, O: Float + RudaElement>(
    a: &Tensor<F>, b: &Tensor<F>, out: &mut Tensor<O>,
    #[comptime] axis: usize, #[comptime] backward: bool, #[comptime] logarithmic: bool) {
    let row = ABSOLUTE_POS as usize;
    let width = a.shape(axis);
    if row < a.len() / width {
        let ab = row_offset(a, row, axis);
        let bb = row_offset(b, row, axis);
        let ob = row_offset(out, row, axis);
        if comptime!(backward) {
            let mut total = 0.0f32;
            for i in 0..width {
                let grad = f32::cast_from(a[ab + i * a.stride(axis)]);
                if comptime!(logarithmic) { total += grad; }
                else { total += grad * f32::cast_from(b[bb + i * b.stride(axis)]); }
            }
            for i in 0..width {
                let grad = f32::cast_from(a[ab + i * a.stride(axis)]);
                let y = f32::cast_from(b[bb + i * b.stride(axis)]);
                if comptime!(logarithmic) { out[ob + i * out.stride(axis)] = O::cast_from(grad - y.exp() * total); }
                else { out[ob + i * out.stride(axis)] = O::cast_from(y * (grad - total)); }
            }
        } else {
            let mut maximum = f32::cast_from(a[ab]);
            for i in 1..width {
                let value = f32::cast_from(a[ab + i * a.stride(axis)]);
                if value != value || value > maximum { maximum = value; }
            }
            let mut total = 0.0f32;
            for i in 0..width { total += (f32::cast_from(a[ab + i * a.stride(axis)]) - maximum).exp(); }
            for i in 0..width {
                let shifted = f32::cast_from(a[ab + i * a.stride(axis)]) - maximum;
                if comptime!(logarithmic) { out[ob + i * out.stride(axis)] = O::cast_from(shifted - total.ln()); }
                else { out[ob + i * out.stride(axis)] = O::cast_from(shifted.exp() / total); }
            }
        }
    }
}

#[ruda(launch)]
pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let row = pos / out.shape(1);
        let col = pos % out.shape(1);
        let mut value = 0.0f32;
        for k in 0..a.shape(1) {
            value += a[row * a.stride(0) + k * a.stride(1)] * b[k * b.stride(0) + col * b.stride(1)];
        }
        out[row * out.stride(0) + col * out.stride(1)] = value;
    }
}

#[ruda(launch)]
pub fn reduce(a: &Tensor<f32>, out: &mut Tensor<f32>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut reduction_size = 1usize;
        let mut source_base = 0usize;
        let mut remaining = pos;
        let mut dim = a.rank();
        while dim > 0 {
            dim -= 1;
            if out.shape(dim) == 1 { reduction_size *= a.shape(dim); }
            else { source_base += (remaining % out.shape(dim)) * a.stride(dim); }
            remaining /= out.shape(dim);
        }
        let mut value = 0.0f32;
        for index in 0..reduction_size {
            let mut reduction_index = index;
            let mut source = source_base;
            let mut dim = a.rank();
            while dim > 0 {
                dim -= 1;
                if out.shape(dim) == 1 {
                    source += (reduction_index % a.shape(dim)) * a.stride(dim);
                    reduction_index /= a.shape(dim);
                }
            }
            value += a[source];
        }
        out[offset(out, pos)] = value;
    }
}

// Storage stays in F16/BF16/F32. This compatibility kernel widens individual
// loads, not the entire inputs; it is also the explicit diagnostic baseline.
#[ruda(launch)]
pub fn matmul_storage<F: Float + RudaElement, O: Float + RudaElement>(
    a: &Tensor<F>, b: &Tensor<F>, out: &mut Tensor<O>, #[comptime] batched: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let rank = out.rank();
        let m_axis = rank - 2;
        let n_axis = rank - 1;
        let n = out.shape(n_axis);
        let m = out.shape(m_axis);
        let col = pos % n;
        let row = (pos / n) % m;
        let mut ab = row * a.stride(m_axis);
        let mut bb = col * b.stride(n_axis);
        let mut ob = row * out.stride(m_axis) + col * out.stride(n_axis);
        if comptime!(batched) {
            let batch = pos / (m * n);
            ab += batch * a.stride(0);
            bb += batch * b.stride(0);
            ob += batch * out.stride(0);
        }
        let mut value = 0.0f32;
        for k in 0..a.shape(n_axis) {
            value += f32::cast_from(a[ab + k * a.stride(n_axis)])
                * f32::cast_from(b[bb + k * b.stride(m_axis)]);
        }
        out[ob] = O::cast_from(value);
    }
}

// The accumulator can alias the output only for F32. Each unit reads its own
// accumulator before storing; the C++ entry rejects overlap with all inputs.
#[ruda(launch)]
pub fn addmm_epilogue<F: Float + RudaElement>(
    acc: &Tensor<f32>, bias: &Tensor<F>, out: &mut Tensor<F>,
    alpha: f32, beta: f32, #[comptime] use_product: bool, #[comptime] use_bias: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut value = 0.0f32;
        if comptime!(use_product) { value = alpha * acc[offset(acc, pos)]; }
        if comptime!(use_bias) { value += beta * f32::cast_from(bias[offset(bias, pos)]); }
        out[offset(out, pos)] = F::cast_from(value);
    }
}

#[ruda(launch)]
pub fn addmm_bias<F: Float + RudaElement>(
    bias: &Tensor<F>, out: &mut Tensor<F>, beta: f32, #[comptime] use_bias: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut value = 0.0f32;
        if comptime!(use_bias) { value = beta * f32::cast_from(bias[offset(bias, pos)]); }
        out[offset(out, pos)] = F::cast_from(value);
    }
}

// Four complete 32-lane CUDA warps per block, one logical row per warp.
// Every active lane takes part in both reductions, even past the row tail.
#[ruda(launch)]
pub fn softmax_warp<F: Float + RudaElement, O: Float + RudaElement>(
    a: &Tensor<F>, b: &Tensor<F>, out: &mut Tensor<O>,
    #[comptime] axis: usize, #[comptime] backward: bool, #[comptime] logarithmic: bool,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let width = a.shape(axis);
    if row < a.len() / width {
        let ab = row_offset(a, row, axis);
        let bb = row_offset(b, row, axis);
        let ob = row_offset(out, row, axis);
        if comptime!(backward) {
            let mut local = 0.0f32;
            let mut i = lane;
            while i < width {
                let grad = f32::cast_from(a[ab + i * a.stride(axis)]);
                if comptime!(logarithmic) { local += grad; }
                else { local += grad * f32::cast_from(b[bb + i * b.stride(axis)]); }
                i += 32;
            }
            let total = plane_sum(local);
            let mut i = lane;
            while i < width {
                let grad = f32::cast_from(a[ab + i * a.stride(axis)]);
                let y = f32::cast_from(b[bb + i * b.stride(axis)]);
                if comptime!(logarithmic) {
                    out[ob + i * out.stride(axis)] = O::cast_from(grad - y.exp() * total);
                } else {
                    out[ob + i * out.stride(axis)] = O::cast_from(y * (grad - total));
                }
                i += 32;
            }
        } else {
            let mut maximum = f32::NEG_INFINITY;
            let mut has_nan = 0.0f32;
            let mut i = lane;
            while i < width {
                let value = f32::cast_from(a[ab + i * a.stride(axis)]);
                if value != value { has_nan = 1.0; }
                if value > maximum { maximum = value; }
                i += 32;
            }
            maximum = plane_max(maximum);
            // Hardware max may ignore NaNs. Restore PyTorch's row-wide NaN
            // propagation explicitly before the exponential/sum reduction.
            let nan_count = plane_sum(has_nan);
            if nan_count > 0.0 { maximum = f32::NAN; }
            let mut local = 0.0f32;
            let mut i = lane;
            while i < width {
                local += (f32::cast_from(a[ab + i * a.stride(axis)]) - maximum).exp();
                i += 32;
            }
            let total = plane_sum(local);
            let mut i = lane;
            while i < width {
                let shifted = f32::cast_from(a[ab + i * a.stride(axis)]) - maximum;
                if comptime!(logarithmic) {
                    out[ob + i * out.stride(axis)] = O::cast_from(shifted - total.ln());
                } else {
                    out[ob + i * out.stride(axis)] = O::cast_from(shifted.exp() / total);
                }
                i += 32;
            }
        }
    }
}

// Fused last-axis LayerNorm used by the native PyTorch inference bridge.
// One warp owns one logical row. Statistics are accumulated in F32 and only
// the final outputs are cast back to storage precision. Weight/bias are
// optional and, when present, have the same storage dtype as the input.
#[ruda(launch)]
pub fn layer_norm_warp<F: Float + RudaElement>(
    input: &Tensor<F>, weight: &Tensor<F>, bias: &Tensor<F>,
    out: &mut Tensor<F>, mean: &mut Tensor<F>, rstd: &mut Tensor<F>,
    epsilon: f32, #[comptime] has_weight: bool, #[comptime] has_bias: bool,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let width = input.shape(input.rank() - 1);
    let rows = input.len() / width;
    if row < rows {
        let base = row * width;
        let mut local_sum = 0.0f32;
        let mut i = lane;
        while i < width {
            local_sum += f32::cast_from(input[base + i]);
            i += 32;
        }
        let mu = plane_sum(local_sum) / width as f32;
        // A second pass over the row is more stable than E[x^2]-E[x]^2 for
        // transformer activations with a large offset. It still stays inside
        // one kernel launch and never materializes a centered tensor.
        let mut local_variance = 0.0f32;
        let mut j = lane;
        while j < width {
            let delta = f32::cast_from(input[base + j]) - mu;
            local_variance += delta * delta;
            j += 32;
        }
        let variance = plane_sum(local_variance) / width as f32;
        let inv = (variance + epsilon).inverse_sqrt();
        if lane == 0 {
            mean[row] = F::cast_from(mu);
            rstd[row] = F::cast_from(inv);
        }
        let mut i = lane;
        while i < width {
            let mut value = (f32::cast_from(input[base + i]) - mu) * inv;
            if comptime!(has_weight) { value *= f32::cast_from(weight[i]); }
            if comptime!(has_bias) { value += f32::cast_from(bias[i]); }
            out[base + i] = F::cast_from(value);
            i += 32;
        }
    }
}

// Fused last-axis RMSNorm used by the native PyTorch inference bridge.
// One warp owns one logical row. Squares and the reciprocal RMS are accumulated
// in F32; storage is widened only in registers and the final value is cast once.
#[ruda(launch)]
pub fn rms_norm_warp<F: Float + RudaElement>(
    input: &Tensor<F>, weight: &Tensor<F>, out: &mut Tensor<F>,
    epsilon: f32, #[comptime] has_weight: bool,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let width = input.shape(input.rank() - 1);
    let rows = input.len() / width;
    if row < rows {
        let base = row * width;
        let mut local_square = 0.0f32;
        let mut i = lane;
        while i < width {
            let value = f32::cast_from(input[base + i]);
            local_square += value * value;
            i += 32;
        }
        let mean_square = plane_sum(local_square) / width as f32;
        let inv = (mean_square + epsilon).inverse_sqrt();
        let mut i = lane;
        while i < width {
            let mut value = f32::cast_from(input[base + i]) * inv;
            if comptime!(has_weight) { value *= f32::cast_from(weight[i]); }
            out[base + i] = F::cast_from(value);
            i += 32;
        }
    }
}

// Storage-aware generic sum reduction. Loads are widened individually and the
// accumulator remains F32, avoiding a tensor-wide promotion allocation.
#[ruda(launch)]
pub fn reduce_sum_storage<F: Float + RudaElement, O: Float + RudaElement>(
    a: &Tensor<F>, out: &mut Tensor<O>,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let mut reduction_size = 1usize;
        let mut source_base = 0usize;
        let mut remaining = pos;
        let mut dim = a.rank();
        while dim > 0 {
            dim -= 1;
            if out.shape(dim) == 1 { reduction_size *= a.shape(dim); }
            else {
                source_base += (remaining % out.shape(dim)) * a.stride(dim);
                remaining /= out.shape(dim);
            }
        }
        let mut value = 0.0f32;
        for index in 0..reduction_size {
            let mut reduction_index = index;
            let mut source = source_base;
            let mut dim = a.rank();
            while dim > 0 {
                dim -= 1;
                if out.shape(dim) == 1 {
                    source += (reduction_index % a.shape(dim)) * a.stride(dim);
                    reduction_index /= a.shape(dim);
                }
            }
            value += f32::cast_from(a[source]);
        }
        out[offset(out, pos)] = O::cast_from(value);
    }
}

// Fast path for the reduction shape used by transformer statistics:
// contiguous [..., width] -> [..., 1]. One warp reduces one row in F32.
#[ruda(launch)]
pub fn reduce_sum_last_warp<F: Float + RudaElement, O: Float + RudaElement>(
    a: &Tensor<F>, out: &mut Tensor<O>,
) {
    let row = (ABSOLUTE_POS / 32) as usize;
    let lane = (ABSOLUTE_POS % 32) as usize;
    let width = a.shape(a.rank() - 1);
    let rows = a.len() / width;
    if row < rows {
        let base = row * width;
        let mut local = 0.0f32;
        let mut column = lane;
        while column < width {
            local += f32::cast_from(a[base + column]);
            column += 32;
        }
        let total = plane_sum(local);
        if lane == 0 { out[row] = O::cast_from(total); }
    }
}
