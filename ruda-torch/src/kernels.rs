use ruda_kernel::dsl::prelude::*;

#[ruda]
fn offset<F: Float>(tensor: &Tensor<F>, position: usize) -> usize {
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
pub fn convert<I: Float + RudaElement, O: Float + RudaElement>(a: &Tensor<I>, out: &mut Tensor<O>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        out[offset(out, pos)] = O::cast_from(a[offset(a, pos)]);
    }
}

#[ruda(launch)]
pub fn pointwise(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>, scalar: f32, #[comptime] op: u32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let target = offset(out, pos);
        if comptime!(op == 5) { out[target] = scalar; }
        else {
            let x = a[offset(a, pos)];
            if comptime!(op == 0) { out[target] = x; }
            else if comptime!(op == 1) { out[target] = x + scalar * b[offset(b, pos)]; }
            else if comptime!(op == 2) { out[target] = x * b[offset(b, pos)]; }
            else if comptime!(op == 3) {
                let mut result = x;
                if x < 0.0 { result = 0.0; }
                out[target] = result;
            }
            else if comptime!(op == 4) {
                let mut result = x;
                if b[offset(b, pos)] <= scalar { result = 0.0; }
                out[target] = result;
            }
            else if comptime!(op == 8) { out[target] = x / b[offset(b, pos)]; }
            else if comptime!(op == 9) { out[target] = x.exp(); }
            else if comptime!(op == 10) { out[target] = x.ln(); }
            else if comptime!(op == 11) { out[target] = x.sqrt(); }
            else if comptime!(op == 12) { out[target] = 1.0 / x.sqrt(); }
            else if comptime!(op == 13) { out[target] = 1.0 / (1.0 + (-x).exp()); }
            else if comptime!(op == 14) { out[target] = x / (1.0 + (-x).exp()); }
            else if comptime!(op == 15) {
                let input = b[offset(b, pos)];
                let sigmoid = 1.0f32 / (1.0f32 + (-input).exp());
                out[target] = x * (sigmoid * (1.0f32 + input * (1.0f32 - sigmoid)));
            }
            else if comptime!(op == 16) {
                let y = b[offset(b, pos)];
                out[target] = x * y * (1.0f32 - y);
            }
            else if comptime!(op == 17) { out[target] = x.tanh(); }
            else if comptime!(op == 18) {
                let y = b[offset(b, pos)];
                out[target] = x * (1.0f32 - y * y);
            }
        }
    }
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
fn row_offset(tensor: &Tensor<f32>, row: usize, #[comptime] axis: usize) -> usize {
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
pub fn softmax(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>,
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
                let grad = a[ab + i * a.stride(axis)];
                if comptime!(logarithmic) { total += grad; }
                else { total += grad * b[bb + i * b.stride(axis)]; }
            }
            for i in 0..width {
                let grad = a[ab + i * a.stride(axis)];
                let y = b[bb + i * b.stride(axis)];
                if comptime!(logarithmic) { out[ob + i * out.stride(axis)] = grad - y.exp() * total; }
                else { out[ob + i * out.stride(axis)] = y * (grad - total); }
            }
        } else {
            let mut maximum = a[ab];
            for i in 1..width {
                let value = a[ab + i * a.stride(axis)];
                if value != value || value > maximum { maximum = value; }
            }
            let mut total = 0.0f32;
            for i in 0..width { total += (a[ab + i * a.stride(axis)] - maximum).exp(); }
            for i in 0..width {
                let shifted = a[ab + i * a.stride(axis)] - maximum;
                if comptime!(logarithmic) { out[ob + i * out.stride(axis)] = shifted - total.ln(); }
                else { out[ob + i * out.stride(axis)] = shifted.exp() / total; }
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
