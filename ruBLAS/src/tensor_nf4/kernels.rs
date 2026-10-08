//! Shared original RUDA NF4 decoder and 16x16x16 cooperative GEMM.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Decode a contiguous flat NF4 interval with high-nibble-first byte order.
#[ruda(launch)]
pub fn decode<F: Float + RudaElement>(
    packed: &Tensor<u8>, scales: &Tensor<f32>, table: &Tensor<f32>,
    output: &mut Tensor<F>, start: u32, block: u32,
) {
    let position = ABSOLUTE_POS as u32;
    if (position as usize) < output.len() {
        let index = position + start;
        let byte = u32::cast_from(packed[(index / 2) as usize]);
        let mut code = byte & 15;
        if index % 2 == 0 { code = byte >> 4; }
        output[position as usize] = F::cast_from(table[code as usize] * scales[(index / block) as usize]);
    }
}

/// Original fused half/BF16 NF4 forward or input gradient, with FP32 accumulators.
#[ruda(launch)]
pub fn gemm<F: Float + RudaElement, O: Float + RudaElement>(
    input: &Tensor<F>, packed: &Tensor<u8>, scales: &Tensor<f32>, table: &Tensor<f32>,
    output: &mut Tensor<O>, rows: u32, columns: u32,
    width: u32, block: u32, element_offset: u32, #[comptime] backward: bool,
) {
    let lane = UNIT_POS as usize;
    let row_base = RUDA_POS_Y as usize * 16;
    let column_base = RUDA_POS_X as usize * 16;
    let k = width as usize;
    let n = columns as usize;
    let mut inner = k;
    let mut cols = n;
    if comptime!(backward) { inner = n; cols = k; }
    let mut left = SharedMemory::<F>::new_aligned(256usize, 32usize);
    let mut right = SharedMemory::<F>::new_aligned(256usize, 32usize);
    let mut result = SharedMemory::<f32>::new_aligned(256usize, 32usize);
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut base = 0usize;
    while base < inner {
        #[unroll]
        for i in 0usize..8usize {
            let t = lane + i * 32;
            let tr = t / 16;
            let tc = t % 16;
            let row = row_base + tr;
            let source_col = base + tc;
            let mut a = F::cast_from(0.0f32);
            if row < rows as usize && source_col < inner { a = input[row * inner + source_col]; }
            left[t] = a;
            let mut wr = column_base + tr;
            let mut wc = base + tc;
            if comptime!(backward) { wr = base + tr; wc = column_base + tc; }
            let mut value = F::cast_from(0.0f32);
            if wr < n && wc < k {
                let index = element_offset as usize + wr * k + wc;
                let byte = u32::cast_from(packed[index / 2]);
                let mut code = byte & 15;
                if index % 2 == 0 { code = byte >> 4; }
                value = F::cast_from(table[code as usize] * scales[index / block as usize]);
            }
            if comptime!(backward) { right[tc * 16 + tr] = value; }
            else { right[t] = value; }
        }
        sync_ruda();
        let a = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
            16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &left.to_slice(), 16);
        let b = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
            16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &right.to_slice(), 16);
        cmma::execute::<F,F,f32,f32>(&a, &b, &acc, &acc);
        sync_ruda(); base += 16;
    }
    cmma::store(&mut result.to_slice_mut(), &acc, 16, cmma::MatrixLayout::RowMajor);
    sync_ruda();
    #[unroll]
    for i in 0usize..8usize {
        let t = lane + i * 32;
        let row = row_base + t / 16;
        let column = column_base + t % 16;
        if row < rows as usize && column < cols {
            output[row * cols + column] = O::cast_from(result[t]);
        }
    }
}

#[ruda(launch)]
pub(super) fn gather_columns<F:Float>(input:&Array<F>,out:&mut Array<F>,columns:u32,start:u32,tile:u32,#[define(F)] _dtype:StorageType) {
    let p=ABSOLUTE_POS;if p>=out.len() {terminate!();}
    let row=p/tile as usize;let column=p%tile as usize;
    out[p]=input[row*columns as usize+start as usize+column];
}
#[ruda(launch)]
pub(super) fn store_columns<F:Float>(input:&Array<F>,out:&mut Array<F>,columns:u32,start:u32,tile:u32,#[define(F)] _dtype:StorageType) {
    let p=ABSOLUTE_POS;if p>=input.len() {terminate!();}
    let row=p/tile as usize;let column=p%tile as usize;
    out[row*columns as usize+start as usize+column]=input[p];
}
#[ruda(launch)]
pub(super) fn zero_accumulator(out:&mut Array<f32>) {let p=ABSOLUTE_POS;if p<out.len() {out[p]=0.0;}}
#[ruda(launch)]
pub(super) fn add_partial(input:&Array<f32>,out:&mut Array<f32>) {let p=ABSOLUTE_POS;if p<out.len() {out[p]+=input[p];}}
#[ruda(launch)]
pub(super) fn add_bias<F:Float,B:Float>(out:&mut Array<F>,bias:&Array<B>,columns:u32,
    #[define(F)] _dtype:StorageType,#[define(B)] _bias_dtype:StorageType) {
    let p=ABSOLUTE_POS;if p<out.len() {out[p]+=F::cast_from(bias[p%columns as usize]);}
}
