//! Segmented grouped GEMM backward, one full 32-thread plane per tile.
//!
//! Native storage is FP16/BF16; accumulators and dWeight are FP32. All threads
//! execute every cooperative operation/barrier. Tail lanes load zeros rather
//! than exiting. No float atomics, transposed global buffers, or split-K scratch.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// dX[e] = dY[e] @ W[e]. Rows belonging to an expert are processed in tiles.
#[ruda(launch)]
pub(super) fn dinput<F: Float>(
    grad: &Array<F>, weights: &Array<F>, offsets: &Array<u32>, out: &mut Array<F>,
    columns: u32, inner: u32, #[define(F)] _dtype: StorageType,
) {
    let expert = RUDA_POS_Y as usize;
    let k_base = RUDA_POS_X as usize * 16;
    let begin = offsets[expert] as usize;
    let end = offsets[expert + 1] as usize;
    let lane = UNIT_POS as usize;
    let mut left = SharedMemory::<F>::new_aligned(256usize, 32usize);
    let mut right = SharedMemory::<F>::new_aligned(256usize, 32usize);
    let mut result = SharedMemory::<f32>::new_aligned(256usize, 32usize);
    let mut row_base = begin;
    while row_base < end {
        let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
            16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
        let mut n_base = 0usize;
        while n_base < columns as usize {
            #[unroll]
            for i in 0usize..8usize {
                let t = lane + i * 32;
                let tr = t / 16;
                let tc = t % 16;
                let row = row_base + tr;
                let n_left = n_base + tc;
                let n_right = n_base + tr;
                let k = k_base + tc;
                let mut a = F::cast_from(0.0f32);
                let mut b = F::cast_from(0.0f32);
                if tr < end - row_base && n_left < columns as usize {
                    a = grad[row * columns as usize + n_left];
                }
                if n_right < columns as usize && k < inner as usize {
                    b = weights[(expert * columns as usize + n_right) * inner as usize + k];
                }
                left[t] = a;
                // Coalesced global W load, transposed only inside the shared tile.
                right[tc * 16 + tr] = b;
            }
            sync_ruda();
            let a = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
                16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
            let b = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
                16usize,16usize,16usize,cmma::MatrixLayout::ColMajor,&right.to_slice(),16);
            cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);
            sync_ruda();
            if columns as usize - n_base <= 16 {n_base = columns as usize;}else {n_base += 16;}
        }
        cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);
        sync_ruda();
        #[unroll]
        for i in 0usize..8usize {
            let t = lane + i * 32;
            let row = row_base + t / 16;
            let k = k_base + t % 16;
            if t / 16 < end - row_base && k < inner as usize {
                out[row * inner as usize + k] = F::cast_from(result[t]);
            }
        }
        sync_ruda();
        if end - row_base <= 16 {row_base = end;}else {row_base += 16;}
    }
}

/// dW[e] = dY[e].T @ X[e]. One block owns each 16x16 weight-gradient tile.
/// Empty experts still write explicit zeros to every valid dWeight element.
#[ruda(launch)]
pub(super) fn dweight<F: Float>(
    input: &Array<F>, grad: &Array<F>, offsets: &Array<u32>, out: &mut Array<f32>,
    columns: u32, inner: u32, #[define(F)] _dtype: StorageType,
) {
    let expert = RUDA_POS_Y as usize;
    let k_base = RUDA_POS_X as usize * 16;
    let n_base = RUDA_POS_Z as usize * 16;
    let begin = offsets[expert] as usize;
    let end = offsets[expert + 1] as usize;
    let lane = UNIT_POS as usize;
    let mut left = SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right = SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result = SharedMemory::<f32>::new_aligned(256usize,32usize);
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
        16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let mut row_base = begin;
    while row_base < end {
        #[unroll]
        for i in 0usize..8usize {
            let t = lane + i * 32;
            let tr = t / 16;
            let tc = t % 16;
            let row = row_base + tr;
            let n = n_base + tc;
            let k = k_base + tc;
            let mut a = F::cast_from(0.0f32);
            let mut b = F::cast_from(0.0f32);
            if tr < end - row_base && n < columns as usize {
                a = grad[row * columns as usize + n];
            }
            if tr < end - row_base && k < inner as usize {
                b = input[row * inner as usize + k];
            }
            // A[n,row] is row-major; B[row,k] is column-major.
            // Both loads are contiguous globally; only shared-memory indices swap.
            left[tc * 16 + tr] = a;
            right[tc * 16 + tr] = b;
        }
        sync_ruda();
        let a = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
            16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
        let b = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
            16usize,16usize,16usize,cmma::MatrixLayout::ColMajor,&right.to_slice(),16);
        cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);
        sync_ruda();
        if end - row_base <= 16 {row_base = end;}else {row_base += 16;}
    }
    cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);
    sync_ruda();
    #[unroll]
    for i in 0usize..8usize {
        let t = lane + i * 32;
        let n = n_base + t / 16;
        let k = k_base + t % 16;
        if n < columns as usize && k < inner as usize {
            out[(expert * columns as usize + n) * inner as usize + k] = result[t];
        }
    }
}
