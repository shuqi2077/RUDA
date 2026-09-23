//! Small-expert-batch 16x16x16 cooperative matrix path.
//! One plane per (expert, output-column tile); rows and K are tiled inside it.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn segmented<F: Float>(
    input: &Array<F>, weights: &Array<F>, offsets: &Array<u32>, out: &mut Array<F>,
    columns: u32, inner: u32,
    #[define(F)] _dtype: StorageType,
) {
    let expert=RUDA_POS_Y as usize;
    let col_base=RUDA_POS_X as usize*16;
    let begin=offsets[expert] as usize;
    let end=offsets[expert+1] as usize;
    let lane=UNIT_POS as usize;
    let mut left=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    let mut row_base=begin;
    while row_base<end {
        let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
            16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
        let mut base_k=0usize;
        while base_k<inner as usize {
            #[unroll]
            for i in 0usize..8usize {
                let t=lane+i*32;
                let row=row_base+t/16;
                let column=col_base+t/16;
                let kk=base_k+t%16;
                let mut a=F::cast_from(0.0f32);
                let mut b=F::cast_from(0.0f32);
                if row<end && kk<inner as usize { a=input[row*inner as usize+kk]; }
                if column<columns as usize && kk<inner as usize {
                    b=weights[(expert*columns as usize+column)*inner as usize+kk];
                }
                left[t]=a; right[t]=b;
            }
            sync_ruda();
            let a=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
                16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
            let b=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
                16usize,16usize,16usize,cmma::MatrixLayout::ColMajor,&right.to_slice(),16);
            cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);
            sync_ruda();
            base_k+=16;
        }
        cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);
        sync_ruda();
        #[unroll]
        for i in 0usize..8usize {
            let t=lane+i*32;
            let row=row_base+t/16;
            let column=col_base+t%16;
            if row<end && column<columns as usize { out[row*columns as usize+column]=F::cast_from(result[t]); }
        }
        sync_ruda();
        row_base+=16;
    }
}
