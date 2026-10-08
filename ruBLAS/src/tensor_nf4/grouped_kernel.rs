use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Original NF4 cooperative arithmetic over private native expert segments.
#[ruda(launch)]
pub(super) fn segmented<F:Float+RudaElement,O:Float+RudaElement>(
    input:&Tensor<F>,packed:&Tensor<u8>,scales:&Tensor<f32>,table:&Tensor<f32>,
    offsets:&Tensor<u32>,output:&mut Tensor<O>,columns:u32,width:u32,block:u32,element_offset:u32,#[comptime] backward:bool,
) {
    let expert=RUDA_POS_Y as usize;let column_base=RUDA_POS_X as usize*16;
    let begin=offsets[expert] as usize;let end=offsets[expert+1] as usize;let lane=UNIT_POS as usize;
    let k=width as usize;let n=columns as usize;let mut inner=k;let mut cols=n;
    if comptime!(backward) {inner=n;cols=k;}
    let mut left=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut right=SharedMemory::<F>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);
    let mut row_base=begin;
    while row_base<end {
        let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
            16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
        let mut base=0usize;
        while base<inner {
            #[unroll]
            for i in 0usize..8usize {
                let t=lane+i*32;let tr=t/16;let tc=t%16;let row=row_base+tr;let source_col=base+tc;
                let mut a=F::cast_from(0.0f32);
                if tr<end-row_base && source_col<inner {a=input[row*inner+source_col];}left[t]=a;
                let mut wr=column_base+tr;let mut wc=base+tc;
                if comptime!(backward) {wr=base+tr;wc=column_base+tc;}
                let mut value=F::cast_from(0.0f32);
                if wr<n && wc<k {
                    let index=element_offset as usize+(expert*n+wr)*k+wc;
                    let byte=u32::cast_from(packed[index/2]);let mut code=byte & 15;
                    if index%2==0 {code=byte>>4;}
                    value=F::cast_from(table[code as usize]*scales[index/block as usize]);
                }
                if comptime!(backward) {right[tc*16+tr]=value;}else {right[t]=value;}
            }
            sync_ruda();
            let a=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
                16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&left.to_slice(),16);
            let b=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
                16usize,16usize,16usize,cmma::MatrixLayout::ColMajor,&right.to_slice(),16);
            cmma::execute::<F,F,f32,f32>(&a,&b,&acc,&acc);sync_ruda();
            if inner-base<=16 {base=inner;}else {base+=16;}
        }
        cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
        #[unroll]
        for i in 0usize..8usize {
            let t=lane+i*32;let row=row_base+t/16;let column=column_base+t%16;
            if t/16<end-row_base && column<cols {output[row*cols+column]=O::cast_from(result[t]);}
        }
        sync_ruda();if end-row_base<=16 {row_base=end;}else {row_base+=16;}
    }
}

#[ruda(launch)]
pub(super) fn copy_rows<F:Float>(input:&Array<F>,output:&mut Array<F>,begin:u32,#[define(F)] _dtype:StorageType) {
    let i=ABSOLUTE_POS;if i<output.len() {output[i]=input[begin as usize+i];}
}
#[ruda(launch)]
pub(super) fn store_rows_columns<F:Float>(input:&Array<F>,output:&mut Array<F>,row_begin:u32,columns:u32,start:u32,tile:u32,#[define(F)] _dtype:StorageType) {
    let i=ABSOLUTE_POS;if i<input.len() {let row=i/tile as usize;let col=i%tile as usize;
        output[(row_begin as usize+row)*columns as usize+start as usize+col]=input[i];}
}
