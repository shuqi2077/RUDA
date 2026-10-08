use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn localize(ids:&Array<u32>,local:&mut Array<u32>,invalid:&mut Array<Atomic<u32>>,begin:u32,end:u32) {
    let row=ABSOLUTE_POS;if row>=ids.len() {terminate!();}
    let expert=ids[row];if expert<begin || expert>=end {invalid[0].fetch_add(1u32);local[row]=0u32;} else {local[row]=expert-begin;}
}
#[ruda(launch)]
pub(crate) fn permute<F:Float>(values:&Array<F>,rows:&Array<u32>,output:&mut Array<F>,width:u32,#[define(F)] _dtype:StorageType) {
    let position=ABSOLUTE_POS;if position>=output.len() {terminate!();}
    let row=position/width as usize;let column=position%width as usize;
    output[position]=values[rows[row] as usize*width as usize+column];
}
