use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn forward<G:Float,W:Float>(input:&Array<G>,qweight:&Array<i32>,qzeros:&Array<i32>,scales:&Array<W>,bias:&Array<W>,
    ids:&Array<u32>,output:&mut Array<G>,experts:u32,width:u32,columns:u32,group_size:u32,has_bias:u32,
    #[define(G)] _activation:StorageType,#[define(W)] _weight:StorageType) {
    let p=ABSOLUTE_POS;if p>=output.len() {terminate!();}
    let k=width as usize;let n=columns as usize;let row=p/n;let column=p%n;let expert=ids[row] as usize;
    if expert>=experts as usize {output[p]=G::cast_from(0.0f32);terminate!();}
    let packed_columns=n/8;let groups=k/group_size as usize;let lane=column%8;
    let shift=((lane%2)*4+lane/2) as u32*4;let mut sum=0.0f32;let mut inner=0usize;
    while inner<k {
        let group=inner/group_size as usize;
        let word=u32::cast_from(qweight[(expert*k+inner)*packed_columns+column/8]);
        let zero_word=u32::cast_from(qzeros[(expert*groups+group)*packed_columns+column/8]);
        let quant=(word>>shift)&15u32;let zero=(zero_word>>shift)&15u32;
        let weight=W::cast_from((f32::cast_from(quant)-f32::cast_from(zero))*f32::cast_from(scales[(expert*groups+group)*n+column]));
        sum+=f32::cast_from(input[row*k+inner])*f32::cast_from(weight);inner+=1;
    }
    let mut value=G::cast_from(sum);if has_bias!=0 {value+=G::cast_from(bias[expert*n+column]);}output[p]=value;
}
#[ruda(launch)]
pub(super) fn input_backward<G:Float,W:Float>(gradient:&Array<G>,qweight:&Array<i32>,qzeros:&Array<i32>,scales:&Array<W>,ids:&Array<u32>,
    output:&mut Array<G>,experts:u32,width:u32,columns:u32,group_size:u32,#[define(G)] _activation:StorageType,#[define(W)] _weight:StorageType) {
    let p=ABSOLUTE_POS;if p>=output.len() {terminate!();}
    let k=width as usize;let n=columns as usize;let row=p/k;let inner=p%k;let expert=ids[row] as usize;
    if expert>=experts as usize {output[p]=G::cast_from(0.0f32);terminate!();}
    let packed_columns=n/8;let groups=k/group_size as usize;let group=inner/group_size as usize;let mut sum=0.0f32;let mut column=0usize;
    while column<n {
        let lane=column%8;let shift=((lane%2)*4+lane/2) as u32*4;
        let word=u32::cast_from(qweight[(expert*k+inner)*packed_columns+column/8]);
        let zero_word=u32::cast_from(qzeros[(expert*groups+group)*packed_columns+column/8]);
        let quant=(word>>shift)&15u32;let zero=(zero_word>>shift)&15u32;
        let weight=W::cast_from((f32::cast_from(quant)-f32::cast_from(zero))*f32::cast_from(scales[(expert*groups+group)*n+column]));
        sum+=f32::cast_from(gradient[row*n+column])*f32::cast_from(weight);column+=1;
    }
    output[p]=G::cast_from(sum);
}
