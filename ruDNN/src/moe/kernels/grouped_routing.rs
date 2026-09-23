//! Parameterized sigmoid + grouped expert selection. This is a correctness-first
//! row kernel, not a claim of optimized warp routing or FP8 model support.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn route<F: Float>(logits:&Array<F>, bias:&Array<f32>, indices:&mut Array<u32>,
    weights:&mut Array<F>, scale:f32,
    #[comptime] experts:usize, #[comptime] groups:usize,
    #[comptime] selected_groups:usize, #[comptime] top_k:usize,
    #[comptime] group_top_two:bool, #[comptime] has_bias:bool,
    #[comptime] renormalize:bool, #[define(F)] _dtype:StorageType,
) {
    let token=ABSOLUTE_POS;
    if token<logits.len()/experts {
        let mut probability=Array::<f32>::new(experts);
        let mut corrected=Array::<f32>::new(experts);
        let mut group_score=Array::<f32>::new(groups);
        let mut group_selected=Array::<u32>::new(groups);
        let mut taken=Array::<u32>::new(experts);
        let per_group=experts/groups;
        for i in 0usize..experts {
            let x=f32::cast_from(logits[token*experts+i]);
            let mut p=0.0f32;
            if x>=0.0 { p=1.0/(1.0+(-x).exp()); }
            else {let e=x.exp();p=e/(1.0+e);}
            probability[i]=p;
            let mut s=p;
            if comptime!(has_bias) {s+=bias[i];}
            corrected[i]=s;taken[i]=0;
        }
        for g in 0usize..groups {
            let mut first=0.0f32;let mut second=0.0f32;
            for j in 0usize..per_group {
                let s=corrected[g*per_group+j];
                if j==0 {first=s;}
                else if j==1 {
                    if s>first {second=first;first=s;} else {second=s;}
                } else if s>first {second=first;first=s;}
                else if s>second {second=s;}
            }
            let mut s=first;
            if comptime!(group_top_two) {s+=second;}
            group_score[g]=s;group_selected[g]=0;
        }
        for _slot in 0usize..selected_groups {
            let mut found=false;let mut best=0.0f32;let mut id=0usize;
            for g in 0usize..groups {
                if group_selected[g]==0 && (!found || group_score[g]>best) {
                    found=true;best=group_score[g];id=g;
                }
            }
            group_selected[id]=1;
        }
        let mut denominator=0.0f32;
        for slot in 0usize..top_k {
            let mut found=false;let mut best=0.0f32;let mut id=0usize;
            for i in 0usize..experts {
                if group_selected[i/per_group]!=0 && taken[i]==0 && (!found || corrected[i]>best) {
                    found=true;best=corrected[i];id=i;
                }
            }
            indices[token*top_k+slot]=id as u32;
            taken[id]=1;denominator+=probability[id];
        }
        for slot in 0usize..top_k {
            let id=indices[token*top_k+slot] as usize;
            let mut w=probability[id];
            if comptime!(renormalize) {w/=denominator;}
            weights[token*top_k+slot]=F::cast_from(w*scale);
        }
    }
}
