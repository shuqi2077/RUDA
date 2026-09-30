//! Atomic-free history backward: one plane owns one physical token / KV head.
//! Fixed sequence, query, head traversal; one final store per gradient element.
//! Row statistics are computed once by the query kernel, not per history token.
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn history_backward<F: Float>(
    q:&Array<F>, k:&Array<F>, v:&Array<F>, qp:&Array<F>, kp:&Array<F>, grad:&Array<F>,
    metadata:&Array<u32>, inverse:&Array<u32>, statistics:&Array<f32>, history_pages:&Array<u32>,
    dk:&mut Array<F>, dv:&mut Array<F>, dkp:&mut Array<F>,
    scale:f32, queries:u32, sequences:u32, page_size:u32, heads:u32, kv_heads:u32,
    #[comptime] key_dim:usize, #[comptime] value_dim:usize,
    #[comptime] position_dim:usize, #[comptime] lanes:usize,
    #[comptime] query_block_rows:usize, #[comptime] prune_queries:bool,
    #[comptime] cache_history:bool, #[comptime] compact_history:bool,
    #[comptime] cache_key_slots:usize, #[comptime] cache_value_slots:usize,
    #[comptime] cache_position_slots:usize,
    #[comptime] causal:bool, #[comptime] mla:bool,
    #[comptime] need_dk:bool, #[comptime] need_dv:bool, #[comptime] need_dkp:bool,
    #[define(F)] _dtype:StorageType,
) {
    let mut slot=RUDA_POS_X as usize;
    if comptime!(compact_history) {
        slot=history_pages[slot/page_size as usize] as usize*page_size as usize+slot%page_size as usize;
    }
    let kv_head=RUDA_POS_Y as usize;
    let lane=UNIT_POS as usize;
    let page=slot / page_size as usize;
    let offset=slot % page_size as usize;
    let cache_row=slot * kv_heads as usize + kv_head;
    let group=(heads / kv_heads) as usize;
    let entries_base=inverse[0] as usize;
    let sequence_base=inverse[1] as usize;
    let query_base=inverse[2] as usize;
    let visibility_base=query_base+queries as usize;
    let block_offsets_base=visibility_base+2*sequences as usize;
    let block_maxima_base=block_offsets_base+sequences as usize+1;
    let mut dk_acc=Array::<f32>::new((key_dim+lanes-1)/lanes);
    let mut dv_acc=Array::<f32>::new((value_dim+lanes-1)/lanes);
    let mut dkp_acc=Array::<f32>::new((position_dim+lanes-1)/lanes);
    #[unroll]
    for j in 0usize..((key_dim+lanes-1)/lanes) { dk_acc[j]=0.0; }
    #[unroll]
    for j in 0usize..((value_dim+lanes-1)/lanes) { dv_acc[j]=0.0; }
    if comptime!(mla) {
        #[unroll]
        for j in 0usize..((position_dim+lanes-1)/lanes) { dkp_acc[j]=0.0; }
    }
    // Lazy physical-row cache. Do not eagerly read reserved/tail pages:
    // a row is loaded only inside its FIRST nonzero-denominator contribution.
    // It remains valid across query heads and links sharing this physical row.
    let mut cached_key=Array::<f32>::new(cache_key_slots);
    let mut cached_value=Array::<f32>::new(cache_value_slots);
    let mut cached_position=Array::<f32>::new(cache_position_slots);
    #[unroll]
    for j in 0usize..cache_key_slots { cached_key[j]=0.0; }
    #[unroll]
    for j in 0usize..cache_value_slots { cached_value[j]=0.0; }
    #[unroll]
    for j in 0usize..cache_position_slots { cached_position[j]=0.0; }
    let mut row_loaded=false;
    // All lanes take the same branches and participate in every plane_sum.
    let mut link=inverse[3+page] as usize;
    let link_end=inverse[4+page] as usize;
    while link<link_end {
        let sequence=inverse[entries_base+2*link] as usize;
        let logical_page=inverse[entries_base+2*link+1] as usize;
        let token=logical_page*page_size as usize+offset;
        let length=metadata[2*queries as usize+sequence] as usize;
        if token<length {
            let row_begin=inverse[sequence_base+sequence] as usize;
            let row_end=inverse[sequence_base+sequence+1] as usize;
            let mut row_index=row_begin;
            let mut monotonic=false;
            if comptime!(causal && prune_queries) {
                monotonic=inverse[visibility_base+sequence]!=0;
                let maximum=inverse[visibility_base+sequences as usize+sequence] as usize;
                if token>maximum { row_index=row_end; }
                // Binary search only a sequence verified nondecreasing on host.
                // The retained suffix is in the ORIGINAL packed row order.
                if monotonic && row_index<row_end {
                    let first=inverse[query_base+row_index] as usize;
                    // Common decode/chunked-prefill prefix: every query can
                    // see this token. Avoid a binary search in that case.
                    if token>metadata[queries as usize+first] as usize {
                        let mut right=row_end;
                        while row_index<right {
                            let mid=row_index+(right-row_index)/2;
                            let candidate=inverse[query_base+mid] as usize;
                            let pos=metadata[queries as usize+candidate] as usize;
                            if pos<token { row_index=mid+1; } else { right=mid; }
                        }
                    }
                }
            }
            while row_index<row_end {
                let mut row_stop=row_end;
                let mut scan=true;
                if comptime!(causal && prune_queries) {
                    if !monotonic {
                        let block=(row_index-row_begin)/query_block_rows;
                        row_stop=usize::min(row_end,row_begin+(block+1)*query_block_rows);
                        let base=inverse[block_offsets_base+sequence] as usize;
                        scan=token<=inverse[block_maxima_base+base+block] as usize;
                    }
                }
                // These integer branches are identical in every lane; no lane
                // bypasses a plane_sum while a neighbour participates in it.
                if scan {
                    while row_index<row_stop {
                        let row=inverse[query_base+row_index] as usize;
                        let mut visible=true;
                        if comptime!(causal) {
                            if !monotonic { visible=token<=metadata[queries as usize+row] as usize; }
                        }
                        if visible {
                            let mut head=kv_head*group;
                            let head_end=head+group;
                            while head<head_end {
                                let stat=(row*heads as usize+head)*3;
                                let denominator=statistics[stat+1];
                                if denominator!=0.0 {
                                    if comptime!(cache_history) {
                                        if !row_loaded {
                                            #[unroll]
                                            for j in 0usize..((key_dim+lanes-1)/lanes) {
                                                let d=lane+j*lanes;
                                                if d<key_dim { cached_key[j]=f32::cast_from(k[cache_row*key_dim+d]); }
                                            }
                                            if comptime!(!mla && (need_dk || need_dkp)) {
                                                #[unroll]
                                                for j in 0usize..((value_dim+lanes-1)/lanes) {
                                                    let d=lane+j*lanes;
                                                    if d<value_dim { cached_value[j]=f32::cast_from(v[cache_row*value_dim+d]); }
                                                }
                                            }
                                            if comptime!(mla) {
                                                #[unroll]
                                                for j in 0usize..((position_dim+lanes-1)/lanes) {
                                                    let d=lane+j*lanes;
                                                    if d<position_dim { cached_position[j]=f32::cast_from(kp[cache_row*position_dim+d]); }
                                                }
                                            }
                                            row_loaded=true;
                                        }
                                    }
                                    let qrow=(row*heads as usize+head)*key_dim;
                                    let grow=(row*heads as usize+head)*value_dim;
                                    let prow=(row*heads as usize+head)*position_dim;
                                    let mut dot=0.0f32;
                                    #[unroll]
                                    for j in 0usize..((key_dim+lanes-1)/lanes) {
                                        let d=lane+j*lanes;
                                        if d<key_dim {
                                            let mut key=0.0f32;
                                            if comptime!(cache_history) { key=cached_key[j]; }
                                            else { key=f32::cast_from(k[cache_row*key_dim+d]); }
                                            dot=fma(f32::cast_from(q[qrow+d]),key,dot);
                                        }
                                    }
                                    if comptime!(mla) {
                                        #[unroll]
                                        for j in 0usize..((position_dim+lanes-1)/lanes) {
                                            let d=lane+j*lanes;
                                            if d<position_dim {
                                                let mut position=0.0f32;
                                                if comptime!(cache_history) { position=cached_position[j]; }
                                                else { position=f32::cast_from(kp[cache_row*position_dim+d]); }
                                                dot=fma(f32::cast_from(qp[prow+d]),position,dot);
                                            }
                                        }
                                    }
                                    let score=plane_sum(dot)*scale;
                                    let probability=(score-statistics[stat]).exp()/denominator;
                                    let mut ds=0.0f32;
                                    if comptime!(need_dk || need_dkp) {
                                        let mut partial=0.0f32;
                                        #[unroll]
                                        for j in 0usize..((value_dim+lanes-1)/lanes) {
                                            let d=lane+j*lanes;
                                            if d<value_dim {
                                                let mut value=0.0f32;
                                                if comptime!(cache_history) {
                                                    if comptime!(mla) { value=cached_key[j]; }
                                                    else { value=cached_value[j]; }
                                                } else { value=f32::cast_from(v[cache_row*value_dim+d]); }
                                                partial=fma(f32::cast_from(grad[grow+d]),value,partial);
                                            }
                                        }
                                        ds=probability*(plane_sum(partial)-statistics[stat+2])*scale;
                                    }
                                    if comptime!(need_dk) {
                                        #[unroll]
                                        for j in 0usize..((key_dim+lanes-1)/lanes) {
                                            let d=lane+j*lanes;
                                            if d<key_dim {
                                                dk_acc[j]=fma(ds,f32::cast_from(q[qrow+d]),dk_acc[j]);
                                                if comptime!(mla) {
                                                    // A latent is BOTH key and value; retain both derivatives.
                                                    dk_acc[j]=fma(probability,f32::cast_from(grad[grow+d]),dk_acc[j]);
                                                }
                                            }
                                        }
                                    }
                                    if comptime!(need_dv && !mla) {
                                        #[unroll]
                                        for j in 0usize..((value_dim+lanes-1)/lanes) {
                                            let d=lane+j*lanes;
                                            if d<value_dim { dv_acc[j]=fma(probability,f32::cast_from(grad[grow+d]),dv_acc[j]); }
                                        }
                                    }
                                    if comptime!(need_dkp && mla) {
                                        #[unroll]
                                        for j in 0usize..((position_dim+lanes-1)/lanes) {
                                            let d=lane+j*lanes;
                                            if d<position_dim { dkp_acc[j]=fma(ds,f32::cast_from(qp[prow+d]),dkp_acc[j]); }
                                        }
                                    }
                                }
                                head+=1;
                            }
                        }
                        row_index+=1;
                    }
                } else { row_index=row_stop; }
            }
        }
        link+=1;
    }
    // Inactive pages/tail slots also store ZERO. No clearing pass is required.
    if comptime!(need_dk) {
        #[unroll]
        for j in 0usize..((key_dim+lanes-1)/lanes) {
            let d=lane+j*lanes;
            if d<key_dim { dk[cache_row*key_dim+d]=F::cast_from(dk_acc[j]); }
        }
    }
    if comptime!(need_dv && !mla) {
        #[unroll]
        for j in 0usize..((value_dim+lanes-1)/lanes) {
            let d=lane+j*lanes;
            if d<value_dim { dv[cache_row*value_dim+d]=F::cast_from(dv_acc[j]); }
        }
    }
    if comptime!(need_dkp && mla) {
        #[unroll]
        for j in 0usize..((position_dim+lanes-1)/lanes) {
            let d=lane+j*lanes;
            if d<position_dim { dkp[cache_row*position_dim+d]=F::cast_from(dkp_acc[j]); }
        }
    }
}
