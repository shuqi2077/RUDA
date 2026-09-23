//! One plane per packed query/head/partition. A single-partition specialization
//! writes the final output; multi-partition specializations write FP32 online
//! statistics for a separate stable merge. No score tensor or KV head repeat.
//! Page-table lookup and integer division are hoisted out of each token loop.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn attention<F: Float, O: Float>(
    q: &Array<F>, k: &Array<F>, v: &Array<F>, qp: &Array<F>, kp: &Array<F>,
    metadata: &Array<u32>, out: &mut Array<O>, scale: f32,
    queries: u32, sequences: u32, table_width: u32, page_size: u32,
    heads: u32, kv_heads: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize,
    #[comptime] position_dim: usize, #[comptime] lanes: usize,
    #[comptime] causal: bool, #[comptime] mla: bool,
    #[comptime] splits: usize,
    #[define(F)] _dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let sequence = metadata[row] as usize;
    let pos = metadata[queries as usize + row] as usize;
    let length = metadata[2*queries as usize + sequence] as usize;
    let mut end = length;
    if comptime!(causal) { if pos < end { end = pos + 1; } }
    // Quotient/remainder partitioning avoids end * split overflow at U32 limits.
    // Every query partitions its own visible prefix, including ragged prefill.
    let split = RUDA_POS_Z as usize;
    let base = end / splits;
    let extra = end % splits;
    let begin = split * base + usize::min(split, extra);
    let mut span = base;
    if split < extra { span += 1; }
    end = begin + span;
    let kv_head = head / (heads / kv_heads) as usize;
    let mut query = Array::<f32>::new(((key_dim+lanes-1)/lanes));
    let mut query_pos = Array::<f32>::new(((position_dim+lanes)/lanes));
    let mut acc = Array::<f32>::new(((value_dim+lanes-1)/lanes));
    #[unroll]
    for j in 0usize..((key_dim+lanes-1)/lanes) {
        let d = lane + j*lanes;
        query[j] = 0.0;
        if d < key_dim { query[j] = f32::cast_from(q[(row*heads as usize+head)*key_dim+d]); }
    }
    if comptime!(mla) {
        #[unroll]
        for j in 0usize..((position_dim+lanes-1)/lanes) {
            let d=lane+j*lanes;
            query_pos[j]=0.0;
            if d<position_dim { query_pos[j]=f32::cast_from(qp[(row*heads as usize+head)*position_dim+d]); }
        }
    }
    #[unroll]
    for j in 0usize..((value_dim+lanes-1)/lanes) { acc[j]=0.0; }
    let mut maximum = 0.0f32;
    let mut denominator = 0.0f32;
    let mut token = begin;
    while token < end {
        let logical_page = token / page_size as usize;
        let page_offset = token % page_size as usize;
        let page = metadata[2*queries as usize+sequences as usize+sequence*table_width as usize+logical_page] as usize;
        // end - token makes the addition safe, even near the U32 index limit.
        let stop = token + usize::min(page_size as usize - page_offset, end - token);
        let mut cache_row = (page*page_size as usize+page_offset)*kv_heads as usize+kv_head;
        while token < stop {
            let mut partial = 0.0f32;
            #[unroll]
            for j in 0usize..((key_dim+lanes-1)/lanes) {
                let d=lane+j*lanes;
                if d<key_dim { partial=fma(query[j],f32::cast_from(k[cache_row*key_dim+d]),partial); }
            }
            if comptime!(mla) {
                #[unroll]
                for j in 0usize..((position_dim+lanes-1)/lanes) {
                    let d=lane+j*lanes;
                    if d<position_dim { partial=fma(query_pos[j],f32::cast_from(kp[cache_row*position_dim+d]),partial); }
                }
            }
            let score=plane_sum(partial)*scale;
            // First-item initialization avoids -inf - -inf for a valid finite row.
            let mut next_max=score;
            if token!=begin { next_max=f32::max(maximum,score); }
            let mut old_scale=0.0f32;
            if token!=begin { old_scale=(maximum-next_max).exp(); }
            let weight=(score-next_max).exp();
            denominator=fma(denominator,old_scale,weight);
            #[unroll]
            for j in 0usize..((value_dim+lanes-1)/lanes) {
                let d=lane+j*lanes;
                if d<value_dim { acc[j]=fma(weight,f32::cast_from(v[cache_row*value_dim+d]),acc[j]*old_scale); }
            }
            maximum=next_max;
            token+=1;
            if token < stop { cache_row+=kv_heads as usize; }
        }
    }
    let output_row = row*heads as usize+head;
    #[unroll]
    for j in 0usize..((value_dim+lanes-1)/lanes) {
        let d=lane+j*lanes;
        if d<value_dim {
            if comptime!(splits == 1) {
                let mut result=0.0f32;
                if span!=0 { result=acc[j]/denominator; }
                out[output_row*value_dim+d]=O::cast_from(result);
            } else {
                // Do not normalize/cast to the model dtype here. Both sums and
                // values stay FP32 until the final inter-partition reduction.
                out[(output_row*splits+split)*(value_dim+2)+d]=O::cast_from(acc[j]);
            }
        }
    }
    if comptime!(splits > 1) {
        if lane == 0 {
            let offset=(output_row*splits+split)*(value_dim+2)+value_dim;
            // Empty partitions have denominator=0, finite maximum=0 and zero
            // accumulators. All workspace slots are written on every launch.
            out[offset]=O::cast_from(maximum);
            out[offset+1]=O::cast_from(denominator);
        }
    }
}

/// Merge unnormalized (maximum, denominator, accumulator) tuples. Empty splits
/// are ignored *before* exponentiation; an entirely empty row returns zeros.
#[ruda(launch)]
pub(super) fn merge<F: Float>(
    partials: &Array<f32>, out: &mut Array<F>, heads: u32,
    #[comptime] value_dim: usize, #[comptime] lanes: usize,
    #[comptime] splits: usize, #[define(F)] _dtype: StorageType,
) {
    let row=RUDA_POS_X as usize;
    let head=RUDA_POS_Y as usize;
    let lane=UNIT_POS as usize;
    let output_row=row*heads as usize+head;
    let mut acc=Array::<f32>::new((value_dim+lanes-1)/lanes);
    #[unroll]
    for j in 0usize..((value_dim+lanes-1)/lanes) { acc[j]=0.0; }
    let mut maximum=0.0f32;
    let mut denominator=0.0f32;
    // Do not unroll up to 32 partitions: that would inflate the instruction body.
    for split in 0usize..splits {
        let offset=(output_row*splits+split)*(value_dim+2);
        let part_den=partials[offset+value_dim+1];
        // Propagate NaNs instead of silently dropping a non-finite partition.
        if part_den != 0.0 {
            let part_max=partials[offset+value_dim];
            let mut next_max=part_max;
            let mut old_scale=0.0f32;
            if denominator != 0.0 {
                next_max=f32::max(maximum,part_max);
                old_scale=(maximum-next_max).exp();
            }
            let part_scale=(part_max-next_max).exp();
            #[unroll]
            for j in 0usize..((value_dim+lanes-1)/lanes) {
                let d=lane+j*lanes;
                if d<value_dim { acc[j]=fma(partials[offset+d],part_scale,acc[j]*old_scale); }
            }
            denominator=fma(part_den,part_scale,denominator*old_scale);
            maximum=next_max;
        }
    }
    #[unroll]
    for j in 0usize..((value_dim+lanes-1)/lanes) {
        let d=lane+j*lanes;
        if d<value_dim {
            let result=if denominator!=0.0 { acc[j]/denominator } else { 0.0f32 };
            out[output_row*value_dim+d]=F::cast_from(result);
        }
    }
}

#[ruda(launch)]
pub(super) fn append<F: Float>(
    new_k: &Array<F>, new_v: &Array<F>, k: &mut Array<F>, v: &mut Array<F>,
    metadata: &Array<u32>, queries: u32, sequences: u32, table_width: u32,
    page_size: u32, kv_heads: u32, key_dim: u32, value_dim: u32,
    #[define(F)] _dtype: StorageType,
) {
    let index=ABSOLUTE_POS;
    let width=kv_heads as usize*usize::max(key_dim as usize,value_dim as usize);
    let row=index/width;
    if row < queries as usize {
        let offset=index%width;
        let seq=metadata[row] as usize;
        let pos=metadata[queries as usize+row] as usize;
        let page=metadata[2*queries as usize+sequences as usize+seq*table_width as usize+pos/page_size as usize] as usize;
        let slot=page*page_size as usize+pos%page_size as usize;
        if offset < kv_heads as usize*key_dim as usize {
            k[slot*kv_heads as usize*key_dim as usize+offset]=new_k[row*kv_heads as usize*key_dim as usize+offset];
        }
        if offset < kv_heads as usize*value_dim as usize {
            v[slot*kv_heads as usize*value_dim as usize+offset]=new_v[row*kv_heads as usize*value_dim as usize+offset];
        }
    }
}
