use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};
use crate::collective::record::RudaRecord;
use crate::collective::{RudaCompare, RudaCompareExpand};
use crate::collective::record::{RudaRecordArray, RudaRecordShared, RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};

/// Record scan in logical-lane order. Every logical lane participates,
/// including inactive data lanes; valid is uniform and in 1..=width.
#[ruda]
pub fn inclusive_scan<T: RudaRecord, O: RudaBinaryOp<T>>(
    value: T, op: &O, valid: u32, #[comptime] width: u32,
) -> T {
    let lane = UNIT_POS_PLANE % width;
    let base = UNIT_POS_PLANE - lane;
    let mut result = value;
    let mut distance = 1u32;
    while distance < width {
        let source = base + select(lane >= distance, lane - distance, lane);
        let left = T::shuffle(result, source);
        if lane >= distance && lane < valid { result = op.combine(left, result); }
        distance *= 2;
    }
    result
}

#[ruda]
pub fn exclusive_scan<T: RudaRecord, O: RudaBinaryOp<T>>(
    value: T, initial: T, op: &O, valid: u32, #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid, width);
    let lane = UNIT_POS_PLANE % width;
    let previous = T::shuffle(inclusive, UNIT_POS_PLANE - select(lane > 0, 1u32, 0u32));
    let mut result = initial;
    if lane > 0 && lane < valid { result = op.combine(initial, previous); }
    result
}

/// Unseeded exclusive scan; output at logical lane zero is unspecified.
#[ruda]
pub fn exclusive_unseeded<T: RudaRecord, O: RudaBinaryOp<T>>(
    value: T, op: &O, valid: u32, #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid, width);
    let lane = UNIT_POS_PLANE % width;
    T::shuffle(inclusive, UNIT_POS_PLANE - select(lane > 0, 1u32, 0u32))
}

#[ruda]
pub fn scan<T: RudaRecord, O: RudaBinaryOp<T>>(
    value: T, initial: T, inclusive_output: &mut T, exclusive_output: &mut T,
    op: &O, valid: u32, #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid, width);
    let lane = UNIT_POS_PLANE % width;
    let previous = T::shuffle(inclusive, UNIT_POS_PLANE - select(lane > 0, 1u32, 0u32));
    let mut exclusive = initial;
    if lane > 0 && lane < valid { exclusive = op.combine(initial, previous); }
    *inclusive_output = inclusive;
    *exclusive_output = exclusive;
    T::shuffle(inclusive, UNIT_POS_PLANE - lane + valid - 1)
}

#[ruda]
pub fn reduce<T: RudaRecord, O: RudaBinaryOp<T>>(
    value: T, op: &O, valid: u32, #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid, width);
    T::shuffle(inclusive, UNIT_POS_PLANE - UNIT_POS_PLANE % width + valid - 1)
}

#[ruda]
pub fn broadcast<T: RudaRecord>(value: T, source: u32, #[comptime] width: u32) -> T {
    T::shuffle(value, UNIT_POS_PLANE - UNIT_POS_PLANE % width + source)
}

/// Segmented reduction at segment heads; head/tail flags use the same
/// conventions as the scalar collectives and width need not be 32.
#[ruda]
pub fn segmented_reduce<T: RudaRecord, O: RudaBinaryOp<T>>(
    value: T, flag: bool, op: &O, #[comptime] head_flags: bool, #[comptime] width: u32,
) -> T {
    let lane = UNIT_POS_PLANE % width;
    let base = UNIT_POS_PLANE - lane;
    let next = plane_shuffle(flag, base + min(lane + 1, width - 1));
    let mut boundary = (if head_flags { next } else { flag }) || lane + 1 == width;
    let mut result = value;
    let mut distance = 1u32;
    while distance < width {
        let source = base + min(lane + distance, width - 1);
        let right = T::shuffle(result, source);
        let right_boundary = plane_shuffle(boundary, source);
        if lane + distance < width {
            if !boundary { result = op.combine(result, right); }
            boundary = boundary || right_boundary;
        }
        distance *= 2;
    }
    result
}

/// Logical-warp stable record pair sort. Shared storage covers all block
/// threads; logical groups use disjoint regions and every native lane calls it.
#[ruda]
pub fn merge_sort_pairs<K: RudaRecord, V: RudaRecord, C: RudaCompare<K>>(
    keys: &mut RudaRecordArray<K>, values: &mut RudaRecordArray<V>,
    source: &mut RudaRecordShared<K>, destination: &mut RudaRecordShared<K>,
    source_values: &mut RudaRecordShared<V>, destination_values: &mut RudaRecordShared<V>,
    compare: &C, valid: usize, #[comptime] width: u32, #[comptime] items: usize,
) {
    let lane = UNIT_POS_PLANE % width;
    let base = (UNIT_POS - lane) as usize * items;
    let start = lane as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid {
            source.write(base + start + item, keys.read(item));
            source_values.write(base + start + item, values.read(item));
        }
    }
    sync_plane();
    let mut run = 1usize;
    while run < width as usize * items {
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index < valid {
                let rank = crate::block::record::merge_rank::<K, C>(source, compare, index, valid, run, base);
                destination.write(base + rank, source.read(base + index));
                destination_values.write(base + rank, source_values.read(base + index));
            }
        }
        sync_plane();
        #[unroll]
        for item in 0..items {
            if start + item < valid {
                source.write(base + start + item, destination.read(base + start + item));
                source_values.write(base + start + item, destination_values.read(base + start + item));
            }
        }
        sync_plane();
        run *= 2;
    }
    #[unroll]
    for item in 0..items {
        if start + item < valid {
            keys.write(item, source.read(base + start + item));
            values.write(item, source_values.read(base + start + item));
        }
    }
    sync_plane();
}

#[ruda]
pub fn merge_sort_keys<K: RudaRecord, C: RudaCompare<K>>(
    keys: &mut RudaRecordArray<K>, source: &mut RudaRecordShared<K>, destination: &mut RudaRecordShared<K>,
    compare: &C, valid: usize, #[comptime] width: u32, #[comptime] items: usize,
) {
    let lane = UNIT_POS_PLANE % width;
    let base = (UNIT_POS - lane) as usize * items;
    let start = lane as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid { source.write(base + start + item, keys.read(item)); }
    }
    sync_plane();
    let mut run = 1usize;
    while run < width as usize * items {
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index < valid {
                let rank = crate::block::record::merge_rank::<K, C>(source, compare, index, valid, run, base);
                destination.write(base + rank, source.read(base + index));
            }
        }
        sync_plane();
        #[unroll]
        for item in 0..items {
            if start + item < valid { source.write(base + start + item, destination.read(base + start + item)); }
        }
        sync_plane();
        run *= 2;
    }
    #[unroll]
    for item in 0..items {
        if start + item < valid { keys.write(item, source.read(base + start + item)); }
    }
    sync_plane();
}

/// One record per batch per lane; output capacity is ceil(batches / width).
#[ruda]
pub fn reduce_batched<T: RudaRecord, O: RudaBinaryOp<T>>(
    input: &RudaRecordArray<T>, output: &mut RudaRecordArray<T>, op: &O,
    #[comptime] width: u32, #[comptime] batches: usize, #[comptime] striped: bool,
) {
    let lane = (UNIT_POS_PLANE % width) as usize;
    let per_lane = (batches + width as usize - 1) / width as usize;
    #[unroll]
    for batch in 0..batches {
        let value = reduce::<T, O>(input.read(batch), op, width, width);
        let owner = if striped { batch % width as usize } else { batch / per_lane };
        let slot = if striped { batch / width as usize } else { batch % per_lane };
        if lane == owner { output.write(slot, value); }
    }
}
