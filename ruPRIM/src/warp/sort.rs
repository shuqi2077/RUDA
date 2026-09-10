use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaCompare, merge};

/// Stable logical-warp key/value merge sort, in blocked register order.
/// Scratch regions cover the block's threads times `items_per_lane`; each
/// logical warp uses a disjoint region. Only subgroup barriers are used.
#[ruda]
pub fn merge_sort_pairs<K: RudaPrimitive, V: RudaPrimitive, C: RudaCompare<K>>(
    keys: &mut Array<K>,
    values: &mut Array<V>,
    source_keys: &mut SharedMemory<K>,
    destination_keys: &mut SharedMemory<K>,
    source_values: &mut SharedMemory<V>,
    destination_values: &mut SharedMemory<V>,
    compare: &C,
    valid_items: usize,
    #[comptime] width: u32,
    #[comptime] items_per_lane: usize,
) {
    let lane = UNIT_POS_PLANE % width;
    let base = (UNIT_POS - lane) as usize * items_per_lane;
    let start = lane as usize * items_per_lane;
    #[unroll]
    for item in 0..items_per_lane {
        if start + item < valid_items {
            source_keys[base + start + item] = keys[item];
            source_values[base + start + item] = values[item];
        }
    }
    sync_plane();
    let mut run = 1usize;
    while run < valid_items {
        #[unroll]
        for item in 0..items_per_lane {
            let index = start + item;
            if index < valid_items {
                let rank = merge::rank::<K, C>(source_keys, compare, index, valid_items, run, base);
                destination_keys[base + rank] = source_keys[base + index];
                destination_values[base + rank] = source_values[base + index];
            }
        }
        sync_plane();
        #[unroll]
        for item in 0..items_per_lane {
            let index = start + item;
            if index < valid_items {
                source_keys[base + index] = destination_keys[base + index];
                source_values[base + index] = destination_values[base + index];
            }
        }
        sync_plane();
        run *= 2;
    }
    #[unroll]
    for item in 0..items_per_lane {
        if start + item < valid_items {
            keys[item] = source_keys[base + start + item];
            values[item] = source_values[base + start + item];
        }
    }
    sync_plane();
}

/// Stable logical-warp merge sort for keys only.
#[ruda]
pub fn merge_sort_keys<K: RudaPrimitive, C: RudaCompare<K>>(
    keys: &mut Array<K>,
    source: &mut SharedMemory<K>,
    destination: &mut SharedMemory<K>,
    compare: &C,
    valid_items: usize,
    #[comptime] width: u32,
    #[comptime] items_per_lane: usize,
) {
    let lane = UNIT_POS_PLANE % width;
    let base = (UNIT_POS - lane) as usize * items_per_lane;
    let start = lane as usize * items_per_lane;
    #[unroll]
    for item in 0..items_per_lane {
        if start + item < valid_items {
            source[base + start + item] = keys[item];
        }
    }
    sync_plane();
    let mut run = 1usize;
    while run < valid_items {
        #[unroll]
        for item in 0..items_per_lane {
            let index = start + item;
            if index < valid_items {
                let rank = merge::rank::<K, C>(source, compare, index, valid_items, run, base);
                destination[base + rank] = source[base + index];
            }
        }
        sync_plane();
        #[unroll]
        for item in 0..items_per_lane {
            if start + item < valid_items {
                source[base + start + item] = destination[base + start + item];
            }
        }
        sync_plane();
        run *= 2;
    }
    #[unroll]
    for item in 0..items_per_lane {
        if start + item < valid_items {
            keys[item] = source[base + start + item];
        }
    }
    sync_plane();
}
