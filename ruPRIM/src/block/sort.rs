use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaCompare, merge};

/// Stable merge sort in blocked register order. Scratch arrays each contain
/// `threads * items_per_thread` items and must not alias one another.
#[ruda]
pub fn merge_sort_keys<K: RudaPrimitive, C: RudaCompare<K>>(
    keys: &mut Array<K>,
    source: &mut SharedMemory<K>,
    destination: &mut SharedMemory<K>,
    compare: &C,
    valid_items: usize,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
) {
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            source[start + item] = keys[item];
        }
    }
    sync_ruda();
    let mut run = 1usize;
    while run < valid_items {
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                let rank = merge::rank::<K, C>(source, compare, index, valid_items, run, 0);
                destination[rank] = source[index];
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                source[index] = destination[index];
            }
        }
        sync_ruda();
        run *= 2;
    }
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            keys[item] = source[start + item];
        }
    }
    sync_ruda();
}

/// Stable key/value merge sort; every value follows its original key.
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
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
) {
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            source_keys[start + item] = keys[item];
            source_values[start + item] = values[item];
        }
    }
    sync_ruda();
    let mut run = 1usize;
    while run < valid_items {
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                let rank = merge::rank::<K, C>(source_keys, compare, index, valid_items, run, 0);
                destination_keys[rank] = source_keys[index];
                destination_values[rank] = source_values[index];
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                source_keys[index] = destination_keys[index];
                source_values[index] = destination_values[index];
            }
        }
        sync_ruda();
        run *= 2;
    }
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            keys[item] = source_keys[start + item];
            values[item] = source_values[start + item];
        }
    }
    sync_ruda();
}
