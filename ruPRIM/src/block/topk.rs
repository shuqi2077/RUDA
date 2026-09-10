use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaSum, radix::RudaRadixKey};

#[ruda]
fn select_ranks<K: RudaRadixKey>(
    keys: &Array<K>, selected: &mut Array<u32>, ranks: &mut Array<u32>,
    scratch: &mut SharedMemory<u32>, k: usize, valid_items: usize,
    #[comptime] threads: usize, #[comptime] items_per_thread: usize,
    #[comptime] begin_bit: u32, #[comptime] end_bit: u32, #[comptime] largest: bool,
) {
    let start = UNIT_POS as usize * items_per_thread;
    let capacity = threads * items_per_thread;
    let mut candidates = Array::<u32>::new(items_per_thread);
    let mut preferred = Array::<u32>::new(items_per_thread);
    let sum = RudaSum {};
    let mut remaining = min(k, valid_items);
    #[unroll]
    for item in 0..items_per_thread {
        candidates[item] = u32::cast_from(start + item < valid_items);
        selected[item] = 0;
    }
    for step in 0..end_bit - begin_bit {
        let bit = end_bit - 1 - step;
        #[unroll]
        for item in 0..items_per_thread {
            let mut flag = 0u32;
            if candidates[item] != 0 {
                let one = ((K::ordered_bits(keys[item]) >> u64::cast_from(bit)) & 1u64) != 0;
                flag = u32::cast_from(one == largest);
            }
            preferred[item] = flag;
        }
        crate::block::inclusive_scan::<u32, RudaSum>(&preferred, ranks, scratch, &sum, capacity, threads, items_per_thread);
        let count = scratch[capacity - 1] as usize;
        let accept = count <= remaining;
        #[unroll]
        for item in 0..items_per_thread {
            if candidates[item] != 0 {
                if accept {
                    if preferred[item] != 0 { selected[item] = 1; candidates[item] = 0; }
                } else {
                    candidates[item] = preferred[item];
                }
            }
        }
        if accept { remaining -= count; }
        sync_ruda();
    }
    crate::block::inclusive_scan::<u32, RudaSum>(&candidates, ranks, scratch, &sum, capacity, threads, items_per_thread);
    #[unroll]
    for item in 0..items_per_thread {
        if candidates[item] != 0 && ranks[item] as usize <= remaining { selected[item] = 1; }
    }
    crate::block::inclusive_scan::<u32, RudaSum>(selected, ranks, scratch, &sum, capacity, threads, items_per_thread);
}

/// Select min/max K by radix refinement, without sorting the tile. The output
/// prefix has min(k, valid_items) items in input order. Equal keys at the cutoff
/// are selected in input order. Other output positions are unspecified.
#[ruda]
pub fn keys<K: RudaRadixKey>(
    keys: &mut Array<K>, key_scratch: &mut SharedMemory<K>, rank_scratch: &mut SharedMemory<u32>,
    k: usize, valid_items: usize, #[comptime] threads: usize, #[comptime] items_per_thread: usize,
    #[comptime] begin_bit: u32, #[comptime] end_bit: u32, #[comptime] largest: bool,
) {
    let mut selected = Array::<u32>::new(items_per_thread);
    let mut ranks = Array::<u32>::new(items_per_thread);
    select_ranks(keys, &mut selected, &mut ranks, rank_scratch, k, valid_items,
        threads, items_per_thread, begin_bit, end_bit, largest);
    #[unroll]
    for item in 0..items_per_thread {
        if selected[item] != 0 { key_scratch[ranks[item] as usize - 1] = keys[item]; }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let index = UNIT_POS as usize * items_per_thread + item;
        if index < min(k, valid_items) { keys[item] = key_scratch[index]; }
    }
    sync_ruda();
}

#[ruda]
pub fn pairs<K: RudaRadixKey, V: RudaPrimitive>(
    keys: &mut Array<K>, values: &mut Array<V>, key_scratch: &mut SharedMemory<K>,
    value_scratch: &mut SharedMemory<V>, rank_scratch: &mut SharedMemory<u32>,
    k: usize, valid_items: usize, #[comptime] threads: usize, #[comptime] items_per_thread: usize,
    #[comptime] begin_bit: u32, #[comptime] end_bit: u32, #[comptime] largest: bool,
) {
    let mut selected = Array::<u32>::new(items_per_thread);
    let mut ranks = Array::<u32>::new(items_per_thread);
    select_ranks(keys, &mut selected, &mut ranks, rank_scratch, k, valid_items,
        threads, items_per_thread, begin_bit, end_bit, largest);
    #[unroll]
    for item in 0..items_per_thread {
        if selected[item] != 0 {
            let rank = ranks[item] as usize - 1;
            key_scratch[rank] = keys[item];
            value_scratch[rank] = values[item];
        }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let index = UNIT_POS as usize * items_per_thread + item;
        if index < min(k, valid_items) { keys[item] = key_scratch[index]; values[item] = value_scratch[index]; }
    }
    sync_ruda();
}
