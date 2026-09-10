use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaSum, radix::RudaRadixKey};

/// Stable bitwise radix sort. Key/value and rank scratch each contain the
/// entire block tile. The half-open bit interval applies after radix encoding.
#[ruda]
pub fn sort_pairs<K: RudaRadixKey, V: RudaPrimitive>(
    keys: &mut Array<K>, values: &mut Array<V>,
    key_scratch: &mut SharedMemory<K>, value_scratch: &mut SharedMemory<V>,
    rank_scratch: &mut SharedMemory<u32>, valid_items: usize,
    #[comptime] threads: usize, #[comptime] items_per_thread: usize,
    #[comptime] begin_bit: u32, #[comptime] end_bit: u32,
    #[comptime] descending: bool,
) {
    let start = UNIT_POS as usize * items_per_thread;
    let mut flags = Array::<u32>::new(items_per_thread);
    let mut prefixes = Array::<u32>::new(items_per_thread);
    let sum = RudaSum {};
    for bit in begin_bit..end_bit {
        #[unroll]
        for item in 0..items_per_thread {
            let mut flag = 0u32;
            if start + item < valid_items {
                flag = u32::cast_from((K::ordered_bits(keys[item]) >> u64::cast_from(bit)) & 1u64);
                if descending { flag ^= 1; }
            }
            flags[item] = flag;
        }
        // Padded flags are zero, so scanning the entire tile also handles empty input.
        crate::block::inclusive_scan::<u32, RudaSum>(&flags, &mut prefixes, rank_scratch, &sum,
            threads * items_per_thread, threads, items_per_thread);
        let ones = rank_scratch[threads * items_per_thread - 1] as usize;
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                let preceding = (prefixes[item] - flags[item]) as usize;
                let rank = if flags[item] == 0 { index - preceding } else { valid_items - ones + preceding };
                key_scratch[rank] = keys[item];
                value_scratch[rank] = values[item];
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                keys[item] = key_scratch[index];
                values[item] = value_scratch[index];
            }
        }
        sync_ruda();
    }
}

/// Stable key-only radix sort, with no auxiliary value payload.
#[ruda]
pub fn sort_keys<K: RudaRadixKey>(
    keys: &mut Array<K>, key_scratch: &mut SharedMemory<K>,
    rank_scratch: &mut SharedMemory<u32>, valid_items: usize,
    #[comptime] threads: usize, #[comptime] items_per_thread: usize,
    #[comptime] begin_bit: u32, #[comptime] end_bit: u32,
    #[comptime] descending: bool,
) {
    let start = UNIT_POS as usize * items_per_thread;
    let mut flags = Array::<u32>::new(items_per_thread);
    let mut prefixes = Array::<u32>::new(items_per_thread);
    let sum = RudaSum {};
    for bit in begin_bit..end_bit {
        #[unroll]
        for item in 0..items_per_thread {
            let mut flag = 0u32;
            if start + item < valid_items {
                flag = u32::cast_from((K::ordered_bits(keys[item]) >> u64::cast_from(bit)) & 1u64);
                if descending { flag ^= 1; }
            }
            flags[item] = flag;
        }
        crate::block::inclusive_scan::<u32, RudaSum>(&flags, &mut prefixes, rank_scratch, &sum,
            threads * items_per_thread, threads, items_per_thread);
        let ones = rank_scratch[threads * items_per_thread - 1] as usize;
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                let preceding = (prefixes[item] - flags[item]) as usize;
                let rank = if flags[item] == 0 { index - preceding } else { valid_items - ones + preceding };
                key_scratch[rank] = keys[item];
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items_per_thread {
            if start + item < valid_items { keys[item] = key_scratch[start + item]; }
        }
        sync_ruda();
    }
}
