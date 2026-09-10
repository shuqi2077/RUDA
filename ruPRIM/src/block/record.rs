use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand, RudaCompare, RudaCompareExpand};
use crate::collective::record::{RudaRecord, RudaRecordArray, RudaRecordShared, RudaRead, RudaWrite, RudaReadExpand, RudaWriteExpand};
use crate::collective::decompose::{RudaDecomposer, RudaDecomposerExpand};
use super::{RudaBlockPrefix, RudaBlockPrefixExpand};

/// Stateful prefix callback; scratch has threads * items + 1 slots. As with
/// the scalar API, the first native subgroup calls the callback and thread
/// zero's returned prefix is used. Valid input is positive.
#[ruda]
pub fn scan_with_prefix<T: RudaRecord, O: RudaBinaryOp<T>, P: RudaBlockPrefix<T>>(
    input: &RudaRecordArray<T>, output: &mut RudaRecordArray<T>, scratch: &mut RudaRecordShared<T>,
    op: &O, callback: &mut P, valid: usize, #[comptime] threads: usize, #[comptime] items: usize,
    #[comptime] exclusive: bool,
) -> T {
    inclusive_scan::<T, O>(input, output, scratch, op, valid, threads, items);
    let aggregate = scratch.read(valid - 1);
    if UNIT_POS < PLANE_DIM {
        let value = callback.prefix(aggregate);
        if UNIT_POS == 0 { scratch.write(threads * items, value); }
    }
    sync_ruda();
    let prefix = scratch.read(threads * items);
    #[unroll]
    for item in 0..items {
        let index = UNIT_POS as usize * items + item;
        if index < valid {
            let mut value = prefix;
            if exclusive {
                if index > 0 { value = op.combine(prefix, scratch.read(index - 1)); }
            } else { value = op.combine(prefix, scratch.read(index)); }
            output.write(item, value);
        }
    }
    sync_ruda();
    aggregate
}

/// Ordered blocked scan of records. All block threads participate; valid is
/// positive and at most threads * items. Scratch has that many record slots.
#[ruda]
pub fn inclusive_scan<T: RudaRecord, O: RudaBinaryOp<T>>(
    input: &RudaRecordArray<T>, output: &mut RudaRecordArray<T>,
    scratch: &mut RudaRecordShared<T>, op: &O, valid: usize,
    #[comptime] threads: usize, #[comptime] items: usize,
) {
    let start = UNIT_POS as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid { scratch.write(start + item, input.read(item)); }
    }
    sync_ruda();
    let mut distance = 1usize;
    while distance < threads * items {
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index < valid {
                let mut value = scratch.read(index);
                if index >= distance { value = op.combine(scratch.read(index - distance), value); }
                output.write(item, value);
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items {
            if start + item < valid { scratch.write(start + item, output.read(item)); }
        }
        sync_ruda();
        distance *= 2;
    }
    #[unroll]
    for item in 0..items {
        if start + item < valid { output.write(item, scratch.read(start + item)); }
    }
    sync_ruda();
}

#[ruda]
pub fn exclusive_scan<T: RudaRecord, O: RudaBinaryOp<T>>(
    input: &RudaRecordArray<T>, output: &mut RudaRecordArray<T>, scratch: &mut RudaRecordShared<T>,
    initial: T, op: &O, valid: usize, #[comptime] threads: usize, #[comptime] items: usize,
) -> T {
    inclusive_scan::<T, O>(input, output, scratch, op, valid, threads, items);
    let aggregate = scratch.read(valid - 1);
    #[unroll]
    for item in 0..items {
        let index = UNIT_POS as usize * items + item;
        if index < valid {
            let mut value = initial;
            if index > 0 { value = op.combine(initial, scratch.read(index - 1)); }
            output.write(item, value);
        }
    }
    sync_ruda();
    aggregate
}

#[ruda]
pub fn reduce<T: RudaRecord, O: RudaBinaryOp<T>>(
    input: &RudaRecordArray<T>, scratch: &mut RudaRecordShared<T>, op: &O,
    valid: usize, #[comptime] threads: usize, #[comptime] items: usize,
) -> T {
    let start = UNIT_POS as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid { scratch.write(start + item, input.read(item)); }
    }
    sync_ruda();
    let mut distance = 1usize;
    while distance < threads * items {
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index % (distance * 2) == 0 && index + distance < valid {
                scratch.write(index, op.combine(scratch.read(index), scratch.read(index + distance)));
            }
        }
        sync_ruda();
        distance *= 2;
    }
    let aggregate = scratch.read(0);
    sync_ruda();
    aggregate
}

#[ruda]
pub fn merge_rank<K: RudaRecord, C: RudaCompare<K>>(
    source: &RudaRecordShared<K>, compare: &C, index: usize, valid: usize, run: usize,
    base: usize,
) -> usize {
    let group = index / (run * 2) * (run * 2);
    let middle = min(group + run, valid);
    let end = min(group + run * 2, valid);
    let key = source.read(base + index);
    let left = index < middle;
    let mut low = if left { middle } else { group };
    let mut high = if left { end } else { middle };
    let begin = low;
    while low < high {
        let probe = low + (high - low) / 2;
        let other = source.read(base + probe);
        let before = if left { compare.before(other, key) } else { !compare.before(key, other) };
        if before { low = probe + 1; } else { high = probe; }
    }
    group + (if left { index - group } else { index - middle }) + low - begin
}

/// Stable merge sorting; every key and value may itself be a nested record.
#[ruda]
pub fn merge_sort_pairs<K: RudaRecord, V: RudaRecord, C: RudaCompare<K>>(
    keys: &mut RudaRecordArray<K>, values: &mut RudaRecordArray<V>,
    source: &mut RudaRecordShared<K>, destination: &mut RudaRecordShared<K>,
    source_values: &mut RudaRecordShared<V>, destination_values: &mut RudaRecordShared<V>,
    compare: &C, valid: usize, #[comptime] items: usize,
) {
    let start = UNIT_POS as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid {
            source.write(start + item, keys.read(item));
            source_values.write(start + item, values.read(item));
        }
    }
    sync_ruda();
    let mut run = 1usize;
    while run < valid {
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index < valid {
                let rank = merge_rank::<K, C>(source, compare, index, valid, run, 0);
                destination.write(rank, source.read(index));
                destination_values.write(rank, source_values.read(index));
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index < valid {
                source.write(index, destination.read(index));
                source_values.write(index, destination_values.read(index));
            }
        }
        sync_ruda();
        run *= 2;
    }
    #[unroll]
    for item in 0..items {
        if start + item < valid {
            keys.write(item, source.read(start + item));
            values.write(item, source_values.read(start + item));
        }
    }
    sync_ruda();
}

/// Stable LSD radix sorting over an arbitrary-width decomposition. Scratch
/// arrays have threads * items entries; bit interval is half-open and valid.
#[ruda]
pub fn radix_sort_pairs<K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K>>(
    keys: &mut RudaRecordArray<K>, values: &mut RudaRecordArray<V>,
    scratch: &mut RudaRecordShared<K>, scratch_values: &mut RudaRecordShared<V>,
    ranks: &mut SharedMemory<u32>, decomposer: &D, valid: usize,
    #[comptime] threads: usize, #[comptime] items: usize,
    #[comptime] begin_bit: usize, #[comptime] end_bit: usize, #[comptime] descending: bool,
) {
    let mut flags = Array::<u32>::new(items);
    let mut prefixes = Array::<u32>::new(items);
    let start = UNIT_POS as usize * items;
    let sum = crate::collective::RudaSum {};
    for bit in begin_bit..end_bit {
        #[unroll]
        for item in 0..items {
            flags[item] = 0;
            if start + item < valid {
                let digit = decomposer.bit(keys.read(item), bit);
                flags[item] = u32::cast_from(digit == descending);
            }
        }
        crate::block::inclusive_scan::<u32, crate::collective::RudaSum>(
            &flags, &mut prefixes, ranks, &sum, threads * items, threads, items);
        let total = ranks[threads * items - 1] as usize;
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index < valid {
                let prefix = prefixes[item] as usize;
                let rank = if flags[item] != 0 { prefix - 1 } else { total + index - prefix };
                scratch.write(rank, keys.read(item));
                scratch_values.write(rank, values.read(item));
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items {
            if start + item < valid {
                keys.write(item, scratch.read(start + item));
                values.write(item, scratch_values.read(start + item));
            }
        }
        sync_ruda();
    }
}

#[ruda]
pub fn merge_sort_keys<K: RudaRecord, C: RudaCompare<K>>(
    keys: &mut RudaRecordArray<K>, source: &mut RudaRecordShared<K>, destination: &mut RudaRecordShared<K>,
    compare: &C, valid: usize, #[comptime] items: usize,
) {
    let start = UNIT_POS as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid { source.write(start + item, keys.read(item)); }
    }
    sync_ruda();
    let mut run = 1usize;
    while run < valid {
        #[unroll]
        for item in 0..items {
            if start + item < valid {
                let rank = merge_rank::<K, C>(source, compare, start + item, valid, run, 0);
                destination.write(rank, source.read(start + item));
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items {
            if start + item < valid { source.write(start + item, destination.read(start + item)); }
        }
        sync_ruda();
        run *= 2;
    }
    #[unroll]
    for item in 0..items {
        if start + item < valid { keys.write(item, source.read(start + item)); }
    }
    sync_ruda();
}

#[ruda]
pub fn radix_sort_keys<K: RudaRecord, D: RudaDecomposer<K>>(
    keys: &mut RudaRecordArray<K>, scratch: &mut RudaRecordShared<K>, ranks: &mut SharedMemory<u32>,
    decomposer: &D, valid: usize, #[comptime] threads: usize, #[comptime] items: usize,
    #[comptime] begin_bit: usize, #[comptime] end_bit: usize, #[comptime] descending: bool,
) {
    let mut flags = Array::<u32>::new(items);
    let mut prefixes = Array::<u32>::new(items);
    let start = UNIT_POS as usize * items;
    let sum = crate::collective::RudaSum {};
    for bit in begin_bit..end_bit {
        #[unroll]
        for item in 0..items {
            flags[item] = 0;
            if start + item < valid { flags[item] = u32::cast_from(decomposer.bit(keys.read(item), bit) == descending); }
        }
        crate::block::inclusive_scan::<u32, crate::collective::RudaSum>(
            &flags, &mut prefixes, ranks, &sum, threads * items, threads, items);
        let total = ranks[threads * items - 1] as usize;
        #[unroll]
        for item in 0..items {
            let index = start + item;
            if index < valid {
                let prefix = prefixes[item] as usize;
                let rank = if flags[item] != 0 { prefix - 1 } else { total + index - prefix };
                scratch.write(rank, keys.read(item));
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items {
            if start + item < valid { keys.write(item, scratch.read(start + item)); }
        }
        sync_ruda();
    }
}

#[ruda]
pub fn topk_ranks<K: RudaRecord, D: RudaDecomposer<K>>(
    keys: &RudaRecordArray<K>, selected: &mut Array<u32>, ranks: &mut Array<u32>, scratch: &mut SharedMemory<u32>,
    decomposer: &D, k: usize, valid: usize, #[comptime] threads: usize, #[comptime] items: usize,
    #[comptime] begin_bit: usize, #[comptime] end_bit: usize, #[comptime] largest: bool,
) {
    let start = UNIT_POS as usize * items;
    let capacity = threads * items;
    let mut candidates = Array::<u32>::new(items);
    let mut preferred = Array::<u32>::new(items);
    let sum = crate::collective::RudaSum {};
    let mut remaining = min(k, valid);
    #[unroll]
    for item in 0..items {
        candidates[item] = u32::cast_from(start + item < valid);
        selected[item] = 0;
    }
    for step in 0..end_bit - begin_bit {
        let bit = end_bit - 1 - step;
        #[unroll]
        for item in 0..items {
            preferred[item] = 0;
            if candidates[item] != 0 { preferred[item] = u32::cast_from(decomposer.bit(keys.read(item), bit) == largest); }
        }
        crate::block::inclusive_scan::<u32, crate::collective::RudaSum>(&preferred, ranks, scratch, &sum, capacity, threads, items);
        let count = scratch[capacity - 1] as usize;
        let accept = count <= remaining;
        #[unroll]
        for item in 0..items {
            if candidates[item] != 0 {
                if accept {
                    if preferred[item] != 0 { selected[item] = 1; candidates[item] = 0; }
                } else { candidates[item] = preferred[item]; }
            }
        }
        if accept { remaining -= count; }
        sync_ruda();
    }
    crate::block::inclusive_scan::<u32, crate::collective::RudaSum>(&candidates, ranks, scratch, &sum, capacity, threads, items);
    #[unroll]
    for item in 0..items {
        if candidates[item] != 0 && ranks[item] as usize <= remaining { selected[item] = 1; }
    }
    crate::block::inclusive_scan::<u32, crate::collective::RudaSum>(selected, ranks, scratch, &sum, capacity, threads, items);
}

#[ruda]
pub fn topk_pairs<K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K>>(
    keys: &mut RudaRecordArray<K>, values: &mut RudaRecordArray<V>, scratch: &mut RudaRecordShared<K>,
    scratch_values: &mut RudaRecordShared<V>, scratch_ranks: &mut SharedMemory<u32>, decomposer: &D,
    k: usize, valid: usize, #[comptime] threads: usize, #[comptime] items: usize,
    #[comptime] begin_bit: usize, #[comptime] end_bit: usize, #[comptime] largest: bool,
) {
    let mut selected = Array::<u32>::new(items);
    let mut ranks = Array::<u32>::new(items);
    topk_ranks::<K, D>(keys, &mut selected, &mut ranks, scratch_ranks, decomposer, k, valid, threads, items, begin_bit, end_bit, largest);
    #[unroll]
    for item in 0..items {
        if selected[item] != 0 {
            let rank = ranks[item] as usize - 1;
            scratch.write(rank, keys.read(item));
            scratch_values.write(rank, values.read(item));
        }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items {
        let index = UNIT_POS as usize * items + item;
        if index < min(k, valid) { keys.write(item, scratch.read(index)); values.write(item, scratch_values.read(index)); }
    }
    sync_ruda();
}

#[ruda]
pub fn topk_keys<K: RudaRecord, D: RudaDecomposer<K>>(
    keys: &mut RudaRecordArray<K>, scratch: &mut RudaRecordShared<K>, scratch_ranks: &mut SharedMemory<u32>,
    decomposer: &D, k: usize, valid: usize, #[comptime] threads: usize, #[comptime] items: usize,
    #[comptime] begin_bit: usize, #[comptime] end_bit: usize, #[comptime] largest: bool,
) {
    let mut selected = Array::<u32>::new(items);
    let mut ranks = Array::<u32>::new(items);
    topk_ranks::<K, D>(keys, &mut selected, &mut ranks, scratch_ranks, decomposer, k, valid, threads, items, begin_bit, end_bit, largest);
    #[unroll]
    for item in 0..items {
        if selected[item] != 0 { scratch.write(ranks[item] as usize - 1, keys.read(item)); }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items {
        let index = UNIT_POS as usize * items + item;
        if index < min(k, valid) { keys.write(item, scratch.read(index)); }
    }
    sync_ruda();
}
