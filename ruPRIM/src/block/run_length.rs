use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::RudaSum;
use crate::collective::record::{RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};

#[ruda]
pub fn prepare_access<T: RudaType, L: Numeric, I: RudaRead<T>, N: RudaRead<L>, S: RudaWrite<T>>(
    run_values: &I, run_lengths: &N, values: &mut S, ends: &mut SharedMemory<u64>,
    num_runs: usize, #[comptime] threads: usize, #[comptime] runs_per_thread: usize,
) -> u64 {
    let start = UNIT_POS as usize * runs_per_thread;
    let mut lengths = Array::<u64>::new(runs_per_thread);
    let mut prefixes = Array::<u64>::new(runs_per_thread);
    #[unroll]
    for item in 0..runs_per_thread {
        lengths[item] = 0;
        if start + item < num_runs {
            values.write(start + item, run_values.read(item));
            lengths[item] = u64::cast_from(run_lengths.read(item));
        }
    }
    let sum = RudaSum {};
    crate::block::inclusive_scan::<u64, RudaSum>(&lengths, &mut prefixes, ends, &sum,
        threads * runs_per_thread, threads, runs_per_thread);
    let total = ends[threads * runs_per_thread - 1];
    sync_ruda();
    total
}

#[ruda]
pub fn decode_access<T: RudaType, C: Numeric, S: RudaRead<T>, W: RudaWrite<T>, O: RudaWrite<C>>(
    values: &S, ends: &SharedMemory<u64>, output: &mut W, relative_offsets: &mut O,
    window_offset: u64, num_runs: usize, #[comptime] items_per_thread: usize,
) {
    let start = UNIT_POS as u64 * items_per_thread as u64;
    let mut total = 0u64;
    if num_runs > 0 { total = ends[num_runs - 1]; }
    #[unroll]
    for item in 0..items_per_thread {
        let relative = start + item as u64;
        if window_offset < total && relative < total - window_offset {
            let position = window_offset + relative;
            let mut low = 0usize;
            let mut high = num_runs;
            while low < high {
                let middle = low + (high - low) / 2;
                if ends[middle] <= position { low = middle + 1; } else { high = middle; }
            }
            let mut begin = 0u64;
            if low > 0 { begin = ends[low - 1]; }
            output.write(item, values.read(low));
            relative_offsets.write(item, C::cast_from(position - begin));
        }
    }
}

/// Prepare inclusive run ends once, then reuse them for any decoded window.
/// Returns the total decoded length. The run lengths' sum must fit in U64.
#[ruda]
pub fn prepare<T: RudaPrimitive>(
    run_values: &Array<T>, run_lengths: &Array<u64>,
    values: &mut SharedMemory<T>, ends: &mut SharedMemory<u64>,
    num_runs: usize, #[comptime] threads: usize, #[comptime] runs_per_thread: usize,
) -> u64 {
    let start = UNIT_POS as usize * runs_per_thread;
    let mut lengths = Array::<u64>::new(runs_per_thread);
    let mut prefixes = Array::<u64>::new(runs_per_thread);
    #[unroll]
    for item in 0..runs_per_thread {
        let mut length = 0u64;
        if start + item < num_runs {
            values[start + item] = run_values[item];
            length = run_lengths[item];
        }
        lengths[item] = length;
    }
    let sum = RudaSum {};
    crate::block::inclusive_scan::<u64, RudaSum>(&lengths, &mut prefixes, ends, &sum,
        threads * runs_per_thread, threads, runs_per_thread);
    let total = ends[threads * runs_per_thread - 1];
    sync_ruda();
    total
}

/// Decode a blocked window, also returning each item's offset within its run.
/// Entries beyond the total decoded length are left unchanged.
#[ruda]
pub fn decode_window<T: RudaPrimitive>(
    values: &SharedMemory<T>, ends: &SharedMemory<u64>,
    output: &mut Array<T>, relative_offsets: &mut Array<u64>,
    window_offset: u64, num_runs: usize,
    #[comptime] items_per_thread: usize,
) {
    let start = UNIT_POS as u64 * items_per_thread as u64;
    let mut total = 0u64;
    if num_runs > 0 { total = ends[num_runs - 1]; }
    #[unroll]
    for item in 0..items_per_thread {
        let relative = start + item as u64;
        if window_offset < total && relative < total - window_offset {
            let position = window_offset + relative;
            let mut low = 0usize;
            let mut high = num_runs;
            while low < high {
                let middle = low + (high - low) / 2;
                if ends[middle] <= position { low = middle + 1; } else { high = middle; }
            }
            let mut begin = 0u64;
            if low > 0 { begin = ends[low - 1]; }
            output[item] = values[low];
            relative_offsets[item] = position - begin;
        }
    }
}
