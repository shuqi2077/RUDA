use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Sort-based histogram without atomic counter requirements. Shared sort
/// scratch holds the entire tile. Samples are valid integral bin indices.
#[ruda]
pub fn sort_histogram<T: Numeric, C: Numeric>(
    samples: &Array<T>, histogram: &mut SharedMemory<C>,
    source: &mut SharedMemory<T>, destination: &mut SharedMemory<T>, valid: usize,
    #[comptime] bins: usize, #[comptime] threads: usize, #[comptime] items: usize,
    #[comptime] accumulate: bool,
) {
    let mut sorted = Array::<T>::new(items);
    #[unroll]
    for item in 0..items {
        if UNIT_POS as usize * items + item < valid { sorted[item] = samples[item]; }
    }
    let compare = crate::collective::RudaAscending {};
    crate::block::sort::merge_sort_keys::<T, crate::collective::RudaAscending>(
        &mut sorted, source, destination, &compare, valid, threads, items);
    let mut bin = UNIT_POS as usize;
    while bin < bins {
        let mut lower = 0usize;
        let mut upper = valid;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            if usize::cast_from(source[middle]) < bin { lower = middle + 1; } else { upper = middle; }
        }
        let begin = lower;
        upper = valid;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            if usize::cast_from(source[middle]) <= bin { lower = middle + 1; } else { upper = middle; }
        }
        let count = C::cast_from(lower - begin);
        if accumulate { histogram[bin] += count; } else { histogram[bin] = count; }
        bin += threads;
    }
    sync_ruda();
}

/// Initialise a shared histogram. All block threads participate.
#[ruda]
pub fn initialise<C: Numeric>(histogram: &mut SharedMemory<Atomic<C>>, #[comptime] bins: usize, #[comptime] threads: usize) {
    let mut bin = UNIT_POS as usize;
    while bin < bins {
        histogram[bin].store(C::from_int(0));
        bin += threads;
    }
    sync_ruda();
}

/// Add a blocked register tile to an existing shared histogram. Valid samples
/// are integral bin indices in range; the backend must support atomic add on C.
#[ruda]
pub fn composite<T: Numeric, C: Numeric>(
    samples: &Array<T>, histogram: &mut SharedMemory<Atomic<C>>, valid_items: usize,
    #[comptime] items_per_thread: usize,
) {
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            histogram[usize::cast_from(samples[item])].fetch_add(C::from_int(1));
        }
    }
    sync_ruda();
}

#[ruda]
pub fn histogram<T: Numeric, C: Numeric>(
    samples: &Array<T>, histogram: &mut SharedMemory<Atomic<C>>, valid_items: usize,
    #[comptime] bins: usize, #[comptime] threads: usize, #[comptime] items_per_thread: usize,
) {
    initialise(histogram, bins, threads);
    composite(samples, histogram, valid_items, items_per_thread);
}
