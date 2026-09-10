use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
pub trait RudaDigitExtractor<K: RudaPrimitive>: RudaType {
    fn digit(&self, key: K) -> u32;
}

/// Stable ranks of extracted digits, without rearranging the input keys.
/// Extracted digits fit `digit_bits` (1..=31). Tile scratch arrays each contain
/// the complete tile; `digit_offsets` contains `(1 << digit_bits) + 1` entries
/// in output bucket order. Adjacent offsets also give bucket populations.
#[ruda]
pub fn rank_keys<K: RudaPrimitive, E: RudaDigitExtractor<K>>(
    keys: &Array<K>, ranks: &mut Array<u32>, extractor: &E,
    digit_scratch: &mut SharedMemory<u32>, index_scratch: &mut SharedMemory<u32>,
    scan_scratch: &mut SharedMemory<u32>, digit_offsets: &mut SharedMemory<u32>,
    valid_items: usize, #[comptime] threads: usize, #[comptime] items_per_thread: usize,
    #[comptime] digit_bits: u32, #[comptime] descending: bool,
) {
    let start = UNIT_POS as usize * items_per_thread;
    let mut digits = Array::<u32>::new(items_per_thread);
    let mut indices = Array::<u32>::new(items_per_thread);
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            digits[item] = extractor.digit(keys[item]);
            indices[item] = (start + item) as u32;
        }
    }
    crate::block::radix::sort_pairs::<u32, u32>(&mut digits, &mut indices, digit_scratch, index_scratch,
        scan_scratch, valid_items, threads, items_per_thread, 0u32, digit_bits, descending);
    let bins = 1usize << digit_bits;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            let mut digit = digits[item];
            if descending { digit = bins as u32 - 1 - digit; }
            digit_scratch[start + item] = digit;
            scan_scratch[indices[item] as usize] = (start + item) as u32;
        }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items { ranks[item] = scan_scratch[start + item]; }
    }
    let mut bucket = UNIT_POS as usize;
    while bucket <= bins {
        let mut low = 0usize;
        let mut high = valid_items;
        while low < high {
            let middle = low + (high - low) / 2;
            if (digit_scratch[middle] as usize) < bucket { low = middle + 1; } else { high = middle; }
        }
        digit_offsets[bucket] = low as u32;
        bucket += threads;
    }
    sync_ruda();
}
