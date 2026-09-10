use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use super::{RudaCompare, RudaCompareExpand};

/// Stable destination of one item when merging adjacent sorted runs.
#[ruda]
pub(crate) fn rank<K: RudaPrimitive, C: RudaCompare<K>>(
    keys: &SharedMemory<K>,
    compare: &C,
    index: usize,
    valid: usize,
    run: usize,
    base: usize,
) -> usize {
    let begin = index / (run * 2) * (run * 2);
    let middle = min(begin + run, valid);
    let end = min(begin + run * 2, valid);
    let left = index < middle;
    let own_begin = select(left, begin, middle);
    let other_begin = select(left, middle, begin);
    let mut low = other_begin;
    let mut high = select(left, end, middle);
    let key = keys[base + index];
    while low < high {
        let mid = low + (high - low) / 2;
        let other = keys[base + mid];
        let advance = if left { compare.before(other, key) } else { !compare.before(key, other) };
        if advance {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    begin + (index - own_begin) + (low - other_begin)
}
