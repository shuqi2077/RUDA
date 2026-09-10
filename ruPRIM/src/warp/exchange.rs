use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Convert blocked/striped register layouts using subgroup shuffles only.
/// Input and output are distinct local arrays of `items_per_lane` elements.
#[ruda]
pub fn transpose<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut Array<T>,
    #[comptime] items_per_lane: usize,
    #[comptime] width: u32,
    #[comptime] blocked_to_striped: bool,
) {
    let lane = UNIT_POS_PLANE % width;
    let base = UNIT_POS_PLANE - lane;
    #[unroll]
    for item in 0..items_per_lane {
        let destination = if blocked_to_striped {
            item * width as usize + lane as usize
        } else {
            lane as usize * items_per_lane + item
        };
        let source_lane = if blocked_to_striped {
            destination / items_per_lane
        } else {
            destination % width as usize
        };
        let source_item = if blocked_to_striped {
            destination % items_per_lane
        } else {
            destination / width as usize
        };
        // Every lane executes the same shuffles, including when source_item differs.
        #[unroll]
        for candidate in 0..items_per_lane {
            let value = plane_shuffle(input[candidate], base + source_lane as u32);
            if source_item == candidate {
                output[item] = value;
            }
        }
    }
}

/// Scatter a permutation of ranks into striped register order.
/// Scratch has one disjoint `width * items_per_lane` region per logical warp.
/// Ranks must be a permutation of `0..width * items_per_lane`.
#[ruda]
pub fn scatter_to_striped<T: RudaPrimitive>(
    input: &Array<T>,
    ranks: &Array<u32>,
    output: &mut Array<T>,
    scratch: &mut SharedMemory<T>,
    #[comptime] items_per_lane: usize,
    #[comptime] width: u32,
) {
    let lane = UNIT_POS_PLANE % width;
    let warp_base = (UNIT_POS - lane) as usize * items_per_lane;
    #[unroll]
    for item in 0..items_per_lane {
        let rank = ranks[item] as usize;
        scratch[warp_base + rank] = input[item];
    }
    sync_plane();
    #[unroll]
    for item in 0..items_per_lane {
        output[item] = scratch[warp_base + item * width as usize + lane as usize];
    }
    sync_plane();
}
