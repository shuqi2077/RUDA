use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Load a logical warp tile into per-lane registers. Invalid items receive
/// `padding`. The tile starts at `offset`; `valid_items` is its valid length.
#[ruda]
pub fn load<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut Array<T>,
    offset: usize,
    valid_items: usize,
    padding: T,
    #[comptime] items_per_lane: usize,
    #[comptime] width: u32,
    #[comptime] striped: bool,
) {
    let lane = (UNIT_POS_PLANE % width) as usize;
    #[unroll]
    for item in 0..items_per_lane {
        let index = if striped { item * width as usize + lane } else { lane * items_per_lane + item };
        let mut value = padding;
        if index < valid_items {
            value = input[offset + index];
        }
        output[item] = value;
    }
}

/// Store a register tile in blocked or striped order, preserving invalid tails.
#[ruda]
pub fn store<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut Array<T>,
    offset: usize,
    valid_items: usize,
    #[comptime] items_per_lane: usize,
    #[comptime] width: u32,
    #[comptime] striped: bool,
) {
    let lane = (UNIT_POS_PLANE % width) as usize;
    #[unroll]
    for item in 0..items_per_lane {
        let index = if striped { item * width as usize + lane } else { lane * items_per_lane + item };
        if index < valid_items {
            output[offset + index] = input[item];
        }
    }
}
