use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Convert blocked and striped layouts. Scratch holds the entire block tile.
#[ruda]
pub fn transpose<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut Array<T>,
    scratch: &mut SharedMemory<T>,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
    #[comptime] blocked_to_striped: bool,
) {
    let lane = UNIT_POS as usize;
    #[unroll]
    for item in 0..items_per_thread {
        let index = if blocked_to_striped { lane * items_per_thread + item } else { item * threads + lane };
        scratch[index] = input[item];
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let index = if blocked_to_striped { item * threads + lane } else { lane * items_per_thread + item };
        output[item] = scratch[index];
    }
    sync_ruda();
}

/// Scatter a permutation of tile ranks into blocked or striped register order.
#[ruda]
pub fn scatter<T: RudaPrimitive>(
    input: &Array<T>,
    ranks: &Array<u32>,
    output: &mut Array<T>,
    scratch: &mut SharedMemory<T>,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
    #[comptime] striped: bool,
) {
    #[unroll]
    for item in 0..items_per_thread {
        let rank = ranks[item] as usize;
        scratch[rank] = input[item];
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let index = if striped { item * threads + UNIT_POS as usize } else { UNIT_POS as usize * items_per_thread + item };
        output[item] = scratch[index];
    }
    sync_ruda();
}

/// Blocked <-> warp-striped exchange, including a partial final logical warp.
/// `width` is an explicit layout width rather than a fixed hardware warp size.
#[ruda]
pub fn warp_transpose<T: RudaPrimitive>(
    input: &Array<T>, output: &mut Array<T>, scratch: &mut SharedMemory<T>,
    #[comptime] threads: usize, #[comptime] items_per_thread: usize,
    #[comptime] width: usize, #[comptime] blocked_to_warp_striped: bool,
) {
    let thread = UNIT_POS as usize;
    let first = thread / width * width;
    let lane = thread - first;
    let lanes = min(width, threads - first);
    #[unroll]
    for item in 0..items_per_thread {
        let blocked = thread * items_per_thread + item;
        let striped = first * items_per_thread + item * lanes + lane;
        let index = if blocked_to_warp_striped { blocked } else { striped };
        scratch[index] = input[item];
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let blocked = thread * items_per_thread + item;
        let striped = first * items_per_thread + item * lanes + lane;
        let index = if blocked_to_warp_striped { striped } else { blocked };
        output[item] = scratch[index];
    }
    sync_ruda();
}

/// Scatter valid items to striped output. Valid ranks must be unique and in
/// tile range. Scratch positions not written by a valid item are unspecified.
#[ruda]
pub fn scatter_flagged<T: RudaPrimitive, I: Int>(
    input: &Array<T>, ranks: &Array<I>, valid: &Array<bool>, output: &mut Array<T>,
    scratch: &mut SharedMemory<T>, #[comptime] threads: usize, #[comptime] items_per_thread: usize,
) {
    #[unroll]
    for item in 0..items_per_thread {
        if valid[item] {
            let rank = usize::cast_from(ranks[item]);
            scratch[rank] = input[item];
        }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread { output[item] = scratch[item * threads + UNIT_POS as usize]; }
    sync_ruda();
}

/// Negative ranks are excluded from the exchange.
#[ruda]
pub fn scatter_guarded<T: RudaPrimitive, I: Int>(
    input: &Array<T>, ranks: &Array<I>, output: &mut Array<T>, scratch: &mut SharedMemory<T>,
    #[comptime] threads: usize, #[comptime] items_per_thread: usize,
) {
    let mut valid = Array::<bool>::new(items_per_thread);
    #[unroll]
    for item in 0..items_per_thread { valid[item] = ranks[item] >= I::from_int(0); }
    scatter_flagged(input, ranks, &valid, output, scratch, threads, items_per_thread);
}
