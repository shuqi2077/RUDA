use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Offset one scalar per thread. Out-of-block destinations preserve `output`.
#[ruda]
pub fn offset<T: RudaPrimitive>(
    input: T,
    output: &mut T,
    scratch: &mut SharedMemory<T>,
    distance: i32,
    #[comptime] threads: u32,
) {
    scratch[UNIT_POS as usize] = input;
    sync_ruda();
    let source = UNIT_POS as i32 + distance;
    if source >= 0 && source < threads as i32 {
        *output = scratch[source as usize];
    }
    sync_ruda();
}

/// Circular offset one scalar per thread.
#[ruda]
pub fn rotate<T: RudaPrimitive>(
    input: T,
    scratch: &mut SharedMemory<T>,
    distance: u32,
    #[comptime] threads: u32,
) -> T {
    scratch[UNIT_POS as usize] = input;
    sync_ruda();
    let source = (UNIT_POS + distance % threads) % threads;
    let output = scratch[source as usize];
    sync_ruda();
    output
}

/// Shift a blocked register tile one item. Boundary output is preserved.
/// Returns the block's discarded prefix (down) or suffix (up).
#[ruda]
pub fn shift<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut Array<T>,
    scratch: &mut SharedMemory<T>,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
    #[comptime] down: bool,
) -> T {
    let start = UNIT_POS as usize * items_per_thread;
    let total = threads * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread { scratch[start + item] = input[item]; }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let index = start + item;
        if down {
            if index + 1 < total { output[item] = scratch[index + 1]; }
        } else {
            if index > 0 { output[item] = scratch[index - 1]; }
        }
    }
    let boundary = if down { scratch[0] } else { scratch[total - 1] };
    sync_ruda();
    boundary
}
