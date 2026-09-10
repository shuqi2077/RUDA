use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl::prelude::barrier::{Barrier, BarrierToken};

/// Enqueue a collective asynchronous global-to-shared copy. All block threads
/// use the same initialized block barrier and slices. Source length must not
/// exceed the destination; destination is not readable until wait completes.
#[ruda]
pub fn copy_async<T: RudaPrimitive>(barrier: &Barrier, input: &Slice<T>, output: &mut SliceMut<T>) {
    barrier.memcpy_async_cooperative(input, output);
}

/// Commit one phase after one or more copy_async calls. Every participating
/// thread commits once; the returned token identifies that phase for wait.
#[ruda]
pub fn commit(barrier: &Barrier) -> BarrierToken { barrier.arrive() }

#[ruda]
pub fn wait(barrier: &Barrier, token: BarrierToken) { barrier.wait(token); }

/// Load a tile in blocked or striped order, assigning `padding` to the tail.
#[ruda]
pub fn load<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut Array<T>,
    offset: usize,
    valid_items: usize,
    padding: T,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
    #[comptime] striped: bool,
) {
    #[unroll]
    for item in 0..items_per_thread {
        let index = if striped { item * threads + UNIT_POS as usize } else { UNIT_POS as usize * items_per_thread + item };
        let mut value = padding;
        if index < valid_items {
            value = input[offset + index];
        }
        output[item] = value;
    }
}

/// Store only valid elements of a blocked or striped register tile.
#[ruda]
pub fn store<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut Array<T>,
    offset: usize,
    valid_items: usize,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
    #[comptime] striped: bool,
) {
    #[unroll]
    for item in 0..items_per_thread {
        let index = if striped { item * threads + UNIT_POS as usize } else { UNIT_POS as usize * items_per_thread + item };
        if index < valid_items {
            output[offset + index] = input[item];
        }
    }
}

/// Cooperatively load a tile into shared memory; every thread participates.
#[ruda]
pub fn load_to_shared<T: RudaPrimitive>(
    input: &Array<T>,
    output: &mut SharedMemory<T>,
    offset: usize,
    valid_items: usize,
    padding: T,
    #[comptime] threads: usize,
    #[comptime] tile_items: usize,
) {
    let mut index = UNIT_POS as usize;
    while index < tile_items {
        let mut value = padding;
        if index < valid_items {
            value = input[offset + index];
        }
        output[index] = value;
        index += threads;
    }
    sync_ruda();
}
