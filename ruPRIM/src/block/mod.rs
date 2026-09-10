//! Block collectives. Every thread in the block participates in every call.

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};

pub mod exchange;
pub mod io;
pub mod adjacent;
pub mod sort;
pub mod shuffle;
pub mod radix;
pub mod histogram;
pub mod run_length;
pub mod rank;
pub mod topk;

#[ruda]
pub trait RudaBlockPrefix<T: RudaPrimitive>: RudaType {
    fn prefix(&mut self, aggregate: T) -> T;
}

/// Ordered inclusive scan of a blocked register tile.
/// Scratch contains `threads * items_per_thread` elements. `valid_items` is
/// uniform and positive; invalid output elements are unspecified.
#[ruda]
pub fn inclusive_scan<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    input: &Array<T>,
    output: &mut Array<T>,
    scratch: &mut SharedMemory<T>,
    op: &O,
    valid_items: usize,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
) {
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items { scratch[start + item] = input[item]; }
    }
    sync_ruda();
    let mut distance = 1usize;
    while distance < threads * items_per_thread {
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index < valid_items {
                let mut value = scratch[index];
                if index >= distance {
                    value = op.combine(scratch[index - distance], value);
                }
                output[item] = value;
            }
        }
        sync_ruda();
        #[unroll]
        for item in 0..items_per_thread {
            if start + item < valid_items { scratch[start + item] = output[item]; }
        }
        sync_ruda();
        distance *= 2;
    }
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items { output[item] = scratch[start + item]; }
    }
    sync_ruda();
}

/// Ordered exclusive scan. Returns the unseeded aggregate to every thread.
#[ruda]
pub fn exclusive_scan<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    input: &Array<T>,
    output: &mut Array<T>,
    scratch: &mut SharedMemory<T>,
    initial: T,
    op: &O,
    valid_items: usize,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
) -> T {
    inclusive_scan::<T, O>(input, output, scratch, op, valid_items, threads, items_per_thread);
    let aggregate = scratch[valid_items - 1];
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        let index = start + item;
        let mut value = initial;
        if index > 0 && index < valid_items {
            value = op.combine(initial, scratch[index - 1]);
        }
        output[item] = value;
    }
    sync_ruda();
    aggregate
}

/// Ordered tree reduction; the aggregate is broadcast to all block threads.
/// `valid_items` must be positive. Input is in blocked register order.
#[ruda]
pub fn reduce<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    input: &Array<T>,
    scratch: &mut SharedMemory<T>,
    op: &O,
    valid_items: usize,
    #[comptime] threads: usize,
    #[comptime] items_per_thread: usize,
) -> T {
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items { scratch[start + item] = input[item]; }
    }
    sync_ruda();
    let mut distance = 1usize;
    while distance < threads * items_per_thread {
        #[unroll]
        for item in 0..items_per_thread {
            let index = start + item;
            if index % (distance * 2) == 0 && index + distance < valid_items {
                scratch[index] = op.combine(scratch[index], scratch[index + distance]);
            }
        }
        sync_ruda();
        distance *= 2;
    }
    let aggregate = scratch[0];
    sync_ruda();
    aggregate
}

/// Inclusive/exclusive scan with a stateful block prefix callback. The first
/// native subgroup invokes the callback, and thread zero's result is used.
/// Scratch requires `threads * items_per_thread + 1` entries. Valid input is
/// positive and callback state updates occur on the participating threads.
#[ruda]
pub fn scan_with_prefix<T: RudaPrimitive, O: RudaBinaryOp<T>, P: RudaBlockPrefix<T>>(
    input: &Array<T>, output: &mut Array<T>, scratch: &mut SharedMemory<T>,
    op: &O, prefix: &mut P, valid_items: usize,
    #[comptime] threads: usize, #[comptime] items_per_thread: usize, #[comptime] exclusive: bool,
) -> T {
    inclusive_scan::<T, O>(input, output, scratch, op, valid_items, threads, items_per_thread);
    let aggregate = scratch[valid_items - 1];
    let prefix_index = threads * items_per_thread;
    if UNIT_POS < PLANE_DIM {
        let value = prefix.prefix(aggregate);
        if UNIT_POS == 0 { scratch[prefix_index] = value; }
    }
    sync_ruda();
    let seed = scratch[prefix_index];
    #[unroll]
    for item in 0..items_per_thread {
        let index = UNIT_POS as usize * items_per_thread + item;
        if index < valid_items {
            let mut value = seed;
            if exclusive {
                if index > 0 { value = op.combine(seed, scratch[index - 1]); }
            } else {
                value = op.combine(seed, output[item]);
            }
            output[item] = value;
        }
    }
    sync_ruda();
    aggregate
}
