use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand, RudaKeyEqual, RudaKeyEqualExpand};

/// Adjacent transform in blocked order. With no external neighbour, preserve
/// the boundary input. The operator receives (current, neighbouring).
#[ruda]
pub fn difference<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    input: &Array<T>,
    output: &mut Array<T>,
    scratch: &mut SharedMemory<T>,
    op: &O,
    neighbour: T,
    valid_items: usize,
    #[comptime] items_per_thread: usize,
    #[comptime] right: bool,
    #[comptime] has_neighbour: bool,
) {
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items {
            scratch[start + item] = input[item];
        }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let index = start + item;
        if index < valid_items {
            let current = scratch[index];
            let boundary = if right { index + 1 == valid_items } else { index == 0 };
            let mut value = current;
            if boundary {
                if has_neighbour { value = op.combine(current, neighbour); }
            } else {
                let adjacent = if right { scratch[index + 1] } else { scratch[index - 1] };
                value = op.combine(current, adjacent);
            }
            output[item] = value;
        }
    }
    sync_ruda();
}

/// Flag heads and tails of equal-key runs. Optional neighbours connect tiles.
#[ruda]
pub fn discontinuity<T: RudaPrimitive, E: RudaKeyEqual<T>>(
    input: &Array<T>,
    heads: &mut Array<bool>,
    tails: &mut Array<bool>,
    scratch: &mut SharedMemory<T>,
    equal: &E,
    predecessor: T,
    successor: T,
    valid_items: usize,
    #[comptime] items_per_thread: usize,
    #[comptime] has_predecessor: bool,
    #[comptime] has_successor: bool,
) {
    let start = UNIT_POS as usize * items_per_thread;
    #[unroll]
    for item in 0..items_per_thread {
        if start + item < valid_items { scratch[start + item] = input[item]; }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items_per_thread {
        let index = start + item;
        let mut head = false;
        let mut tail = false;
        if index < valid_items {
            let value = scratch[index];
            if index > 0 {
                head = !equal.equal(scratch[index - 1], value);
            } else {
                head = true;
                if has_predecessor { head = !equal.equal(predecessor, value); }
            }
            if index + 1 < valid_items {
                tail = !equal.equal(value, scratch[index + 1]);
            } else {
                tail = true;
                if has_successor { tail = !equal.equal(value, successor); }
            }
        }
        heads[item] = head;
        tails[item] = tail;
    }
    sync_ruda();
}
