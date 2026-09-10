use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand, RudaKeyEqual, RudaKeyEqualExpand};
use crate::collective::record::{RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};

#[ruda]
pub fn difference_access<T: RudaType<ExpandType: Assign> + Copy, I: RudaRead<T>, W: RudaWrite<T>, S: RudaWrite<T>, O: RudaBinaryOp<T>>(
    input: &I, output: &mut W, scratch: &mut S, op: &O, neighbour: T, valid: usize,
    #[comptime] items: usize, #[comptime] right: bool, #[comptime] has_neighbour: bool,
) {
    let start = UNIT_POS as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid { scratch.write(start + item, input.read(item)); }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items {
        let index = start + item;
        if index < valid {
            let current = scratch.read(index);
            let mut value = current;
            let boundary = if right { index + 1 == valid } else { index == 0 };
            if boundary {
                if has_neighbour { value = op.combine(current, neighbour); }
            } else {
                let next = if right { index + 1 } else { index - 1 };
                value = op.combine(current, scratch.read(next));
            }
            output.write(item, value);
        }
    }
    sync_ruda();
}

#[ruda]
pub trait RudaDiscontinuity<T: RudaType>: RudaType {
    fn flag(&self, left: T, right: T, right_index: usize) -> bool;
}

#[ruda]
pub fn discontinuity_access<T: RudaType<ExpandType: Assign> + Copy,
    I: RudaRead<T>, H: RudaWrite<bool>, W: RudaWrite<bool>, S: RudaWrite<T>, O: RudaDiscontinuity<T>>(
    input: &I, heads: &mut H, tails: &mut W, scratch: &mut S,
    op: &O, predecessor: T, successor: T, valid: usize,
    #[comptime] items: usize, #[comptime] has_predecessor: bool, #[comptime] has_successor: bool,
) {
    let start = UNIT_POS as usize * items;
    #[unroll]
    for item in 0..items {
        if start + item < valid { scratch.write(start + item, input.read(item)); }
    }
    sync_ruda();
    #[unroll]
    for item in 0..items {
        let index = start + item;
        let mut head = false;
        let mut tail = false;
        if index < valid {
            let value = scratch.read(index);
            if index == 0 {
                head = true;
                if has_predecessor { head = op.flag(predecessor, value, index); }
            } else { head = op.flag(scratch.read(index - 1), value, index); }
            if index + 1 == valid {
                tail = true;
                if has_successor { tail = op.flag(value, successor, index + 1); }
            } else { tail = op.flag(value, scratch.read(index + 1), index + 1); }
        }
        heads.write(item, head);
        tails.write(item, tail);
    }
    sync_ruda();
}

/// Index-aware head/tail flags; right_index is the right item's tile rank,
/// including valid_items for an external successor.
#[ruda]
pub fn discontinuity_indexed<T: RudaPrimitive, O: RudaDiscontinuity<T>>(
    input: &Array<T>, heads: &mut Array<bool>, tails: &mut Array<bool>, scratch: &mut SharedMemory<T>,
    op: &O, predecessor: T, successor: T, valid_items: usize,
    #[comptime] items_per_thread: usize, #[comptime] has_predecessor: bool, #[comptime] has_successor: bool,
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
            if index == 0 {
                head = true;
                if has_predecessor { head = op.flag(predecessor, scratch[index], index); }
            } else { head = op.flag(scratch[index - 1], scratch[index], index); }
            if index + 1 == valid_items {
                tail = true;
                if has_successor { tail = op.flag(scratch[index], successor, index + 1); }
            } else { tail = op.flag(scratch[index], scratch[index + 1], index + 1); }
        }
        heads[item] = head;
        tails[item] = tail;
    }
    sync_ruda();
}

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
