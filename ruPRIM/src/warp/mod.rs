//! Collectives over logical warps within a native device subgroup.
//!
//! All lanes of a logical warp participate in each call. `width` is positive,
//! does not exceed the native subgroup width, and groups must not straddle a
//! native subgroup. No NVIDIA warp width is assumed by these algorithms.

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use crate::collective::{RudaBinaryOp, RudaBinaryOpExpand};

pub mod exchange;
pub mod io;
pub mod sort;
pub mod batched;
pub mod record;

#[ruda]
pub fn logical_lane_id(#[comptime] width: u32) -> u32 { UNIT_POS_PLANE % width }

#[ruda]
pub fn logical_warp_id(#[comptime] width: u32) -> u32 { UNIT_POS_PLANE / width }

#[ruda]
pub fn logical_warp_base_id(#[comptime] width: u32) -> u32 { UNIT_POS_PLANE / width * width }

/// Compute both scan forms with one scan and return the unseeded aggregate
/// to every participating logical lane.
#[ruda]
pub fn scan<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    value: T, initial: T, inclusive_output: &mut T, exclusive_output: &mut T,
    op: &O, valid_lanes: u32, #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid_lanes, width);
    let lane = logical_lane_id(width);
    let base = logical_warp_base_id(width);
    let previous = plane_shuffle(inclusive, base + select(lane > 0, lane - 1, lane));
    *inclusive_output = inclusive;
    let mut exclusive = initial;
    if lane > 0 && lane < valid_lanes { exclusive = op.combine(initial, previous); }
    *exclusive_output = exclusive;
    plane_shuffle(inclusive, base + valid_lanes - 1)
}

/// Inclusive scan in lane order. Values beyond `valid_lanes` are unspecified.
/// `valid_lanes` is uniform within a logical warp and lies in `1..=width`.
#[ruda]
pub fn inclusive_scan<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    value: T,
    op: &O,
    valid_lanes: u32,
    #[comptime] width: u32,
) -> T {
    let lane = UNIT_POS_PLANE % width;
    let base = UNIT_POS_PLANE - lane;
    let mut result = value;
    let mut distance = 1u32;
    while distance < width {
        let source = base + select(lane >= distance, lane - distance, lane);
        let left = plane_shuffle(result, source);
        if lane >= distance && lane < valid_lanes {
            result = op.combine(left, result);
        }
        distance *= 2;
    }
    result
}

/// Exclusive scan seeded by `initial`, with the same participation contract.
#[ruda]
pub fn exclusive_scan<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    value: T,
    initial: T,
    op: &O,
    valid_lanes: u32,
    #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid_lanes, width);
    let lane = UNIT_POS_PLANE % width;
    let previous = plane_shuffle(inclusive, UNIT_POS_PLANE - select(lane > 0, 1u32, 0u32));
    let mut result = initial;
    if lane > 0 && lane < valid_lanes {
        result = op.combine(initial, previous);
    }
    result
}

/// Unseeded exclusive scan; the first logical lane's output is unspecified.
#[ruda]
pub fn exclusive_unseeded<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    value: T, op: &O, valid_lanes: u32, #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid_lanes, width);
    let lane = UNIT_POS_PLANE % width;
    plane_shuffle(inclusive, UNIT_POS_PLANE - select(lane > 0, 1u32, 0u32))
}

/// Ordered reduction, broadcast to all lanes of the logical warp.
#[ruda]
pub fn reduce<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    value: T,
    op: &O,
    valid_lanes: u32,
    #[comptime] width: u32,
) -> T {
    let inclusive = inclusive_scan::<T, O>(value, op, valid_lanes, width);
    let base = UNIT_POS_PLANE - UNIT_POS_PLANE % width;
    plane_shuffle(inclusive, base + valid_lanes - 1)
}

/// Broadcast from a logical lane, not a native subgroup lane.
#[ruda]
pub fn broadcast<T: RudaPrimitive>(value: T, source: u32, #[comptime] width: u32) -> T {
    let base = UNIT_POS_PLANE - UNIT_POS_PLANE % width;
    plane_shuffle(value, base + source)
}

/// Segmented reduction whose result is valid at each segment's head lane.
/// The first lane implicitly starts a segment.
#[ruda]
pub fn head_segmented_reduce<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    value: T,
    head: bool,
    op: &O,
    #[comptime] width: u32,
) -> T {
    let lane = UNIT_POS_PLANE % width;
    let base = UNIT_POS_PLANE - lane;
    let mut result = value;
    let next_lane = select(lane + 1 < width, lane + 1, lane);
    let next_head = plane_shuffle(head, base + next_lane);
    let mut boundary = next_head || lane + 1 == width;
    let mut distance = 1u32;
    while distance < width {
        let source = base + select(lane + distance < width, lane + distance, lane);
        let right = plane_shuffle(result, source);
        let right_boundary = plane_shuffle(boundary, source);
        if lane + distance < width {
            if !boundary {
                result = op.combine(result, right);
            }
            boundary = boundary || right_boundary;
        }
        distance *= 2;
    }
    result
}

/// Segmented reduction whose result is valid at each segment's head lane.
/// A true `tail` marks the last lane of a segment; the final lane is implicit.
#[ruda]
pub fn tail_segmented_reduce<T: RudaPrimitive, O: RudaBinaryOp<T>>(
    value: T,
    tail: bool,
    op: &O,
    #[comptime] width: u32,
) -> T {
    let lane = UNIT_POS_PLANE % width;
    let base = UNIT_POS_PLANE - lane;
    let mut result = value;
    let mut boundary = tail || lane + 1 == width;
    let mut distance = 1u32;
    while distance < width {
        let source = base + select(lane + distance < width, lane + distance, lane);
        let right = plane_shuffle(result, source);
        let right_boundary = plane_shuffle(boundary, source);
        if lane + distance < width {
            if !boundary {
                result = op.combine(result, right);
            }
            boundary = boundary || right_boundary;
        }
        distance *= 2;
    }
    result
}
