use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use super::{RudaUnaryOp, RudaUnaryOpExpand};
use super::record::{RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};

/// Map a register to blocked, striped, or logical-warp-striped tile order.
#[ruda]
pub fn index(lane: usize, item: usize, #[comptime] lanes: usize, #[comptime] items: usize,
    #[comptime] striped: bool, #[comptime] warp_striped: bool, #[comptime] width: usize,
) -> usize {
    if warp_striped { lane / width * width * items + item * width + lane % width }
    else if striped { item * lanes + lane }
    else { lane * items + item }
}

/// Iterator load with explicit conversion and tail semantics. lane is a block
/// lane or logical-warp lane, respectively; input/output must not overlap.
#[ruda]
pub fn load<T: RudaType, U: RudaType + Copy, I: RudaRead<T>, W: RudaWrite<U>, O: RudaUnaryOp<T, U>>(
    input: &I, output: &mut W, convert: &O, offset: usize, valid: usize, lane: usize, padding: U,
    #[comptime] lanes: usize, #[comptime] items: usize, #[comptime] striped: bool,
    #[comptime] warp_striped: bool, #[comptime] width: usize, #[comptime] pad: bool,
) {
    #[unroll]
    for item in 0..items {
        let position = index(lane, item, lanes, items, striped, warp_striped, width);
        if position < valid { output.write(item, convert.apply(input.read(offset + position))); }
        else if pad { output.write(item, padding); }
    }
}

#[ruda]
pub fn store<T: RudaType, U: RudaType, I: RudaRead<T>, W: RudaWrite<U>, O: RudaUnaryOp<T, U>>(
    input: &I, output: &mut W, convert: &O, offset: usize, valid: usize, lane: usize,
    #[comptime] lanes: usize, #[comptime] items: usize, #[comptime] striped: bool,
    #[comptime] warp_striped: bool, #[comptime] width: usize,
) {
    #[unroll]
    for item in 0..items {
        let position = index(lane, item, lanes, items, striped, warp_striped, width);
        if position < valid { output.write(offset + position, convert.apply(input.read(item))); }
    }
}

/// Collective layout conversion for scalar/record storage and arbitrary
/// readers/writers. Scratch has lanes * items entries. For warp scope its
/// base is disjoint for each logical group; all native subgroup lanes call it.
#[ruda]
pub fn exchange<T: RudaType, I: RudaRead<T>, W: RudaWrite<T>, S: RudaWrite<T>>(
    input: &I, output: &mut W, scratch: &mut S, lane: usize, base: usize,
    #[comptime] lanes: usize, #[comptime] items: usize, #[comptime] width: usize,
    #[comptime] input_striped: bool, #[comptime] input_warp_striped: bool,
    #[comptime] output_striped: bool, #[comptime] output_warp_striped: bool, #[comptime] warp_scope: bool,
) {
    #[unroll]
    for item in 0..items {
        let position = index(lane, item, lanes, items, input_striped, input_warp_striped, width);
        scratch.write(base + position, input.read(item));
    }
    if warp_scope { sync_plane(); } else { sync_ruda(); }
    #[unroll]
    for item in 0..items {
        let position = index(lane, item, lanes, items, output_striped, output_warp_striped, width);
        output.write(item, scratch.read(base + position));
    }
    if warp_scope { sync_plane(); } else { sync_ruda(); }
}

/// Scatter-to-blocked/striped/warp-striped with optional valid flags and
/// negative/out-of-range rank guarding. Written ranks must be unique. A caller
/// using guarded holes must initialize scratch at every subsequently read hole.
#[ruda]
pub fn scatter<T: RudaType, I: RudaRead<T>, W: RudaWrite<T>, S: RudaWrite<T>>(
    input: &I, ranks: &Array<i64>, flags: &Array<bool>, output: &mut W, scratch: &mut S,
    lane: usize, base: usize, #[comptime] lanes: usize, #[comptime] items: usize, #[comptime] width: usize,
    #[comptime] striped: bool, #[comptime] warp_striped: bool, #[comptime] guarded: bool,
    #[comptime] flagged: bool, #[comptime] warp_scope: bool,
) {
    #[unroll]
    for item in 0..items {
        let rank = ranks[item];
        let mut valid = true;
        if guarded { valid = rank >= 0 && (rank as u64) < (lanes * items) as u64; }
        if flagged { valid = valid && flags[item]; }
        if valid { scratch.write(base + rank as usize, input.read(item)); }
    }
    if warp_scope { sync_plane(); } else { sync_ruda(); }
    #[unroll]
    for item in 0..items {
        let position = index(lane, item, lanes, items, striped, warp_striped, width);
        output.write(item, scratch.read(base + position));
    }
    if warp_scope { sync_plane(); } else { sync_ruda(); }
}

#[derive(Clone, Copy, RudaType, RudaLaunch)]
pub struct RudaIdentity;

impl<R: Runtime> Clone for RudaIdentityLaunch<R> {
    fn clone(&self) -> Self { Self::new() }
}

#[ruda]
impl<T: RudaType> RudaUnaryOp<T, T> for RudaIdentity {
    fn apply(&self, value: T) -> T { value }
}
