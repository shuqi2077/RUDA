//! Coalesced zero-only work for pages excluded from history computation.
//! Never reads Q/K/V, statistics or an existing output value. Active and idle
//! page sets are disjoint, so this kernel cannot erase a computed gradient.
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn zero_inactive<F: Float>(
    pages: &Array<u32>, dk: &mut Array<F>, dv: &mut Array<F>, dkp: &mut Array<F>,
    active_pages: u32, page_size: u32, kv_heads: u32, elements: u32,
    #[comptime] key_dim: usize, #[comptime] value_dim: usize,
    #[comptime] position_dim: usize, #[comptime] width: usize,
    #[comptime] need_dk: bool, #[comptime] need_dv: bool, #[comptime] need_dkp: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < elements as usize {
        let d = i % width;
        let row = i / width;
        let rows_per_page = page_size as usize * kv_heads as usize;
        let page = pages[active_pages as usize + row / rows_per_page] as usize;
        let physical_row = page * rows_per_page + row % rows_per_page;
        if comptime!(need_dk) {
            if d < key_dim { dk[physical_row * key_dim + d] = F::cast_from(0.0f32); }
        }
        if comptime!(need_dv) {
            if d < value_dim { dv[physical_row * value_dim + d] = F::cast_from(0.0f32); }
        }
        if comptime!(need_dkp) {
            if d < position_dim { dkp[physical_row * position_dim + d] = F::cast_from(0.0f32); }
        }
    }
}
