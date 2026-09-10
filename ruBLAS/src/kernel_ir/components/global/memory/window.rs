use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::tiling::{MatrixLayout, stage::StageMemoryConfig};

use crate::kernel_ir::components::global::memory::GlobalMemoryConfig;

/// Reads data from the tensor view as a window, i.e. a slice of global memory
/// Also returns the length of the slice
///
/// The length of the slice is the width of the tile
///
/// # Note
///
/// If the slice would be partly out-of-bounds, it will simply be shorter.
/// The caller must do the padding if necessary.
#[ruda]
pub fn load_window_in_tile<EG: Numeric>(
    view: &View<EG, Coords2d>,
    tile: Coords2d,
    nth_window: u32,
    #[comptime] smem_config: StageMemoryConfig,
    #[comptime] gmem_config: GlobalMemoryConfig,
) -> Slice<EG> {
    let (tile_row, tile_col) = tile;
    let tile_size_row = smem_config.elements_per_tile_along_row;
    let tile_size_col = smem_config.elements_per_tile_along_col;

    let size = match smem_config.matrix_layout {
        MatrixLayout::RowMajor => (1u32, tile_size_col).runtime(),
        MatrixLayout::ColMajor => (tile_size_row, 1u32).runtime(),
    };

    let offset = (tile_row * tile_size_row, tile_col * tile_size_col);
    let tile_size = (tile_size_row, tile_size_col).runtime();

    load_window(
        &view.slice(offset, tile_size),
        nth_window,
        size,
        gmem_config,
    )
}

/// Reads data from the tensor view as a window, i.e. a slice of global memory
///
/// The length of the slice is the width of the tile
///
/// # Note
///
/// If the slice would be partly out-of-bounds, it will simply be shorter.
/// The caller must do the padding if necessary.
#[ruda]
pub fn load_window_in_stage<EG: RudaPrimitive>(
    view: &View<EG, Coords2d>,
    nth_window: u32,
    #[comptime] smem_config: StageMemoryConfig,
    #[comptime] gmem_config: GlobalMemoryConfig,
) -> Slice<EG> {
    let size = match smem_config.matrix_layout {
        MatrixLayout::RowMajor => (1u32, smem_config.elements_per_stage_along_col()).runtime(),
        MatrixLayout::ColMajor => (smem_config.elements_per_stage_along_row(), 1u32).runtime(),
    };

    load_window(view, nth_window, size, gmem_config)
}

#[ruda]
fn load_window<EG: RudaPrimitive>(
    view: &View<EG, Coords2d>,
    nth_window: u32,
    size: Coords2d,
    #[comptime] gmem_config: GlobalMemoryConfig,
) -> Slice<EG> {
    let offset = match gmem_config.matrix_layout {
        MatrixLayout::RowMajor => (nth_window, 0),
        MatrixLayout::ColMajor => (0, nth_window),
    };

    if gmem_config.check_row_bounds || gmem_config.check_col_bounds {
        view.slice(offset, size).to_linear_slice()
    } else {
        view.slice_unchecked(offset, size).to_linear_slice()
    }
}
