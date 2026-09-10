//! Matmul-specific interpretation of the generic [RudaMapping] from `ruda-kernel::tiling`.
//!
//! The partitioned matmul interprets the `(x, y, z)` problem-space axes
//! as `(m, n, batch)`. GEMV variants use [ruda_pos_to_matrix_batch] instead.

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

pub use ruda_kernel::tiling::ruda_count::{RudaMapping, RudaMappingLaunch, ruda_mapping_launch};

#[ruda]
/// Reads the ruda position as matmul tensor coordinates `(m, n, batch)`.
pub fn ruda_pos_to_m_n_batch(ruda_mapping: &RudaMapping) -> (u32, u32, u32) {
    ruda_mapping.ruda_pos_to_xyz()
}

#[ruda]
/// Reads the ruda position as GEMV `(matrix_axis, batch)` coordinates.
///
/// GEMV is 2D in problem space (the matrix axis + batch). The routine builds
/// its [RudaCountPlan] with `y = 1`, so the meaningful matrix-axis lives in `x`.
pub fn ruda_pos_to_matrix_batch(ruda_mapping: &RudaMapping) -> (u32, u32) {
    let (matrix, _, batch) = ruda_mapping.ruda_pos_to_xyz();
    (matrix, batch)
}
