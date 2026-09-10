use ruda_kernel::tiling::MatrixLayout;

use crate::kernel_ir::components::{batch::BatchConfig, global::memory::GlobalLayoutConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NaiveMatmulConfig {
    pub fused_accumulation: bool,
    pub sequential_accumulation: bool,
}

impl BatchConfig for NaiveMatmulConfig {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }
}
