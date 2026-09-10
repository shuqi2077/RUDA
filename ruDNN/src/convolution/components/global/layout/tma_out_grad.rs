use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::library::tensor::layout::Layout;
use ruda_kernel::library::tensor::layout::LayoutExpand;
use rublas::kernel_ir::launch::BatchedCoords;

use crate::convolution::components::ConvolutionProblem;

/// Weight backwards needs a consolidated layout to work properly across the combined `k` dimension.
/// Padding to an even tile shape on width isn't valid, because `im2col` doesn't do this.
/// Wouldn't be necessary with `im2colWide`, should investigate at some point.
#[derive(RudaType, RudaLaunch)]
pub struct TmaOutGradLayout {
    rows: u32,
    cols: u32,
}

#[ruda]
impl Layout for TmaOutGradLayout {
    type Coordinates = BatchedCoords;
    type SourceCoordinates = Coords2d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (_, row, col) = pos;
        (row, col)
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        (1, self.rows, self.cols)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

impl<R: Runtime> TmaOutGradLayoutLaunch<R> {
    pub fn from_problem(problem: &ConvolutionProblem) -> Self {
        TmaOutGradLayoutLaunch::new(problem.k as u32, problem.m as u32)
    }
}
