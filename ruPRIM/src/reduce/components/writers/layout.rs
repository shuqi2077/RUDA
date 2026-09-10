use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::Coords1d;
use ruda_kernel::library::tensor::layout::Coords2d;
use ruda_kernel::library::tensor::layout::Layout;
use ruda_kernel::library::tensor::layout::LayoutExpand;
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;

use crate::reduce::components::args::NumericVector;
use crate::reduce::components::layout::ReductionLayout;

/// Maps a `(write_index, k_iter)` coordinate to a flat vector position in the
/// output buffer. Strides are expressed in vector units (one step along the
/// output's SIMD axis = one unit in `write_stride`).
///
/// For rank-1 outputs (or any case where `reduce_axis == out_vec_axis`), the
/// caller should pass `write_stride = 0` and `num_writes = 1`, so the layout
/// collapses to `position = k_iter * k_stride`.
#[derive(RudaType, Clone)]
pub struct ReduceOutputLayout {
    k_stride: usize,
    write_stride: usize,
    num_writes: usize,
    accumulator_length: usize,
}

#[ruda]
impl ReduceOutputLayout {
    pub fn new(
        k_stride: usize,
        write_stride: usize,
        num_writes: usize,
        accumulator_length: usize,
    ) -> ReduceOutputLayout {
        ReduceOutputLayout {
            k_stride,
            write_stride,
            num_writes,
            accumulator_length,
        }
    }
}

#[ruda]
impl Layout for ReduceOutputLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> Coords1d {
        let write_index = coords.0 as usize;
        let k_iter = coords.1 as usize;
        k_iter * self.k_stride + write_index * self.write_stride
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (Coords1d, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.num_writes as u32, self.accumulator_length as u32)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos.0 < self.num_writes as u32 && pos.1 < self.accumulator_length as u32
    }
}

/// Build the output layout from reduction ordinals and the tensor's actual strides.
#[ruda]
pub(crate) fn build_reduce_output_layout<Out: NumericVector>(
    output: &VirtualTensor<Out::T, Out::N, ReadWrite>,
    reduce_axis: usize,
    _out_vec_axis: usize,
    #[comptime] accumulator_length: usize,
) -> ReductionLayout<Out::T, Out::N> {
    ReductionLayout::new(output, reduce_axis, accumulator_length)
}
