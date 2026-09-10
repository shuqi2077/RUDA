use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::{Coords1d, Coords2d, Layout, LayoutExpand};
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;

#[ruda]
pub(crate) fn reduction_count<E: Numeric, N: Size>(
    output: &VirtualTensor<E, N, ReadWrite>,
    reduce_axis: usize,
) -> usize {
    let mut count = 1usize;
    for axis in 0..output.rank() {
        if axis != reduce_axis {
            count *= output.shape(axis);
        }
    }
    count
}

#[ruda]
fn reduction_coordinate<E: Numeric, N: Size>(
    output: &VirtualTensor<E, N, ReadWrite>,
    reduce_axis: usize,
    axis: usize,
    reduction_index: usize,
) -> usize {
    let stride = output.stride(axis);
    let mut logical_stride = 1usize;
    for inner in 0..output.rank() {
        let inner_stride = output.stride(inner);
        if inner != reduce_axis
            && (inner_stride < stride || (inner_stride == stride && inner < axis))
        {
            logical_stride *= output.shape(inner);
        }
    }
    (reduction_index / logical_stride) % output.shape(axis)
}

#[ruda]
pub(crate) fn reduction_input_offset<In: Numeric, InSize: Size, Out: Numeric, OutSize: Size>(
    input: &VirtualTensor<In, InSize>,
    output: &VirtualTensor<Out, OutSize, ReadWrite>,
    reduce_axis: usize,
    reduction_index: usize,
) -> usize {
    let mut offset = 0usize;
    for axis in 0..input.rank() {
        if axis != reduce_axis {
            let coordinate = reduction_coordinate(output, reduce_axis, axis, reduction_index);
            offset += coordinate * input.stride(axis);
        }
    }
    offset / input.vector_size()
}

#[derive(RudaType, Clone)]
pub(crate) struct ReductionLayout<E: Numeric, N: Size> {
    output: VirtualTensor<E, N, ReadWrite>,
    reduce_axis: usize,
    num_writes: usize,
    accumulator_length: usize,
}

#[ruda]
impl<E: Numeric, N: Size> ReductionLayout<E, N> {
    pub(crate) fn new(
        output: &VirtualTensor<E, N, ReadWrite>,
        reduce_axis: usize,
        accumulator_length: usize,
    ) -> Self {
        ReductionLayout::<E, N> {
            output: output.clone(),
            reduce_axis,
            num_writes: reduction_count(output, reduce_axis) / output.vector_size(),
            accumulator_length,
        }
    }
}

#[ruda]
impl<E: Numeric, N: Size> Layout for ReductionLayout<E, N> {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> Coords1d {
        let vector_size = self.output.vector_size();
        let reduction_index = coords.0 as usize * vector_size;
        let mut offset = coords.1 as usize * self.output.stride(self.reduce_axis);
        for axis in 0..self.output.rank() {
            if axis != self.reduce_axis {
                let coordinate = reduction_coordinate(
                    &self.output, self.reduce_axis, axis, reduction_index,
                );
                offset += coordinate * self.output.stride(axis);
            }
        }
        offset / vector_size
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (Coords1d, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.num_writes as u32, self.accumulator_length as u32)
    }

    fn is_in_bounds(&self, coords: Self::Coordinates) -> bool {
        (coords.0 as usize) < self.num_writes && (coords.1 as usize) < self.accumulator_length
    }
}
