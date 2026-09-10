use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::{
    BoundChecks, ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        instructions::{Item, ReduceRequirements},
        readers::{bound_checks::ReaderBoundChecks, new_coordinates},
    },
};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords1d;
use ruda_kernel::library::tensor::layout::plain::PlainLayout;
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;
use crate::reduce::components::layout::reduction_input_offset;

#[derive(RudaType)]
pub struct PerpendicularReader<P: ReducePrecision> {
    view: View<Vector<P::EI, P::SI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    batch_offset: usize,
    vector_offset_stride: usize,
    requirements: ReduceRequirements,
    bound_checks: ReaderBoundChecks<P>,
    shape: usize,
    effective_plane_dim: u32,
}

#[ruda]
impl<P: ReducePrecision> PerpendicularReader<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<I: ReduceInstruction<P>, Out: NumericVector>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        inst: &I,
        reduce_axis: usize,
        reduce_index: usize,
        idle: ComptimeOption<bool>,
        effective_plane_dim: u32,
        #[comptime] bound_checks: BoundChecks,
    ) -> PerpendicularReader<P> {
        let vector_size = input.vector_size();
        let output_index = reduce_index * vector_size;

        let batch_offset = reduction_input_offset(input, output, reduce_axis, output_index);

        let requirements = I::requirements(inst);
        let vector_offset_stride = input.stride(reduce_axis) / vector_size;
        let shape = input.shape(reduce_axis);

        let bound_checks = ReaderBoundChecks::new::<I>(inst, shape, idle, bound_checks);

        PerpendicularReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            batch_offset,
            vector_offset_stride,
            requirements,
            bound_checks,
            shape,
            effective_plane_dim,
        }
    }

    pub fn length_unit(&self) -> usize {
        self.shape
    }

    pub fn length_plane(&self) -> usize {
        self.shape.div_ceil(self.effective_plane_dim as usize)
    }

    pub fn length_ruda(&self) -> usize {
        self.shape.div_ceil(RUDA_DIM as usize)
    }

    pub fn read_ruda(&self, vector_index: usize) -> Item<P> {
        let plane_pos = vector_index * RUDA_DIM as usize;
        let unit_pos = UNIT_POS as usize;
        let pos = plane_pos + unit_pos;
        let offset = plane_pos * self.vector_offset_stride
            + unit_pos * self.vector_offset_stride
            + self.batch_offset;

        let elements = self.bound_checks.read(pos, offset, &self.view);

        let args = new_coordinates(
            plane_pos + unit_pos,
            self.requirements,
            VectorizationMode::Perpendicular,
        );

        Item::<P> { elements, args }
    }

    pub fn read_plane(&self, vector_index: usize) -> Item<P> {
        self.read_plane_at(vector_index, self.effective_plane_dim, UNIT_POS_PLANE)
    }

    pub(crate) fn read_plane_at(&self, vector_index: usize, plane_dim: u32, unit_pos: u32) -> Item<P> {
        let plane_pos = vector_index * plane_dim as usize;
        let unit_pos = unit_pos as usize;
        let pos = plane_pos + unit_pos;
        let offset = plane_pos * self.vector_offset_stride
            + unit_pos * self.vector_offset_stride
            + self.batch_offset;

        let elements = self.bound_checks.read(pos, offset, &self.view);

        let args = new_coordinates(
            plane_pos + unit_pos,
            self.requirements,
            VectorizationMode::Perpendicular,
        );

        Item::<P> { elements, args }
    }

    pub fn read_unit(&self, vector_index: usize) -> Item<P> {
        let offset = self.batch_offset + vector_index * self.vector_offset_stride;
        let elements = self.bound_checks.read(vector_index, offset, &self.view);

        let args = new_coordinates(
            vector_index,
            self.requirements,
            VectorizationMode::Perpendicular,
        );

        Item::<P> { elements, args }
    }
}
