use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::{
    ReducePrecision,
    components::{
        instructions::Item,
        readers::{Reader, ReaderExpand},
    },
};
use ruda_kernel::dsl::prelude::*;

#[derive(RudaType)]
pub struct PlaneReader<P: ReducePrecision> {
    reader: Reader<P>,
    plane_dim: u32,
    unit_pos: u32,
}

#[ruda]
impl<P: ReducePrecision> PlaneReader<P> {
    pub fn new(reader: Reader<P>) -> PlaneReader<P> {
        PlaneReader::<P> {
            reader,
            plane_dim: plane_sum(1u32),
            unit_pos: plane_exclusive_sum(1u32),
        }
    }

    pub fn read(&self, vector_index: usize) -> Item<P> {
        match &self.reader {
            Reader::Parallel(reader) => reader.read_plane_at(vector_index, self.plane_dim, self.unit_pos),
            Reader::Perpendicular(reader) => reader.read_plane_at(vector_index, self.plane_dim, self.unit_pos),
        }
    }

    pub fn length(&self) -> usize {
        match &self.reader {
            Reader::Parallel(reader) => reader.length_unit().div_ceil(self.plane_dim as usize),
            Reader::Perpendicular(reader) => reader.length_unit().div_ceil(self.plane_dim as usize),
        }
    }
}
