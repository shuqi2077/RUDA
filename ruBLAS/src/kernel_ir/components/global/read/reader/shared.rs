use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::RudaType;

use crate::kernel_ir::components::{
    global::read::{FullLoadingStrategy, PartialLoadingStrategy},
    stage::StageFamily,
};

#[derive(Copy, Clone, RudaType)]
/// Identifier for the stage in global double buffering
pub enum StageBuffer {
    /// First buffer
    A,
    /// Second buffer
    B,
}

impl StageBuffer {
    pub fn to_index(&self) -> u32 {
        match self {
            StageBuffer::A => 0,
            StageBuffer::B => 1,
        }
    }
}

#[derive(RudaType, Clone)]
/// Comptime counter for loading tasks
pub struct TaskCounter {
    #[ruda(comptime)]
    pub counter: u32,
}

pub type FullLoaderStage<RC, L, E, N> =
    <<L as FullLoadingStrategy<RC>>::Stage as StageFamily>::Stage<
        E,
        N,
        <L as FullLoadingStrategy<RC>>::TilingLayout,
    >;
pub type PartialLoaderStage<RC, L, E, N> =
    <<L as PartialLoadingStrategy<RC>>::Stage as StageFamily>::Stage<
        E,
        N,
        <L as PartialLoadingStrategy<RC>>::TilingLayout,
    >;
