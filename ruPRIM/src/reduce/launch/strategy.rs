use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::routines::{
    BlueprintStrategy, ruda::RudaRoutine, plane::PlaneRoutine, unit::UnitRoutine,
};
use ruda_kernel::dsl::ir::features::Plane;
use ruda_kernel::dsl::prelude::*;

#[derive(Debug, Clone)]
pub struct ReduceStrategy {
    pub routine: RoutineStrategy,
    pub vectorization: VectorizationStrategy,
}

#[derive(Debug, Clone)]
pub enum RoutineStrategy {
    /// A unit is responsible to reduce a full vector.
    Unit(BlueprintStrategy<UnitRoutine>),
    /// A plane is responsible to reduce a full vector.
    Plane(BlueprintStrategy<PlaneRoutine>),
    /// A ruda is responsible to reduce a full vector.
    Ruda(BlueprintStrategy<RudaRoutine>),
}

#[derive(Debug, Clone, Copy)]
pub struct VectorizationStrategy {
    /// When the vectorization is parallel, enable vectorization of the output so that each
    /// unit can perform N reductions, where N is the output `vector_size`.
    pub parallel_output_vectorization: bool,
}

pub(crate) fn support_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().features.plane.contains(Plane::Ops)
}
