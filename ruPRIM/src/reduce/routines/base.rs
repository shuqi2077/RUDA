use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::{ReduceDtypes, ReduceError, VectorizationMode, routines::ReduceBlueprint};
use ruda_kernel::dsl::prelude::*;

#[derive(Debug)]
pub struct ReduceVectorSettings {
    pub vectorization_mode: VectorizationMode,
    pub vector_size_input: VectorSize,
    pub vector_size_output: VectorSize,
}

#[derive(Debug)]
pub struct ReduceLaunchSettings {
    pub ruda_dim: RudaDim,
    pub ruda_count: RudaCount,
    pub address_type: AddressType,
    pub vector: ReduceVectorSettings,
}

#[derive(Debug)]
pub struct ReduceProblem {
    /// Number of elements in reduce axis
    pub reduce_len: usize,
    /// Number of instances of the reduce axis
    pub reduce_count: usize,
    pub axis: usize,
    pub dtypes: ReduceDtypes,
    /// The address type, defined by the max of each handle's `required_address_type`
    pub address_type: AddressType,
}

#[derive(Debug, Clone)]
pub enum BlueprintStrategy<R: Routine> {
    Forced(R::Blueprint, RudaDim),
    Inferred(R::Strategy),
}

pub trait Routine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError>;
}

pub(crate) fn validate_ruda_dim<R: Runtime>(
    client: &ComputeClient<R>,
    ruda_dim: RudaDim,
) -> Result<(), ReduceError> {
    let hardware = &client.properties().hardware;
    let units = ruda_dim.x.checked_mul(ruda_dim.y).and_then(|xy| xy.checked_mul(ruda_dim.z));
    if ruda_dim.x == 0 || ruda_dim.y == 0 || ruda_dim.z == 0 {
        return Err(ReduceError::Validation {
            details: "Ruda dimensions must be nonzero",
        });
    }
    if ruda_dim.x > hardware.max_ruda_dim.0
        || ruda_dim.y > hardware.max_ruda_dim.1
        || ruda_dim.z > hardware.max_ruda_dim.2
        || units.is_none_or(|units| units > hardware.max_units_per_ruda)
    {
        return Err(ReduceError::Validation {
            details: "Ruda dimensions exceed device limits",
        });
    }
    Ok(())
}
