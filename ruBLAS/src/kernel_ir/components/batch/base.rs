use ruda_kernel::dsl as kernel_dsl;
use crate::kernel_ir::{
    components::stage::NumStages,
    definition::{
        AccG, Blueprint, LhsG, MatmulElems, MatmulProblem, MatmulSetupError, MatmulTypes,
        MatmulVectorSizes, RhsG,
    },
};
use crate::kernel_ir::{
    definition::{RudaMapping, RudaMappingLaunch},
    launch::{InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
    {components::RudaDimResource, launch::RuntimeConfig},
    {components::global::memory::GlobalLayoutConfig, launch::ConfigRuntimeArg},
};
use ruda_kernel::dsl::ir::DeviceProperties;
use ruda_kernel::dsl::prelude::*;
use std::{fmt::Debug, hash::Hash};

/// A family of [matmuls](BatchMatmul) working with any [precision](MatmulPrecision).
pub trait BatchMatmulFamily<RC: RuntimeConfig>: 'static + Send + Sync {
    /// The specific [BatchMatmul] implementation associated with this family.
    type Matmul<MP: MatmulTypes>: BatchMatmul<RC, MP, Config = Self::Config>;

    /// The configuration type associated with this matmul family.
    type Config: BatchConfig;

    type Blueprint: Blueprint;

    /// Constructs the configuration based on the matmul problem, selection, and vector sizes.
    ///
    /// This function may return an error if the configuration cannot be supported on the current runtime.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError>;

    fn num_stages() -> NumStages;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<MA: MatmulArgs<Config = RC>, R: Runtime>(
        client: &ComputeClient<R>,
        ruda_dim: RudaDim,
        ruda_count: RudaCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        config: ConfigRuntimeArg<MA, R>,
        ruda_mapping: RudaMappingLaunch<R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError>;

    /// Returns the compute resources required to run this matmul.
    fn rudadim_resource(
        blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<RudaDimResource, MatmulSetupError>;

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError>;
}

#[ruda]
/// Provides matrix multiplication operations at the batch level.
///
/// At the batch level,
///  - Inputs are whole tensors in global memory.
///  - All Rudas are used to solve the problem
///  - Dimensions M, N and K can be arbitrary large,
///    as well as the number of batches.
///
/// # Assumptions
/// - Vector sizes of the inputs evenly divide the dimension they are aligned with.
///
/// # Safety
///
/// - It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
///   It is therefore important to use an underlying global matmul that performs check bounds,
/// - It is accepted to launch more Ruda than necessary, providing a RudaCountInput that states
///   the max ruda position
pub trait BatchMatmul<RC: RuntimeConfig, MP: MatmulTypes>: 'static + Send + Sync {
    type Config: BatchConfig;

    /// Performs batchwise matrix multiplication over tensors.
    fn execute<Args: MatmulArgs<Config = RC>>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        ruda_mapping: RudaMapping,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the [batch matmul](BatchMatmul) level.
pub trait BatchConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig;
    fn out_global_layout_config(&self) -> GlobalLayoutConfig;
}

/// How oobs are handled.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CheckBounds {
    /// No bound check is necessary.
    None,
    /// Use checked reads and writes.
    Checked,
    /// Terminate idle work units early.
    Terminate,
}
