use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::ir::DeviceProperties;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;

use crate::attention::kernel_ir::definition::{
    AttentionElems, AttentionPrecision, AttentionSetupError, RudaMapping, RudaMappingLaunch,
    InputRuntimeArg, OutputRuntimeArg,
};
use crate::attention::kernel_ir::{
    definition::attention_types::*,
    launch::AttentionArgs,
    {components::global::GlobalAttentionConfig, definition::AttentionVectorSizes},
};
use std::{fmt::Debug, hash::Hash};

/// A family of [BatchAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait BatchAttentionFamily: Send + Sync + 'static {
    /// The specific [BatchAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: BatchAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: BatchAttentionConfig;
    type Blueprint;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_unchecked<AA: AttentionArgs, R: Runtime>(
        client: &ComputeClient<R>,
        ruda_dim: RudaDim,
        ruda_count: RudaCount,
        address_type: AddressType,
        input: InputRuntimeArg<AA, R>,
        output: OutputRuntimeArg<AA, R>,
        ruda_mapping: RudaMappingLaunch<R>,
        dtypes: &AttentionElems,
        vector_sizes: &AttentionVectorSizes,
        attention_blueprint: Self::Blueprint,
    ) -> Result<(), LaunchError>;

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: Self::Blueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError>;
}

#[ruda]
pub trait BatchAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// The configuration type associated with this Attention.
    type Config: BatchAttentionConfig;

    fn execute(
        query: VirtualTensor<QG<AP>, QGS<AP>>,
        key: VirtualTensor<KG<AP>, KGS<AP>>,
        value: VirtualTensor<VG<AP>, VGS<AP>>,
        mask: ComptimeOption<VirtualTensor<MSK<AP>, MSKS<AP>>>,
        out: VirtualTensor<OG<AP>, OGS<AP>, ReadWrite>,
        ruda_mapping: RudaMapping,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the Batch Attention level
pub trait BatchAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type GlobalConfig: GlobalAttentionConfig;

    fn global_config(&self) -> Self::GlobalConfig;

    fn ruda_dim(&self) -> RudaDim;
}
