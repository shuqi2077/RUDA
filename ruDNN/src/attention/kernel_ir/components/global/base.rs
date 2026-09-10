use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::ir::DeviceProperties;

use crate::attention::kernel_ir::{
    components::global::simple::AttentionWriter,
    definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        attention_types::*,
    },
};
use ruda_kernel::library::tensor::r#virtual::VirtualTensor;

use crate::attention::kernel_ir::components::{global::simple::QueryReader, stage::StageAttentionConfig};
use std::{fmt::Debug, hash::Hash};

/// A family of [GlobalAttention] implementations that operate with any [precision](AttentionPrecision).
pub trait GlobalAttentionFamily: Send + Sync + 'static {
    /// The specific [GlobalAttention] implementation associated with this family.
    type Attention<AP: AttentionPrecision>: GlobalAttention<AP, Config = Self::Config>;

    /// The configuration type associated with this Attention family.
    type Config: GlobalAttentionConfig;

    /// Constructs the configuration based on the algorithm's blueprint.
    ///
    /// This function may return an error if the configuration cannot be supported.
    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &AttentionBlueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError>;
}

#[ruda]
pub trait GlobalAttention<AP: AttentionPrecision>: 'static + Send + Sync {
    /// Writes to Out at the same offset it loaded Query
    type Writer: AttentionWriter<OS<AP>, OSS<AP>, OG<AP>, OGS<AP>>;

    /// Loads to SMEM as is (transposed later)
    type KeyReader: RudaType;
    /// Loads to SMEM as is
    type ValueReader: RudaType;
    /// Loads to SMEM as is
    type MaskReader: RudaType;

    /// The configuration type associated with this Attention.
    type Config: GlobalAttentionConfig;

    fn execute(
        query_reader: QueryReader<AP>,
        key_reader: Self::KeyReader,
        value_reader: Self::ValueReader,
        mask_reader: Self::MaskReader,
        writer: Self::Writer,
        seq_q: u32,
        seq_kv: u32,
        #[comptime] config: Self::Config,
    );

    fn init_query_reader(
        batch_index: u32,
        stage_q_offset: u32,
        query: VirtualTensor<QG<AP>, QGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> QueryReader<AP>;

    fn init_key_reader(
        batch_index: u32,
        key: VirtualTensor<KG<AP>, KGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::KeyReader;

    fn init_value_reader(
        batch_index: u32,
        value: VirtualTensor<VG<AP>, VGS<AP>>,
        #[comptime] config: Self::Config,
    ) -> Self::ValueReader;

    fn init_mask_reader(
        batch_index: u32,
        stage_q_offset: u32,
        mask: ComptimeOption<VirtualTensor<MSK<AP>, MSKS<AP>>>,
        seq_kv_shape: u32,
        #[comptime] config: Self::Config,
    ) -> Self::MaskReader;

    fn init_writer(
        batch_index: u32,
        stage_q_offset: u32,
        out: VirtualTensor<OG<AP>, OGS<AP>, ReadWrite>,
        #[comptime] config: Self::Config,
    ) -> Self::Writer;
}

/// Configuration for the Global Attention level
pub trait GlobalAttentionConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type StageConfig: StageAttentionConfig;

    fn stage_config(&self) -> Self::StageConfig;
    fn ruda_dim(&self) -> RudaDim;
}
