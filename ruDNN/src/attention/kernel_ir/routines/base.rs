use ruda_kernel::dsl as kernel_dsl;
use std::fmt::Debug;

use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::ir::AddressType;

use crate::attention::kernel_ir::components::tile::TileAttentionKind;
use crate::attention::kernel_ir::components::{
    batch::BatchAttentionFamily, global::GlobalAttentionFamily, stage::StageAttentionFamily,
};
use crate::attention::kernel_ir::definition::{
    AttentionElems, AttentionProblem, AttentionSetupError, AttentionVectorSizes, RudaCountPlan,
};
use crate::attention::kernel_ir::launch::BlueprintStrategy;

pub trait Routine: Debug + Clone {
    /// Tile-level strategy this routine selects.
    const TILE_KIND: TileAttentionKind;

    type StageAttention: StageAttentionFamily;
    type GlobalAttention: GlobalAttentionFamily;
    type BatchAttention: BatchAttentionFamily<Blueprint = Self::Blueprint>;

    type Strategy;
    type Blueprint: Clone;

    fn prepare<R: Runtime>(
        problem: &AttentionProblem,
        device_settings: &DeviceSettings<R>,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError>;
}

pub struct LaunchInfo<B> {
    pub blueprint: B,
    pub dtypes: AttentionElems,
    pub ruda_dim: RudaDim,
    pub ruda_count_plan: RudaCountPlan,
    pub address_type: AddressType,
}

pub struct DeviceSettings<R: Runtime> {
    pub plane_dim: u32,
    pub max_ruda_count: (u32, u32, u32),
    pub vector_sizes: AttentionVectorSizes,
    pub client: ComputeClient<R>,
}

impl<R: Runtime> core::fmt::Debug for DeviceSettings<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSettings")
            .field("plane_dim", &self.plane_dim)
            .field("max_ruda_count", &self.max_ruda_count)
            .field("vector_sizes", &self.vector_sizes)
            .finish()
    }
}

impl<R: Runtime> DeviceSettings<R> {
    pub fn new(client: &ComputeClient<R>, problem: &AttentionProblem) -> Self {
        DeviceSettings {
            plane_dim: client.properties().hardware.plane_size_max,
            max_ruda_count: client.properties().hardware.max_ruda_count,
            vector_sizes: AttentionVectorSizes::new_max_for_problem(client, problem),
            client: client.clone(),
        }
    }
}
