use super::*;
use alloc::string::String;
use ruda_model::{module::ModuleDisplay, record::{Record, PrecisionSettings}};
use crate::expert_parallel::ExpertOwnership;
use crate::transformer::{ExpertOwnedModelGeometry, ExpertAdapterOwnershipEntry};

/// Original expert-world metadata readable without data gathers or native
/// expert execution. This deliberately does not substitute data rank/world.
pub trait ShardedOwnedExpertMetadata<B: Backend>: FullyShardedModule<B> {
    fn expert_ownership(&self) -> &ExpertOwnership;
    fn expert_rank(&self) -> usize;
}
macro_rules! owner_metadata {
    ($target:ident) => {
        impl<B: Backend> ShardedOwnedExpertMetadata<B> for $target<B> {
            fn expert_ownership(&self) -> &ExpertOwnership { &self.ownership }
            fn expert_rank(&self) -> usize { self.rank }
        }
    };
}
owner_metadata!(FullyShardedOwnedSwiGluExperts);
owner_metadata!(FullyShardedOwnedFloatingExpertAdapters);
owner_metadata!(FullyShardedOwnedAwqExperts);
owner_metadata!(FullyShardedOwnedPackedExperts);
macro_rules! selected_metadata {
    ($target:ident, $first:ident, $second:ident) => {
        impl<B: Backend> ShardedOwnedExpertMetadata<B> for $target<B> {
            fn expert_ownership(&self) -> &ExpertOwnership {
                match self { Self::$first(value) => value.expert_ownership(), Self::$second(value) => value.expert_ownership() }
            }
            fn expert_rank(&self) -> usize {
                match self { Self::$first(value) => value.expert_rank(), Self::$second(value) => value.expert_rank() }
            }
        }
    };
}
selected_metadata!(FullyShardedSelectableOwnedExperts, Original, Adapted);
selected_metadata!(FullyShardedMixedOwnedExperts, Floating, Packed);

fn owner<B: Backend, E: ShardedOwnedExpertMetadata<B>>(layer: usize, experts: &E) -> ExpertAdapterOwnershipEntry {
    ExpertAdapterOwnershipEntry { layer, prefix: experts.expert_ownership().prefix().to_vec(), rank: experts.expert_rank() }
}

impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: ShardedOwnedExpertMetadata<B> + ModuleDisplay>
    ExpertOwnedModelGeometry<B> for FullyShardedExpertParallelTransformerModel<B, P, E> {
    fn expert_model_layers(&self) -> usize { self.layers.len() }
    fn expert_model_ownership(&self) -> Vec<ExpertAdapterOwnershipEntry> {
        self.layers.iter().enumerate().filter_map(|(index, layer)| match layer {
            FullyShardedExpertParallelTransformerLayer::Local(_) => None,
            FullyShardedExpertParallelTransformerLayer::Parallel(value) => Some(owner(index, &value.routed.experts)),
        }).collect()
    }
}

impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, L: FullyShardedModule<B> + ModuleDisplay,
    Q: FullyShardedModule<B> + ModuleDisplay, E: ShardedOwnedExpertMetadata<B> + ModuleDisplay> ExpertOwnedModelGeometry<B>
    for FullyShardedMhcResidualStack<B, P, FullyShardedMixedMhcFeedForward<B, L, Q, E>> {
    fn expert_model_layers(&self) -> usize { self.layers.len() }
    fn expert_model_ownership(&self) -> Vec<ExpertAdapterOwnershipEntry> {
        self.layers.iter().enumerate().filter_map(|(index, layer)| match &layer.feed_forward {
            FullyShardedMixedMhcFeedForward::Local(_) => None,
            FullyShardedMixedMhcFeedForward::Parallel(value) => Some(owner(index, &value.routed.experts)),
        }).collect()
    }
}
impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, Q: FullyShardedModule<B> + ModuleDisplay,
    E: ShardedOwnedExpertMetadata<B> + ModuleDisplay> ExpertOwnedModelGeometry<B>
    for FullyShardedMhcResidualStack<B, P, FullyShardedExpertParallelMhcFeedForward<B, Q, E>> {
    fn expert_model_layers(&self) -> usize { self.layers.len() }
    fn expert_model_ownership(&self) -> Vec<ExpertAdapterOwnershipEntry> {
        self.layers.iter().enumerate().map(|(index, layer)| owner(index, &layer.feed_forward.routed.experts)).collect()
    }
}
impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, F: FullyShardedModule<B> + ModuleDisplay, H: FullyShardedModule<B> + ModuleDisplay>
    ExpertOwnedModelGeometry<B> for FullyShardedMhcResidualModel<B, P, F, H>
where FullyShardedMhcResidualStack<B, P, F>: ExpertOwnedModelGeometry<B> {
    fn expert_model_layers(&self) -> usize { self.stack.expert_model_layers() }
    fn expert_model_ownership(&self) -> Vec<ExpertAdapterOwnershipEntry> { self.stack.expert_model_ownership() }
}

/// Exact local floating/packed FSDP storage, bound to the caller's prepared
/// architecture contract and original expert ownership. No complete global
/// expert tensors, optimizer state, scheduler, RNG or data position are implied.
#[derive(Clone)]
pub struct FullyShardedExpertOwnedStorageRecord<B: Backend> {
    version: u32,
    contract_id: String,
    layers: usize,
    ownership: Vec<ExpertAdapterOwnershipEntry>,
    storage: FullyShardedStorageRecord<B>,
}

impl<B: Backend> FullyShardedExpertOwnedStorageRecord<B> {
    pub fn capture<M: FullyShardedModule<B> + ExpertOwnedModelGeometry<B>>(model: &M, contract_id: &str)
        -> Result<Self, FullyShardedParameterError> {
        let value = Self { version: 1, contract_id: contract_id.into(), layers: model.expert_model_layers(),
            ownership: model.expert_model_ownership(), storage: FullyShardedStorageRecord::capture(model)? };
        value.validate()?; Ok(value)
    }
    pub fn contract_id(&self) -> &str { &self.contract_id }
    pub fn ownership(&self) -> &[ExpertAdapterOwnershipEntry] { &self.ownership }
    pub fn storage(&self) -> &FullyShardedStorageRecord<B> { &self.storage }

    pub fn validate(&self) -> Result<(), FullyShardedParameterError> {
        if self.version != 1 || self.contract_id.is_empty() { return Err(FullyShardedParameterError::Record); }
        let mut previous = None;
        for entry in &self.ownership {
            if entry.layer >= self.layers || previous.is_some_and(|layer| layer >= entry.layer)
                || entry.prefix.len() < 2 || entry.rank >= entry.prefix.len() - 1 || entry.prefix[0] != 0
                || entry.prefix.windows(2).any(|pair| pair[0] > pair[1])
                || entry.prefix.len() - 1 > u32::MAX as usize
                || entry.prefix.last().is_none_or(|total| *total == 0 || *total > u32::MAX as usize) {
                return Err(FullyShardedParameterError::Geometry("invalid original saved expert ownership"));
            }
            previous = Some(entry.layer);
        }
        self.storage.validate()
    }

    /// Check ownership and every actual local ID/shape/precision/training/tie
    /// contract before mapping any destination leaf. Expert repartition is not
    /// silently treated as ordinary data-axis repartition.
    pub fn validate_for<M: FullyShardedModule<B> + ExpertOwnedModelGeometry<B>>(&self, model: &M, contract_id: &str)
        -> Result<(), FullyShardedParameterError> {
        self.validate()?;
        if self.contract_id != contract_id || self.layers != model.expert_model_layers() || self.ownership != model.expert_model_ownership() {
            return Err(FullyShardedParameterError::Record);
        }
        self.storage.validate_for(model)
    }
    pub fn restore_into<M: FullyShardedModule<B> + ExpertOwnedModelGeometry<B>>(self, model: M, contract_id: &str)
        -> Result<M, FullyShardedParameterError> {
        self.validate_for(&model, contract_id)?;
        self.storage.restore_into(model)
    }

    /// Change only the explicit data-axis topology from a complete set of
    /// original data shards. All records must describe the same expert owner;
    /// NF4 windows/AWQ words and original floating storage remain unchanged.
    pub fn repartition_from_data_ranks(sources: &[Self], data_rank: usize, data_world: usize)
        -> Result<Self, FullyShardedParameterError> {
        let first = sources.first().ok_or(FullyShardedParameterError::Geometry("complete source data ranks are required"))?;
        for source in sources {
            source.validate()?;
            if source.contract_id != first.contract_id || source.layers != first.layers || source.ownership != first.ownership {
                return Err(FullyShardedParameterError::Record);
            }
        }
        let storage = sources.iter().map(|source| source.storage.clone()).collect::<Vec<_>>();
        let value = Self { version: 1, contract_id: first.contract_id.clone(), layers: first.layers, ownership: first.ownership.clone(),
            storage: FullyShardedStorageRecord::repartition_from_ranks(&storage, data_rank, data_world)? };
        value.validate()?; Ok(value)
    }
}
impl<B: Backend> Record<B> for FullyShardedExpertOwnedStorageRecord<B> {
    type Item<P: PrecisionSettings> = (u32, String, usize, Vec<ExpertAdapterOwnershipEntry>, <FullyShardedStorageRecord<B> as Record<B>>::Item<P>);
    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        (self.version, self.contract_id, self.layers, self.ownership, self.storage.into_item::<P>())
    }
    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self { version: item.0, contract_id: item.1, layers: item.2, ownership: item.3,
            storage: FullyShardedStorageRecord::from_item::<P>(item.4, device) }
    }
}
