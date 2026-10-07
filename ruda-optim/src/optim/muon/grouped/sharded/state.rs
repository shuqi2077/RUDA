use super::{AutodiffBackend,AutodiffModule,BroadcastTensorCollective,MuonShardedAdamW,MuonError,Manifest,Placement,ShardRecords,AdamRecords,
    Record,PrecisionSettings,HashMap,ParamId,String,validate_adam_records,Optimizer};

/// Native rank-local sharded Muon and existing AdamW state, with exact original routing metadata.
/// Save the actual model record with it so original parameter IDs survive restoration.
#[derive(Clone)]
pub struct MuonShardedAdamWRecord<B: AutodiffBackend> {
    pub(super) version:u32,pub(super) config_key:String,pub(super) manifest:Manifest,pub(super) placement:Placement,
    pub(super) muon:ShardRecords<B>,pub(super) adamw:AdamRecords<B>,
}
impl<B: AutodiffBackend> MuonShardedAdamWRecord<B> {
    /// Number of actual saved native matrix-shard momentum states.
    pub fn muon_state_count(&self) -> usize {self.muon.len()}
    /// Number of actual saved auxiliary AdamW states, excluding frozen parameters.
    pub fn adamw_state_count(&self) -> usize {self.adamw.len()}
}
impl<B: AutodiffBackend> Record<B> for MuonShardedAdamWRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,Manifest,Placement,<ShardRecords<B> as Record<B::InnerBackend>>::Item<S>,<AdamRecords<B> as Record<B>>::Item<S>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.config_key,self.manifest,self.placement,<ShardRecords<B> as Record<B::InnerBackend>>::into_item::<S>(self.muon),
            <AdamRecords<B> as Record<B>>::into_item::<S>(self.adamw))
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,config_key:item.1,manifest:item.2,placement:item.3,muon:<ShardRecords<B> as Record<B::InnerBackend>>::from_item::<S>(item.4,device),
            adamw:<AdamRecords<B> as Record<B>>::from_item::<S>(item.5,device)}
    }
}

impl<M,B,C> MuonShardedAdamW<M,B,C>
    where B: AutodiffBackend,M: AutodiffModule<B>,C: BroadcastTensorCollective<B::InnerBackend> {
    /// Restore only corresponding native configuration, original IDs/roles and actual rank/layout.
    pub fn try_load_record(mut self,record: MuonShardedAdamWRecord<B>) -> Result<Self,MuonError> {
        if record.version != 1 || record.config_key != self.config_key || record.manifest != self.manifest || record.placement != self.placement() {
            return Err(MuonError::IncompatibleRecord);
        }
        let known: HashMap<_,_> = self.manifest.iter().map(|entry|(ParamId::from(entry.0),entry)).collect();
        for (id,state) in &record.muon {
            let index = self.indices.get(id).ok_or(MuonError::IncompatibleRecord)?;let binding = &self.bindings[*index];
            let expected = known.get(id).ok_or(MuonError::IncompatibleRecord)?;
            let shape: [usize;2] = expected.1.as_slice().try_into().map_err(|_|MuonError::IncompatibleRecord)?;
            let global = binding.layout.global_shape(binding.communicator.rank(),binding.communicator.world_size(),shape)?;
            state.validate_placement(binding.communicator.rank(),&binding.layout,global)?;
            if state.momentum().dims() != shape || alloc::format!("{:?}",state.momentum().dtype()) != expected.3 {
                return Err(MuonError::IncompatibleRecord);
            }
        }
        if record.adamw.keys().any(|id|!known.contains_key(id) || self.indices.contains_key(id)) {return Err(MuonError::IncompatibleRecord);}
        validate_adam_records(&record.adamw,&self.manifest)?;
        self.states = record.muon;self.adamw = self.adamw.load_record(record.adamw);
        Ok(self)
    }
}
