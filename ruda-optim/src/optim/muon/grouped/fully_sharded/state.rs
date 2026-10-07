use super::*;
use ruda_model::tensor::backend::Backend;
mod partition;

/// Actual one-role local optimizer state: native Muon/AdamW or explicitly selected FP32 master variants.
/// Payload choice is checked against the original parameter routing and numerical configuration on load.
#[derive(Clone)]
pub struct FullyShardedMuonAdamWState<B:Backend> {
    storage:DType,
    pub(super) muon:Option<MuonFlatShardedState<B>>,
    pub(super) adamw:Option<AdamWState<B,1>>,
    pub(super) master_muon:Option<Fp32MasterState<B,1,MuonFlatShardedState<B>>>,
    pub(super) master_adamw:Option<Fp32MasterState<B,1,AdamWState<B,1>>>,
}
impl<B:Backend> FullyShardedMuonAdamWState<B> {
    pub(super) fn native_muon(state:MuonFlatShardedState<B>) -> Self {
        Self {storage:state.momentum().dtype(),muon:Some(state),adamw:None,master_muon:None,master_adamw:None}
    }
    pub(super) fn native_adamw(state:AdamWState<B,1>,storage:DType) -> Self {
        Self {storage,muon:None,adamw:Some(state),master_muon:None,master_adamw:None}
    }
    pub(super) fn master_muon(state:Fp32MasterState<B,1,MuonFlatShardedState<B>>) -> Self {
        Self {storage:DType::F32,muon:None,adamw:None,master_muon:Some(state),master_adamw:None}
    }
    pub(super) fn master_adamw(state:Fp32MasterState<B,1,AdamWState<B,1>>) -> Self {
        Self {storage:DType::F32,muon:None,adamw:None,master_muon:None,master_adamw:Some(state)}
    }
    /// Actual local Muon history, never a reconstructed complete global matrix.
    pub fn muon_momentum(&self) -> Option<&Tensor<B,1>> {
        self.muon.as_ref().map(|state|state.momentum()).or_else(||self.master_muon.as_ref().and_then(|state|state.inner.as_ref().map(|inner|inner.momentum())))
    }
    /// Actual authoritative FP32 local parameter when master precision was explicitly selected.
    pub fn master(&self) -> Option<&Tensor<B,1>> {
        self.master_muon.as_ref().map(|state|&state.master).or_else(||self.master_adamw.as_ref().map(|state|&state.master))
    }
    /// Actual native/FP32 AdamW moments for this rank's original element interval.
    pub fn adamw_state(&self) -> Option<&AdamWState<B,1>> {
        self.adamw.as_ref().or_else(||self.master_adamw.as_ref().and_then(|state|state.inner.as_ref()))
    }
    pub(super) fn to_device(mut self,device:&B::Device) -> Self {
        self.muon=self.muon.map(|state|state.to_device(device));
        self.adamw=self.adamw.map(|state|<AdamW as SimpleOptimizer<B>>::to_device(state,device));
        self.master_muon=self.master_muon.map(|state|state.to_flat_shard_device(device));
        self.master_adamw=self.master_adamw.map(|state|<Fp32MasterOptimizer<AdamW> as SimpleOptimizer<B>>::to_device(state,device));self
    }
    pub(super) fn validate<C:BroadcastTensorCollective<B>>(&self,binding:&FullyShardedOptimizerParameter<C>,local:usize,model_dtype:DType,
        master:bool,amsgrad:bool) -> Result<(),MuonError> {
        parameter_geometry(&binding.logical_shape,binding.communicator.rank(),binding.communicator.world_size(),local)?;
        let variants=usize::from(self.muon.is_some())+usize::from(self.adamw.is_some())+usize::from(self.master_muon.is_some())+usize::from(self.master_adamw.is_some());
        if variants!=1 || self.storage!=if master {DType::F32} else {model_dtype} {return Err(MuonError::IncompatibleRecord);}
        if binding.use_muon {
            let shape=binding.logical_shape.as_slice().try_into().map_err(|_|MuonError::IncompatibleRecord)?;let layout=MuonFlatShardLayout::new(shape);
            let state=if master {
                let master=self.master_muon.as_ref().ok_or(MuonError::IncompatibleRecord)?;
                if master.master.dims()!=[local] || master.master.dtype()!=DType::F32 {return Err(MuonError::IncompatibleRecord);}
                master.inner.as_ref()
            } else {Some(self.muon.as_ref().ok_or(MuonError::IncompatibleRecord)?)};
            if let Some(state)=state {
                state.validate_placement(binding.communicator.rank(),binding.communicator.world_size(),&layout)?;
                if state.momentum().dtype()!=self.storage {return Err(MuonError::IncompatibleRecord);}
            }
        } else {
            let state=if master {
                let master=self.master_adamw.as_ref().ok_or(MuonError::IncompatibleRecord)?;
                if master.master.dims()!=[local] || master.master.dtype()!=DType::F32 {return Err(MuonError::IncompatibleRecord);}
                master.inner.as_ref()
            } else {Some(self.adamw.as_ref().ok_or(MuonError::IncompatibleRecord)?)};
            if let Some(state)=state {
                let momentum=&state.momentum;
                if momentum.time==0 || momentum.moment_1.dims()!=[local] || momentum.moment_2.dims()!=[local]
                    || momentum.moment_1.dtype()!=self.storage || momentum.moment_2.dtype()!=self.storage
                    || momentum.max_moment_2.is_some()!=amsgrad || momentum.max_moment_2.as_ref().is_some_and(|value|value.dims()!=[local] || value.dtype()!=self.storage) {
                    return Err(MuonError::IncompatibleRecord);
                }
            }
        }
        Ok(())
    }
}

fn cast_adam<B:Backend>(mut state:AdamWState<B,1>,dtype:DType) -> AdamWState<B,1> {
    state.momentum.moment_1=state.momentum.moment_1.cast(dtype);state.momentum.moment_2=state.momentum.moment_2.cast(dtype);
    state.momentum.max_moment_2=state.momentum.max_moment_2.map(|value|value.cast(dtype));state
}
impl<B:Backend> Record<B> for FullyShardedMuonAdamWState<B> {
    type Item<P:PrecisionSettings>=(DType,<Option<MuonFlatShardedState<B>> as Record<B>>::Item<P>,<Option<AdamWState<B,1>> as Record<B>>::Item<P>,
        <Option<Fp32MasterState<B,1,MuonFlatShardedState<B>>> as Record<B>>::Item<P>,<Option<Fp32MasterState<B,1,AdamWState<B,1>>> as Record<B>>::Item<P>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.storage,self.muon.into_item::<P>(),self.adamw.into_item::<P>(),self.master_muon.into_item::<P>(),self.master_adamw.into_item::<P>())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        let adamw=Option::<AdamWState<B,1>>::from_item::<P>(item.2,device).map(|state|cast_adam(state,item.0));
        let master_adamw=Option::<Fp32MasterState<B,1,AdamWState<B,1>>>::from_item::<P>(item.4,device)
            .map(|mut state| {state.inner=state.inner.map(|inner|cast_adam(inner,DType::F32));state});
        Self {storage:item.0,muon:Option::<MuonFlatShardedState<B>>::from_item::<P>(item.1,device),adamw,
            master_muon:Option::<Fp32MasterState<B,1,MuonFlatShardedState<B>>>::from_item::<P>(item.3,device),master_adamw}
    }
}

/// Exact original mixed-optimizer configuration, parameter roles/topology and actual local momentum/master payloads.
/// Save the corresponding model, data, scheduler and pending gradient records separately at the same boundary.
#[derive(Clone)]
pub struct FullyShardedMuonAdamWRecord<B:AutodiffBackend> {
    pub(super) version:u32,pub(super) config_key:String,pub(super) manifest:FlatManifest,pub(super) placement:Placement,pub(super) states:States<B>,
}
impl<B:AutodiffBackend> FullyShardedMuonAdamWRecord<B> {
    /// Actual native local state per canonical parameter identity.
    pub fn states(&self) -> &States<B> {&self.states}
    /// Number of actual allocated local parameter states, not logical aliases or padded element counts.
    pub fn state_parameter_count(&self) -> usize {self.states.len()}
}
impl<B:AutodiffBackend> Record<B> for FullyShardedMuonAdamWRecord<B> {
    type Item<P:PrecisionSettings>=(u32,String,FlatManifest,Placement,<States<B> as Record<B::InnerBackend>>::Item<P>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.config_key,self.manifest,self.placement,<States<B> as Record<B::InnerBackend>>::into_item::<P>(self.states))
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        Self {version:item.0,config_key:item.1,manifest:item.2,placement:item.3,states:<States<B> as Record<B::InnerBackend>>::from_item::<P>(item.4,device)}
    }
}
impl<M,B,C> FullyShardedMuonAdamW<M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    /// Import actual independently partitioned native local states using the original optimizer settings.
    pub fn try_load_states(self,states:States<B>) -> Result<Self,MuonError> {
        let mut record=self.to_record();record.states=states;self.try_load_record(record)
    }
    /// Restore only matching original local geometry, roles, precision and numerical configuration.
    pub fn try_load_record(mut self,record:FullyShardedMuonAdamWRecord<B>) -> Result<Self,MuonError> {
        if record.version!=1 || record.config_key!=self.config_key || record.manifest!=self.manifest || record.placement!=self.placement() {return Err(MuonError::IncompatibleRecord);}
        for (id,state) in &record.states {
            let index=*self.indices.get(id).ok_or(MuonError::IncompatibleRecord)?;
            let expected=self.manifest.iter().find(|entry|entry.0==id.val()).ok_or(MuonError::IncompatibleRecord)?;
            if !expected.2 {return Err(MuonError::IncompatibleRecord);}
            let model_dtype=if self.master.is_some() {DType::F32} else {
                // Native states keep the actual original storage, compared directly to the prepared manifest.
                if format!("{:?}",state.storage)!=expected.3 {return Err(MuonError::IncompatibleRecord);}state.storage
            };
            state.validate(&self.bindings[index],expected.1,model_dtype,self.master.is_some(),self.adamw.uses_amsgrad())?;
        }
        self.states=record.states;Ok(self)
    }
}
