use super::*;
use crate::{MuonState,repartition_flat_buffer};

fn partition_buffer<B:Backend,const D:usize>(value:Tensor<B,D>,rank:u32,world:u32) -> Result<Tensor<B,1>,MuonError> {
    if world==0 || rank>=world {return Err(MuonError::InvalidConfig("invalid original state buffer rank/world"));}
    let shape=value.dims();
    let elements=shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis)).ok_or(MuonError::InvalidConfig("original state buffer size overflows"))?;
    let size=elements.div_ceil(world as usize);parameter_geometry(&shape,rank,world,size)?;
    let start=rank as usize*size;let real=elements.saturating_sub(start).min(size);
    let mut local=Tensor::zeros([size],(&value.device(),value.dtype()));
    if real>0 {local=local.slice_assign([0..real],value.reshape([elements]).slice([start..start+real]));}Ok(local)
}

fn check_global_adam<B:Backend,const D:usize>(state:&AdamWState<B,D>,shape:[usize;D],dtype:DType,device:&B::Device) -> Result<(),MuonError> {
    let momentum=&state.momentum;
    if momentum.time==0 || momentum.moment_1.dims()!=shape || momentum.moment_2.dims()!=shape
        || momentum.max_moment_2.as_ref().is_some_and(|value|value.dims()!=shape) {return Err(MuonError::IncompatibleRecord);}
    if momentum.moment_1.dtype()!=dtype || momentum.moment_2.dtype()!=dtype
        || momentum.max_moment_2.as_ref().is_some_and(|value|value.dtype()!=dtype) {return Err(MuonError::DTypeMismatch("original complete AdamW state"));}
    if momentum.moment_1.device()!=*device || momentum.moment_2.device()!=*device
        || momentum.max_moment_2.as_ref().is_some_and(|value|value.device()!=*device) {return Err(MuonError::DeviceMismatch("original complete AdamW state"));}
    Ok(())
}

fn partition_adam<B:Backend,const D:usize>(state:AdamWState<B,D>,rank:u32,world:u32) -> Result<AdamWState<B,1>,MuonError> {
    check_global_adam(&state,state.momentum.moment_1.dims(),state.momentum.moment_1.dtype(),&state.momentum.moment_1.device())?;
    let momentum=state.momentum;
    Ok(AdamWState {momentum:crate::AdaptiveMomentumState {time:momentum.time,
        moment_1:partition_buffer(momentum.moment_1,rank,world)?,moment_2:partition_buffer(momentum.moment_2,rank,world)?,
        max_moment_2:momentum.max_moment_2.map(|value|partition_buffer(value,rank,world)).transpose()?}})
}

impl<B:Backend> FullyShardedMuonAdamWState<B> {
    /// Partition actual complete native Muon history; retain the original optimizer configuration when importing it.
    pub fn from_global_muon_state(state:MuonState<B,2>,layout:&MuonFlatShardLayout,rank:u32,world:u32) -> Result<Self,MuonError> {
        Ok(Self::native_muon(MuonFlatShardedState::from_global_state(state,layout,rank,world)?))
    }
    /// Partition actual original AdamW first/second/AMSGrad buffers and retain the exact original per-parameter clock.
    pub fn from_global_adamw_state<const D:usize>(state:AdamWState<B,D>,rank:u32,world:u32) -> Result<Self,MuonError> {
        let dtype=state.momentum.moment_1.dtype();Ok(Self::native_adamw(partition_adam(state,rank,world)?,dtype))
    }
    /// Partition the authoritative loaded original FP32 Muon master/history, not the rounded model weight.
    pub fn from_global_master_muon_state(state:Fp32MasterState<B,2,MuonState<B,2>>,layout:&MuonFlatShardLayout,rank:u32,world:u32)
        -> Result<Self,MuonError> {
        Ok(Self::master_muon(Fp32MasterState::from_global_muon_state(state,layout,rank,world)?))
    }
    /// Partition the actual original FP32 AdamW master and every matching native history buffer together.
    /// A genuinely absent inner state stays absent; no fabricated optimizer clock/history is inserted.
    pub fn from_global_master_adamw_state<const D:usize>(state:Fp32MasterState<B,D,AdamWState<B,D>>,rank:u32,world:u32)
        -> Result<Self,MuonError> {
        if state.master.dtype()!=DType::F32 {return Err(MuonError::DTypeMismatch("master"));}
        if let Some(inner)=&state.inner {check_global_adam(inner,state.master.dims(),DType::F32,&state.master.device())?;}
        Ok(Self::master_adamw(Fp32MasterState {master:partition_buffer(state.master,rank,world)?,
            inner:state.inner.map(|inner|partition_adam(inner,rank,world)).transpose()?}))
    }
}

impl<B:AutodiffBackend> FullyShardedMuonAdamWRecord<B> {
    /// Move actual saved local momentum/masters only, without changing original configuration or placement.
    pub fn to_device(mut self,device:&B::Device) -> Self {
        self.states=self.states.into_iter().map(|(id,state)|(id,state.to_device(device))).collect();self
    }
    /// Offline complete-rank-set conversion of a whole original mixed optimizer checkpoint.
    /// Records must represent the same completed boundary and one complete original data group. The actual
    /// algorithm/configuration, parameter roles, absence of unused state and every per-parameter clock remain
    /// unchanged. Load/repartition matching model/data/scheduler/gradient records and create the new explicit
    /// communicators separately; this does not perform an automatic live-world migration or tensor transport.
    pub fn repartition_from_ranks(sources:&[Self],rank:u32,world:u32) -> Result<Self,MuonError> {
        let first=sources.first().ok_or(MuonError::InvalidConfig("complete original mixed optimizer record set is empty"))?;
        if world==0 || rank>=world {return Err(MuonError::InvalidConfig("invalid destination mixed optimizer record rank/world"));}
        let old_world=u32::try_from(sources.len()).map_err(|_|MuonError::InvalidConfig("original mixed optimizer record rank count overflows"))?;
        let ids=first.states.keys().copied().collect::<HashSet<_>>();
        for (source_rank,source) in sources.iter().enumerate() {
            if source.version!=1 || source.config_key!=first.config_key || source.manifest.len()!=first.manifest.len()
                || source.placement.len()!=first.placement.len() || source.placement.len()!=source.manifest.len()
                || source.states.keys().copied().collect::<HashSet<_>>()!=ids {return Err(MuonError::IncompatibleRecord);}
            let mut previous=None;
            for ((manifest,placement),(original,original_placement)) in source.manifest.iter().zip(&source.placement)
                .zip(first.manifest.iter().zip(&first.placement)) {
                if previous.is_some_and(|id|id>=manifest.0) || placement.0!=manifest.0 || manifest!=original
                    || placement.0!=original_placement.0 || placement.1!=original_placement.1 || placement.4!=original_placement.4
                    || placement.2!=source_rank as u32 || placement.3!=old_world || (placement.4 && !manifest.2) {
                    return Err(MuonError::IncompatibleRecord);
                }
                previous=Some(manifest.0);parameter_geometry(&placement.1,placement.2,placement.3,manifest.1)?;
                if placement.4 && placement.1.len()!=2 {return Err(MuonError::IncompatibleRecord);}
            }
            for (id,state) in &source.states {
                let manifest=source.manifest.iter().find(|entry|entry.0==id.val()).ok_or(MuonError::IncompatibleRecord)?;
                let placement=source.placement.iter().find(|entry|entry.0==id.val()).ok_or(MuonError::IncompatibleRecord)?;
                if !manifest.2 {return Err(MuonError::IncompatibleRecord);}
                let master=state.master_muon.is_some() || state.master_adamw.is_some();
                if !master && format!("{:?}",state.storage)!=manifest.3 {return Err(MuonError::IncompatibleRecord);}
                if master && !matches!(manifest.3.as_str(),"F32"|"F16"|"BF16") {return Err(MuonError::IncompatibleRecord);}
                let variants=usize::from(state.muon.is_some())+usize::from(state.adamw.is_some())+usize::from(state.master_muon.is_some())+usize::from(state.master_adamw.is_some());
                if variants!=1 || placement.4!=(state.muon.is_some() || state.master_muon.is_some()) {return Err(MuonError::IncompatibleRecord);}
                if let Some(momentum)=state.muon_momentum() {
                    if momentum.dims()!=[manifest.1] || momentum.dtype()!=state.storage {return Err(MuonError::IncompatibleRecord);}
                }
                if let Some(master)=state.master() {if master.dims()!=[manifest.1] || master.dtype()!=DType::F32 {return Err(MuonError::IncompatibleRecord);}}
            }
        }
        let mut states=HashMap::new();
        for (id,_) in &first.states {
            let placement=first.placement.iter().find(|entry|entry.0==id.val()).ok_or(MuonError::IncompatibleRecord)?;
            let records=sources.iter().map(|source|source.states.get(id).cloned().ok_or(MuonError::IncompatibleRecord)).collect::<Result<Vec<_>,_>>()?;
            states.insert(*id,FullyShardedMuonAdamWState::repartition_from_ranks(&records,&placement.1,placement.4,rank,world)?);
        }
        let mut manifest=first.manifest.clone();let mut placement=first.placement.clone();
        for (entry,layout) in manifest.iter_mut().zip(&mut placement) {
            let elements=layout.1.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis)).ok_or(MuonError::InvalidConfig("destination mixed optimizer record size overflows"))?;
            entry.1=elements.div_ceil(world as usize);layout.2=rank;layout.3=world;
            parameter_geometry(&layout.1,rank,world,entry.1)?;
        }
        Ok(Self {version:1,config_key:first.config_key.clone(),manifest,placement,states})
    }
}

fn repartition_adam<B:Backend>(sources:&[AdamWState<B,1>],elements:usize,rank:u32,world:u32) -> Result<AdamWState<B,1>,MuonError> {
    let first=sources.first().ok_or(MuonError::InvalidConfig("complete original AdamW rank set is empty"))?;
    for state in sources {
        check_global_adam(state,first.momentum.moment_1.dims(),first.momentum.moment_1.dtype(),&first.momentum.moment_1.device())?;
        if state.momentum.time!=first.momentum.time || state.momentum.max_moment_2.is_some()!=first.momentum.max_moment_2.is_some() {return Err(MuonError::IncompatibleRecord);}
    }
    let moments1=sources.iter().map(|state|state.momentum.moment_1.clone()).collect::<Vec<_>>();
    let moments2=sources.iter().map(|state|state.momentum.moment_2.clone()).collect::<Vec<_>>();
    let maximum=if first.momentum.max_moment_2.is_some() {
        let buffers=sources.iter().map(|state|state.momentum.max_moment_2.clone().ok_or(MuonError::IncompatibleRecord)).collect::<Result<Vec<_>,_>>()?;
        Some(repartition_flat_buffer(&buffers,elements,rank,world)?)
    } else {None};
    Ok(AdamWState {momentum:crate::AdaptiveMomentumState {time:first.momentum.time,
        moment_1:repartition_flat_buffer(&moments1,elements,rank,world)?,moment_2:repartition_flat_buffer(&moments2,elements,rank,world)?,max_moment_2:maximum}})
}

impl<B:Backend> FullyShardedMuonAdamWState<B> {
    /// Offline complete-rank-set conversion of one original parameter's exact native histories and optional master.
    /// Corresponding model/logical placement/data/scheduler state must be converted before actual resumed training.
    pub fn repartition_from_ranks(sources:&[Self],logical_shape:&[usize],use_muon:bool,rank:u32,world:u32) -> Result<Self,MuonError> {
        let first=sources.first().ok_or(MuonError::InvalidConfig("complete original mixed optimizer state set is empty"))?;
        let old_world=u32::try_from(sources.len()).map_err(|_|MuonError::InvalidConfig("original optimizer rank count overflows"))?;
        if world==0 {return Err(MuonError::InvalidConfig("invalid destination optimizer world"));}
        if use_muon && logical_shape.len()!=2 {return Err(MuonError::ExpectedMatrix {rank:logical_shape.len()});}
        let elements=logical_shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis)).ok_or(MuonError::InvalidConfig("original mixed optimizer logical size overflows"))?;
        parameter_geometry(logical_shape,rank,world,elements.div_ceil(world as usize))?;
        let variant=|state:&Self| (state.muon.is_some(),state.adamw.is_some(),state.master_muon.is_some(),state.master_adamw.is_some());
        let selected=variant(first);let count=usize::from(selected.0)+usize::from(selected.1)+usize::from(selected.2)+usize::from(selected.3);
        if count!=1 || use_muon!=(selected.0 || selected.2) {return Err(MuonError::IncompatibleRecord);}
        for source in sources {if source.storage!=first.storage || variant(source)!=selected {return Err(MuonError::IncompatibleRecord);}}
        if let Some(original)=&first.muon {
            if original.layout().shape.as_slice()!=logical_shape {return Err(MuonError::IncompatibleRecord);}
            let states=sources.iter().map(|state|state.muon.clone().ok_or(MuonError::IncompatibleRecord)).collect::<Result<Vec<_>,_>>()?;
            if states.iter().any(|state|state.momentum().dtype()!=first.storage) {return Err(MuonError::DTypeMismatch("original native Muon state"));}
            return Ok(Self::native_muon(MuonFlatShardedState::repartition_from_ranks(&states,rank,world)?));
        }
        if first.adamw.is_some() {
            let states=sources.iter().map(|state|state.adamw.clone().ok_or(MuonError::IncompatibleRecord)).collect::<Result<Vec<_>,_>>()?;
            if states.iter().any(|state|state.momentum.moment_1.dtype()!=first.storage) {return Err(MuonError::DTypeMismatch("original native AdamW state"));}
            return Ok(Self::native_adamw(repartition_adam(&states,elements,rank,world)?,first.storage));
        }
        let masters=sources.iter().map(|state|state.master().cloned().ok_or(MuonError::IncompatibleRecord)).collect::<Result<Vec<_>,_>>()?;
        if first.storage!=DType::F32 || masters.iter().any(|master|master.dtype()!=DType::F32) {return Err(MuonError::DTypeMismatch("master"));}
        let master=repartition_flat_buffer(&masters,elements,rank,world)?;
        if let Some(original)=&first.master_muon {
            let present=original.inner.is_some();let mut inner=Vec::new();
            for (source_rank,source) in sources.iter().enumerate() {
                let state=source.master_muon.as_ref().ok_or(MuonError::IncompatibleRecord)?;
                if state.inner.is_some()!=present {return Err(MuonError::IncompatibleRecord);}
                if let Some(state)=&state.inner {
                    if state.momentum().dtype()!=DType::F32 || state.momentum().device()!=masters[source_rank].device() {return Err(MuonError::IncompatibleRecord);}
                    let shape=logical_shape.try_into().map_err(|_|MuonError::IncompatibleRecord)?;
                    state.validate_placement(source_rank as u32,old_world,&MuonFlatShardLayout::new(shape))?;inner.push(state.clone());
                }
            }
            return Ok(Self::master_muon(Fp32MasterState {master,inner:if present {Some(MuonFlatShardedState::repartition_from_ranks(&inner,rank,world)?)} else {None}}));
        }
        let mut inner=Vec::new();let present=first.master_adamw.as_ref().ok_or(MuonError::IncompatibleRecord)?.inner.is_some();
        for (source_rank,source) in sources.iter().enumerate() {
            let state=source.master_adamw.as_ref().ok_or(MuonError::IncompatibleRecord)?;
            if state.inner.is_some()!=present {return Err(MuonError::IncompatibleRecord);}
            if let Some(state)=&state.inner {
                check_global_adam(state,masters[source_rank].dims(),DType::F32,&masters[source_rank].device())?;inner.push(state.clone());
            }
        }
        Ok(Self::master_adamw(Fp32MasterState {master,inner:if present {Some(repartition_adam(&inner,elements,rank,world)?)} else {None}}))
    }
}
