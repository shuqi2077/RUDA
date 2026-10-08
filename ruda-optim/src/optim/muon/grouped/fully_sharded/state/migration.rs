use super::*;
use alloc::collections::BTreeMap;
use crate::{FullyShardedOptimizerMigration,FullyShardedParameterPlacement};

impl<B:AutodiffBackend> FullyShardedMuonAdamWRecord<B> {
    /// Original logical ownership and explicitly saved Muon role, including
    /// unused leaves with no momentum. Roles are not inferred from state presence
    /// or matrix shape; caller-owned transports are reconstructed separately.
    pub fn parameter_placements(&self) -> Vec<(FullyShardedParameterPlacement,bool)> {
        self.placement.iter().map(|entry|(FullyShardedParameterPlacement::new(ParamId::from(entry.0),entry.1.clone(),entry.2,entry.3),entry.4)).collect()
    }

    /// Exact offline native Muon/AdamW/master migration over explicit source
    /// groups for each original logical parameter. Unrelated TP/expert-owned
    /// local parameter lists and data worlds may differ; original configuration,
    /// roles, dtype, clocks and history presence must still match for each chosen
    /// group. Invokes the original concrete state's repartition implementation,
    /// never a coordinate-fragment replacement for Muon's matrix update.
    /// Source indices select one complete group, ordered internally by saved data
    /// rank. No SUM, optimizer replay, scheduler step or automatic peer recovery.
    /// Destination native buffers use the original repartitioner device; the
    /// existing `to_device` can explicitly move the resulting complete record.
    pub fn repartition_explicit(sources:&[Self],migrations:&[FullyShardedOptimizerMigration]) -> Result<Self,MuonError> {
        let first=sources.first().ok_or(MuonError::InvalidConfig("actual original mixed optimizer archives required"))?;
        let mut indices=Vec::with_capacity(sources.len());
        for source in sources {
            let manifest=source.manifest.iter().map(|entry|(entry.0,entry)).collect::<BTreeMap<_,_>>();
            let placement=source.placement.iter().map(|entry|(entry.0,entry)).collect::<BTreeMap<_,_>>();
            if source.version!=1 || source.config_key!=first.config_key || manifest.len()!=source.manifest.len()
                || placement.len()!=source.placement.len() || !manifest.keys().eq(placement.keys())
                || source.states.keys().any(|id|!manifest.contains_key(&id.val())) {return Err(MuonError::IncompatibleRecord);}
            indices.push((manifest,placement));
        }
        let mut states=HashMap::new();let mut manifest=Vec::with_capacity(migrations.len());let mut placement=Vec::with_capacity(migrations.len());
        let mut ids=HashSet::new();
        for migration in migrations {
            let target=&migration.target;let id=target.parameter;
            if !ids.insert(id) || target.world_size==0 || target.rank>=target.world_size {
                return Err(MuonError::InvalidConfig("invalid or duplicate explicit mixed optimizer target"));
            }
            let count=target.logical_shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis))
                .ok_or(MuonError::InvalidConfig("destination mixed optimizer parameter size overflows"))?;
            let size=count.div_ceil(target.world_size as usize);parameter_geometry(&target.logical_shape,target.rank,target.world_size,size)?;
            let first_index=*migration.source_records.first().ok_or(MuonError::InvalidConfig("explicit original mixed optimizer group required"))?;
            let (saved_manifest,saved_placement)=indices.get(first_index).ok_or(MuonError::IncompatibleRecord)?;
            let original=*saved_placement.get(&id.val()).ok_or(MuonError::IncompatibleRecord)?;
            let metadata=*saved_manifest.get(&id.val()).ok_or(MuonError::IncompatibleRecord)?;
            if original.1!=target.logical_shape || original.3==0 || original.3 as usize!=migration.source_records.len()
                || (original.4 && (!metadata.2 || original.1.len()!=2)) {return Err(MuonError::IncompatibleRecord);}
            parameter_geometry(&original.1,original.2,original.3,metadata.1)?;
            let present=sources[first_index].states.contains_key(&id);let mut ranks=HashSet::new();let mut histories=Vec::new();
            for index in &migration.source_records {
                let source=sources.get(*index).ok_or(MuonError::IncompatibleRecord)?;
                let (source_manifest,source_placement)=&indices[*index];
                let spec=*source_placement.get(&id.val()).ok_or(MuonError::IncompatibleRecord)?;
                let entry=*source_manifest.get(&id.val()).ok_or(MuonError::IncompatibleRecord)?;
                if spec.1!=original.1 || spec.3!=original.3 || spec.4!=original.4 || spec.2>=spec.3 || !ranks.insert(spec.2)
                    || entry.1!=metadata.1 || entry.2!=metadata.2 || entry.3!=metadata.3
                    || source.states.contains_key(&id)!=present {return Err(MuonError::IncompatibleRecord);}
                parameter_geometry(&spec.1,spec.2,spec.3,entry.1)?;
                let Some(state)=source.states.get(&id) else {continue;};
                if !entry.2 {return Err(MuonError::IncompatibleRecord);}
                let master=state.master_muon.is_some() || state.master_adamw.is_some();
                if (!master && format!("{:?}",state.storage)!=entry.3)
                    || (master && !matches!(entry.3.as_str(),"F32"|"F16"|"BF16")) {return Err(MuonError::IncompatibleRecord);}
                histories.push((spec.2,state.clone()));
            }
            if present {
                histories.sort_by_key(|entry|entry.0);
                let histories=histories.into_iter().map(|entry|entry.1).collect::<Vec<_>>();
                states.insert(id,FullyShardedMuonAdamWState::repartition_from_ranks(&histories,&original.1,original.4,target.rank,target.world_size)?);
            }
            manifest.push((id.val(),size,metadata.2,metadata.3.clone()));
            placement.push((id.val(),target.logical_shape.clone(),target.rank,target.world_size,original.4));
        }
        manifest.sort_by_key(|entry|entry.0);placement.sort_by_key(|entry|entry.0);
        Ok(Self {version:1,config_key:first.config_key.clone(),manifest,placement,states})
    }
}
