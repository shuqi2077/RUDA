use super::*;
use crate::{FullyShardedParameterPlacement,FlatOptimizerTensorShard,OptimizerShardError,OptimizerCheckpointScalars};

/// Explicit target ownership and one complete ORIGINAL source history group.
/// Indices address the supplied optimizer archive list, not inferred global ranks.
/// A separate group may be declared for every actual TP/expert-owned parameter.
#[derive(Clone,Debug)]
pub struct FullyShardedOptimizerMigration {
    pub target:FullyShardedParameterPlacement,
    pub source_records:Vec<usize>,
}
impl FullyShardedOptimizerMigration {
    pub fn new(target:FullyShardedParameterPlacement,source_records:Vec<usize>) -> Self {Self {target,source_records}}
}

impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> FullyShardedElementwiseRecord<B,O>
where O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1>+OptimizerCheckpointScalars {
    /// Exact offline coordinate-history migration with per-parameter source and
    /// target data ownership. No common world size or matching local model lists
    /// are imposed on unrelated TP/expert groups. The original scalar clocks,
    /// optional buffers, branch tags, actual buffer dtype and absent/frozen
    /// histories survive unchanged; target rank padding is zero.
    /// Does not update a model, replay an optimizer step, sum histories, infer
    /// peer membership or migrate non-coordinate matrix-optimizer state.
    pub fn reshard_explicit(records:&[Self],migrations:&[FullyShardedOptimizerMigration],device:&B::Device)
        -> Result<Self,OptimizerShardError> {
        let records=records.iter().collect::<Vec<_>>();migrate_optimizer_record::<B,O>(&records,migrations,device)
    }
}

pub(super) fn migrate_optimizer_record<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>>(
    records:&[&FullyShardedElementwiseRecord<B,O>],migrations:&[FullyShardedOptimizerMigration],device:&B::Device)
    -> Result<FullyShardedElementwiseRecord<B,O>,OptimizerShardError>
where O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1>+OptimizerCheckpointScalars {
    if records.is_empty() {return Err(OptimizerShardError::Placement("actual original optimizer archives required"));}
    let mut indices=Vec::with_capacity(records.len());
    for record in records {
        if record.version!=1 {return Err(OptimizerShardError::Placement("unsupported native flat optimizer record"));}
        let index=record.placement.iter().map(|entry|(entry.0,entry)).collect::<BTreeMap<_,_>>();
        if index.len()!=record.placement.len() || record.states.keys().any(|id|!index.contains_key(&id.val())) {
            return Err(OptimizerShardError::Placement("duplicate parameter ownership or unknown saved native history"));
        }
        indices.push(index);
    }
    let mut placement=Vec::with_capacity(migrations.len());let mut states=HashMap::with_capacity(migrations.len());let mut ids=BTreeSet::new();
    for migration in migrations {
        let target=&migration.target;let id=target.parameter;
        if !ids.insert(id) {return Err(OptimizerShardError::Placement("duplicate explicit target optimizer ownership"));}
        let target_shard=FlatOptimizerTensorShard::new(target.logical_shape.clone(),target.rank,target.world_size)?;
        let (count,new_slots)=target_shard.geometry()?;
        let first_index=*migration.source_records.first().ok_or(OptimizerShardError::Placement("explicit complete original history group required"))?;
        let spec=indices.get(first_index).and_then(|index|index.get(&id.val())).copied()
            .ok_or(OptimizerShardError::Placement("selected source archive lacks the original parameter ownership"))?;
        if spec.1!=target.logical_shape || spec.3==0 || spec.3 as usize!=migration.source_records.len()
            || !matches!(spec.4,DType::F32|DType::F16|DType::BF16) {
            return Err(OptimizerShardError::Placement("original source history group/logical axes/storage differ"));
        }
        let (_,old_slots)=FlatOptimizerTensorShard::new(spec.1.clone(),spec.2,spec.3)?.geometry()?;
        let reference=records[first_index].states.get(&id);let mut template=Vec::new();
        if let Some(reference)=reference {reference.visit_checkpoint_buffers(&mut |value|template.push(value.clone()));}
        let scalar=reference.map(OptimizerCheckpointScalars::checkpoint_scalars);
        let mut ranks=BTreeSet::new();let mut sources=Vec::with_capacity(migration.source_records.len());
        for index in &migration.source_records {
            let record=records.get(*index).ok_or(OptimizerShardError::Placement("original source history archive index out of range"))?;
            let original=indices[*index].get(&id.val()).copied().ok_or(OptimizerShardError::Placement("missing selected original parameter ownership"))?;
            if original.1!=spec.1 || original.3!=spec.3 || original.2>=spec.3 || original.4!=spec.4 || original.5!=spec.5
                || !ranks.insert(original.2) {return Err(OptimizerShardError::Placement("original logical ownership/storage/trainability differs or source rank repeats"));}
            let state=record.states.get(&id);
            if state.is_some()!=reference.is_some() {return Err(OptimizerShardError::Placement("native history presence differs across original data shards"));}
            let Some(state)=state else {continue;};
            if Some(state.checkpoint_scalars())!=scalar {return Err(OptimizerShardError::Placement("original optimizer clocks/branches/optional histories differ"));}
            let mut buffers=Vec::new();state.visit_checkpoint_buffers(&mut |value|buffers.push(value.clone()));
            if buffers.len()!=template.len() {return Err(OptimizerShardError::Placement("actual native history buffer structure differs"));}
            for (value,reference) in buffers.iter().zip(&template) {
                if value.dims()!=[old_slots] || value.dtype()!=reference.dtype() || !value.dtype().is_float() || matches!(value.dtype(),DType::QFloat(_)) {
                    return Err(OptimizerShardError::Shape("actual saved optimizer local buffer/precision differs"));
                }
            }
            sources.push((original.2,buffers));
        }
        placement.push((id.val(),target.logical_shape.clone(),target.rank,target.world_size,spec.4,spec.5));
        let Some(reference)=reference else {continue;};
        let start=target.rank as usize*new_slots;let end=start.saturating_add(new_slots).min(count);
        let mut buffers=Vec::with_capacity(template.len());
        for (index,reference) in template.iter().enumerate() {
            let mut value=Tensor::<B::InnerBackend,1>::zeros([new_slots],(device,reference.dtype()));
            for (rank,source) in &sources {
                let old_start=*rank as usize*old_slots;let old_end=old_start.saturating_add(old_slots).min(count);
                let left=start.max(old_start);let right=end.min(old_end);
                if left<right {value=value.slice_assign([left-start..right-start],source[index].clone().slice([left-old_start..right-old_start]).to_device(device));}
            }
            buffers.push(value);
        }
        let mut buffers=buffers.into_iter();
        let state=reference.clone().map_checkpoint_buffers(&mut |_|buffers.next().expect("validated original native history buffer count"));
        states.insert(id,state);
    }
    placement.sort_by_key(|entry|entry.0);
    Ok(FullyShardedElementwiseRecord {version:1,placement,states})
}
