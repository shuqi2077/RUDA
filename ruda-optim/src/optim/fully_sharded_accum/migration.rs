use super::*;

/// One actual logical parameter's pending-gradient ownership migration.
/// `source_records` selects exactly one complete ORIGINAL data-shard group from
/// the supplied archive list. Extra TP/expert-owned partitions and replicated
/// copies are never guessed, merged or counted as additional source shards.
#[derive(Clone,Debug)]
pub struct FullyShardedGradientMigration {
    pub target:FullyShardedParameterPlacement,
    pub source_records:Vec<usize>,
}
impl FullyShardedGradientMigration {
    pub fn new(target:FullyShardedParameterPlacement,source_records:Vec<usize>) -> Self {Self {target,source_records}}
}

impl FullyShardedGradientsRecord {
    /// Offline overlap-copy into explicit heterogeneous target ownership.
    /// Each target parameter chooses its original complete group independently,
    /// allowing different source/target data worlds and locally owned expert/TP
    /// partitions. All supplied archives must come from the same global window.
    /// Original IDs, native saved values, storage/trainable metadata, loss scale,
    /// counts and issued microbatches survive unchanged; absent gradients remain
    /// absent and new rank padding is zero. No gradient/denominator SUM, optimizer
    /// migration, graph replay or session reconstruction is performed here.
    pub fn reshard_explicit<B:Backend>(records:&[Self],migrations:&[FullyShardedGradientMigration],device:&B::Device)
        -> Result<Self,RecorderError> {
        let sources=records.iter().collect::<Vec<_>>();let migrated=migrate::<B>(&sources,migrations,device)?;
        let gradients=migrated.gradients.try_to_record::<B>()?;Ok(migrated.into_record(gradients))
    }

    /// Same original overlap-copy using actual asynchronous native readback.
    /// No blocking-read fallback is required by backends whose execution permits
    /// asynchronous device reads only. Input archives remain caller-owned.
    pub async fn reshard_explicit_async<B:Backend>(records:&[Self],migrations:&[FullyShardedGradientMigration],device:&B::Device)
        -> Result<Self,RecorderError> {
        let sources=records.iter().collect::<Vec<_>>();let migrated=migrate::<B>(&sources,migrations,device)?;
        let gradients=migrated.gradients.to_record_async::<B>().await?;Ok(migrated.into_record(gradients))
    }
}

impl<B:Backend> FullyShardedWeightedGradientsRecord<B> {
    /// Same explicit original shard selection with the already GLOBAL fractional
    /// denominator copied once. Native weighted windows need matching saved
    /// denominators, not an average or a SUM of copies from the source archives.
    pub fn reshard_explicit(records:&[Self],migrations:&[FullyShardedGradientMigration],device:&B::Device)
        -> Result<Self,RecorderError> {
        let first=Self::validate_migration_weights(records)?;
        let windows=records.iter().map(Self::window_record).collect::<Vec<_>>();
        let migrated=migrate::<B>(&windows,migrations,device)?;
        let gradients=migrated.gradients.try_to_record::<B>()?;let window=migrated.into_record(gradients);
        Ok(Self::with_migrated_window(first,window,device))
    }

    /// Async original weighted-window migration, including native denominator
    /// comparison and pending-gradient readback. No scalar is read synchronously.
    pub async fn reshard_explicit_async(records:&[Self],migrations:&[FullyShardedGradientMigration],device:&B::Device)
        -> Result<Self,RecorderError> {
        let first=Self::validate_migration_weights_async(records).await?;
        let windows=records.iter().map(Self::window_record).collect::<Vec<_>>();
        let migrated=migrate::<B>(&windows,migrations,device)?;
        let gradients=migrated.gradients.to_record_async::<B>().await?;let window=migrated.into_record(gradients);
        Ok(Self::with_migrated_window(first,window,device))
    }
}

struct NativeMigration {
    gradients:GradientsParams,
    state:FullyShardedAccumulationState,
    placement:Placement,
}
impl NativeMigration {
    fn into_record(self,gradients:GradientsParamsRecord) -> FullyShardedGradientsRecord {
        FullyShardedGradientsRecord {version:1,gradients,state:self.state,placement:self.placement}
    }
}

fn migrate<B:Backend>(records:&[&FullyShardedGradientsRecord],migrations:&[FullyShardedGradientMigration],device:&B::Device)
    -> Result<NativeMigration,RecorderError> {
    let invalid=|message:&str|RecorderError::Unknown(message.to_string());
    let first=records.first().ok_or_else(||invalid("actual original pending-gradient archives required"))?;
    work_dtype(&first.state).map_err(|error|invalid(&error.to_string()))?;
    if first.state.microbatches==0 && first.state.global_count!=0 {return Err(invalid("invalid empty global pending-gradient window"));}
    let mut indices=Vec::with_capacity(records.len());
    for record in records {
        if record.version!=1 || record.state!=first.state {return Err(invalid("pending-gradient global windows/options differ"));}
        let index: BTreeMap<_,_>=record.placement.iter().map(|entry|(entry.0,entry)).collect();
        if index.len()!=record.placement.len() {return Err(invalid("duplicate original pending-gradient parameter identity"));}
        let ids=record.gradients.parameter_ids()?;
        if ids.iter().any(|id|!index.contains_key(&id.val())) || (record.state.microbatches==0 && !ids.is_empty()) {
            return Err(invalid("archived derivatives do not match the original pending-gradient window"));
        }
        indices.push(index);
    }
    let mut target=GradientsParams::new();let mut placement=Vec::with_capacity(migrations.len());let mut target_ids=BTreeSet::new();
    for migration in migrations {
        let binding=&migration.target;let id=binding.parameter.val();
        if binding.world_size==0 || binding.rank>=binding.world_size || !target_ids.insert(id) {
            return Err(invalid("invalid or duplicate explicit target pending-gradient ownership"));
        }
        let total=elements(&binding.logical_shape).map_err(|error|invalid(&error.to_string()))?;
        let first_index=*migration.source_records.first().ok_or_else(||invalid("explicit complete original source group required"))?;
        let reference=indices.get(first_index).and_then(|index|index.get(&id)).copied()
            .ok_or_else(||invalid("selected archive does not contain the original parameter placement"))?;
        let source_world=reference.3;
        if source_world==0 || source_world as usize!=migration.source_records.len() || reference.1!=binding.logical_shape
            || !matches!(reference.4,DType::F32|DType::F16|DType::BF16) {
            return Err(invalid("original source group size/logical axes/storage differ"));
        }
        let old_slots=total.div_ceil(source_world as usize);let new_slots=total.div_ceil(binding.world_size as usize);
        if old_slots.checked_mul(source_world as usize).is_none() || new_slots.checked_mul(binding.world_size as usize).is_none() {
            return Err(invalid("pending-gradient padded topology overflows"));
        }
        let start=binding.rank as usize*new_slots;let end=start.saturating_add(new_slots).min(total);
        let mut value:Option<Tensor<B,1>>=None;
        let mut ranks=BTreeSet::new();let mut present=None;
        for source_index in &migration.source_records {
            let source=records.get(*source_index).ok_or_else(||invalid("original source archive index out of range"))?;
            let spec=indices[*source_index].get(&id).copied().ok_or_else(||invalid("missing explicitly selected original parameter placement"))?;
            if spec.1!=reference.1 || spec.3!=source_world || spec.2>=source_world || spec.4!=reference.4 || spec.5!=reference.5
                || !ranks.insert(spec.2) {return Err(invalid("original logical ownership/storage/trainability differs or source rank repeats"));}
            let archive=source.gradients.select_parameters(&[binding.parameter])?;
            let metadata=archive.parameter_metadata(binding.parameter)?;
            if present.is_some_and(|previous|previous!=metadata.is_some()) {return Err(invalid("original gradient presence differs across source data shards"));}
            present=Some(metadata.is_some());
            let Some((dtype,shape))=metadata else {continue;};
            if !reference.5 || shape!=[old_slots] || dtype!=first.state.dtype {
                return Err(invalid("actual original pending-gradient storage/precision/trainability differs"));
            }
            let mut selected=GradientsParams::from_record::<B>(archive,device)?;
            let original=selected.remove::<B,1>(binding.parameter).ok_or_else(||invalid("missing restored original pending derivative"))?;
            if value.is_none() {value=Some(Tensor::zeros([new_slots],(device,first.state.dtype)));}
            let old_start=spec.2 as usize*old_slots;let old_end=old_start.saturating_add(old_slots).min(total);
            let overlap_start=start.max(old_start);let overlap_end=end.min(old_end);
            if overlap_start<overlap_end {
                let target=value.take().ok_or_else(||invalid("missing target pending-gradient storage"))?;
                value=Some(target.slice_assign([overlap_start-start..overlap_end-start],original.slice([overlap_start-old_start..overlap_end-old_start])));
            }
        }
        placement.push((id,binding.logical_shape.clone(),binding.rank,binding.world_size,reference.4,reference.5));
        if let Some(value)=value {target.register(binding.parameter,value);}
    }
    placement.sort_by_key(|entry|entry.0);
    Ok(NativeMigration {gradients:target,state:first.state.clone(),placement})
}
