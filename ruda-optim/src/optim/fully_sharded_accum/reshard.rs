use super::*;

impl FullyShardedGradientsRecord {
    /// Offline ownership migration of one actual pending SUM-gradient window from a complete original rank set.
    /// Values are overlap-copied, not summed again: saved FSDP buffers are already globally reduce-scattered.
    /// Original IDs, absent gradients, fixed scale and exact GLOBAL count/window counters remain unchanged.
    /// All saved parameters must share the same original data-rank ordering; unrelated TP groups are not inferred.
    pub fn reshard<B:Backend>(records:&[Self],target_rank:u32,target_world:u32,device:&B::Device) -> Result<Self,RecorderError> {
        let invalid=|message:&str|RecorderError::Unknown(message.to_string());
        let first=records.first().ok_or_else(||invalid("complete original pending-gradient rank set required"))?;
        if first.version!=1 || target_world==0 || target_rank>=target_world {return Err(invalid("invalid pending-gradient reshard version/topology"));}
        work_dtype(&first.state).map_err(|error|invalid(&error.to_string()))?;
        let source_world=u32::try_from(records.len()).map_err(|_|invalid("original rank count exceeds topology range"))?;
        let mut ordered=Vec::with_capacity(records.len());let mut ranks=BTreeSet::new();
        for record in records {
            if record.version!=1 || record.state!=first.state || record.placement.len()!=first.placement.len() {
                return Err(invalid("pending-gradient rank windows/options differ"));
            }
            let rank=record.placement.first().map(|entry|entry.2).ok_or_else(||invalid("actual parameter placements required for ownership migration"))?;
            if rank>=source_world || !ranks.insert(rank) {return Err(invalid("duplicate/missing original pending-gradient rank"));}
            for (spec,reference) in record.placement.iter().zip(&first.placement) {
                if spec.0!=reference.0 || spec.1!=reference.1 || spec.3!=source_world || reference.3!=source_world || spec.2!=rank
                    || spec.4!=reference.4 || spec.5!=reference.5 {return Err(invalid("original logical parameter ownership/storage/trainability differs"));}
                elements(&spec.1).map_err(|error|invalid(&error.to_string()))?;
            }
            ordered.push((rank,record));
        }
        ordered.sort_by_key(|entry|entry.0);
        let mut sources=Vec::with_capacity(records.len());let mut membership=None;
        for (_,record) in &ordered {
            let values=GradientsParams::from_record::<B>(record.gradients.clone(),device)?;
            let ids=values.container.ids().into_iter().map(|id|id.val()).collect::<BTreeSet<_>>();
            if membership.as_ref().is_some_and(|previous|previous!=&ids) {return Err(invalid("globally unused/present pending gradients differ across ranks"));}
            membership=Some(ids);sources.push(values);
        }
        let ids=membership.unwrap_or_default();let declared=first.placement.iter().map(|entry|entry.0).collect::<BTreeSet<_>>();
        if !ids.is_subset(&declared) || (first.state.microbatches==0 && (!ids.is_empty() || first.state.global_count!=0)) {
            return Err(invalid("pending gradients do not match the actual saved window"));
        }
        let mut target=GradientsParams::new();let mut placement=Vec::with_capacity(first.placement.len());
        for reference in &first.placement {
            let total=elements(&reference.1).map_err(|error|invalid(&error.to_string()))?;
            let old_slots=total.div_ceil(source_world as usize);let new_slots=total.div_ceil(target_world as usize);
            if old_slots.checked_mul(source_world as usize).is_none() || new_slots.checked_mul(target_world as usize).is_none() {
                return Err(invalid("pending gradient padded topology overflows"));
            }
            placement.push((reference.0,reference.1.clone(),target_rank,target_world,reference.4,reference.5));
            if !ids.contains(&reference.0) {continue;}
            if !reference.5 {return Err(invalid("frozen parameter has saved pending derivatives"));}
            let start=target_rank as usize*new_slots;let end=start.saturating_add(new_slots).min(total);
            let mut value=Tensor::<B,1>::zeros([new_slots],(device,first.state.dtype));
            for (rank,source) in sources.iter().enumerate() {
                let original=source.get::<B,1>(ParamId::from(reference.0)).ok_or_else(||invalid("missing actual pending gradient"))?;
                if original.dims()!=[old_slots] || original.dtype()!=first.state.dtype {
                    return Err(invalid("actual pending gradient local vector/precision differs"));
                }
                let old_start=rank*old_slots;let old_end=old_start.saturating_add(old_slots).min(total);
                let overlap_start=start.max(old_start);let overlap_end=end.min(old_end);
                if overlap_start<overlap_end {
                    value=value.slice_assign([overlap_start-start..overlap_end-start],original.slice([overlap_start-old_start..overlap_end-old_start]));
                }
            }
            target.register(ParamId::from(reference.0),value);
        }
        Ok(Self {version:1,gradients:target.try_to_record::<B>()?,state:first.state.clone(),placement})
    }
}
