use super::*;
use crate::{FlatShardElementwiseOptimizer,FlatOptimizerTensorShard,OptimizerCheckpointScalars,OptimizerShardError};
use crate::record::AdaptorRecord;

impl<O,M,B,C> FullyShardedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:FlatShardElementwiseOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
        O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    /// Import actual original full native optimizer histories into the prepared model's exact flat owners.
    /// Source clocks/moments/AMSGrad/centered/RMSProp/FP32-master values are retained, never reseeded.
    /// Absent histories remain absent; existing histories for newly frozen roles are retained without updating.
    /// Replace this local history container only after every source record is validated and partitioned.
    pub fn try_import_native_record(&mut self,record:HashMap<ParamId,AdaptorRecord<O,B>>) -> Result<(),OptimizerShardError> {
        self.try_import_native_records(record)
    }
    /// Streaming equivalent: each actual full source history can be released immediately after local slicing.
    /// The iterator supplies the complete original history set; omitted histories stay absent.
    pub fn try_import_native_records<I:IntoIterator<Item=(ParamId,AdaptorRecord<O,B>)>>(&mut self,records:I) -> Result<(),OptimizerShardError> {
        let mut states=HashMap::new();
        for (id,record) in records {
            if states.contains_key(&id) {return Err(OptimizerShardError::Placement("duplicate original native history identity"));}
            let spec=self.placement.iter().find(|entry|entry.0==id.val()).ok_or(OptimizerShardError::Placement("native history belongs to an unknown original parameter"))?;
            let shard=FlatOptimizerTensorShard::new(spec.1.clone(),spec.2,spec.3)?;
            let state=match record {AdaptorRecord::V1(record)=>O::partition_native_history(record,&shard)?};
            let (_,slots)=shard.geometry()?;
            validate_state::<B::InnerBackend,O>(&state,[slots],self.optimizer.shard_gradient_dtype(spec.4)).map_err(OptimizerShardError::DType)?;
            self.optimizer.validate_fully_sharded_history(&state).map_err(OptimizerShardError::Placement)?;
            states.insert(id,state);
        }
        self.states=states;Ok(())
    }
}

impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> FullyShardedElementwiseRecord<B,O>
    where O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    /// Actual local history count, excluding never-used parameters whose original state is absent.
    pub fn state_parameter_count(&self) -> usize {self.states.len()}
    /// Inspect actual original local clocks/buffers without copying or reconstructing global history.
    pub fn state(&self,id:ParamId) -> Option<&O::State<1>> {self.states.get(&id)}
}
impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> FullyShardedElementwiseRecord<B,O>
    where O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1>+OptimizerCheckpointScalars {
    /// Offline exact ownership migration from a complete original rank set to one new data owner.
    /// Coordinate histories are overlap-copied, not summed or recomputed; original clocks/options must match.
    /// The complete saved group must use one original rank ordering for every actual parameter.
    pub fn reshard(records:&[Self],target_rank:u32,target_world:u32,device:&B::Device) -> Result<Self,OptimizerShardError> {
        let first=records.first().ok_or(OptimizerShardError::Placement("complete original optimizer rank set required"))?;
        if first.version!=1 {return Err(OptimizerShardError::Placement("unsupported native flat optimizer record"));}
        let source_world=u32::try_from(records.len()).map_err(|_|OptimizerShardError::Placement("original rank count exceeds topology range"))?;
        if target_world==0 || target_rank>=target_world {return Err(OptimizerShardError::Placement("invalid requested optimizer owner"));}
        let canonical=first.placement.iter().map(|entry|entry.0).collect::<BTreeSet<_>>();
        if canonical.len()!=first.placement.len() || first.placement.is_empty() {return Err(OptimizerShardError::Placement("unique actual parameter ownership required"));}
        let membership=first.states.keys().copied().collect::<BTreeSet<_>>();
        if membership.iter().any(|id|!canonical.contains(&id.val())) {return Err(OptimizerShardError::Placement("unknown saved original optimizer history"));}
        let mut ranks=BTreeSet::new();let mut ordered=Vec::with_capacity(records.len());
        for record in records {
            if record.version!=1 || record.placement.len()!=first.placement.len() || record.states.keys().copied().collect::<BTreeSet<_>>()!=membership {
                return Err(OptimizerShardError::Placement("native optimizer history presence/version/membership differs across ranks"));
            }
            let rank=record.placement[0].2;
            if rank>=source_world || !ranks.insert(rank) {return Err(OptimizerShardError::Placement("duplicate/missing original optimizer owner"));}
            for (spec,reference) in record.placement.iter().zip(&first.placement) {
                if spec.0!=reference.0 || spec.1!=reference.1 || spec.2!=rank || spec.3!=source_world || reference.3!=source_world
                    || spec.4!=reference.4 || spec.5!=reference.5 {
                    return Err(OptimizerShardError::Placement("original logical parameter axes/precision/trainability/topology differ"));
                }
                FlatOptimizerTensorShard::new(spec.1.clone(),rank,source_world)?;
            }
            ordered.push((rank,record));
        }
        ordered.sort_by_key(|entry|entry.0);let mut placement=Vec::with_capacity(first.placement.len());let mut states=HashMap::with_capacity(membership.len());
        for spec in &first.placement {
            let shard=FlatOptimizerTensorShard::new(spec.1.clone(),target_rank,target_world)?;
            let (count,new_slots)=shard.geometry()?;let old_slots=count.div_ceil(source_world as usize);
            placement.push((spec.0,spec.1.clone(),target_rank,target_world,spec.4,spec.5));
            let id=ParamId::from(spec.0);let Some(reference)=first.states.get(&id) else {continue;};
            let scalar=reference.checkpoint_scalars();let mut template=Vec::new();reference.visit_checkpoint_buffers(&mut |value|template.push(value.clone()));
            let mut sources=Vec::with_capacity(records.len());
            for (_,record) in &ordered {
                let source=record.states.get(&id).ok_or(OptimizerShardError::Placement("missing original native optimizer history"))?;
                if source.checkpoint_scalars()!=scalar {return Err(OptimizerShardError::Placement("original optimizer clocks/optional histories differ"));}
                let mut buffers=Vec::new();source.visit_checkpoint_buffers(&mut |value|buffers.push(value.clone()));
                if buffers.len()!=template.len() {return Err(OptimizerShardError::Placement("actual native history buffer structure differs"));}
                for (value,original) in buffers.iter().zip(&template) {
                    if value.dims()!=[old_slots] || value.dtype()!=original.dtype() || !value.dtype().is_float() || matches!(value.dtype(),DType::QFloat(_)) {
                        return Err(OptimizerShardError::Shape("actual saved optimizer local buffer/precision differs"));
                    }
                }
                sources.push(buffers);
            }
            let start=target_rank as usize*new_slots;let end=start.saturating_add(new_slots).min(count);let mut buffers=Vec::with_capacity(template.len());
            for (index,original) in template.iter().enumerate() {
                let mut value=Tensor::<B::InnerBackend,1>::zeros([new_slots],(device,original.dtype()));
                for (rank,source) in sources.iter().enumerate() {
                    let old_start=rank*old_slots;let old_end=old_start.saturating_add(old_slots).min(count);
                    let left=start.max(old_start);let right=end.min(old_end);
                    if left<right {value=value.slice_assign([left-start..right-start],source[index].clone().slice([left-old_start..right-old_start]).to_device(device));}
                }
                buffers.push(value);
            }
            let mut buffers=buffers.into_iter();
            let state=reference.clone().map_checkpoint_buffers(&mut |_|buffers.next().expect("validated original native history buffer count"));
            states.insert(id,state);
        }
        Ok(Self {version:1,placement,states})
    }
}
