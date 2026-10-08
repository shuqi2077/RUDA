use super::*;

/// Actual canonical flat leaf and original logical data-shard ownership.
/// This contains no numerical values, Muon role, transport or normalization
/// policy. A mixed DP/TP/EP model can combine placements obtained from different
/// communicator implementations without replacing its native parameter leaves.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct FullyShardedParameterPlacement {
    pub parameter:ParamId,
    pub logical_shape:Vec<usize>,
    pub rank:u32,
    pub world_size:u32,
}

impl FullyShardedParameterPlacement {
    /// Explicit original partition metadata; module binding validates real storage.
    pub fn new(parameter:ParamId,logical_shape:Vec<usize>,rank:u32,world_size:u32) -> Self {
        Self {parameter,logical_shape,rank,world_size}
    }

    /// Extract the actual original identity/shape/rank from an optimizer binding.
    /// No collective or optimizer initialization is performed.
    pub fn from_shard<B,C>(binding:&FullyShardedOptimizerParameter<C>) -> Self
    where B:AutodiffBackend,C:BroadcastTensorCollective<B::InnerBackend> {
        Self::new(binding.parameter,binding.logical_shape.clone(),binding.communicator.rank(),binding.communicator.world_size())
    }
}

impl<B:Backend> Record<B> for FullyShardedParameterPlacement {
    type Item<P:PrecisionSettings>=(u64,Vec<usize>,u32,u32);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.parameter.val(),self.logical_shape,self.rank,self.world_size)
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,_device:&B::Device) -> Self {
        Self::new(ParamId::from(item.0),item.1,item.2,item.3)
    }
}

fn metadata(placement:&Placement) -> Vec<FullyShardedParameterPlacement> {
    placement.iter().map(|entry|FullyShardedParameterPlacement::new(ParamId::from(entry.0),entry.1.clone(),entry.2,entry.3)).collect()
}

impl FullyShardedAccumulationContract {
    /// Original validated local ownership, independent of the saved native dtype
    /// and requires-grad flags. Restore still checks those original flags/storage.
    /// This snapshot does not read or duplicate model/gradient values.
    pub fn placements(&self) -> Vec<FullyShardedParameterPlacement> {metadata(&self.placement)}
}

impl FullyShardedGradientsRecord {
    /// Saved original placements for reconstructing caller-owned native groups.
    /// This does not repartition pending gradients or guess new peer membership.
    pub fn placements(&self) -> Vec<FullyShardedParameterPlacement> {metadata(&self.placement)}
}

impl<M> FullyShardedGradientsAccumulator<M> {
    /// Bind the actual complete local module using heterogeneous original groups'
    /// explicit metadata. No communicator type or common data-group size is
    /// required: scope-completed gather backward already performs the actual SUM/
    /// reduce-scatter in each original group. Frozen leaves must also be declared.
    /// Logical TP partitions and owned experts remain the caller's actual leaves;
    /// this does not reinterpret a partial tensor as an unpartitioned parameter.
    pub fn from_placements<B:AutodiffBackend>(module:&M,parameters:&[FullyShardedParameterPlacement],
        dtype:FloatDType,loss_scale:f64) -> Result<Self,FullyShardedAccumulationError>
    where M:AutodiffModule<B> {
        validate_work_dtype(dtype)?;
        let state=FullyShardedAccumulationState {dtype:dtype.into(),loss_scale,global_count:0,microbatches:0};
        work_dtype(&state)?;
        let mut placement=Vec::with_capacity(parameters.len());let mut ids=BTreeSet::new();
        for parameter in parameters {
            if !ids.insert(parameter.parameter) {
                return Err(FullyShardedAccumulationError::Placement("duplicate canonical parameter binding"));
            }
            placement.push((parameter.parameter.val(),parameter.logical_shape.clone(),parameter.rank,parameter.world_size,DType::F32,false));
        }
        Self::bind_placement::<B>(module,placement,state)
    }

    /// Actual original placement snapshot, retaining no transport or tensor data.
    pub fn placements(&self) -> Vec<FullyShardedParameterPlacement> {metadata(&self.placement)}
}

impl<M:AutodiffModule<B>,B:AutodiffBackend> FullyShardedWeightedGradientsAccumulator<M,B> {
    /// Fractional GLOBAL-weight accumulation with the same heterogeneous native
    /// ownership. Uses the original integer-window accumulator and weight scalar,
    /// not a second gradient SUM or an inferred per-replica denominator.
    pub fn from_placements(module:&M,parameters:&[FullyShardedParameterPlacement],dtype:FloatDType,
        loss_scale:f64,device:&B::Device) -> Result<Self,FullyShardedAccumulationError> {
        let window=FullyShardedGradientsAccumulator::from_placements::<B>(module,parameters,dtype,loss_scale)?;
        Ok(Self::from_window(window,device))
    }
}

impl HybridShardedGradientNormParameter {
    /// Attach only an explicitly known copy count to actual original ownership.
    /// The count is not derived from local shard length, data world or TP width.
    pub fn from_placement(placement:&FullyShardedParameterPlacement,replicas:u32) -> Self {
        Self::new(placement.parameter,placement.logical_shape.clone(),placement.rank,placement.world_size,replicas)
    }
}
