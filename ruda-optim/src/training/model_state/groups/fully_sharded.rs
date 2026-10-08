use super::*;
use crate::{FullyShardedAccumulationContract,FullyShardedWeightedAccumulationContract,FullyShardedGradientsAccumulator,
    FullyShardedWeightedGradientsAccumulator,training::{RestoredFullyShardedTraining,RestoredFullyShardedWeightedTraining}};

impl<B,M,S,R,G,U> ModelGroupTrainingRecord<B,M,S,R,G,U>
where B:AutodiffBackend,M:AutodiffModule<B>,S:LrScheduler,R:Record<B>,G:Record<B>,U:Record<B> {
    /// Native model/adapter state, heterogeneous original optimizer group records,
    /// scheduler and real FSDP pending gradients share one quiescent boundary.
    /// The logical ownership/global count/scale contract adds no duplicate gradient
    /// payload. Data/sampler/RNG and original group memberships remain caller-owned.
    pub fn capture_fully_sharded(model:&M,model_state:R,groups:G,scheduler:&S,
        accumulator:&FullyShardedGradientsAccumulator<M>,state:U)
        -> Result<ModelGroupTrainingRecord<B,M,S,R,G,(FullyShardedAccumulationContract,U)>,RecorderError> {
        let continuation=accumulator.continuation();
        continuation.validate_for::<B,M>(model).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        ModelGroupTrainingRecord::<B,M,S,R,G,(FullyShardedAccumulationContract,U)>::capture(
            model,model_state,groups,scheduler,accumulator.inner(),(continuation,state))
    }

    /// Actual asynchronous pending-gradient capture with the same original
    /// frozen-base/adapter archive and group records, without reset or replay.
    pub async fn capture_fully_sharded_async(model:&M,model_state:R,groups:G,scheduler:&S,
        accumulator:&FullyShardedGradientsAccumulator<M>,state:U)
        -> Result<ModelGroupTrainingRecord<B,M,S,R,G,(FullyShardedAccumulationContract,U)>,RecorderError> {
        let continuation=accumulator.continuation();
        continuation.validate_for::<B,M>(model).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        ModelGroupTrainingRecord::<B,M,S,R,G,(FullyShardedAccumulationContract,U)>::capture_async(
            model,model_state,groups,scheduler,accumulator.inner(),(continuation,state)).await
    }

    /// Preserve actual fractional GLOBAL normalization alongside integer selected
    /// counts, microbatch count, work precision, loss scale and shard placement.
    /// No replica weight or normalization group is inferred from the group tuple.
    pub fn capture_fully_sharded_weighted(model:&M,model_state:R,groups:G,scheduler:&S,
        accumulator:&FullyShardedWeightedGradientsAccumulator<M,B>,state:U)
        -> Result<ModelGroupTrainingRecord<B,M,S,R,G,(FullyShardedWeightedAccumulationContract<B>,U)>,RecorderError> {
        let continuation=accumulator.continuation();
        continuation.validate_for(model).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        ModelGroupTrainingRecord::<B,M,S,R,G,(FullyShardedWeightedAccumulationContract<B>,U)>::capture(
            model,model_state,groups,scheduler,accumulator.inner(),(continuation,state))
    }
    pub async fn capture_fully_sharded_weighted_async(model:&M,model_state:R,groups:G,scheduler:&S,
        accumulator:&FullyShardedWeightedGradientsAccumulator<M,B>,state:U)
        -> Result<ModelGroupTrainingRecord<B,M,S,R,G,(FullyShardedWeightedAccumulationContract<B>,U)>,RecorderError> {
        let continuation=accumulator.continuation();
        continuation.validate_for(model).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        ModelGroupTrainingRecord::<B,M,S,R,G,(FullyShardedWeightedAccumulationContract<B>,U)>::capture_async(
            model,model_state,groups,scheduler,accumulator.inner(),(continuation,state)).await
    }
}

impl<B,M,S,R,G,U> ModelGroupTrainingRecord<B,M,S,R,G,(FullyShardedAccumulationContract,U)>
where B:AutodiffBackend,M:AutodiffModule<B>,S:LrScheduler,R:Record<B>,G:Record<B>,U:Record<B> {
    /// Restore original model identities before native optimizer/session tuples,
    /// then reattach the real pending local SUM window with its original logical
    /// sharding/global count/scale. No optimizer, scheduler or backward step occurs.
    pub fn restore_fully_sharded<F,H,Q>(self,model:M,scheduler:S,restore_model:F,restore_groups:H)
        -> Result<RestoredFullyShardedTraining<M,Q,S,U>,RecorderError>
    where F:FnOnce(R,M)->Result<M,RecorderError>,H:FnOnce(M,G)->Result<(M,Q),RecorderError> {
        let restored=self.restore(model,scheduler,restore_model,restore_groups)?;
        let (continuation,state)=restored.state;
        let accumulator=FullyShardedGradientsAccumulator::from_accumulator::<B>(&restored.model,restored.accumulator,continuation)
            .map_err(|error|RecorderError::Unknown(error.to_string()))?;
        Ok(RestoredFullyShardedTraining {model:restored.model,optimizer:restored.optimizer,scheduler:restored.scheduler,accumulator,state})
    }
}

impl<B,M,S,R,G,U> ModelGroupTrainingRecord<B,M,S,R,G,(FullyShardedWeightedAccumulationContract<B>,U)>
where B:AutodiffBackend,M:AutodiffModule<B>,S:LrScheduler,R:Record<B>,G:Record<B>,U:Record<B> {
    /// Original fractional weight and actual native pending derivatives survive
    /// heterogeneous group restore. Peer/session reconstruction is an explicit
    /// callback, allowing loaded-state binding without replacing model leaves.
    pub fn restore_fully_sharded_weighted<F,H,Q>(self,model:M,scheduler:S,restore_model:F,restore_groups:H)
        -> Result<RestoredFullyShardedWeightedTraining<B,M,Q,S,U>,RecorderError>
    where F:FnOnce(R,M)->Result<M,RecorderError>,H:FnOnce(M,G)->Result<(M,Q),RecorderError> {
        let restored=self.restore(model,scheduler,restore_model,restore_groups)?;
        let (continuation,state)=restored.state;
        let accumulator=FullyShardedWeightedGradientsAccumulator::<M,B>::from_accumulator(&restored.model,restored.accumulator,continuation)
            .map_err(|error|RecorderError::Unknown(error.to_string()))?;
        Ok(RestoredFullyShardedWeightedTraining {model:restored.model,optimizer:restored.optimizer,scheduler:restored.scheduler,accumulator,state})
    }
}
