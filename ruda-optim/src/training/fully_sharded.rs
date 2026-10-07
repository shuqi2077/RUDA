use super::*;
use crate::{FullyShardedAccumulationContract,FullyShardedWeightedAccumulationContract,FullyShardedGradientsAccumulator,FullyShardedWeightedGradientsAccumulator};

/// Components resumed from one actual local-rank model/optimizer/scheduler/gradient continuation boundary.
pub struct RestoredFullyShardedTraining<M,O,S,U> {
    /// Actual native model state with original canonical parameter IDs.
    pub model:M,
    /// Actual native local optimizer histories, without an update during restore.
    pub optimizer:O,
    /// Original scheduler position, not advanced.
    pub scheduler:S,
    /// Pending SUM derivatives and exact whole-window GLOBAL integer count/scale.
    pub accumulator:FullyShardedGradientsAccumulator<M>,
    /// Original caller-supplied data/sampler/RNG continuation.
    pub state:U,
}
/// Same actual native local training continuation with fractional GLOBAL effective weights.
pub struct RestoredFullyShardedWeightedTraining<B:AutodiffBackend,M:AutodiffModule<B>,O,S,U> {
    /// Actual restored local model and canonical shared leaves.
    pub model:M,
    /// Actual native local optimizer histories.
    pub optimizer:O,
    /// Actual restored scheduler, without a step.
    pub scheduler:S,
    /// Pending derivatives and exact native fractional normalization continuation.
    pub accumulator:FullyShardedWeightedGradientsAccumulator<M,B>,
    /// Actual caller-owned input/sampler/RNG continuation.
    pub state:U,
}

impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,U>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,R:Record<B>,U:Record<B> {
    /// Combine actual local model state, optimizer/scheduler and original FSDP pending window in one payload.
    /// R can be native parameter-only or trainable-delta state; the original frozen base remains caller-owned.
    pub fn capture_fully_sharded(model:&M,model_state:R,optimizer:&O,scheduler:&S,accumulator:&FullyShardedGradientsAccumulator<M>,state:U)
        -> Result<ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedAccumulationContract,U)>,RecorderError> {
        let continuation=accumulator.continuation();continuation.validate_for::<B,M>(model).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        ModelStateTrainingRecord::<B,M,O,S,R,(FullyShardedAccumulationContract,U)>::capture(model,model_state,optimizer,scheduler,accumulator.inner(),(continuation,state))
    }
    /// Async capture of the same original local FSDP window, without duplicate gradient snapshots.
    pub async fn capture_fully_sharded_async(model:&M,model_state:R,optimizer:&O,scheduler:&S,accumulator:&FullyShardedGradientsAccumulator<M>,state:U)
        -> Result<ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedAccumulationContract,U)>,RecorderError> {
        let continuation=accumulator.continuation();continuation.validate_for::<B,M>(model).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        ModelStateTrainingRecord::<B,M,O,S,R,(FullyShardedAccumulationContract,U)>::capture_async(model,model_state,optimizer,scheduler,accumulator.inner(),(continuation,state)).await
    }
    /// Same local-rank boundary with native fractional global weights, retaining original count metadata.
    pub fn capture_fully_sharded_weighted(model:&M,model_state:R,optimizer:&O,scheduler:&S,accumulator:&FullyShardedWeightedGradientsAccumulator<M,B>,state:U)
        -> Result<ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedWeightedAccumulationContract<B>,U)>,RecorderError> {
        ModelStateTrainingRecord::<B,M,O,S,R,(FullyShardedWeightedAccumulationContract<B>,U)>::capture(model,model_state,optimizer,scheduler,accumulator.inner(),(accumulator.continuation(),state))
    }
    /// Async actual pending-gradient readback while retaining exact original fractional weight storage.
    pub async fn capture_fully_sharded_weighted_async(model:&M,model_state:R,optimizer:&O,scheduler:&S,accumulator:&FullyShardedWeightedGradientsAccumulator<M,B>,state:U)
        -> Result<ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedWeightedAccumulationContract<B>,U)>,RecorderError> {
        ModelStateTrainingRecord::<B,M,O,S,R,(FullyShardedWeightedAccumulationContract<B>,U)>::capture_async(model,model_state,optimizer,scheduler,accumulator.inner(),(accumulator.continuation(),state)).await
    }
}
impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedAccumulationContract,U)>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,R:Record<B>,U:Record<B> {
    /// Restore model state before validating original logical ownership and pending derivative normalization.
    /// No backward replay, native collective, optimizer/scheduler step or data/RNG inference occurs.
    pub fn restore_fully_sharded<F>(self,model:M,optimizer:O,scheduler:S,device:&B::Device,restore_model:F)
        -> Result<RestoredFullyShardedTraining<M,O,S,U>,RecorderError> where F:FnOnce(R,M)->Result<M,RecorderError> {
        let restored=self.restore(model,optimizer,scheduler,device,restore_model)?;let (continuation,state)=restored.state;
        let accumulator=FullyShardedGradientsAccumulator::from_accumulator::<B>(&restored.model,restored.accumulator,continuation)
            .map_err(|error|RecorderError::Unknown(error.to_string()))?;
        Ok(RestoredFullyShardedTraining {model:restored.model,optimizer:restored.optimizer,scheduler:restored.scheduler,accumulator,state})
    }
}
impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedWeightedAccumulationContract<B>,U)>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,R:Record<B>,U:Record<B> {
    /// Restore native weighted continuation from the same original local model/history/data/RNG boundary.
    pub fn restore_fully_sharded_weighted<F>(self,model:M,optimizer:O,scheduler:S,device:&B::Device,restore_model:F)
        -> Result<RestoredFullyShardedWeightedTraining<B,M,O,S,U>,RecorderError> where F:FnOnce(R,M)->Result<M,RecorderError> {
        let restored=self.restore(model,optimizer,scheduler,device,restore_model)?;let (continuation,state)=restored.state;
        let accumulator=FullyShardedWeightedGradientsAccumulator::from_accumulator(&restored.model,restored.accumulator,continuation)
            .map_err(|error|RecorderError::Unknown(error.to_string()))?;
        Ok(RestoredFullyShardedWeightedTraining {model:restored.model,optimizer:restored.optimizer,scheduler:restored.scheduler,accumulator,state})
    }
}
