use super::*;
use crate::{FullyShardedAccumulationContract,FullyShardedWeightedAccumulationContract,FullyShardedGradientsAccumulator,
    FullyShardedWeightedGradientsAccumulator,training::{RestoredFullyShardedTraining,RestoredFullyShardedWeightedTraining}};

impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,U>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,R:Record<B>,U:Record<B> {
    /// Restore the actual caller-selected archive into prepared native topology,
    /// validate original trainable IDs/storage, then construct the original
    /// optimizer from that model, caller state and native saved histories.
    /// Pending derivatives restore to each actual leaf's device. No common device
    /// is inferred, and omitted frozen bases are not moved/copied/requantized.
    /// The callback must reconstruct original options/ownership, not new defaults.
    pub fn restore_with_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(R,M)->Result<M,RecorderError>,H:FnOnce(&M,&U,O::Record)->Result<O,RecorderError> {
        if self.version!=1 {return Err(invalid("unsupported format version"));}
        let model=restore_model(self.model_state,model)?;
        self.contract.validate_for::<B,M>(&model)?;
        super::super::factory::restore_components::<B,M,O,S,U,H>(model,scheduler,self.scheduler,self.gradients,self.optimizer,self.state,restore_optimizer)
    }
}

impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,(WeightedAccumulationState,U)>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,R:Record<B>,U:Record<B> {
    /// Actual native archive/group construction plus original weighted pending
    /// window, without normalizing it or advancing the saved scheduler position.
    pub fn restore_weighted_with_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(R,M)->Result<M,RecorderError>,H:FnOnce(&M,&(WeightedAccumulationState,U),O::Record)->Result<O,RecorderError> {
        super::super::restore_weighted::<B,M,O,S,U>(self.restore_with_optimizer(model,scheduler,restore_model,restore_optimizer)?)
    }
}

impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedAccumulationContract,U)>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,R:Record<B>,U:Record<B> {
    /// Native ID-bound FSDP optimizer construction after model restoration, then
    /// attach exact original logical ownership/count/scale to the restored pending
    /// SUM derivatives. No gather backward or gradient reduction is repeated.
    pub fn restore_fully_sharded_with_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredFullyShardedTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(R,M)->Result<M,RecorderError>,H:FnOnce(&M,&(FullyShardedAccumulationContract,U),O::Record)->Result<O,RecorderError> {
        let restored=self.restore_with_optimizer(model,scheduler,restore_model,restore_optimizer)?;
        let (continuation,state)=restored.state;
        let accumulator=FullyShardedGradientsAccumulator::from_accumulator::<B>(&restored.model,restored.accumulator,continuation)
            .map_err(|error|RecorderError::Unknown(error.to_string()))?;
        Ok(RestoredFullyShardedTraining {model:restored.model,optimizer:restored.optimizer,scheduler:restored.scheduler,accumulator,state})
    }
}

impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,(FullyShardedWeightedAccumulationContract<B>,U)>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,R:Record<B>,U:Record<B> {
    /// Same post-model factory for fractional GLOBAL normalization, preserving
    /// actual local derivatives, selected counts, native denominator and scale.
    /// Integer counts are not substituted for the original fractional weight.
    pub fn restore_fully_sharded_weighted_with_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredFullyShardedWeightedTraining<B,M,O,S,U>,RecorderError>
    where F:FnOnce(R,M)->Result<M,RecorderError>,H:FnOnce(&M,&(FullyShardedWeightedAccumulationContract<B>,U),O::Record)->Result<O,RecorderError> {
        let restored=self.restore_with_optimizer(model,scheduler,restore_model,restore_optimizer)?;
        let (continuation,state)=restored.state;
        let accumulator=FullyShardedWeightedGradientsAccumulator::<M,B>::from_accumulator(&restored.model,restored.accumulator,continuation)
            .map_err(|error|RecorderError::Unknown(error.to_string()))?;
        Ok(RestoredFullyShardedWeightedTraining {model:restored.model,optimizer:restored.optimizer,scheduler:restored.scheduler,accumulator,state})
    }
}
