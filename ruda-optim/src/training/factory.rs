use super::*;

pub(super) fn restore_components<B,M,O,S,U,F>(model:M,scheduler:S,scheduler_record:S::Record<B>,gradients:GradientsParamsRecord,
    optimizer:O::Record,state:U,restore_optimizer:F) -> Result<RestoredTraining<M,O,S,U>,RecorderError>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,U:Record<B>,
    F:FnOnce(&M,&U,O::Record)->Result<O,RecorderError> {
    let mut accumulator=GradientsAccumulator::new();
    accumulator.load_record_for_model::<B>(gradients,&model)?;
    let optimizer=restore_optimizer(&model,&state,optimizer)?;
    Ok(RestoredTraining {model,optimizer,scheduler:scheduler.load_record::<B>(scheduler_record),accumulator,state})
}

impl<B,M,O,S,U> TrainingRecord<B,M,O,S,U>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,U:Record<B> {
    /// Restore actual model IDs/storage on the explicit device BEFORE constructing
    /// an ID-bound optimizer or replica session. The factory receives original
    /// caller state and native optimizer records, not guessed group settings.
    /// Pending gradients use the restored trainable leaves' actual devices/shape;
    /// no optimizer/scheduler step, normalization, reset or backward replay occurs.
    pub fn restore_with_optimizer<F>(self,model:M,scheduler:S,device:&B::Device,restore_optimizer:F)
        -> Result<RestoredTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(&M,&U,O::Record)->Result<O,RecorderError> {
        let model=model.load_record(self.model).fork(device);
        restore_components::<B,M,O,S,U,F>(model,scheduler,self.scheduler,self.gradients,self.optimizer,self.state,restore_optimizer)
    }

    /// Native asynchronous weighted-window capture including original mixed
    /// parameter storage. Uses the original accumulator and exact saved counters,
    /// without draining pending gradients or requiring synchronous device reads.
    pub async fn capture_weighted_with_dtypes_async(model:&M,optimizer:&O,scheduler:&S,
        accumulator:&WeightedGradientsAccumulator<M>,state:U)
        -> Result<TrainingRecord<B,M,O,S,(ModuleDTypeRecord,(WeightedAccumulationState,U))>,RecorderError> {
        TrainingRecord::<B,M,O,S,(WeightedAccumulationState,U)>::capture_async_with_dtypes(
            model,optimizer,scheduler,accumulator.inner(),(accumulator.state().clone(),state)).await
    }
}

impl<B,M,O,S,U> TrainingRecord<B,M,O,S,(ModuleDTypeRecord,U)>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,U:Record<B> {
    /// Apply actual saved per-parameter dtypes before the optimizer factory sees
    /// the model. The original dtype record is not passed off as a new precision
    /// policy; pending work-gradient dtype and original optimizer buffers survive.
    pub fn restore_with_dtypes_and_optimizer<F>(self,model:M,scheduler:S,device:&B::Device,restore_optimizer:F)
        -> Result<RestoredTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(&M,&U,O::Record)->Result<O,RecorderError> {
        let (dtypes,state)=self.state;
        let model=dtypes.apply(model.load_record(self.model).fork(device))?;
        restore_components::<B,M,O,S,U,F>(model,scheduler,self.scheduler,self.gradients,self.optimizer,state,restore_optimizer)
    }
}

impl<B,M,O,S,U> TrainingRecord<B,M,O,S,(WeightedAccumulationState,U)>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,U:Record<B> {
    /// Reconstruct original optimizer groups after model loading, with access to
    /// exact normalization/caller state. Restore the existing pending window once.
    pub fn restore_weighted_with_optimizer<F>(self,model:M,scheduler:S,device:&B::Device,restore_optimizer:F)
        -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(&M,&(WeightedAccumulationState,U),O::Record)->Result<O,RecorderError> {
        restore_weighted::<B,M,O,S,U>(self.restore_with_optimizer(model,scheduler,device,restore_optimizer)?)
    }
}

impl<B,M,O,S,U> TrainingRecord<B,M,O,S,(ModuleDTypeRecord,(WeightedAccumulationState,U))>
where B:AutodiffBackend,M:AutodiffModule<B>,O:Optimizer<M,B>,S:LrScheduler,U:Record<B> {
    /// Original storage/IDs are final before native group construction; exact
    /// unequal-window normalization stays separate from optimizer configuration.
    pub fn restore_weighted_with_dtypes_and_optimizer<F>(self,model:M,scheduler:S,device:&B::Device,restore_optimizer:F)
        -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(&M,&(WeightedAccumulationState,U),O::Record)->Result<O,RecorderError> {
        restore_weighted::<B,M,O,S,U>(self.restore_with_dtypes_and_optimizer(model,scheduler,device,restore_optimizer)?)
    }
}
