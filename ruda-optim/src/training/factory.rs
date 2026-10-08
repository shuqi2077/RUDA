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
    /// Restore the actual full native model archive through an explicit prepared
    /// topology callback, then construct original optimizer groups. Supports
    /// caller-owned per-leaf devices without forcing the model through one device.
    /// The model callback establishes its original graph/frozen-base/placement
    /// boundary; no transport, device mapping or repartitioning is inferred here.
    /// Reuses the existing full training payload and pending-gradient validation.
    pub fn restore_with_model_and_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(M::Record,M)->Result<M,RecorderError>,H:FnOnce(&M,&U,O::Record)->Result<O,RecorderError> {
        let model=restore_model(self.model,model)?;
        restore_components::<B,M,O,S,U,H>(model,scheduler,self.scheduler,self.gradients,self.optimizer,self.state,restore_optimizer)
    }
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
    /// Caller-controlled native placement and actual saved mixed floating storage
    /// are both final before the optimizer/session factory receives model IDs.
    /// No dense substitution, common-device fork or master/history precision cast.
    pub fn restore_with_dtypes_model_and_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(M::Record,M)->Result<M,RecorderError>,H:FnOnce(&M,&U,O::Record)->Result<O,RecorderError> {
        let (dtypes,state)=self.state;
        let model=dtypes.apply(restore_model(self.model,model)?)?;
        restore_components::<B,M,O,S,U,H>(model,scheduler,self.scheduler,self.gradients,self.optimizer,state,restore_optimizer)
    }
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
    /// Exact weighted window with caller-restored native topology and post-model
    /// optimizer construction; pending sums/counters are not replayed or reset.
    pub fn restore_weighted_with_model_and_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(M::Record,M)->Result<M,RecorderError>,H:FnOnce(&M,&(WeightedAccumulationState,U),O::Record)->Result<O,RecorderError> {
        restore_weighted::<B,M,O,S,U>(self.restore_with_model_and_optimizer(model,scheduler,restore_model,restore_optimizer)?)
    }
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
    /// Restore original per-leaf placement/storage before native group histories,
    /// then reattach the original unequal-microbatch window and caller state.
    pub fn restore_weighted_with_dtypes_model_and_optimizer<F,H>(self,model:M,scheduler:S,restore_model:F,restore_optimizer:H)
        -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(M::Record,M)->Result<M,RecorderError>,H:FnOnce(&M,&(WeightedAccumulationState,U),O::Record)->Result<O,RecorderError> {
        restore_weighted::<B,M,O,S,U>(self.restore_with_dtypes_model_and_optimizer(model,scheduler,restore_model,restore_optimizer)?)
    }
    /// Original storage/IDs are final before native group construction; exact
    /// unequal-window normalization stays separate from optimizer configuration.
    pub fn restore_weighted_with_dtypes_and_optimizer<F>(self,model:M,scheduler:S,device:&B::Device,restore_optimizer:F)
        -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
    where F:FnOnce(&M,&(WeightedAccumulationState,U),O::Record)->Result<O,RecorderError> {
        restore_weighted::<B,M,O,S,U>(self.restore_with_dtypes_and_optimizer(model,scheduler,device,restore_optimizer)?)
    }
}
