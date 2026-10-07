use core::marker::PhantomData;
use alloc::string::ToString;
use ruda_model::{
    module::{AutodiffModule, ModuleDTypeRecord},
    record::{PrecisionSettings, Record, Recorder, RecorderError},
    tensor::backend::AutodiffBackend,
};

use crate::{GradientsAccumulator, GradientsParamsRecord, Optimizer, WeightedGradientsAccumulator,
    WeightedAccumulationState, lr_scheduler::LrScheduler};

#[cfg(test)]
mod tests;

mod model_state;
pub use model_state::{ModelStateTrainingRecord,TrainableParameterContract};
mod fully_sharded;
pub use fully_sharded::{RestoredFullyShardedTraining,RestoredFullyShardedWeightedTraining};

/// One record containing the trainable state and a caller-defined continuation record.
///
/// Capture at a training boundary with no concurrent updates. The caller's state
/// carries its counters and any available input/RNG records; this type does not
/// infer them or snapshot a DataLoader. Recorder precision settings apply to all
/// component records and must preserve their values for exact continuation.
pub struct TrainingRecord<B, M, O, S, U>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    U: Record<B>,
{
    model: M::Record,
    optimizer: O::Record,
    scheduler: S::Record<B>,
    gradients: GradientsParamsRecord,
    state: U,
    marker: PhantomData<fn() -> (B, M, O, S)>,
}

/// Components restored together, including pending gradients and caller state.
pub struct RestoredTraining<M, O, S, U> {
    /// Model with the recorded parameter IDs.
    pub model: M,
    /// Optimizer with the recorded parameter state.
    pub optimizer: O,
    /// Scheduler at the recorded position.
    pub scheduler: S,
    /// Pending gradients, without an implicit optimizer update or reset.
    pub accumulator: GradientsAccumulator<M>,
    /// Caller-defined continuation state.
    pub state: U,
}

/// Training components restored with actual unequal-microbatch normalization state.
pub struct RestoredWeightedTraining<M,O,S,U> {
    /// Model with the recorded IDs and optional per-parameter storage dtypes.
    pub model: M,
    /// Recorded optimizer state, without a parameter update.
    pub optimizer: O,
    /// Scheduler at the saved position, not advanced during restore.
    pub scheduler: S,
    /// Pending gradients, effective weight, microbatch count and loss scale.
    pub accumulator: WeightedGradientsAccumulator<M>,
    /// Caller-owned source/sampler/RNG state from the same training checkpoint.
    pub state: U,
}

impl<B, M, O, S, U> TrainingRecord<B, M, O, S, U>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    U: Record<B>,
{
    /// Capture weighted accumulation using the existing combined record format.
    /// Counters/options are kept with caller state, not inferred on resume.
    pub fn capture_weighted(
        model: &M,optimizer: &O,scheduler: &S,
        accumulator: &WeightedGradientsAccumulator<M>,state: U,
    ) -> Result<TrainingRecord<B,M,O,S,(WeightedAccumulationState,U)>,RecorderError> {
        TrainingRecord::<B,M,O,S,(WeightedAccumulationState,U)>::capture(
            model,optimizer,scheduler,accumulator.inner(),(accumulator.state().clone(),state))
    }

    /// Asynchronous combined capture of weighted pending gradients and counters.
    pub async fn capture_weighted_async(
        model: &M,optimizer: &O,scheduler: &S,
        accumulator: &WeightedGradientsAccumulator<M>,state: U,
    ) -> Result<TrainingRecord<B,M,O,S,(WeightedAccumulationState,U)>,RecorderError> {
        TrainingRecord::<B,M,O,S,(WeightedAccumulationState,U)>::capture_async(
            model,optimizer,scheduler,accumulator.inner(),(accumulator.state().clone(),state)).await
    }

    /// Capture weighted accumulation plus mixed floating parameter storage dtypes.
    pub fn capture_weighted_with_dtypes(
        model: &M,optimizer: &O,scheduler: &S,
        accumulator: &WeightedGradientsAccumulator<M>,state: U,
    ) -> Result<TrainingRecord<B,M,O,S,(ModuleDTypeRecord,(WeightedAccumulationState,U))>,RecorderError> {
        TrainingRecord::<B,M,O,S,(WeightedAccumulationState,U)>::capture_with_dtypes(
            model,optimizer,scheduler,accumulator.inner(),(accumulator.state().clone(),state))
    }

    /// Capture the components without consuming the model or clearing gradients.
    pub fn capture(
        model: &M,
        optimizer: &O,
        scheduler: &S,
        accumulator: &GradientsAccumulator<M>,
        state: U,
    ) -> Result<Self, RecorderError> {
        let gradients = accumulator.try_to_record::<B>()?;
        Ok(Self {
            model: model.clone().into_record(),
            optimizer: optimizer.to_record(),
            scheduler: scheduler.to_record::<B>(),
            gradients,
            state,
            marker: PhantomData,
        })
    }

    /// Capture with asynchronous gradient readback; no recorder I/O is performed.
    pub async fn capture_async(
        model: &M,
        optimizer: &O,
        scheduler: &S,
        accumulator: &GradientsAccumulator<M>,
        state: U,
    ) -> Result<Self, RecorderError> {
        let gradients = accumulator.to_record_async::<B>().await?;
        Ok(Self {
            model: model.clone().into_record(),
            optimizer: optimizer.to_record(),
            scheduler: scheduler.to_record::<B>(),
            gradients,
            state,
            marker: PhantomData,
        })
    }

    /// Capture with opt-in per-parameter floating storage dtype metadata.
    ///
    /// The metadata is recorded with caller state; ordinary captures and their
    /// serialization format are unchanged. Use `restore_with_dtypes` to restore
    /// mixed storage dtypes independently from the recorder's value precision.
    pub fn capture_with_dtypes(
        model: &M,
        optimizer: &O,
        scheduler: &S,
        accumulator: &GradientsAccumulator<M>,
        state: U,
    ) -> Result<TrainingRecord<B, M, O, S, (ModuleDTypeRecord, U)>, RecorderError> {
        let dtypes = ModuleDTypeRecord::capture(model)?;
        TrainingRecord::<B, M, O, S, (ModuleDTypeRecord, U)>::capture(
            model,
            optimizer,
            scheduler,
            accumulator,
            (dtypes, state),
        )
    }

    /// Asynchronously capture pending gradients and per-parameter storage dtypes.
    pub async fn capture_async_with_dtypes(
        model: &M,
        optimizer: &O,
        scheduler: &S,
        accumulator: &GradientsAccumulator<M>,
        state: U,
    ) -> Result<TrainingRecord<B, M, O, S, (ModuleDTypeRecord, U)>, RecorderError> {
        let dtypes = ModuleDTypeRecord::capture(model)?;
        TrainingRecord::<B, M, O, S, (ModuleDTypeRecord, U)>::capture_async(
            model,
            optimizer,
            scheduler,
            accumulator,
            (dtypes, state),
        )
        .await
    }

    /// Save all components in one recorder payload.
    pub fn save<R: Recorder<B>>(
        self,
        recorder: &R,
        args: R::RecordArgs,
    ) -> Result<R::RecordOutput, RecorderError> {
        recorder.record(self, args)
    }

    /// Read a combined record using the selected recorder and device.
    pub fn load<R: Recorder<B>>(
        recorder: &R,
        args: R::LoadArgs,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        recorder.load(args, device)
    }

    /// Restore onto compatible model, optimizer and scheduler configurations.
    ///
    /// No scheduler step, optimizer step or gradient reset is performed. Apply
    /// the returned caller state before consuming the next training batch.
    pub fn restore(
        self,
        model: M,
        optimizer: O,
        scheduler: S,
        device: &B::Device,
    ) -> Result<RestoredTraining<M, O, S, U>, RecorderError> {
        let mut accumulator = GradientsAccumulator::new();
        accumulator.load_record::<B>(self.gradients, device)?;
        let model = model.load_record(self.model).fork(device);
        let optimizer = optimizer.load_record(self.optimizer);
        let scheduler = scheduler.load_record::<B>(self.scheduler);
        Ok(RestoredTraining {
            model,
            optimizer,
            scheduler,
            accumulator,
            state: self.state,
        })
    }
}

impl<B,M,O,S,U> TrainingRecord<B,M,O,S,(WeightedAccumulationState,U)>
where B: AutodiffBackend,M: AutodiffModule<B>,O: Optimizer<M,B>,S: LrScheduler,U: Record<B> {
    /// Restore pending gradients and their weight/loss-scale state without replay.
    pub fn restore_weighted(
        self,model: M,optimizer: O,scheduler: S,device: &B::Device,
    ) -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError> {
        restore_weighted::<B,M,O,S,U>(self.restore(model,optimizer,scheduler,device)?)
    }
}

impl<B,M,O,S,U> TrainingRecord<B,M,O,S,(ModuleDTypeRecord,(WeightedAccumulationState,U))>
where B: AutodiffBackend,M: AutodiffModule<B>,O: Optimizer<M,B>,S: LrScheduler,U: Record<B> {
    /// Restore both mixed storage and the exact saved accumulation window.
    pub fn restore_weighted_with_dtypes(
        self,model: M,optimizer: O,scheduler: S,device: &B::Device,
    ) -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError> {
        restore_weighted::<B,M,O,S,U>(self.restore_with_dtypes(model,optimizer,scheduler,device)?)
    }
}

fn restore_weighted<B,M,O,S,U>(
    restored: RestoredTraining<M,O,S,(WeightedAccumulationState,U)>,
) -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
where B: AutodiffBackend,M: AutodiffModule<B> {
    let (normalization,state) = restored.state;
    let accumulator = WeightedGradientsAccumulator::from_accumulator::<B>(
        &restored.model,restored.accumulator,normalization)
        .map_err(|error|RecorderError::Unknown(error.to_string()))?;
    Ok(RestoredWeightedTraining {model:restored.model,optimizer:restored.optimizer,
        scheduler:restored.scheduler,accumulator,state})
}

impl<B, M, O, S, U> TrainingRecord<B, M, O, S, (ModuleDTypeRecord, U)>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    U: Record<B>,
{
    /// Restore captured floating storage dtypes, pending gradients and trainable state.
    /// No update, scheduler step, gradient reset or requantization is performed.
    pub fn restore_with_dtypes(
        self,
        model: M,
        optimizer: O,
        scheduler: S,
        device: &B::Device,
    ) -> Result<RestoredTraining<M, O, S, U>, RecorderError> {
        let restored = self.restore(model, optimizer, scheduler, device)?;
        let (dtypes, state) = restored.state;
        Ok(RestoredTraining {
            model: dtypes.apply(restored.model)?,
            optimizer: restored.optimizer,
            scheduler: restored.scheduler,
            accumulator: restored.accumulator,
            state,
        })
    }
}

impl<B, M, O, S, U> Record<B> for TrainingRecord<B, M, O, S, U>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    U: Record<B>,
{
    type Item<P: PrecisionSettings> = (
        <M::Record as Record<B>>::Item<P>,
        <O::Record as Record<B>>::Item<P>,
        <S::Record<B> as Record<B>>::Item<P>,
        <GradientsParamsRecord as Record<B>>::Item<P>,
        U::Item<P>,
    );

    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        (
            self.model.into_item::<P>(),
            self.optimizer.into_item::<P>(),
            self.scheduler.into_item::<P>(),
            <GradientsParamsRecord as Record<B>>::into_item::<P>(self.gradients),
            self.state.into_item::<P>(),
        )
    }

    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self {
            model: <M::Record as Record<B>>::from_item::<P>(item.0, device),
            optimizer: <O::Record as Record<B>>::from_item::<P>(item.1, device),
            scheduler: <S::Record<B> as Record<B>>::from_item::<P>(item.2, device),
            gradients: <GradientsParamsRecord as Record<B>>::from_item::<P>(item.3, device),
            state: U::from_item::<P>(item.4, device),
            marker: PhantomData,
        }
    }
}
