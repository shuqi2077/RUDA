use core::marker::PhantomData;
use ruda_model::{
    module::AutodiffModule,
    record::{PrecisionSettings, Record, Recorder, RecorderError},
    tensor::backend::AutodiffBackend,
};

use crate::{GradientsAccumulator, GradientsParamsRecord, Optimizer, lr_scheduler::LrScheduler};

#[cfg(test)]
mod tests;

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

impl<B, M, O, S, U> TrainingRecord<B, M, O, S, U>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    U: Record<B>,
{
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
