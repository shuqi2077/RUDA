//! Complete continuation with caller-owned heterogeneous optimizer/parallel groups.

use super::*;
mod fully_sharded;

/// Serialize an original inner-backend record alongside autodiff model records.
///
/// Allows native ZeRO-2 slice records in the same tuple as selected ZeRO-1 or
/// ordinary optimizer records. It changes neither buffers nor precision/device
/// semantics: conversion uses the original record implementation exactly once.
pub struct InnerBackendRecord<B: AutodiffBackend, R: Record<B::InnerBackend>> {
    inner: R,
    marker: PhantomData<B>,
}

impl<B: AutodiffBackend, R: Record<B::InnerBackend>> InnerBackendRecord<B, R> {
    /// Attach a backend type to an actual original inner record, without copying state.
    pub fn new(inner: R) -> Self { Self { inner, marker: PhantomData } }

    /// Recover the original record for its existing rank-local restore API.
    pub fn into_inner(self) -> R { self.inner }

    /// Inspect the actual original record without snapshotting or converting it.
    pub fn inner(&self) -> &R { &self.inner }
}

impl<B: AutodiffBackend, R: Record<B::InnerBackend>> Record<B> for InnerBackendRecord<B, R> {
    type Item<P: PrecisionSettings> = R::Item<P>;
    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> { self.inner.into_item::<P>() }
    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self::new(R::from_item::<P>(item, device))
    }
}

/// Original model state, explicit optimizer-group records, scheduler and pending gradients together.
///
/// `R` is the caller's actual native model/adapter archive, not a dense stand-in.
/// `G` may be a tuple of selected ZeRO-1, wrapped ZeRO-2 and other actual optimizer
/// records. Groups need not implement the single-model `Optimizer` trait or share
/// a device/algorithm. Capture all supplied records at the same quiescent boundary.
/// Caller state carries original source/sampler/RNG position; none is inferred.
pub struct ModelGroupTrainingRecord<B, M, S, R, G, U>
where
    B: AutodiffBackend, M: AutodiffModule<B>, S: LrScheduler,
    R: Record<B>, G: Record<B>, U: Record<B>,
{
    version: u32,
    model_state: R,
    contract: TrainableParameterContract,
    groups: G,
    scheduler: S::Record<B>,
    gradients: GradientsParamsRecord,
    state: U,
    marker: PhantomData<fn() -> (B, M, S)>,
}

impl<B, M, S, R, G, U> ModelGroupTrainingRecord<B, M, S, R, G, U>
where
    B: AutodiffBackend, M: AutodiffModule<B>, S: LrScheduler,
    R: Record<B>, G: Record<B>, U: Record<B>,
{
    /// Capture supplied original model/group records and live scheduler/pending values.
    /// Does not clear a window, update a parameter, choose groups or duplicate a frozen base.
    pub fn capture(
        model: &M, model_state: R, groups: G, scheduler: &S,
        accumulator: &GradientsAccumulator<M>, state: U,
    ) -> Result<Self, RecorderError> {
        check_pending::<B, M>(model, accumulator)?;
        let contract = TrainableParameterContract::capture::<B, M>(model)?;
        let gradients = accumulator.try_to_record::<B>()?;
        Ok(Self { version: 1, model_state, contract, groups,
            scheduler: scheduler.to_record::<B>(), gradients, state, marker: PhantomData })
    }

    /// Same capture using asynchronous readback of actual pending derivatives.
    pub async fn capture_async(
        model: &M, model_state: R, groups: G, scheduler: &S,
        accumulator: &GradientsAccumulator<M>, state: U,
    ) -> Result<Self, RecorderError> {
        check_pending::<B, M>(model, accumulator)?;
        let contract = TrainableParameterContract::capture::<B, M>(model)?;
        let gradients = accumulator.to_record_async::<B>().await?;
        Ok(Self { version: 1, model_state, contract, groups,
            scheduler: scheduler.to_record::<B>(), gradients, state, marker: PhantomData })
    }

    /// Capture exact weighted-window counters/scale with supplied original group records.
    pub fn capture_weighted(
        model: &M, model_state: R, groups: G, scheduler: &S,
        accumulator: &WeightedGradientsAccumulator<M>, state: U,
    ) -> Result<ModelGroupTrainingRecord<B, M, S, R, G, (WeightedAccumulationState, U)>, RecorderError> {
        ModelGroupTrainingRecord::capture(model, model_state, groups, scheduler,
            accumulator.inner(), (accumulator.state().clone(), state))
    }

    /// Asynchronous capture preserving the same original weighted accumulation window.
    pub async fn capture_weighted_async(
        model: &M, model_state: R, groups: G, scheduler: &S,
        accumulator: &WeightedGradientsAccumulator<M>, state: U,
    ) -> Result<ModelGroupTrainingRecord<B, M, S, R, G, (WeightedAccumulationState, U)>, RecorderError> {
        ModelGroupTrainingRecord::capture_async(model, model_state, groups, scheduler,
            accumulator.inner(), (accumulator.state().clone(), state)).await
    }

    /// Write the actual supplied components in one existing recorder payload.
    /// Recorder precision settings must preserve all master/moment/pending values.
    pub fn save<C: Recorder<B>>(self, recorder: &C, args: C::RecordArgs)
        -> Result<C::RecordOutput, RecorderError> { recorder.record(self, args) }

    /// Read on the recorder's requested device; group placement is handled by its explicit restore callback.
    pub fn load<C: Recorder<B>>(recorder: &C, args: C::LoadArgs, device: &B::Device)
        -> Result<Self, RecorderError> { recorder.load(args, device) }

    /// Restore original model IDs, then actual group state and per-parameter-device pending values.
    ///
    /// `restore_model` restores the supplied native model archive into the caller's
    /// prepared topology. `restore_groups` receives that actual model and the saved
    /// group records, returning the model and caller-owned live optimizer tuple.
    /// It must restore each group's exact original rank/ownership/options using
    /// the group's own restore API. No communicator or repartitioning is inferred.
    /// The result's `optimizer` field contains this live tuple. Scheduler state is
    /// restored without advancing it; pending derivatives are not normalized/reset.
    pub fn restore<F, H, Q>(
        self, model: M, scheduler: S, restore_model: F, restore_groups: H,
    ) -> Result<RestoredTraining<M, Q, S, U>, RecorderError>
    where
        F: FnOnce(R, M) -> Result<M, RecorderError>,
        H: FnOnce(M, G) -> Result<(M, Q), RecorderError>,
    {
        if self.version != 1 { return Err(invalid("unsupported optimizer-group format version")); }
        let model = restore_model(self.model_state, model)?;
        self.contract.validate_for::<B, M>(&model)?;
        let (model, optimizer) = restore_groups(model, self.groups)?;
        self.contract.validate_for::<B, M>(&model)?;
        let mut accumulator = GradientsAccumulator::new();
        accumulator.load_record_for_model::<B>(self.gradients, &model)?;
        check_pending::<B, M>(&model, &accumulator)?;
        Ok(RestoredTraining { model, optimizer, scheduler: scheduler.load_record::<B>(self.scheduler),
            accumulator, state: self.state })
    }
}

impl<B, M, S, R, G, U> ModelGroupTrainingRecord<B, M, S, R, G, (WeightedAccumulationState, U)>
where
    B: AutodiffBackend, M: AutodiffModule<B>, S: LrScheduler,
    R: Record<B>, G: Record<B>, U: Record<B>,
{
    /// Restore original groups and exact weighted-window counters with mixed-device pending values.
    pub fn restore_weighted<F, H, Q>(
        self, model: M, scheduler: S, restore_model: F, restore_groups: H,
    ) -> Result<RestoredWeightedTraining<M, Q, S, U>, RecorderError>
    where
        F: FnOnce(R, M) -> Result<M, RecorderError>,
        H: FnOnce(M, G) -> Result<(M, Q), RecorderError>,
    {
        super::super::restore_weighted::<B, M, Q, S, U>(
            self.restore(model, scheduler, restore_model, restore_groups)?)
    }
}

impl<B, M, S, R, G, U> Record<B> for ModelGroupTrainingRecord<B, M, S, R, G, U>
where
    B: AutodiffBackend, M: AutodiffModule<B>, S: LrScheduler,
    R: Record<B>, G: Record<B>, U: Record<B>,
{
    type Item<P: PrecisionSettings> = (u32, R::Item<P>, <TrainableParameterContract as Record<B>>::Item<P>,
        G::Item<P>, <S::Record<B> as Record<B>>::Item<P>, <GradientsParamsRecord as Record<B>>::Item<P>, U::Item<P>);
    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        (self.version, self.model_state.into_item::<P>(), <TrainableParameterContract as Record<B>>::into_item::<P>(self.contract),
            self.groups.into_item::<P>(), self.scheduler.into_item::<P>(),
            <GradientsParamsRecord as Record<B>>::into_item::<P>(self.gradients), self.state.into_item::<P>())
    }
    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        Self { version: item.0, model_state: R::from_item::<P>(item.1, device),
            contract: <TrainableParameterContract as Record<B>>::from_item::<P>(item.2, device),
            groups: G::from_item::<P>(item.3, device), scheduler: <S::Record<B> as Record<B>>::from_item::<P>(item.4, device),
            gradients: <GradientsParamsRecord as Record<B>>::from_item::<P>(item.5, device),
            state: U::from_item::<P>(item.6, device), marker: PhantomData }
    }
}
