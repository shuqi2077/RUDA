use alloc::{collections::{BTreeMap,BTreeSet},format,string::ToString,vec::Vec};
use core::marker::PhantomData;
use ruda_model::{
    module::{AutodiffModule,ModuleVisitor,Param},
    record::{PrecisionSettings,Record,Recorder,RecorderError},
    tensor::{DType,Tensor,backend::{AutodiffBackend,Backend}},
};
use serde::{Deserialize,Serialize};
use crate::{GradientsAccumulator,GradientsParams,GradientsParamsRecord,Optimizer,
    WeightedAccumulationState,WeightedGradientsAccumulator,lr_scheduler::LrScheduler};
use super::{RestoredTraining,RestoredWeightedTraining};

/// Exact trainable IDs, logical shapes and storage, without frozen weight values.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct TrainableParameterContract {
    entries: Vec<(u64,Vec<usize>,DType)>,
}

impl<B: Backend> Record<B> for TrainableParameterContract {
    type Item<S: PrecisionSettings> = Self;
    fn into_item<S: PrecisionSettings>(self) -> Self { self }
    fn from_item<S: PrecisionSettings>(item: Self,_device: &B::Device) -> Self { item }
}

struct CaptureContract {
    entries: BTreeMap<u64,(Vec<usize>,DType)>,
    error: bool,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for CaptureContract {
    fn visit_float<const D: usize>(&mut self,param: &Param<Tensor<B,D>>) {
        let tensor = param.val();
        if !tensor.is_require_grad() { return; }
        let metadata = (tensor.dims().to_vec(),tensor.dtype());
        if let Some(previous) = self.entries.insert(param.id.val(),metadata.clone()) {
            self.error |= previous != metadata;
        }
    }
}

impl TrainableParameterContract {
    /// Capture actual trainable metadata once per tied ID; no tensor values are read.
    pub fn capture<B: AutodiffBackend,M: AutodiffModule<B>>(model: &M) -> Result<Self,RecorderError> {
        let mut visitor = CaptureContract {entries:BTreeMap::new(),error:false};
        model.visit(&mut visitor);
        if visitor.error { return Err(invalid("tied trainable parameter geometry/dtype differs")); }
        Ok(Self {entries:visitor.entries.into_iter().map(|(id,(shape,dtype))|(id,shape,dtype)).collect()})
    }

    /// Match original trainable IDs/dtypes after the caller's model-state restoration.
    pub fn validate_for<B: AutodiffBackend,M: AutodiffModule<B>>(&self,model: &M) -> Result<(),RecorderError> {
        let actual = Self::capture::<B,M>(model)?;
        if *self != actual { return Err(invalid("restored trainable parameter IDs, geometry or storage differs")); }
        Ok(())
    }

    /// Actual unique trainable parameter count, not the total base-model size.
    pub fn parameters(&self) -> usize { self.entries.len() }
}

struct PendingCheck<'a> {
    gradients: &'a GradientsParams,
    active_ids: BTreeSet<u64>,
    frozen: bool,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for PendingCheck<'_> {
    fn visit_float<const D: usize>(&mut self,param: &Param<Tensor<B,D>>) {
        if !param.val().is_require_grad() && !self.active_ids.contains(&param.id.val())
            && self.gradients.get::<B::InnerBackend,D>(param.id).is_some() { self.frozen = true; }
    }
}

fn check_pending<B: AutodiffBackend,M: AutodiffModule<B>>(model: &M,accumulator: &GradientsAccumulator<M>)
    -> Result<(),RecorderError> {
    accumulator.pending().validate_for::<B,M>(model).map_err(|error|RecorderError::Unknown(error.to_string()))?;
    let active_ids = TrainableParameterContract::capture::<B,M>(model)?.entries.into_iter().map(|(id,_,_)|id).collect();
    let mut visitor = PendingCheck {gradients:accumulator.pending(),active_ids,frozen:false};
    model.visit(&mut visitor);
    if visitor.frozen { return Err(invalid("pending gradients include a frozen parameter")); }
    Ok(())
}

fn invalid(reason: &str) -> RecorderError {
    RecorderError::Unknown(format!("Invalid model-state training record: {reason}"))
}

/// Caller-selected model state plus actual optimizer/scheduler/pending gradients.
///
/// `R` can be RUDA's A/B-only native adapter record. It must restore every
/// trainable parameter, including IDs/dtypes, and identify any omitted frozen
/// state. Capture `R` and these components at the same boundary, without concurrent
/// updates. No full model record or hidden frozen tensor copy is created here.
pub struct ModelStateTrainingRecord<B,M,O,S,R,U>
where B: AutodiffBackend,M: AutodiffModule<B>,O: Optimizer<M,B>,S: LrScheduler,R: Record<B>,U: Record<B> {
    version: u32,
    model_state: R,
    contract: TrainableParameterContract,
    optimizer: O::Record,
    scheduler: S::Record<B>,
    gradients: GradientsParamsRecord,
    state: U,
    marker: PhantomData<fn()->(B,M,O,S)>,
}

impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,U>
where B: AutodiffBackend,M: AutodiffModule<B>,O: Optimizer<M,B>,S: LrScheduler,R: Record<B>,U: Record<B> {
    /// Capture provided actual model state, optimizer, scheduler and pending values.
    pub fn capture(model: &M,model_state: R,optimizer: &O,scheduler: &S,
        accumulator: &GradientsAccumulator<M>,state: U) -> Result<Self,RecorderError> {
        check_pending::<B,M>(model,accumulator)?;
        let contract = TrainableParameterContract::capture::<B,M>(model)?;
        let gradients = accumulator.try_to_record::<B>()?;
        Ok(Self {version:1,model_state,contract,optimizer:optimizer.to_record(),scheduler:scheduler.to_record::<B>(),
            gradients,state,marker:PhantomData})
    }

    /// Same capture with asynchronous readback of actual pending gradients.
    pub async fn capture_async(model: &M,model_state: R,optimizer: &O,scheduler: &S,
        accumulator: &GradientsAccumulator<M>,state: U) -> Result<Self,RecorderError> {
        check_pending::<B,M>(model,accumulator)?;
        let contract = TrainableParameterContract::capture::<B,M>(model)?;
        let gradients = accumulator.to_record_async::<B>().await?;
        Ok(Self {version:1,model_state,contract,optimizer:optimizer.to_record(),scheduler:scheduler.to_record::<B>(),
            gradients,state,marker:PhantomData})
    }

    /// Capture actual weighted-window counts/scale alongside caller source/RNG state.
    pub fn capture_weighted(model: &M,model_state: R,optimizer: &O,scheduler: &S,
        accumulator: &WeightedGradientsAccumulator<M>,state: U)
        -> Result<ModelStateTrainingRecord<B,M,O,S,R,(WeightedAccumulationState,U)>,RecorderError> {
        ModelStateTrainingRecord::<B,M,O,S,R,(WeightedAccumulationState,U)>::capture(
            model,model_state,optimizer,scheduler,accumulator.inner(),(accumulator.state().clone(),state))
    }

    /// Asynchronous weighted capture, with no implicit window reset or update.
    pub async fn capture_weighted_async(model: &M,model_state: R,optimizer: &O,scheduler: &S,
        accumulator: &WeightedGradientsAccumulator<M>,state: U)
        -> Result<ModelStateTrainingRecord<B,M,O,S,R,(WeightedAccumulationState,U)>,RecorderError> {
        ModelStateTrainingRecord::<B,M,O,S,R,(WeightedAccumulationState,U)>::capture_async(
            model,model_state,optimizer,scheduler,accumulator.inner(),(accumulator.state().clone(),state)).await
    }

    /// Save provided model state and live training components in one recorder payload.
    pub fn save<C: Recorder<B>>(self,recorder: &C,args: C::RecordArgs) -> Result<C::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Read the combined state on the requested device; recreate omitted base state separately.
    pub fn load<C: Recorder<B>>(recorder: &C,args: C::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore caller-selected model state first, then validate actual trainable identities.
    /// `restore_model` may call a native adapter record's restore_into with an independently
    /// supplied frozen base identity. The prepared model must already use the requested
    /// device; the omitted base is never copied or moved implicitly.
    pub fn restore<F>(self,model: M,optimizer: O,scheduler: S,device: &B::Device,restore_model: F)
        -> Result<RestoredTraining<M,O,S,U>,RecorderError>
    where F: FnOnce(R,M)->Result<M,RecorderError> {
        if self.version != 1 { return Err(invalid("unsupported format version")); }
        let model = restore_model(self.model_state,model)?;
        if model.devices().iter().any(|actual|actual != device) { return Err(invalid("restored model must already use the requested device")); }
        self.contract.validate_for::<B,M>(&model)?;
        let mut accumulator = GradientsAccumulator::new();
        accumulator.load_record::<B>(self.gradients,device)?;
        check_pending::<B,M>(&model,&accumulator)?;
        Ok(RestoredTraining {model,optimizer:optimizer.load_record(self.optimizer),scheduler:scheduler.load_record::<B>(self.scheduler),
            accumulator,state:self.state})
    }
}

impl<B,M,O,S,R,U> ModelStateTrainingRecord<B,M,O,S,R,(WeightedAccumulationState,U)>
where B: AutodiffBackend,M: AutodiffModule<B>,O: Optimizer<M,B>,S: LrScheduler,R: Record<B>,U: Record<B> {
    /// Restore actual A/B/model state and the weighted accumulation window together.
    pub fn restore_weighted<F>(self,model: M,optimizer: O,scheduler: S,device: &B::Device,restore_model: F)
        -> Result<RestoredWeightedTraining<M,O,S,U>,RecorderError>
    where F: FnOnce(R,M)->Result<M,RecorderError> {
        super::restore_weighted::<B,M,O,S,U>(self.restore(model,optimizer,scheduler,device,restore_model)?)
    }
}

impl<B,M,O,S,R,U> Record<B> for ModelStateTrainingRecord<B,M,O,S,R,U>
where B: AutodiffBackend,M: AutodiffModule<B>,O: Optimizer<M,B>,S: LrScheduler,R: Record<B>,U: Record<B> {
    type Item<P: PrecisionSettings> = (u32,R::Item<P>,<TrainableParameterContract as Record<B>>::Item<P>,
        <O::Record as Record<B>>::Item<P>,<S::Record<B> as Record<B>>::Item<P>,
        <GradientsParamsRecord as Record<B>>::Item<P>,U::Item<P>);
    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.model_state.into_item::<P>(),<TrainableParameterContract as Record<B>>::into_item::<P>(self.contract),
            self.optimizer.into_item::<P>(),self.scheduler.into_item::<P>(),
            <GradientsParamsRecord as Record<B>>::into_item::<P>(self.gradients),self.state.into_item::<P>())
    }
    fn from_item<P: PrecisionSettings>(item: Self::Item<P>,device: &B::Device) -> Self {
        Self {version:item.0,model_state:R::from_item::<P>(item.1,device),
            contract:<TrainableParameterContract as Record<B>>::from_item::<P>(item.2,device),
            optimizer:<O::Record as Record<B>>::from_item::<P>(item.3,device),
            scheduler:<S::Record<B> as Record<B>>::from_item::<P>(item.4,device),
            gradients:<GradientsParamsRecord as Record<B>>::from_item::<P>(item.5,device),state:U::from_item::<P>(item.6,device),marker:PhantomData}
    }
}
