use alloc::string::ToString;
use core::fmt;
use ruda_model::{
    module::AutodiffModule,
    record::{PrecisionSettings,Record,RecorderError},
    tensor::{DType,FloatDType,backend::{AutodiffBackend,Backend}},
};
use serde::{Deserialize,Serialize};
use super::{GradientsAccumulator,GradientsParams,GradientsParamsRecord,GradientTransformError};
use super::gradient_transform::{representable,validate_work_dtype};

/// Explicit normalization and continuation state of an accumulation window.
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
pub struct WeightedAccumulationState {
    /// F32/F64 arithmetic used before adding each microbatch's gradients.
    #[serde(serialize_with="serialize_dtype",deserialize_with="deserialize_dtype")]
    pub dtype: FloatDType,
    /// Loss multiplier actually applied by the caller throughout this window.
    pub loss_scale: f64,
    /// Sum of caller-supplied effective token/sample weights, not batch count.
    pub total_weight: f64,
    /// Number of accepted microbatches, including explicit zero-weight batches.
    pub microbatches: u64,
}

fn serialize_dtype<S: serde::Serializer>(dtype: &FloatDType,serializer: S) -> Result<S::Ok,S::Error> {
    DType::from(*dtype).serialize(serializer)
}

fn deserialize_dtype<'de,D: serde::Deserializer<'de>>(deserializer: D) -> Result<FloatDType,D::Error> {
    match DType::deserialize(deserializer)? {
        DType::F32 => Ok(FloatDType::F32),
        DType::F64 => Ok(FloatDType::F64),
        _ => Err(serde::de::Error::custom("weighted accumulation dtype must be F32 or F64")),
    }
}

impl<B: Backend> Record<B> for WeightedAccumulationState {
    type Item<S: PrecisionSettings> = Self;
    fn into_item<S: PrecisionSettings>(self) -> Self { self }
    fn from_item<S: PrecisionSettings>(item: Self,_device: &B::Device) -> Self { item }
}

/// Pending gradients and their actual normalization state, saved together.
#[derive(Clone,Debug)]
pub struct WeightedGradientsRecord {
    /// Values keyed by the original model IDs; full precision settings recommended.
    pub gradients: GradientsParamsRecord,
    /// Work dtype, loss scale, effective weight and pending microbatch count.
    pub state: WeightedAccumulationState,
}

impl<B: Backend> Record<B> for WeightedGradientsRecord {
    type Item<S: PrecisionSettings> = (GradientsParamsRecord,WeightedAccumulationState);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (<GradientsParamsRecord as Record<B>>::into_item::<S>(self.gradients),self.state)
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self { gradients:<GradientsParamsRecord as Record<B>>::from_item::<S>(item.0,device),state:item.1 }
    }
}

/// Explicit accumulation argument, geometry or checkpoint error.
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum WeightedAccumulationError {
    /// Invalid work dtype, scalar or gradient metadata.
    Gradient(GradientTransformError),
    /// Effective weights must be nonnegative, finite and representable.
    InvalidWeight,
    /// The actual microbatch counter cannot accept another increment.
    CounterOverflow,
    /// A mean requires positive total weight; the pending window remains intact.
    EmptyWeight,
    /// Stored counters/options describe an inconsistent accumulation window.
    InvalidState,
}

impl From<GradientTransformError> for WeightedAccumulationError {
    fn from(error: GradientTransformError) -> Self { Self::Gradient(error) }
}

impl fmt::Display for WeightedAccumulationError {
    fn fmt(&self,f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gradient(error) => fmt::Display::fmt(error,f),
            Self::InvalidWeight => f.write_str("effective accumulation weight must be finite, nonnegative and representable"),
            Self::CounterOverflow => f.write_str("accumulation microbatch counter overflow"),
            Self::EmptyWeight => f.write_str("cannot normalize an accumulation window with zero effective weight"),
            Self::InvalidState => f.write_str("inconsistent weighted accumulation checkpoint state"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WeightedAccumulationError {}

/// Result of an explicitly completed accumulation window.
pub struct AccumulatedGradients {
    /// Present gradients only; globally/local unused parameters are not invented.
    pub gradients: GradientsParams,
    /// Actual counts and loss scale of the window that produced these gradients.
    pub state: WeightedAccumulationState,
}

/// Backend-independent unequal-microbatch accumulation with resumable weights.
///
/// The caller supplies gradients of either local loss sums or local loss means,
/// plus the effective weight used by that loss. Loss scaling is fixed at setup;
/// it is never inferred from values. No optimizer, scheduler, data iterator,
/// dynamic scaler, clipping or distributed reduction is advanced implicitly.
pub struct WeightedGradientsAccumulator<M> {
    accumulator: GradientsAccumulator<M>,
    state: WeightedAccumulationState,
}

impl<M> WeightedGradientsAccumulator<M> {
    /// Start an empty window. Use one loss scale for every forward/backward in it.
    pub fn new(dtype: FloatDType,loss_scale: f64) -> Result<Self,WeightedAccumulationError> {
        let state = WeightedAccumulationState { dtype,loss_scale,total_weight:0.,microbatches:0 };
        validate_state(&state)?;
        Ok(Self { accumulator:GradientsAccumulator::new(),state })
    }

    /// Current actual counters; inspecting them does not clear gradients.
    pub fn state(&self) -> &WeightedAccumulationState { &self.state }

    pub(crate) fn inner(&self) -> &GradientsAccumulator<M> { &self.accumulator }

    /// Reattach saved normalization state to already-restored pending gradients.
    /// The supplied module must have the same IDs, dimensions and device.
    pub fn from_accumulator<B: AutodiffBackend>(
        module: &M,accumulator: GradientsAccumulator<M>,state: WeightedAccumulationState,
    ) -> Result<Self,WeightedAccumulationError> where M: AutodiffModule<B> {
        validate_state(&state)?;
        if state.total_weight == 0. && !accumulator.pending().is_empty() {
            return Err(WeightedAccumulationError::InvalidState);
        }
        let values = accumulator.pending().cast_for::<B,M>(module,state.dtype)?;
        let mut restored = GradientsAccumulator::new();
        restored.accumulate_with_dtype::<B>(module,values,state.dtype);
        Ok(Self {accumulator:restored,state})
    }

    /// Accumulate gradients of a caller's weighted loss SUM.
    /// `weight` is its effective token/sample count, not the loss multiplier.
    /// An explicit zero-weight batch contributes no gradients but counts as issued.
    pub fn accumulate_sum<B: AutodiffBackend>(
        &mut self,module: &M,gradients: &GradientsParams,weight: f64,
    ) -> Result<(),WeightedAccumulationError> where M: AutodiffModule<B> {
        self.accumulate::<B>(module,gradients,weight,false)
    }

    /// Accumulate a MEAN loss by multiplying gradients by its actual weight
    /// after work-dtype conversion. Unequal batch/token counts are not averaged equally.
    pub fn accumulate_mean<B: AutodiffBackend>(
        &mut self,module: &M,gradients: &GradientsParams,weight: f64,
    ) -> Result<(),WeightedAccumulationError> where M: AutodiffModule<B> {
        self.accumulate::<B>(module,gradients,weight,true)
    }

    fn accumulate<B: AutodiffBackend>(
        &mut self,module: &M,gradients: &GradientsParams,weight: f64,mean: bool,
    ) -> Result<(),WeightedAccumulationError> where M: AutodiffModule<B> {
        if weight < 0. || !representable(weight,self.state.dtype) ||
            (weight > 0. && self.state.dtype == FloatDType::F32 && weight as f32 == 0.) {
            return Err(WeightedAccumulationError::InvalidWeight);
        }
        let total = self.state.total_weight + weight;
        if !representable(total,self.state.dtype) { return Err(WeightedAccumulationError::InvalidWeight); }
        let count = self.state.microbatches.checked_add(1).ok_or(WeightedAccumulationError::CounterOverflow)?;
        // Validate every supplied ID before any scaling/addition. A rejected
        // argument leaves pending gradients and normalization counters untouched.
        gradients.validate_for::<B,M>(module)?;
        if weight > 0. {
            let multiplier = if mean { weight } else { 1. };
            let values = gradients.scaled_for::<B,M>(module,multiplier,self.state.dtype)?;
            self.accumulator.accumulate_with_dtype::<B>(module,values,self.state.dtype);
        }
        self.state.total_weight = total;
        self.state.microbatches = count;
        Ok(())
    }

    /// Return the actual scaled loss sums and their local weight, then reset.
    /// Useful before DDP's existing sum/weight reduction; no communication occurs.
    /// Divide the reduced gradients by `state.loss_scale` exactly once yourself.
    pub fn finish_sums(&mut self) -> AccumulatedGradients {
        let state = self.state.clone();
        let gradients = self.accumulator.grads();
        self.state.total_weight = 0.;
        self.state.microbatches = 0;
        AccumulatedGradients {gradients,state}
    }

    /// Normalize by loss scale and total effective weight, then clear the window.
    /// Division is sequential, avoiding an overflowing scale*weight product.
    /// Zero total weight returns an error without clearing anything.
    pub fn finish_mean<B: AutodiffBackend>(
        &mut self,module: &M,
    ) -> Result<AccumulatedGradients,WeightedAccumulationError> where M: AutodiffModule<B> {
        if self.state.total_weight <= 0. { return Err(WeightedAccumulationError::EmptyWeight); }
        let gradients = self.accumulator.pending().unscaled_for::<B,M>(
            module,self.state.loss_scale,self.state.dtype)?.unscaled_for::<B,M>(
            module,self.state.total_weight,self.state.dtype)?;
        let state = self.state.clone();
        self.accumulator.grads();
        self.state.total_weight = 0.;
        self.state.microbatches = 0;
        Ok(AccumulatedGradients {gradients,state})
    }

    /// Snapshot actual gradient values/options/counters without resetting.
    pub fn try_to_record<B: AutodiffBackend>(&self) -> Result<WeightedGradientsRecord,RecorderError>
    where M: AutodiffModule<B> {
        Ok(WeightedGradientsRecord {gradients:self.accumulator.try_to_record::<B>()?,state:self.state.clone()})
    }

    /// Snapshot with asynchronous device readback; source position remains caller state.
    pub async fn to_record_async<B: AutodiffBackend>(&self) -> Result<WeightedGradientsRecord,RecorderError>
    where M: AutodiffModule<B> {
        Ok(WeightedGradientsRecord {gradients:self.accumulator.to_record_async::<B>().await?,state:self.state.clone()})
    }

    /// Restore counters and pending values together for the checkpoint's model IDs.
    /// Invalid state/membership leaves this accumulator unchanged. No batch replay.
    pub fn load_record<B: AutodiffBackend>(
        &mut self,module: &M,record: WeightedGradientsRecord,device: &B::Device,
    ) -> Result<(),RecorderError> where M: AutodiffModule<B> {
        validate_state(&record.state).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        let gradients = GradientsParams::from_record::<B::InnerBackend>(record.gradients,device)?;
        if record.state.total_weight == 0. && !gradients.is_empty() {
            return Err(RecorderError::Unknown(WeightedAccumulationError::InvalidState.to_string()));
        }
        gradients.validate_for::<B,M>(module).map_err(|error|RecorderError::Unknown(error.to_string()))?;
        let mut restored = GradientsAccumulator::new();
        let values = gradients.cast_for::<B,M>(module,record.state.dtype)
            .map_err(|error|RecorderError::Unknown(error.to_string()))?;
        restored.accumulate_with_dtype::<B>(module,values,record.state.dtype);
        self.accumulator = restored;
        self.state = record.state;
        Ok(())
    }
}

fn validate_state(state: &WeightedAccumulationState) -> Result<(),WeightedAccumulationError> {
    validate_work_dtype(state.dtype)?;
    if !representable(state.loss_scale,state.dtype) || state.loss_scale <= 0. ||
        (state.dtype == FloatDType::F32 && state.loss_scale as f32 == 0.) {
        return Err(GradientTransformError::InvalidScalar.into());
    }
    if state.total_weight < 0. || !representable(state.total_weight,state.dtype) ||
        (state.total_weight > 0. && state.dtype == FloatDType::F32 && state.total_weight as f32 == 0.) ||
        (state.microbatches == 0 && state.total_weight != 0.) {
        return Err(WeightedAccumulationError::InvalidState);
    }
    Ok(())
}
