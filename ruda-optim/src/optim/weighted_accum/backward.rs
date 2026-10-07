use core::fmt;
use ruda_model::{module::AutodiffModule,tensor::{DType,Tensor,TensorData,Transaction,
    backend::{AutodiffBackend,ExecutionError}}};
use super::{GradientsParams,WeightedAccumulationError,WeightedGradientsAccumulator};

/// Actual unscaled native loss statistics for one issued microbatch.
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct LossAccumulationReport {
    /// Actual numerator in its original computation dtype, read into host FP64.
    pub loss_sum: f64,
    /// Caller-supplied selected token/sample/class-weight denominator.
    pub effective_weight: f64,
    /// Zero-weight batches count as issued but do not execute backward.
    pub backward_performed: bool,
}

impl LossAccumulationReport {
    /// Actual batch mean; an empty-weight batch has no defined weighted mean.
    pub fn mean(&self) -> Option<f64> {
        if self.effective_weight > 0. {Some(self.loss_sum/self.effective_weight)} else {None}
    }
}

/// Invalid scalar-loss arguments, native readback or accumulation failure.
#[derive(Debug)]
pub enum LossAccumulationError {
    /// Numerator and denominator must be [1] tensors on the same device.
    InvalidGeometry,
    /// Native floating storage is required; quantized loss is not dequantized.
    InvalidDType,
    /// A backend failed while reading the two actual scalar statistics.
    Execution(ExecutionError),
    /// Existing effective-weight, counter or module-gradient contract failed.
    Accumulation(WeightedAccumulationError),
}

impl From<WeightedAccumulationError> for LossAccumulationError {
    fn from(error: WeightedAccumulationError) -> Self {Self::Accumulation(error)}
}

impl From<ExecutionError> for LossAccumulationError {
    fn from(error: ExecutionError) -> Self {Self::Execution(error)}
}

impl fmt::Display for LossAccumulationError {
    fn fmt(&self,f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry => f.write_str("loss sum and effective weight must be same-device [1] tensors"),
            Self::InvalidDType => f.write_str("loss sum and effective weight require native floating storage"),
            Self::Execution(error) => fmt::Display::fmt(error,f),
            Self::Accumulation(error) => fmt::Display::fmt(error,f),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LossAccumulationError {}

fn floating(dtype: DType) -> bool {
    matches!(dtype,DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64)
}

fn statistics<B: AutodiffBackend>(loss_sum: &Tensor<B,1>,effective_weight: Tensor<B,1>)
    -> Result<Transaction<B::InnerBackend>,LossAccumulationError> {
    if loss_sum.dims() != [1] || effective_weight.dims() != [1] || loss_sum.device() != effective_weight.device() {
        return Err(LossAccumulationError::InvalidGeometry);
    }
    if !floating(loss_sum.dtype()) || !floating(effective_weight.dtype()) {
        return Err(LossAccumulationError::InvalidDType);
    }
    Ok(Transaction::default().register(loss_sum.clone().inner()).register(effective_weight.inner()))
}

fn report(values: &[TensorData]) -> LossAccumulationReport {
    LossAccumulationReport {loss_sum:values[0].iter::<f64>().next().expect("loss sum scalar was validated"),
        effective_weight:values[1].iter::<f64>().next().expect("loss weight scalar was validated"),backward_performed:false}
}

impl<M> WeightedGradientsAccumulator<M> {
    /// Backpropagate a native SUM loss and accumulate using its actual effective weight.
    /// Compatible with native causal and unreduced classification/regression sums;
    /// the denominator is explicit and is not inferred from shape or batch size.
    /// This method applies this window's loss_scale exactly once. Supply an unscaled
    /// numerator, not a batch mean. Denominator gradients are intentionally not taken.
    /// Both statistics use one native read transaction before backward; backend FP64
    /// data is read as FP64 even when its static FloatElem type is FP32.
    /// No optimizer/scheduler/collective is called and no nonfinite loss is hidden.
    /// On targets without blocking readback use backward_sum_async instead.
    pub fn backward_sum<B: AutodiffBackend>(&mut self,module: &M,loss_sum: Tensor<B,1>,
        effective_weight: Tensor<B,1>) -> Result<LossAccumulationReport,LossAccumulationError>
        where M: AutodiffModule<B> {
        let values = statistics(&loss_sum,effective_weight)?.try_execute()?;
        self.backward_report::<B>(module,loss_sum,report(&values))
    }

    /// Async readback variant of backward_sum with identical scaling and counters.
    /// Readback failure leaves the pending window unchanged and performs no backward.
    /// Backward computation itself follows the backend's normal execution semantics.
    pub async fn backward_sum_async<B: AutodiffBackend>(&mut self,module: &M,loss_sum: Tensor<B,1>,
        effective_weight: Tensor<B,1>) -> Result<LossAccumulationReport,LossAccumulationError>
        where M: AutodiffModule<B> {
        let values = statistics(&loss_sum,effective_weight)?.execute_async().await?;
        self.backward_report::<B>(module,loss_sum,report(&values))
    }

    fn backward_report<B: AutodiffBackend>(&mut self,module: &M,loss_sum: Tensor<B,1>,mut report: LossAccumulationReport)
        -> Result<LossAccumulationReport,LossAccumulationError> where M: AutodiffModule<B> {
        self.next_counts(report.effective_weight)?;
        let gradients = if report.effective_weight == 0. {GradientsParams::new()} else {
            let scaled = loss_sum.cast(DType::from(self.state.dtype)).mul_scalar(self.state.loss_scale);
            report.backward_performed = true;
            GradientsParams::from_grads(scaled.backward(),module)
        };
        self.accumulate_sum::<B>(module,&gradients,report.effective_weight)?;
        Ok(report)
    }
}
