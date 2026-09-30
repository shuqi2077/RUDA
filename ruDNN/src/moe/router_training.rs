//! Continuous router weights for a FIXED expert selection. Selection/grouping
//! remains the model's responsibility. No CPU readback or full probability tensor.
use super::{elements, empty, float_tensor, kernels, MoeError, RoutingPlan};
use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScoring { Softmax, Sigmoid }

#[derive(Debug, Clone, Copy)]
pub struct RouterWeightOptions {
    pub scoring: RouterScoring,
    pub renormalize: bool,
    /// Applied AFTER optional selected-weight normalization; finite and positive.
    pub scale: f32,
}

impl RouterWeightOptions {
    pub fn validate(&self) -> Result<(), MoeError> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(MoeError("router scale must be finite positive FP32"));
        }
        Ok(())
    }
}

fn same_queue<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>) -> Result<(), MoeError> {
    if !b.is_contiguous() || b.qparams.is_some() || a.device.to_id() != b.device.to_id()
        || !a.client.same_execution_queue(&b.client) {
        return Err(MoeError("router operands require contiguous, same-device, same-queue storage"));
    }
    elements(b.meta.shape())?;
    Ok(())
}

fn validate<R: Runtime>(logits: &RudaTensor<R>, indices: &RudaTensor<R>, options: RouterWeightOptions)
    -> Result<(usize, usize, usize, u32), MoeError>
{
    options.validate()?;
    float_tensor(logits)?;
    same_queue(logits, logits)?;
    same_queue(logits, indices)?;
    if logits.meta.num_dims() != 2 || indices.meta.num_dims() != 2
        || !matches!(indices.dtype, DType::U32 | DType::I32 | DType::I64) {
        return Err(MoeError("router requires [tokens,experts] logits and [tokens,top_k] integer indices"));
    }
    let (tokens, experts, k) = (logits.meta.shape()[0], logits.meta.shape()[1], indices.meta.shape()[1]);
    if experts == 0 || indices.meta.shape()[0] != tokens || k == 0 || k > experts || k > 64 {
        return Err(MoeError("router requires 1 <= top_k <= min(experts,64) and matching token counts"));
    }
    let hw = &logits.client.properties().hardware;
    let lanes = hw.plane_size_max;
    if !matches!(lanes, 32 | 64) || lanes > hw.max_ruda_dim.0 || tokens > hw.max_ruda_count.0 as usize {
        return Err(MoeError("router requires a full 32/64-lane plane and a legal row grid"));
    }
    Ok((tokens, experts, k, lanes))
}

/// Returns FP32 weights, including for low-precision logits. Repeated indices
/// use gather semantics: each occurrence contributes to the denominator.
/// Invalid device indices produce NaN for the ENTIRE row, without reading an
/// invalid address. No host bounds-check synchronization is inserted.
/// Zero normalization denominator/non-finite softmax arithmetic is not repaired.
pub fn selected_router_weights<R: Runtime>(logits: &RudaTensor<R>, indices: &RudaTensor<R>,
    options: RouterWeightOptions) -> Result<RudaTensor<R>, MoeError>
{
    validate(logits, indices, options)?;
    let output = empty(logits, indices.meta.shape().to_vec(), DType::F32);
    // SAFETY: freshly allocated output does not alias the read-only operands.
    unsafe { selected_router_weights_into(logits, indices, &output, options)?; }
    Ok(output)
}

/// # Safety
/// `output` must be disjoint from both inputs and not accessed concurrently.
/// References must remain valid through submission; the ordered runtime owns
/// the submitted allocation handles until their work is complete.
pub unsafe fn selected_router_weights_into<R: Runtime>(logits: &RudaTensor<R>, indices: &RudaTensor<R>,
    output: &RudaTensor<R>, options: RouterWeightOptions) -> Result<(), MoeError>
{
    let (tokens, experts, k, lanes) = validate(logits, indices, options)?;
    same_queue(logits, output)?;
    if output.meta.shape() != indices.meta.shape() || output.dtype != DType::F32 {
        return Err(MoeError("router weight output must be matching FP32 storage"));
    }
    if tokens != 0 {
        kernels::router_training::weights::launch::<R>(
            &logits.client, RudaCount::Static(tokens as u32, 1, 1), RudaDim::new_1d(lanes),
            logits.clone().into_array_arg(), indices.clone().into_array_arg(), output.clone().into_array_arg(),
            experts, k, lanes as usize, options.scoring == RouterScoring::Softmax,
            options.renormalize, options.scale, [logits.dtype.into(), indices.dtype.into()]);
    }
    Ok(())
}

/// First-order vector-Jacobian product. `grad_weights` is FP32 and gradient
/// output has the logits dtype. No floating atomics or [tokens,experts] FP32
/// probability/seed workspace; probabilities are recomputed in registers.
pub fn selected_router_backward<R: Runtime>(logits: &RudaTensor<R>, indices: &RudaTensor<R>,
    grad_weights: &RudaTensor<R>, options: RouterWeightOptions) -> Result<RudaTensor<R>, MoeError>
{
    validate(logits, indices, options)?;
    let output = empty(logits, logits.meta.shape().to_vec(), logits.dtype);
    unsafe { selected_router_backward_into(logits, indices, grad_weights, &output, options)?; }
    Ok(output)
}

/// # Safety
/// Output must be disjoint from logits, indices and grad_weights, without
/// concurrent access. Inputs must be unchanged from the matching forward.
pub unsafe fn selected_router_backward_into<R: Runtime>(logits: &RudaTensor<R>, indices: &RudaTensor<R>,
    grad_weights: &RudaTensor<R>, output: &RudaTensor<R>, options: RouterWeightOptions)
    -> Result<(), MoeError>
{
    let (tokens, experts, k, lanes) = validate(logits, indices, options)?;
    same_queue(logits, grad_weights)?; same_queue(logits, output)?;
    if grad_weights.meta.shape() != indices.meta.shape() || grad_weights.dtype != DType::F32
        || output.meta.shape() != logits.meta.shape() || output.dtype != logits.dtype {
        return Err(MoeError("router backward gradient shape/dtype mismatch"));
    }
    if tokens != 0 {
        kernels::router_training::backward::launch::<R>(
            &logits.client, RudaCount::Static(tokens as u32, 1, 1), RudaDim::new_1d(lanes),
            logits.clone().into_array_arg(), indices.clone().into_array_arg(), grad_weights.clone().into_array_arg(),
            output.clone().into_array_arg(), experts, k, lanes as usize,
            options.scoring == RouterScoring::Softmax, options.renormalize, options.scale,
            [logits.dtype.into(), indices.dtype.into()]);
    }
    Ok(())
}

/// Holds the same logits/selection for forward and backward. Rust callers must
/// not mutate shared input handles before backward (PyTorch enforces versions).
#[derive(Debug, Clone)]
pub struct RouterTrainingPlan<R: Runtime> {
    plan: RoutingPlan<R>,
    logits: RudaTensor<R>,
    options: RouterWeightOptions,
}
impl<R: Runtime> RoutingPlan<R> {
    /// Keep existing top-k/group/correction-bias selection, but recompute its
    /// continuous weights in FP32 using the explicit supplied scoring policy.
    /// Correction bias is NOT added to logits here: it is selection-only.
    pub fn into_training(mut self, logits: RudaTensor<R>, options: RouterWeightOptions)
        -> Result<RouterTrainingPlan<R>, MoeError>
    {
        if logits.meta.shape() != [self.tokens, self.experts] {
            return Err(MoeError("router training logits do not match routing plan"));
        }
        self.weights = selected_router_weights(&logits, &self.indices, options)?;
        Ok(RouterTrainingPlan { plan: self, logits, options })
    }
}
impl<R: Runtime> RouterTrainingPlan<R> {
    pub fn routing(&self) -> &RoutingPlan<R> { &self.plan }
    pub fn backward(&self, grad_weights: &RudaTensor<R>) -> Result<RudaTensor<R>, MoeError> {
        selected_router_backward(&self.logits, &self.plan.indices, grad_weights, self.options)
    }
}
