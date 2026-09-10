use super::{MoeError, elements, empty, float_tensor, kernels};
use ruda_core::tensor::DType;
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, contiguous::into_contiguous},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingOptions {
    pub top_k: usize,
    pub renormalize: bool,
}

#[derive(Debug, Clone)]
pub struct RoutingPlan<R: Runtime> {
    pub(super) indices: RudaTensor<R>,
    pub(super) weights: RudaTensor<R>,
    pub(super) tokens: usize,
    pub(super) experts: usize,
    pub(super) top_k: usize,
}

impl<R: Runtime> RoutingPlan<R> {
    pub fn expert_indices(&self) -> &RudaTensor<R> {
        &self.indices
    }
    pub fn weights(&self) -> &RudaTensor<R> {
        &self.weights
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn experts(&self) -> usize {
        self.experts
    }
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

/// FP32 softmax over all experts, then top-k and optional selected-weight normalization.
/// Weights are cast to the logits dtype after normalization. Ties select lower expert IDs.
/// Rows containing NaN/+inf or only -inf retain NaN weights; no uniform fallback is applied.
pub fn route<R: Runtime>(
    logits: RudaTensor<R>,
    options: RoutingOptions,
) -> Result<RoutingPlan<R>, MoeError> {
    float_tensor(&logits)?;
    if logits.meta.num_dims() != 2 {
        return Err(MoeError("MoE logits must have shape [tokens, experts]"));
    }
    let tokens = logits.meta.shape()[0];
    let experts = logits.meta.shape()[1];
    if experts == 0 || options.top_k == 0 || options.top_k > experts {
        return Err(MoeError("MoE requires 1 <= top_k <= experts"));
    }
    elements(&[tokens, experts])?;
    elements(&[tokens, options.top_k])?;
    let indices = empty(&logits, [tokens, options.top_k], DType::U32);
    let weights = empty(&logits, [tokens, options.top_k], logits.dtype);
    if tokens != 0 {
        let logits = into_contiguous(logits);
        let probabilities = empty(&logits, [tokens, experts], DType::F32);
        let dim = RudaDim::new(logits.client.properties(), tokens);
        let count = calculate_ruda_count_elemwise(&logits.client, tokens, dim);
        kernels::routing::softmax::launch::<R>(
            &logits.client,
            count.clone(),
            dim,
            logits.clone().into_array_arg(),
            probabilities.clone().into_array_arg(),
            experts as u32,
            logits.dtype.into(),
        );
        kernels::routing::topk::launch::<R>(
            &logits.client,
            count,
            dim,
            probabilities.into_array_arg(),
            indices.clone().into_array_arg(),
            weights.clone().into_array_arg(),
            experts as u32,
            options.top_k as u32,
            u32::from(options.renormalize),
            logits.dtype.into(),
        );
    }
    Ok(RoutingPlan {
        indices,
        weights,
        tokens,
        experts,
        top_k: options.top_k,
    })
}
