use super::{Tensor,Int};
use crate::{TensorPrimitive,moe::{MoeOptions,MoeCombineGradientStrategy},moe_exchange::{MoeDispatchOps,MoeReceivedOps,MoeReceivedOptions}};

/// Actual typed native source-side dispatched rows and differentiable FP32 continuous weights.
#[derive(Debug)]
pub struct NativeMoeDispatched<B:MoeDispatchOps> {
    /// Original expert-sorted activation COPY rows.
    pub values:Tensor<B,2>,
    /// Original FP32 continuous weights with the original source logits graph.
    pub weights:Tensor<B,2>,
    /// Original selected native U32 IDs per source token.
    pub selected_experts:Tensor<B,2,Int>,
    /// Original native U32 expert IDs aligned with sorted assignment rows.
    pub row_experts:Tensor<B,1,Int>,
    /// Original valid private native dispatch and router metadata.
    pub state:B::MoeDispatchState,
}
/// Run actual original device routing and COPY dispatch before expert-parallel transport.
pub fn dispatch_moe<B:MoeDispatchOps>(input:Tensor<B,2>,logits:Tensor<B,2>,bias:Option<Tensor<B,1>>,options:MoeOptions)
    -> Result<NativeMoeDispatched<B>,B::MoeError> {
    let result=B::moe_dispatch(input.into_primitive().tensor(),logits.into_primitive().tensor(),bias.map(|value|value.into_primitive().tensor()),options)?;
    Ok(NativeMoeDispatched {values:Tensor::from_primitive(TensorPrimitive::Float(result.values)),weights:Tensor::from_primitive(TensorPrimitive::Float(result.weights)),
        selected_experts:Tensor::from_primitive(result.selected_experts),row_experts:Tensor::from_primitive(result.row_experts),state:result.state})
}
/// Apply original expert-ID-ordered combine after actual rows have returned to their original source.
pub fn combine_moe<B:MoeDispatchOps>(state:B::MoeDispatchState,expert_values:Tensor<B,2>,weights:Tensor<B,2>,strategy:MoeCombineGradientStrategy)
    -> Result<Tensor<B,2>,B::MoeError> {
    B::moe_combine(state,expert_values.into_primitive().tensor(),weights.into_primitive().tensor(),strategy)
        .map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
}
/// Execute only the actual local expert cubes and restore original source-rank receive order.
/// Native inference retains no backward cache; actual AD parents retain their required VJPs.
pub fn received_moe_experts<B:MoeReceivedOps>(input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,gate:Tensor<B,3>,up:Tensor<B,3>,down:Tensor<B,3>,options:MoeReceivedOptions)
    -> Result<Tensor<B,2>,B::MoeError> {
    B::moe_received_inference(input.into_primitive().tensor(),global_ids.into_primitive(),gate.into_primitive().tensor(),up.into_primitive().tensor(),down.into_primitive().tensor(),options)
        .map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
}
