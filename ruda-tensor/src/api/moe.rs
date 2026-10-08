use super::{Tensor,Int};
use crate::{TensorPrimitive,moe::{MoeOps,MoeRouterWeightOptions}};

/// Original native FP32 continuous router weights for fixed expert IDs. This does
/// not perform discrete selection or detach the actual source logits graph.
pub fn selected_router_weights<B:MoeOps>(logits:Tensor<B,2>,indices:Tensor<B,2,Int>,options:MoeRouterWeightOptions) -> Result<Tensor<B,2>,B::MoeError> {
    B::moe_selected_weights(logits.into_primitive().tensor(),indices.into_primitive(),options)
        .map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
}
/// Original native first-order VJP for a fixed selection. FP32 gradient weights
/// and original source-logits storage are required by the native kernel contract.
pub fn selected_router_weights_backward<B:MoeOps>(logits:Tensor<B,2>,indices:Tensor<B,2,Int>,gradient:Tensor<B,2>,options:MoeRouterWeightOptions)
    -> Result<Tensor<B,2>,B::MoeError> {
    B::moe_selected_weights_backward(logits.into_primitive().tensor(),indices.into_primitive(),gradient.into_primitive().tensor(),options)
        .map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
}
