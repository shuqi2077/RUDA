//! Text projections can own packed device weights without a dense shadow copy.
use super::*;
use std::sync::Arc;

pub(super) enum Projection<B: Backend> {
    Dense(Linear<B>),
    Packed(Arc<dyn Fn(B::FloatTensorPrimitive) -> B::FloatTensorPrimitive + Send + Sync>),
}
impl<B: Backend> Projection<B> {
    pub(super) fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Dense(linear) => linear.forward(input),
            Self::Packed(run) => Tensor::from_primitive(TensorPrimitive::Float(run(input.into_primitive().tensor()))),
        }
    }
    pub(super) fn dense(&self) -> Option<&Linear<B>> {
        match self { Self::Dense(linear) => Some(linear), Self::Packed(_) => None }
    }
}
