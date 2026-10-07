use super::{SimpleOptimizer,Adam,AdamW,Sgd,AdaGrad,RmsProp,Adan};
use ruda_model::tensor::{DType,backend::Backend};

/// An optimizer whose original update is independent for each coordinate of a parameter.
/// Matrix optimizers such as Muon must not implement this: independent fragments change their algorithm.
/// Tensor-wide clipping/normalization must retain original logical parameter geometry outside this update.
pub trait ElementwiseShardOptimizer<B:Backend>:SimpleOptimizer<B> {
    /// Reject embedded tensor-wide operations whose meaning changes on a local slice.
    fn validate_element_sharding(&self) -> Result<(),&'static str> {Ok(())}
    /// Explicit native dtype for already globally normalized and reduce-scattered local derivatives.
    fn shard_gradient_dtype(&self,storage:DType) -> DType {storage}
}
impl<B:Backend> ElementwiseShardOptimizer<B> for Adam {}
impl<B:Backend> ElementwiseShardOptimizer<B> for AdamW {}
impl<B:Backend> ElementwiseShardOptimizer<B> for Sgd<B> {}
impl<B:Backend> ElementwiseShardOptimizer<B> for AdaGrad {}
impl<B:Backend> ElementwiseShardOptimizer<B> for RmsProp {}
impl<B:Backend> ElementwiseShardOptimizer<B> for Adan {}
