use super::{SimpleOptimizer,Adam,AdamW,Sgd,AdaGrad,RmsProp,Adan};
use ruda_model::tensor::{Tensor,DType,BroadcastTensorCollective,backend::Backend};

/// An optimizer whose original update is independent for each coordinate of a parameter.
/// Matrix optimizers such as Muon must not implement this: independent fragments change their algorithm.
/// Tensor-wide clipping/normalization must retain original logical parameter geometry outside this update.
pub trait ElementwiseShardOptimizer<B:Backend>:SimpleOptimizer<B> {
    /// Reject embedded tensor-wide operations whose meaning changes on a local slice.
    fn validate_element_sharding(&self) -> Result<(),&'static str> {Ok(())}
    /// Explicit native dtype for already globally normalized and reduce-scattered local derivatives.
    fn shard_gradient_dtype(&self,storage:DType) -> DType {storage}
    /// Validate original configuration for native FSDP, where tensor-wide transforms can use logical gathers.
    /// Existing ZeRO-2 validation remains separate and does not implicitly enable local-fragment clipping.
    fn validate_fully_sharded_execution(&self) -> Result<(),&'static str> {self.validate_element_sharding()}
    /// Check original native history identity before updating or restoring this concrete configured algorithm.
    fn validate_fully_sharded_history(&self,_state:&Self::State<1>) -> Result<(),&'static str> {Ok(())}
    /// Original coordinate update over one actual native FSDP owner, with any wrapper-level logical transforms.
    fn step_fully_sharded<C:BroadcastTensorCollective<B>>(&self,lr:crate::LearningRate,tensor:Tensor<B,1>,gradient:Tensor<B,1>,
        state:Option<Self::State<1>>, _binding:&crate::FullyShardedOptimizerParameter<C>)
        -> Result<(Tensor<B,1>,Option<Self::State<1>>),crate::FullyShardedElementwiseError<C::Error>> {
        Ok(self.step(lr,tensor,gradient,state))
    }
}
impl<B:Backend> ElementwiseShardOptimizer<B> for Adam {}
impl<B:Backend> ElementwiseShardOptimizer<B> for AdamW {}
impl<B:Backend> ElementwiseShardOptimizer<B> for Sgd<B> {}
impl<B:Backend> ElementwiseShardOptimizer<B> for AdaGrad {}
impl<B:Backend> ElementwiseShardOptimizer<B> for RmsProp {}
impl<B:Backend> ElementwiseShardOptimizer<B> for Adan {}
