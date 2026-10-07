use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::{module::Module,tensor::{Tensor,backend::Backend,module::linear}};
use crate::{Dropout,Linear,transformer::AdaptedProjection};
use super::{AttentionParallelGroups,BroadcastTensorCollective};

mod attention;
mod feed_forward;
mod transformer;
mod stack;
mod packed;
pub use attention::*;
pub use feed_forward::*;
pub use transformer::*;
pub use stack::*;

fn geometry<B: Backend>(layer: &AdaptedProjection<B>) -> [usize;2] {
    match layer {AdaptedProjection::Dense(layer)=>layer.weight.val().dims(),AdaptedProjection::LoRA(layer)=>layer.base.weight.val().dims()}
}

fn bias<B: Backend,const D: usize>(input: Tensor<B,D>,bias: Option<Tensor<B,1>>) -> Tensor<B,D> {
    if let Some(bias) = bias {let mut shape = [1;D];shape[D-1] = bias.dims()[0];input+bias.reshape(shape)} else {input}
}

fn sum_inference<B,C,const D: usize>(value: Tensor<B,D>,communicator: &C) -> Result<Tensor<B,D>,C::Error>
    where B: Backend,C: BroadcastTensorCollective<B> {
    Ok(Tensor::from_primitive(ruda_model::tensor::TensorPrimitive::Float(communicator.all_reduce_sum(value.into_primitive().tensor())?)))
}

fn row_inference<B,C,const D: usize>(layer: &AdaptedProjection<B>,input: Tensor<B,D>,communicator: &C) -> Result<Tensor<B,D>,C::Error>
    where B: Backend,C: BroadcastTensorCollective<B> {
    let base_layer = match layer {AdaptedProjection::Dense(layer)=>layer,AdaptedProjection::LoRA(layer)=>&layer.base};
    let base = bias(sum_inference(linear(input.clone(),base_layer.weight.val(),None),communicator)?,base_layer.bias.as_ref().map(|bias|bias.val()));
    if let AdaptedProjection::LoRA(layer) = layer {
        let storage = base.dtype();
        let adapted = layer.dropout.forward(input.cast(layer.adapter_a.weight.val().dtype()));
        let hidden = sum_inference(linear(adapted,layer.adapter_a.weight.val(),None),communicator)?;
        let hidden = bias(hidden,layer.adapter_a.bias.as_ref().map(|bias|bias.val())).cast(layer.adapter_b.weight.val().dtype());
        Ok(base+layer.adapter_b.forward(hidden).mul_scalar(layer.scale).cast(storage))
    } else {Ok(base)}
}

fn replicated<B,S,C,const D: usize>(value: Tensor<Autodiff<B,S>,D>,communicator: &C)
    -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
    // A frozen quantized value must keep its original native storage, not enter
    // the float-only derivative region through implicit full dequantization.
    if value.is_require_grad() {region::copy_to_region(value,communicator.clone())} else {Ok(value)}
}

fn column_dense<B,S,C,K,const D: usize>(layer: &Linear<Autodiff<B,S>>,input: Tensor<Autodiff<B,S>,D>,
    communicator: &C,replicas: Option<&K>) -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
    let input = region::copy_to_region(input,communicator.clone())?;
    let mut weight = layer.weight.val();let mut bias = layer.bias.as_ref().map(|bias|bias.val());
    if let Some(replicas) = replicas {
        weight = replicated(weight,replicas)?;
        bias = bias.map(|bias|replicated(bias,replicas)).transpose()?;
    }
    Ok(linear(input,weight,bias))
}

fn column<B,S,C,K,F,const D: usize>(layer: &AdaptedProjection<Autodiff<B,S>>,input: Tensor<Autodiff<B,S>,D>,
    communicator: &C,replicas: Option<&K>,dropout: F) -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
        F: FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
    match layer {
        AdaptedProjection::Dense(layer) => column_dense(layer,input,communicator,replicas),
        AdaptedProjection::LoRA(layer) => {
            let source = input.clone().cast(layer.adapter_a.weight.val().dtype());let shape = source.dims();let device = source.device();let dtype = source.dtype();
            let adapted = dropout(&layer.dropout,source);
            assert!(adapted.dims() == shape && adapted.device() == device && adapted.dtype() == dtype,"parallel adapter dropout changed actual input geometry/device/storage");
            let base = column_dense(&layer.base,input,communicator,replicas)?;let storage = base.dtype();
            let adapted = region::copy_to_region(adapted,communicator.clone())?;
            let a = replicated(layer.adapter_a.weight.val(),communicator)?;
            let a_bias = layer.adapter_a.bias.as_ref().map(|bias|replicated(bias.val(),communicator)).transpose()?;
            let hidden = linear(adapted,a,a_bias).cast(layer.adapter_b.weight.val().dtype());
            let mut b = layer.adapter_b.weight.val();let mut b_bias = layer.adapter_b.bias.as_ref().map(|bias|bias.val());
            if let Some(replicas) = replicas {
                b = replicated(b,replicas)?;
                b_bias = b_bias.map(|bias|replicated(bias,replicas)).transpose()?;
            }
            Ok(base+linear(hidden,b,b_bias).mul_scalar(layer.scale).cast(storage))
        }
    }
}

fn row_dense<B,S,C,const D: usize>(layer: &Linear<Autodiff<B,S>>,input: Tensor<Autodiff<B,S>,D>,communicator: &C)
    -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
    let partial = linear(input,layer.weight.val(),None);
    Ok(bias(region::reduce_from_region(partial,communicator.clone())?,layer.bias.as_ref().map(|bias|bias.val())))
}

fn row<B,S,C,F,const D: usize>(layer: &AdaptedProjection<Autodiff<B,S>>,input: Tensor<Autodiff<B,S>,D>,communicator: &C,dropout: F)
    -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B>,F: FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
    match layer {
        AdaptedProjection::Dense(layer) => row_dense(layer,input,communicator),
        AdaptedProjection::LoRA(layer) => {
            let source = input.clone().cast(layer.adapter_a.weight.val().dtype());let shape = source.dims();let device = source.device();let dtype = source.dtype();
            let adapted = dropout(&layer.dropout,source);
            assert!(adapted.dims() == shape && adapted.device() == device && adapted.dtype() == dtype,"parallel output adapter dropout changed actual local input");
            let base = row_dense(&layer.base,input,communicator)?;let storage = base.dtype();
            let hidden = linear(adapted,layer.adapter_a.weight.val(),None);
            let hidden = region::reduce_from_region(hidden,communicator.clone())?;
            let hidden = bias(hidden,layer.adapter_a.bias.as_ref().map(|bias|bias.val())).cast(layer.adapter_b.weight.val().dtype());
            Ok(base+layer.adapter_b.forward(hidden).mul_scalar(layer.scale).cast(storage))
        }
    }
}
