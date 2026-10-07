use super::*;
use alloc::{collections::BTreeMap,format,vec::Vec};
use crate::{Linear,transformer::AdaptedProjection};
use super::super::{TensorParallelAdaptedEncoderDecoderStack,TensorParallelAdaptedDecoderCrossAttention};
use ruda_model::{module::ParamId,record::RecorderError};

pub(super) trait AdapterTree<B:Backend> {
    fn adapters(&self) -> Vec<&Linear<B>>;
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F);
}

fn projection<'a,B:Backend>(projection:&'a AdaptedProjection<B>,values:&mut Vec<&'a Linear<B>>) {
    if let AdaptedProjection::LoRA(layer) = projection {values.push(&layer.adapter_a);values.push(&layer.adapter_b);}
}
fn map_projection<B:Backend,F:FnMut(&mut Linear<B>)>(projection:&mut AdaptedProjection<B>,apply:&mut F) {
    if let AdaptedProjection::LoRA(layer) = projection {apply(&mut layer.adapter_a);apply(&mut layer.adapter_b);}
}

impl<B:Backend> AdapterTree<B> for TensorParallelAdaptedStackLayer<B> {
    fn adapters(&self) -> Vec<&Linear<B>> {
        let mut values = Vec::new();
        if let Self::Adapted(block) = self {
            let attention = &block.attention.local;let feed = &block.feed_forward.local;
            for item in [&attention.query,&attention.key,&attention.value,&attention.output,&feed.up,&feed.down] {projection(item,&mut values);}
            if let Some(gate) = &feed.gate {projection(gate,&mut values);}
        }
        values
    }
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F) {
        if let Self::Adapted(block) = self {
            let attention = &mut block.attention.local;let feed = &mut block.feed_forward.local;
            for item in [&mut attention.query,&mut attention.key,&mut attention.value,&mut attention.output,&mut feed.up,&mut feed.down] {map_projection(item,apply);}
            if let Some(gate) = &mut feed.gate {map_projection(gate,apply);}
        }
    }
}

impl<B:Backend> AdapterTree<B> for TensorParallelAdaptedTransformerStack<B> {
    fn adapters(&self) -> Vec<&Linear<B>> {self.layers.iter().flat_map(|layer|layer.adapters()).collect()}
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F) {for layer in &mut self.layers {layer.map_adapters(apply);}}
}

impl<B:Backend> AdapterTree<B> for TensorParallelAdaptedDecoderCrossAttention<B> {
    fn adapters(&self) -> Vec<&Linear<B>> {
        let mut values = Vec::new();
        if let Self::Adapted(block) = self {
            let attention = &block.attention.local;
            for item in [&attention.query,&attention.key,&attention.value,&attention.output] {projection(item,&mut values);}
        }
        values
    }
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F) {
        if let Self::Adapted(block) = self {
            let attention = &mut block.attention.local;
            for item in [&mut attention.query,&mut attention.key,&mut attention.value,&mut attention.output] {map_projection(item,apply);}
        }
    }
}

impl<B:Backend> AdapterTree<B> for TensorParallelAdaptedEncoderDecoderStack<B> {
    fn adapters(&self) -> Vec<&Linear<B>> {
        let mut values = Vec::new();for layer in &self.layers {values.extend(layer.backbone.adapters());values.extend(layer.cross_attention.adapters());}values
    }
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F) {
        for layer in &mut self.layers {layer.backbone.map_adapters(apply);layer.cross_attention.map_adapters(apply);}
    }
}

impl<B:Backend> AdapterTree<B> for TensorParallelOutputHead<B> {
    fn adapters(&self) -> Vec<&Linear<B>> {
        match self {
            Self::AdaptedLinear(head)=>alloc::vec![&head.local.projection.adapter_a,&head.local.projection.adapter_b],
            Self::AdaptedVocabulary(head)=>alloc::vec![&head.projection.adapter_a,&head.projection.adapter_b],
            _=>Vec::new(),
        }
    }
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F) {
        match self {
            Self::AdaptedLinear(head)=>{apply(&mut head.local.projection.adapter_a);apply(&mut head.local.projection.adapter_b);},
            Self::AdaptedVocabulary(head)=>{apply(&mut head.projection.adapter_a);apply(&mut head.projection.adapter_b);},
            _=>{},
        }
    }
}

impl<B:Backend> AdapterTree<B> for TensorParallelTransformerModel<B> {
    fn adapters(&self) -> Vec<&Linear<B>> {let mut values = self.backbone.adapters();values.extend(self.head.adapters());values}
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F) {self.backbone.map_adapters(apply);self.head.map_adapters(apply);}
}
impl<B:Backend> AdapterTree<B> for TensorParallelEncoderDecoderModel<B> {
    fn adapters(&self) -> Vec<&Linear<B>> {let mut values = self.encoder.adapters();values.extend(self.decoder.adapters());values.extend(self.head.adapters());values}
    fn map_adapters<F:FnMut(&mut Linear<B>)>(&mut self,apply:&mut F) {self.encoder.map_adapters(apply);self.decoder.map_adapters(apply);self.head.map_adapters(apply);}
}

fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid parallel model adapter record: {reason}"))}

pub(super) fn capture<B:Backend,M:AdapterTree<B>>(model:&M) -> Result<Vec<usize>,RecorderError> {
    let mut seen:BTreeMap<(ParamId,bool),(usize,Tensor<B,2>)> = BTreeMap::new();let mut aliases = Vec::new();
    for adapter in model.adapters() {
        let value = adapter.weight.val();let key = (adapter.weight.id,value.is_require_grad());let index = aliases.len();
        if let Some((canonical,previous)) = seen.get(&key) {
            if previous.dims() != value.dims() || previous.dtype() != value.dtype() || previous.device() != value.device() {
                return Err(invalid("shared adapter IDs have incompatible geometry/storage/device"));
            }
            aliases.push(*canonical);
        } else {seen.insert(key,(index,value));aliases.push(index);}
    }
    Ok(aliases)
}

pub(super) fn rejoin<B:Backend,M:AdapterTree<B>>(mut model:M,expected:&[usize]) -> Result<M,RecorderError> {
    if capture(&model)?.as_slice() != expected {return Err(invalid("restored adapter identities no longer match original parameter sharing"));}
    let values:Vec<_> = model.adapters().iter().map(|adapter|adapter.weight.val()).collect();let mut index = 0usize;
    model.map_adapters(&mut |adapter| {let value = values[expected[index]].clone();adapter.weight = adapter.weight.clone().map(|_|value);index += 1;});
    Ok(model)
}
