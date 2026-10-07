use alloc::vec::Vec;
use ruda_model::{module::Module,record::RecorderError,
    tensor::{Tensor,TensorPrimitive,Bool,Int,DType,backend::Backend}};
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy};
use crate::{Linear,Dropout,LoRALinear,attention::PackedSequenceLayout,
    pool::{pool_sequence,pool_packed_sequences,SequencePooling,SequencePoolOutput},
    transformer::{TransformerHead,AdaptedTransformerHead,TransformerHeadAdapterRecord,AdaptedProjection,SequenceHeadOutput}};
use super::{BroadcastTensorCollective,ColumnParallelLinear,VocabParallelLossLayout,VocabParallelCrossEntropy,
    partition_parallel_projection,TensorParallelProjectionAxis};

mod training;
mod vocabulary;
pub use vocabulary::VocabParallelTransformerHead;
mod inference;
mod dispatch;
pub use dispatch::TensorParallelOutputHead;
mod vocabulary_adapter;
pub use vocabulary_adapter::*;

/// Actual native Transformer head with rank-local output classes and replicated hidden states.
/// The supplied class layout and transport determine the logical vocabulary, never a model name.
#[derive(Module,Debug)]
pub struct TensorParallelTransformerHead<B:Backend> {
    /// Original local projection, replicated normalization and head-input dropout.
    pub local:TransformerHead<B>,
}

/// Actual native low-rank head: local frozen base/B outputs and replicated adapter A.
/// Norm/input dropout precede the original base and adapter branches as in the native head.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedTransformerHead<B:Backend> {
    /// Original native local head, retaining base/A/B IDs, dtypes, multiplier and dropout.
    pub local:AdaptedTransformerHead<B>,
}

fn check_projection<B:Backend>(projection:&Linear<B>,layout:&VocabParallelLossLayout,rank:usize) {
    let [hidden,classes] = projection.weight.val().dims();
    assert!(hidden > 0,"parallel head hidden width must be positive");
    assert_eq!(classes,layout.interval(rank).len(),"head output width differs from actual rank class storage");
    if let Some(bias) = &projection.bias {
        assert_eq!(bias.val().dims(),[classes],"parallel head bias width differs");
        assert_eq!(bias.val().device(),projection.weight.val().device(),"parallel head bias/weight devices differ");
    }
}

fn check_input<B:Backend,const D:usize>(input:&Tensor<B,D>,projection:&Linear<B>,
    layout:&VocabParallelLossLayout,rank:u32,world:u32) {
    assert!(D > 0,"parallel head requires a hidden feature axis");
    assert_eq!(world as usize,layout.world_size(),"parallel head topology/layout differ");
    check_projection(projection,layout,rank as usize);
    assert_eq!(input.dims()[D-1],projection.weight.val().dims()[0],"parallel head input hidden width differs");
    assert_eq!(input.device(),projection.weight.val().device(),"parallel head input/weight devices differ");
}

pub(super) fn transformed<B:Backend,F,const D:usize>(dropout:&Dropout,input:Tensor<B,D>,apply:F) -> Tensor<B,D>
    where F:FnOnce(&Dropout,Tensor<B,D>)->Tensor<B,D> {
    let shape = input.dims();let device = input.device();let dtype = input.dtype();let output = apply(dropout,input);
    assert!(output.dims() == shape && output.device() == device && output.dtype() == dtype,
        "parallel head dropout changed actual hidden shape/device/storage");
    output
}

fn mask_padding<B:Backend>(mut projection:Linear<B>,layout:&VocabParallelLossLayout,rank:usize) -> Linear<B> {
    let interval = layout.interval(rank);let width = interval.len();
    let real = interval.end.min(layout.vocabulary_size()).saturating_sub(interval.start);
    if real == width {return projection;}
    let hidden = projection.weight.val().dims()[0];
    assert!(!matches!(projection.weight.val().dtype(),DType::QFloat(_)),
        "packed head padding must be removed/repacked explicitly before loading native local shards");
    let padding = Tensor::<B,1,Int>::arange(0..width as i64,(&projection.weight.val().device(),DType::I64)).greater_equal_elem(real as i64);
    projection.weight = projection.weight.map(|weight|weight.mask_fill(padding.clone().reshape([1,width]).expand([hidden,width]),0));
    projection.bias = projection.bias.map(|bias|bias.map(|value|value.mask_fill(padding,0)));
    projection
}

fn adapted_projection<B:Backend>(mut projection:LoRALinear<B>,layout:&VocabParallelLossLayout,rank:usize) -> AdaptedProjection<B> {
    projection.base = mask_padding(projection.base,layout,rank);
    projection.adapter_b = mask_padding(projection.adapter_b,layout,rank);
    AdaptedProjection::LoRA(projection)
}

impl<B:Backend> TensorParallelTransformerHead<B> {
    /// Load declared local class columns without initialization, gathering or ID replacement.
    /// Packed frozen base storage is retained on shards containing only real classes.
    pub fn from_shard(local:TransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        check_projection(&local.projection,layout,rank);
        let local = TransformerHead::from_projection(local.projection,local.normalization,local.dropout);
        Self {local}
    }

    /// Slice actual loaded floating full output columns and matching bias, preserving IDs/flags.
    /// The full source includes exactly the declared storage columns; no padded weights are invented.
    pub fn from_full(mut head:TransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        assert_eq!(head.projection.weight.val().dims()[1],layout.storage_size(),"full head and declared class storage differ");
        head.projection = ColumnParallelLinear::from_full(head.projection,layout.interval(rank)).local;
        Self::from_shard(head,layout,rank)
    }

    /// Return the original local native head container without weight gathering.
    pub fn into_local_head(self) -> TransformerHead<B> {self.local}

    /// Native inference on rank-local classes; gathering real global logits is explicitly optional.
    /// Use the native inference backend/valid module to retain the original dropout mode.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<B,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,C::Error> {
        check_input(&hidden,&self.local.projection,layout,communicator.rank(),communicator.world_size());
        let hidden = if let Some(norm) = &self.local.normalization {norm.forward(hidden)} else {hidden};
        let projection = mask_padding(self.local.projection.clone(),layout,communicator.rank() as usize);
        let logits = projection.forward(self.local.dropout.forward(hidden));
        if gather_output {layout.gather_logits_inference(logits,communicator)} else {Ok(logits)}
    }
}

impl<B:Backend> TensorParallelAdaptedTransformerHead<B> {
    /// Load real native local base/B columns and replicated A without changing their roles/values.
    pub fn from_shard(local:AdaptedTransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        check_projection(&local.projection.base,layout,rank);
        let [hidden,width] = local.projection.base.weight.val().dims();let [a_hidden,adapter_rank] = local.projection.adapter_a.weight.val().dims();
        assert!(adapter_rank > 0 && local.projection.scale.is_finite(),"invalid parallel head adapter rank/scale");
        assert_eq!(hidden,a_hidden,"parallel head adapter A hidden width differs");
        assert_eq!(local.projection.adapter_b.weight.val().dims(),[adapter_rank,width],"parallel head adapter B class width differs");
        assert!(local.projection.adapter_a.bias.is_none() && local.projection.adapter_b.bias.is_none(),"native head adapters must be bias-free");
        assert!(local.projection.dropout.prob.is_finite() && (0.0..1.0).contains(&local.projection.dropout.prob),"invalid head adapter dropout");
        if let Some(norm) = &local.normalization {assert_eq!(norm.width(),hidden,"parallel head normalization width differs");}
        assert!(local.dropout.prob.is_finite() && (0.0..=1.0).contains(&local.dropout.prob),"invalid native head dropout");
        Self {local}
    }

    /// Slice original full base/B class columns, preserving native A, scales and both dropouts.
    pub fn from_full(mut head:AdaptedTransformerHead<B>,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        assert_eq!(head.projection.base.weight.val().dims()[1],layout.storage_size(),"full adapted head and class storage differ");
        let projection = partition_parallel_projection(AdaptedProjection::LoRA(head.projection),TensorParallelProjectionAxis::Column,layout.interval(rank));
        head.projection = match projection {AdaptedProjection::LoRA(projection)=>projection,AdaptedProjection::Dense(_)=>unreachable!()};
        Self::from_shard(head,layout,rank)
    }

    /// Return native local head values and flags without merging its adapter.
    pub fn into_local_head(self) -> AdaptedTransformerHead<B> {self.local}

    /// Native rank-local inference with real-class-only optional global logits gathering.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<B,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,C::Error> {
        check_input(&hidden,&self.local.projection.base,layout,communicator.rank(),communicator.world_size());
        let hidden = if let Some(norm) = &self.local.normalization {norm.forward(hidden)} else {hidden};
        let projection = adapted_projection(self.local.projection.clone(),layout,communicator.rank() as usize);
        let logits = projection.forward(self.local.dropout.forward(hidden));
        if gather_output {layout.gather_logits_inference(logits,communicator)} else {Ok(logits)}
    }

    /// Reuse the exact native local A/B-only record, rejecting omitted trainable norm state.
    pub fn adapter_record(&self,base_id:&str) -> Result<TransformerHeadAdapterRecord<B>,RecorderError> {self.local.adapter_record(base_id)}

    /// Restore this rank's original native adapter record without replacing its base or norm.
    pub fn restore_adapter(mut self,record:TransformerHeadAdapterRecord<B>,base_id:&str) -> Result<Self,RecorderError> {
        self.local = record.restore_into(self.local,base_id)?;Ok(self)
    }
}

impl VocabParallelLossLayout {
    /// Gather actual unequal native-backend logit shards, removing transport/storage padding.
    /// This deliberately materializes complete real logits and does not replace a sharded loss.
    pub fn gather_logits_inference<B,C,const D:usize>(&self,logits:Tensor<B,D>,communicator:C) -> Result<Tensor<B,D>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        assert!(D > 0,"native vocabulary gather requires a class axis");
        assert_eq!(communicator.world_size() as usize,self.world_size(),"native vocabulary gather topology/layout differ");
        let width = self.interval(communicator.rank() as usize).len();let mut shape = logits.dims();
        assert_eq!(shape[D-1],width,"native gather logits differ from actual rank storage");
        if shape[..D-1].contains(&0) {shape[D-1] = self.vocabulary_size();return Ok(logits.reshape(shape));}
        let maximum = (0..self.world_size()).map(|rank|self.interval(rank).len()).max().unwrap();
        let total = maximum.checked_mul(self.world_size()).expect("native padded vocabulary gather geometry overflow");
        let leading = logits.swap_dims(0,D-1);let mut padding_shape = leading.dims();padding_shape[0] = maximum-width;
        let padded = if width == maximum {leading} else {
            let zeros = Tensor::<B,D>::zeros(padding_shape,(&leading.device(),leading.dtype()));Tensor::cat(alloc::vec![leading,zeros],0)
        };
        let mut gathered_shape = padded.dims();gathered_shape[0] = total;
        let gathered = communicator.all_gather_float(padded.into_primitive().tensor())?;
        let gathered = Tensor::<B,D>::from_primitive(TensorPrimitive::Float(gathered));
        assert_eq!(gathered.dims(),gathered_shape,"native transport returned incompatible vocabulary geometry");
        let mut parts = Vec::new();
        for rank in 0..self.world_size() {
            let interval = self.interval(rank);if interval.start >= self.vocabulary_size() {break;}
            let real = interval.end.min(self.vocabulary_size())-interval.start;
            parts.push(gathered.clone().slice_dim(0,rank*maximum..rank*maximum+real));
        }
        Ok(Tensor::cat(parts,0).swap_dims(0,D-1))
    }
}
