//! Local column/row shards using the model-parallel derivatives of ruda-autodiff.
use crate::{Linear, Embedding};
use ruda_autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor_parallel as region};
use region::BroadcastTensorCollective;
use ruda_model::{
    module::{Module, Param},
    tensor::{Tensor, Int, DType, TensorPrimitive, ElementConversion, backend::Backend, module::{linear, embedding}, activation::silu},
};

mod loss;
pub use loss::*;
mod uneven_vocabulary;
mod attention;
pub use attention::*;
mod feed_forward;
pub use feed_forward::*;
mod transformer;
pub use transformer::*;
mod encoder_decoder;
pub use encoder_decoder::*;
mod adapted;
pub use adapted::*;
mod partition;
pub use partition::*;
mod replicated_module;
pub use replicated_module::*;
mod head;
pub use head::*;
mod causal_loss;
mod kl_loss;
mod selection;
pub use selection::VocabParallelGreedySelection;
mod topk;
pub use topk::VocabParallelTopKSelection;
mod normalization_inference;
mod embeddings;
pub use embeddings::TensorParallelTransformerEmbeddings;
mod model;
pub use model::*;

/// Output-feature shard of a global projection, with an optional local bias shard.
/// Construct local weights directly or load an explicitly partitioned checkpoint.
#[derive(Module, Debug)]
pub struct ColumnParallelLinear<B: Backend> {
    /// Local weight `[global_input, local_output]` and local-output bias.
    pub local: Linear<B>,
}

/// Input-feature shard of a global projection, with a replicated output bias.
#[derive(Module, Debug)]
pub struct RowParallelLinear<B: Backend> {
    /// Local weight `[local_input, global_output]` and optional full-output bias.
    pub local: Linear<B>,
}

impl<B: Backend> ColumnParallelLinear<B> {
    /// Use caller-provided rank-local weights. Does not construct a full global weight.
    pub fn from_shard(local: Linear<B>) -> Self { Self { local } }
}
impl<B: Backend> RowParallelLinear<B> {
    /// Use caller-provided rank-local weights and, if present, identical replicated biases.
    pub fn from_shard(local: Linear<B>) -> Self { Self { local } }
}

impl<B: Backend, S: CheckpointStrategy> ColumnParallelLinear<Autodiff<B, S>> {
    /// Project a replicated input; sum its shard gradients and optionally gather output features.
    /// `gather_output=false` connects directly to a row-parallel layer's sharded input.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C, gather_output: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        assert!(D > 0, "linear projection requires a feature axis");
        let input = region::copy_to_region(input, communicator.clone())?;
        let output = self.local.forward(input);
        if gather_output { region::gather_from_region(output, communicator, D - 1) }
        else { Ok(output) }
    }
}

impl<B: Backend, S: CheckpointStrategy> RowParallelLinear<Autodiff<B, S>> {
    /// Project a rank-local input shard or explicitly scatter a replicated input.
    /// Sum partial outputs before adding bias, so bias is not multiplied by world size.
    /// Replicated losses must be identical on the ranks in this tensor-parallel group.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C, input_is_parallel: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        assert!(D > 0, "linear projection requires a feature axis");
        let input = if input_is_parallel { input }
            else { region::scatter_to_region(input, communicator.clone(), D - 1)? };
        let partial = linear(input, self.local.weight.val(), None);
        let output = region::reduce_from_region(partial, communicator)?;
        Ok(match &self.local.bias {
            Some(bias) => {
                let mut shape = [1; D];
                shape[D - 1] = bias.val().dims()[0];
                output + bias.val().reshape(shape)
            }
            None => output,
        })
    }
}

/// Column-parallel frozen base and local B, with a replicated trainable A.
#[derive(Module, Debug)]
pub struct ColumnParallelLoRA<B: Backend> {
    /// Frozen output-column base shard.
    pub base: ColumnParallelLinear<B>,
    /// Replicated `[global_input, rank]` adapter without bias.
    pub adapter_a: Linear<B>,
    /// Local `[rank, local_output]` adapter without bias.
    pub adapter_b: Linear<B>,
    /// Explicit alpha/rank or rank-stabilized multiplier.
    pub scale: f64,
}

/// Row-parallel frozen base and local A, with a replicated trainable B.
#[derive(Module, Debug)]
pub struct RowParallelLoRA<B: Backend> {
    /// Frozen input-row base shard, including any replicated output bias.
    pub base: RowParallelLinear<B>,
    /// Local `[local_input, rank]` adapter without bias.
    pub adapter_a: Linear<B>,
    /// Replicated `[rank, global_output]` adapter without bias.
    pub adapter_b: Linear<B>,
    /// Explicit adapter multiplier.
    pub scale: f64,
}

impl<B: Backend> ColumnParallelLoRA<B> {
    /// Attach already initialized compatible adapter shards; do not initialize a full base.
    pub fn from_shards(mut base: ColumnParallelLinear<B>, adapter_a: Linear<B>, adapter_b: Linear<B>, scale: f64) -> Self {
        let [input, output] = base.local.weight.val().dims();
        let [a_input, rank] = adapter_a.weight.val().dims();
        assert_eq!(input, a_input, "column adapter input width differs");
        assert_eq!(adapter_b.weight.val().dims(), [rank, output], "column adapter output shard differs");
        assert!(adapter_a.bias.is_none() && adapter_b.bias.is_none(), "LoRA adapters must be bias-free");
        assert!(rank > 0 && scale.is_finite(), "invalid adapter rank/scale");
        base.local = base.local.no_grad();
        Self { base, adapter_a, adapter_b, scale }
    }

    /// Consume a dropout-free adapter and merge only this rank's dense weight shard.
    pub fn merge(self) -> ColumnParallelLinear<B> {
        let update = self.adapter_a.weight.val().matmul(self.adapter_b.weight.val()).mul_scalar(self.scale).detach();
        let mut base = self.base;
        base.local.weight = base.local.weight.map(|weight| (weight + update).detach().set_require_grad(false));
        base
    }
}

impl<B: Backend> RowParallelLoRA<B> {
    /// Attach local A and replicated B without breaking the supplied base parameter IDs.
    pub fn from_shards(mut base: RowParallelLinear<B>, adapter_a: Linear<B>, adapter_b: Linear<B>, scale: f64) -> Self {
        let [input, output] = base.local.weight.val().dims();
        let [a_input, rank] = adapter_a.weight.val().dims();
        assert_eq!(input, a_input, "row adapter input shard differs");
        assert_eq!(adapter_b.weight.val().dims(), [rank, output], "row adapter output width differs");
        assert!(adapter_a.bias.is_none() && adapter_b.bias.is_none(), "LoRA adapters must be bias-free");
        assert!(rank > 0 && scale.is_finite(), "invalid adapter rank/scale");
        base.local = base.local.no_grad();
        Self { base, adapter_a, adapter_b, scale }
    }

    /// Merge the local input-row update, without gathering the complete base.
    pub fn merge(self) -> RowParallelLinear<B> {
        let update = self.adapter_a.weight.val().matmul(self.adapter_b.weight.val()).mul_scalar(self.scale).detach();
        let mut base = self.base;
        base.local.weight = base.local.weight.map(|weight| (weight + update).detach().set_require_grad(false));
        base
    }
}

impl<B: Backend, S: CheckpointStrategy> ColumnParallelLoRA<Autodiff<B, S>> {
    /// Project a replicated logical input with SUM derivatives for both input and A.
    /// Optional adapter input carries caller-applied synchronized dropout on the same device.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, adapter_input: Option<Tensor<Autodiff<B, S>, D>>,
        communicator: C, gather_output: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        let adapted = adapter_input.unwrap_or_else(|| input.clone());
        assert_eq!(adapted.dims(), input.dims(), "adapter input/dropout geometry differs");
        let base = self.base.forward(input, communicator.clone(), false)?;
        let dtype = base.dtype();
        let weight = region::copy_to_region(self.adapter_a.weight.val(), communicator.clone())?;
        let adapted = region::copy_to_region(adapted.cast(weight.dtype()), communicator.clone())?;
        let hidden = linear(adapted, weight, None);
        let hidden = hidden.cast(self.adapter_b.weight.val().dtype());
        let output = base + self.adapter_b.forward(hidden).mul_scalar(self.scale).cast(dtype);
        if gather_output { region::gather_from_region(output, communicator, D - 1) } else { Ok(output) }
    }
}

impl<B: Backend, S: CheckpointStrategy> RowParallelLoRA<Autodiff<B, S>> {
    /// Reduce rank activations before B; add the base bias only once after reduction.
    /// Caller dropout is either replicated before scatter or explicitly input-sharded.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, adapter_input: Option<Tensor<Autodiff<B, S>, D>>,
        communicator: C, input_is_parallel: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        let adapted = adapter_input.unwrap_or_else(|| input.clone());
        assert_eq!(adapted.dims(), input.dims(), "adapter input/dropout geometry differs");
        let (input, adapted) = if input_is_parallel { (input, adapted) } else {
            (region::scatter_to_region(input, communicator.clone(), D - 1)?,
             region::scatter_to_region(adapted, communicator.clone(), D - 1)?)
        };
        let base = self.base.forward(input, communicator.clone(), true)?;
        let dtype = base.dtype();
        let hidden = self.adapter_a.forward(adapted.cast(self.adapter_a.weight.val().dtype()));
        let hidden = region::reduce_from_region(hidden, communicator)?;
        let hidden = hidden.cast(self.adapter_b.weight.val().dtype());
        Ok(base + self.adapter_b.forward(hidden).mul_scalar(self.scale).cast(dtype))
    }
}

/// Gated MLP whose intermediate feature axis remains rank-local.
#[derive(Module, Debug)]
pub struct TensorParallelGatedMlp<B: Backend> {
    /// Local gate output columns.
    pub gate: ColumnParallelLinear<B>,
    /// Local value output columns.
    pub up: ColumnParallelLinear<B>,
    /// Local intermediate input rows and a replicated output bias.
    pub down: RowParallelLinear<B>,
}

impl<B: Backend> TensorParallelGatedMlp<B> {
    /// Connect explicit shards without guessing a model's feed-forward expansion.
    pub fn from_shards(gate: ColumnParallelLinear<B>, up: ColumnParallelLinear<B>, down: RowParallelLinear<B>) -> Self {
        assert_eq!(gate.local.weight.val().dims(), up.local.weight.val().dims(), "gate/value shard widths differ");
        assert_eq!(up.local.weight.val().dims()[1], down.local.weight.val().dims()[0], "MLP intermediate shard widths differ");
        Self { gate, up, down }
    }
}

impl<B: Backend, S: CheckpointStrategy> TensorParallelGatedMlp<Autodiff<B, S>> {
    /// SwiGLU forward with no full intermediate all-gather.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        self.forward_with(input, communicator, silu)
    }

    /// Use an explicitly chosen gate activation, retaining the same parallel derivatives.
    pub fn forward_with<C: BroadcastTensorCollective<B>, F, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C, activation: F,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error>
    where F: FnOnce(Tensor<Autodiff<B, S>, D>) -> Tensor<Autodiff<B, S>, D> {
        let gate = self.gate.forward(input.clone(), communicator.clone(), false)?;
        let value = self.up.forward(input, communicator.clone(), false)?;
        self.down.forward(activation(gate) * value, communicator, true)
    }
}

/// Vocabulary-sharded lookup with replicated token IDs and padding-gradient semantics.
#[derive(Module, Debug)]
pub struct VocabParallelEmbedding<B: Backend> {
    /// Local vocabulary rows.
    pub local: Embedding<B>,
    /// First global token ID owned by this shard.
    pub vocabulary_start: usize,
    /// Complete logical vocabulary size.
    pub vocabulary_size: usize,
    /// Optional global padding row; its value is retained and its gradient is suppressed.
    pub padding_index: Option<usize>,
}

impl<B: Backend> VocabParallelEmbedding<B> {
    /// Use supplied vocabulary rows; embeddings and heads can retain shared parameter IDs.
    pub fn from_shard(local: Embedding<B>, vocabulary_start: usize, vocabulary_size: usize, padding_index: Option<usize>) -> Self {
        assert!(vocabulary_size > 0 && vocabulary_size <= i64::MAX as usize, "invalid logical vocabulary");
        assert!(padding_index.is_none_or(|index| index < vocabulary_size), "invalid padding index");
        Self { local, vocabulary_start, vocabulary_size, padding_index }
    }
}

impl<B: Backend, S: CheckpointStrategy> VocabParallelEmbedding<Autodiff<B, S>> {
    /// Sum owned rows into a replicated output; repeated valid IDs accumulate gradients.
    pub fn forward<C: BroadcastTensorCollective<B>>(
        &self, tokens: Tensor<Autodiff<B, S>, 2, Int>, communicator: C,
    ) -> Result<Tensor<Autodiff<B, S>, 3>, C::Error> {
        let [local_vocab, features] = self.local.weight.val().dims();
        assert!(local_vocab > 0 && local_vocab.checked_mul(communicator.world_size() as usize) == Some(self.vocabulary_size), "vocabulary shards differ from topology");
        assert_eq!(self.vocabulary_start, local_vocab * communicator.rank() as usize, "vocabulary shard rank differs");
        let start = self.vocabulary_start as i64;
        let end = start + local_vocab as i64;
        let outside = tokens.clone().lower_elem(start).bool_or(tokens.clone().greater_equal_elem(end));
        let invalid = tokens.clone().lower_elem(0).bool_or(tokens.clone().greater_equal_elem(self.vocabulary_size as i64));
        let indices = tokens.sub_scalar(start).mask_fill(outside.clone(), 0).mask_fill(invalid, local_vocab as i64);
        let mut weight = self.local.weight.val();
        if let Some(padding) = self.padding_index.filter(|index| *index >= self.vocabulary_start && *index < self.vocabulary_start + local_vocab) {
            let rows = Tensor::<Autodiff<B, S>, 1, Int>::arange(0..local_vocab as i64, &weight.device());
            let mask = rows.equal_elem((padding - self.vocabulary_start) as i64).reshape([local_vocab, 1]).repeat_dim(1, features);
            weight = weight.clone().mask_where(mask, weight.detach());
        }
        let output = embedding(weight, indices).mask_fill(outside.unsqueeze_dim::<3>(2).repeat_dim(2, features), 0.0);
        region::reduce_from_region(output, communicator)
    }
}

/// Vocabulary head sharing `[local_vocab, hidden]` storage with its embedding.
#[derive(Module, Debug)]
pub struct VocabParallelProjection<B: Backend> {
    /// The exact embedding parameter and ID, not a separately transposed copy.
    pub weight: Param<Tensor<B, 2>>,
    /// Optional local-vocabulary bias.
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> VocabParallelProjection<B> {
    /// Retain embedding/head ties with a differentiable transpose only in forward.
    pub fn from_embedding(embedding: &VocabParallelEmbedding<B>, bias: Option<Param<Tensor<B, 1>>>) -> Self {
        if let Some(bias) = &bias { assert_eq!(bias.val().dims()[0], embedding.local.weight.val().dims()[0]); }
        Self { weight: embedding.local.weight.clone(), bias }
    }
}

impl<B: Backend, S: CheckpointStrategy> VocabParallelProjection<Autodiff<B, S>> {
    /// Compute local logits or explicitly gather the complete vocabulary axis.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C, gather_output: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        let input = region::copy_to_region(input, communicator.clone())?;
        let logits = linear(input, self.weight.val().transpose(), self.bias.as_ref().map(|bias| bias.val()));
        if gather_output { region::gather_from_region(logits, communicator, D - 1) } else { Ok(logits) }
    }
}

/// Token-wise full-vocabulary cross entropy with rank-local logits and derivatives.
/// Labels/token order must be identical across the TP group. Equal vocabulary shards
/// cover the complete vocabulary; FP32 softmax statistics are reduced, not logits.
/// The transport's existing gather obtains maxima with O(world * tokens) storage.
pub fn vocab_parallel_cross_entropy<B, S, C>(
    logits: Tensor<Autodiff<B, S>, 2>, labels: Tensor<Autodiff<B, S>, 1, Int>,
    communicator: C, ignore_index: i64, label_smoothing: f64,
) -> Result<Tensor<Autodiff<B, S>, 1>, C::Error>
where B: Backend, S: CheckpointStrategy, C: BroadcastTensorCollective<B> {
    let [tokens, local_vocab] = logits.dims();
    let world = communicator.world_size() as usize;
    let rank = communicator.rank() as usize;
    assert!(local_vocab > 0 && world > 0 && rank < world, "invalid vocabulary partition");
    assert_eq!(labels.dims()[0], tokens, "logit/target rows differ");
    assert!(label_smoothing.is_finite() && (0.0..=1.0).contains(&label_smoothing), "invalid label smoothing");
    let vocabulary = local_vocab.checked_mul(world).expect("vocabulary size overflow");
    assert!(vocabulary <= i64::MAX as usize, "vocabulary IDs exceed integer range");
    let ignored = labels.clone().equal_elem(ignore_index);
    let invalid = labels.clone().lower_elem(0).bool_or(labels.clone().greater_equal_elem(vocabulary as i64))
        .bool_and(ignored.clone().bool_not());
    assert!(!invalid.any().into_scalar().elem::<bool>(), "target lies outside the full vocabulary");
    let logits = logits.cast(DType::F32);
    if tokens == 0 { return Ok(logits.sum_dim(1).reshape([tokens])); }
    let local_max = logits.clone().detach().max_dim(1).inner();
    let maxima = communicator.all_gather_float(local_max.into_primitive().tensor())?;
    let maximum = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(maxima))
        .reshape([world, tokens]).max_dim(0).reshape([tokens, 1]);
    let maximum = Tensor::<Autodiff<B, S>, 2>::from_inner(maximum);
    let sum = (logits.clone() - maximum.clone()).exp().sum_dim(1);
    let sum = region::reduce_from_region(sum, communicator.clone())?;
    let normalizer = sum.log() + maximum;
    let start = (rank * local_vocab) as i64;
    let outside = labels.clone().lower_elem(start).bool_or(labels.clone().greater_equal_elem(start + local_vocab as i64))
        .bool_or(ignored.clone());
    let indices = labels.sub_scalar(start).mask_fill(outside.clone(), 0).reshape([tokens, 1]);
    let selected = logits.clone().gather(1, indices).mask_fill(outside.reshape([tokens, 1]), 0.0);
    let selected = region::reduce_from_region(selected, communicator.clone())?;
    let loss = if label_smoothing == 0.0 { normalizer - selected } else {
        let uniform = region::reduce_from_region(logits.sum_dim(1), communicator)?;
        normalizer - selected.mul_scalar(1.0 - label_smoothing) - uniform.mul_scalar(label_smoothing / vocabulary as f64)
    };
    Ok(loss.reshape([tokens]).mask_fill(ignored, 0.0))
}
