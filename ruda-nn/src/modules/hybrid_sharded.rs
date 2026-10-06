//! Native TP projections whose persistent model/state storage is data-sharded.
use ruda_autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor_parallel as region};
use ruda_model::{module::{Module, Param}, tensor::{Tensor, Int, backend::Backend, module::linear, activation::silu}};
use region::BroadcastTensorCollective;
use super::{fully_sharded::{ShardedParameter, FullyShardedLinear}, tensor_parallel};

/// Preserve errors from the separate data and tensor transports.
#[derive(Debug)]
pub enum HybridParallelError<D, T> {
    /// Parameter gather on the data axis failed.
    Data(D),
    /// Projection communication on the tensor axis failed.
    Tensor(T),
}

/// Output-column TP projection with FSDP slices of its local matrix and bias.
#[derive(Module, Debug)]
pub struct FullyShardedColumnParallelLinear<B: Backend> {
    /// Data slices of `[global_input, local_output]` and local-output bias.
    pub local: FullyShardedLinear<B>,
}

/// Input-row TP projection with data slices of its matrix and replicated bias.
#[derive(Module, Debug)]
pub struct FullyShardedRowParallelLinear<B: Backend> {
    /// Data slices of `[local_input, global_output]` and full-output bias.
    pub local: FullyShardedLinear<B>,
}

impl<B: Backend> FullyShardedColumnParallelLinear<B> {
    /// Partition a caller-loaded TP shard, preserving its parameter identities.
    pub fn from_tensor_shard(layer: tensor_parallel::ColumnParallelLinear<B>, data_rank: usize, data_world: usize) -> Self {
        Self { local: FullyShardedLinear::from_full(layer.local, data_rank, data_world) }
    }
}

impl<B: Backend> FullyShardedRowParallelLinear<B> {
    /// Partition the local input rows; the bias remains replicated only on TP.
    pub fn from_tensor_shard(layer: tensor_parallel::RowParallelLinear<B>, data_rank: usize, data_world: usize) -> Self {
        Self { local: FullyShardedLinear::from_full(layer.local, data_rank, data_world) }
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedColumnParallelLinear<Autodiff<B, S>> {
    /// Gather only this TP matrix; input derivatives sum on TP, weight derivatives on DP.
    /// Independent data losses must be normalized by their global weight before backward.
    pub fn forward<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, data: C, tensor: T, gather_output: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, HybridParallelError<C::Error, T::Error>> {
        assert!(D > 0, "projection needs a feature axis");
        let input = region::copy_to_region(input, tensor.clone()).map_err(HybridParallelError::Tensor)?;
        let output = self.local.forward(input, data).map_err(HybridParallelError::Data)?;
        if gather_output { region::gather_from_region(output, tensor, D - 1).map_err(HybridParallelError::Tensor) }
        else { Ok(output) }
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedRowParallelLinear<Autodiff<B, S>> {
    /// Reduce partial outputs before adding the data-gathered full-output bias once.
    pub fn forward<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, data: C, tensor: T, input_is_parallel: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, HybridParallelError<C::Error, T::Error>> {
        assert!(D > 0, "projection needs a feature axis");
        let input = if input_is_parallel { input } else {
            region::scatter_to_region(input, tensor.clone(), D - 1).map_err(HybridParallelError::Tensor)?
        };
        let weight = self.local.weight.gather::<C, 2>(data.clone()).map_err(HybridParallelError::Data)?;
        let partial = linear(input, weight, None);
        let output = region::reduce_from_region(partial, tensor).map_err(HybridParallelError::Tensor)?;
        Ok(match &self.local.bias {
            Some(bias) => {
                let bias = bias.gather::<C, 1>(data).map_err(HybridParallelError::Data)?;
                let mut shape = [1; D];
                shape[D - 1] = bias.dims()[0];
                output + bias.reshape(shape)
            }
            None => output,
        })
    }
}

/// FSDP/TP LoRA with local B columns and a TP-replicated, data-sharded A.
#[derive(Module, Debug)]
pub struct FullyShardedColumnParallelLoRA<B: Backend> {
    /// Frozen base storage, already partitioned on both axes.
    pub base: FullyShardedColumnParallelLinear<B>,
    /// Data slices of the replicated `[global_input, rank]` adapter.
    pub adapter_a: FullyShardedLinear<B>,
    /// Data slices of the local `[rank, local_output]` adapter.
    pub adapter_b: FullyShardedLinear<B>,
    /// Caller-selected adapter multiplier.
    pub scale: f64,
}

/// FSDP/TP LoRA with local A rows and a TP-replicated, data-sharded B.
#[derive(Module, Debug)]
pub struct FullyShardedRowParallelLoRA<B: Backend> {
    /// Frozen base matrix and its data-sharded replicated output bias.
    pub base: FullyShardedRowParallelLinear<B>,
    /// Data slices of the `[local_input, rank]` adapter.
    pub adapter_a: FullyShardedLinear<B>,
    /// Data slices of the `[rank, global_output]` adapter.
    pub adapter_b: FullyShardedLinear<B>,
    /// Caller-selected adapter multiplier.
    pub scale: f64,
}

impl<B: Backend> FullyShardedColumnParallelLoRA<B> {
    /// Consume an initialized TP adapter without gathering or reinitializing its base.
    pub fn from_tensor_shard(layer: tensor_parallel::ColumnParallelLoRA<B>, rank: usize, world: usize) -> Self {
        Self { base: FullyShardedColumnParallelLinear::from_tensor_shard(layer.base, rank, world),
            adapter_a: FullyShardedLinear::from_full(layer.adapter_a, rank, world),
            adapter_b: FullyShardedLinear::from_full(layer.adapter_b, rank, world), scale: layer.scale }
    }
}

impl<B: Backend> FullyShardedRowParallelLoRA<B> {
    /// Consume compatible TP adapter shards, retaining frozen/trainable flags and IDs.
    pub fn from_tensor_shard(layer: tensor_parallel::RowParallelLoRA<B>, rank: usize, world: usize) -> Self {
        Self { base: FullyShardedRowParallelLinear::from_tensor_shard(layer.base, rank, world),
            adapter_a: FullyShardedLinear::from_full(layer.adapter_a, rank, world),
            adapter_b: FullyShardedLinear::from_full(layer.adapter_b, rank, world), scale: layer.scale }
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedColumnParallelLoRA<Autodiff<B, S>> {
    /// Gather data slices of A before its TP-SUM derivative; mixed-precision adapters retain the base output dtype.
    /// Optional adapter input is caller-applied synchronized dropout, not a new random mask.
    pub fn forward<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, adapter_input: Option<Tensor<Autodiff<B, S>, D>>,
        data: C, tensor: T, gather_output: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, HybridParallelError<C::Error, T::Error>> {
        assert!(D > 0, "adapter needs a feature axis");
        let adapted = adapter_input.unwrap_or_else(|| input.clone());
        assert_eq!(adapted.dims(), input.dims(), "adapter input geometry differs");
        let base = self.base.forward(input, data.clone(), tensor.clone(), false)?;
        let dtype = base.dtype();
        let a = self.adapter_a.weight.gather::<C, 2>(data.clone()).map_err(HybridParallelError::Data)?;
        let adapted = region::copy_to_region(adapted.cast(a.dtype()), tensor.clone()).map_err(HybridParallelError::Tensor)?;
        let a = region::copy_to_region(a, tensor.clone()).map_err(HybridParallelError::Tensor)?;
        let hidden = linear(adapted, a, None).cast(self.adapter_b.weight.local.val().dtype());
        let update = self.adapter_b.forward(hidden, data).map_err(HybridParallelError::Data)?;
        let output = base + update.mul_scalar(self.scale).cast(dtype);
        if gather_output { region::gather_from_region(output, tensor, D - 1).map_err(HybridParallelError::Tensor) }
        else { Ok(output) }
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedRowParallelLoRA<Autodiff<B, S>> {
    /// Sum A's TP-local activations before B; DP gather derivatives return only local data slices.
    pub fn forward<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, adapter_input: Option<Tensor<Autodiff<B, S>, D>>,
        data: C, tensor: T, input_is_parallel: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, HybridParallelError<C::Error, T::Error>> {
        assert!(D > 0, "adapter needs a feature axis");
        let adapted = adapter_input.unwrap_or_else(|| input.clone());
        assert_eq!(adapted.dims(), input.dims(), "adapter input geometry differs");
        let (input, adapted) = if input_is_parallel { (input, adapted) } else {
            (region::scatter_to_region(input, tensor.clone(), D - 1).map_err(HybridParallelError::Tensor)?,
             region::scatter_to_region(adapted, tensor.clone(), D - 1).map_err(HybridParallelError::Tensor)?)
        };
        let base = self.base.forward(input, data.clone(), tensor.clone(), true)?;
        let dtype = base.dtype();
        let adapted = adapted.cast(self.adapter_a.weight.local.val().dtype());
        let hidden = self.adapter_a.forward(adapted, data.clone()).map_err(HybridParallelError::Data)?;
        let hidden = region::reduce_from_region(hidden, tensor).map_err(HybridParallelError::Tensor)?;
        let hidden = hidden.cast(self.adapter_b.weight.local.val().dtype());
        let update = self.adapter_b.forward(hidden, data).map_err(HybridParallelError::Data)?;
        Ok(base + update.mul_scalar(self.scale).cast(dtype))
    }
}

/// Native gated MLP retaining both data storage shards and local TP intermediate activations.
#[derive(Module, Debug)]
pub struct FullyShardedTensorParallelGatedMlp<B: Backend> {
    /// Local gate output-column storage.
    pub gate: FullyShardedColumnParallelLinear<B>,
    /// Local value output-column storage.
    pub up: FullyShardedColumnParallelLinear<B>,
    /// Local intermediate input-row storage.
    pub down: FullyShardedRowParallelLinear<B>,
}

impl<B: Backend> FullyShardedTensorParallelGatedMlp<B> {
    /// Partition an existing, explicitly connected TP MLP over its data group.
    pub fn from_tensor_shard(layer: tensor_parallel::TensorParallelGatedMlp<B>, rank: usize, world: usize) -> Self {
        Self { gate: FullyShardedColumnParallelLinear::from_tensor_shard(layer.gate, rank, world),
            up: FullyShardedColumnParallelLinear::from_tensor_shard(layer.up, rank, world),
            down: FullyShardedRowParallelLinear::from_tensor_shard(layer.down, rank, world) }
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedTensorParallelGatedMlp<Autodiff<B, S>> {
    /// SwiGLU with a TP-local intermediate and data-reduce-scattered parameter gradients.
    pub fn forward<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, data: C, tensor: T,
    ) -> Result<Tensor<Autodiff<B, S>, D>, HybridParallelError<C::Error, T::Error>> {
        self.forward_with(input, data, tensor, silu)
    }

    /// Apply an explicit alternative gate activation without changing partition semantics.
    pub fn forward_with<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>, F, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, data: C, tensor: T, activation: F,
    ) -> Result<Tensor<Autodiff<B, S>, D>, HybridParallelError<C::Error, T::Error>>
    where F: FnOnce(Tensor<Autodiff<B, S>, D>) -> Tensor<Autodiff<B, S>, D> {
        let gate = self.gate.forward(input.clone(), data.clone(), tensor.clone(), false)?;
        let up = self.up.forward(input, data.clone(), tensor.clone(), false)?;
        self.down.forward(activation(gate) * up, data, tensor, true)
    }
}

/// FSDP slices of a TP-local vocabulary table.
#[derive(Module, Debug)]
pub struct FullyShardedVocabParallelEmbedding<B: Backend> {
    /// Data-sharded local `[local_vocabulary, hidden]` table.
    pub weight: ShardedParameter<B>,
    /// First logical vocabulary row owned by this tensor coordinate.
    pub vocabulary_start: usize,
    /// Complete vocabulary size.
    pub vocabulary_size: usize,
    /// Optional global row whose gradient is suppressed.
    pub padding_index: Option<usize>,
}

impl<B: Backend> FullyShardedVocabParallelEmbedding<B> {
    /// Keep existing vocabulary ownership while partitioning its local table on DP.
    pub fn from_tensor_shard(layer: tensor_parallel::VocabParallelEmbedding<B>, rank: usize, world: usize) -> Self {
        Self { weight: ShardedParameter::from_full(layer.local.weight, rank, world),
            vocabulary_start: layer.vocabulary_start, vocabulary_size: layer.vocabulary_size, padding_index: layer.padding_index }
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedVocabParallelEmbedding<Autodiff<B, S>> {
    /// Gather only local vocabulary rows; reuse the native lookup and TP derivative.
    pub fn forward<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>>(
        &self, tokens: Tensor<Autodiff<B, S>, 2, Int>, data: C, tensor: T,
    ) -> Result<Tensor<Autodiff<B, S>, 3>, HybridParallelError<C::Error, T::Error>> {
        let weight = self.weight.gather::<C, 2>(data).map_err(HybridParallelError::Data)?;
        let layer = tensor_parallel::VocabParallelEmbedding {
            local: crate::Embedding { weight: Param::initialized(self.weight.local.id, weight) },
            vocabulary_start: self.vocabulary_start, vocabulary_size: self.vocabulary_size, padding_index: self.padding_index,
        };
        layer.forward(tokens, tensor).map_err(HybridParallelError::Tensor)
    }
}

/// Vocabulary projection retaining the exact embedding's local flat parameter and ID.
#[derive(Module, Debug)]
pub struct FullyShardedVocabParallelProjection<B: Backend> {
    /// The same physical storage as its sharded embedding.
    pub weight: ShardedParameter<B>,
    /// Optional data-sharded local-vocabulary bias.
    pub bias: Option<ShardedParameter<B>>,
}

impl<B: Backend> FullyShardedVocabParallelProjection<B> {
    /// Tie the embedding/head storage; no transpose is retained as a second parameter.
    pub fn from_embedding(embedding: &FullyShardedVocabParallelEmbedding<B>, bias: Option<ShardedParameter<B>>) -> Self {
        if let Some(bias) = &bias {
            assert_eq!(bias.logical_shape.as_slice(), &[embedding.weight.logical_shape[0]], "vocabulary bias width differs");
            assert_eq!((bias.rank, bias.world_size), (embedding.weight.rank, embedding.weight.world_size), "bias data topology differs");
        }
        Self { weight: embedding.weight.clone(), bias }
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedVocabParallelProjection<Autodiff<B, S>> {
    /// Gather DP slices and then reuse the TP head's native differentiable transpose.
    pub fn forward<C: BroadcastTensorCollective<B>, T: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, data: C, tensor: T, gather_output: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, HybridParallelError<C::Error, T::Error>> {
        assert!(D > 0, "projection needs a feature axis");
        let weight = self.weight.gather::<C, 2>(data.clone()).map_err(HybridParallelError::Data)?;
        let bias = match &self.bias {
            Some(bias) => Some(Param::initialized(bias.local.id, bias.gather::<C, 1>(data).map_err(HybridParallelError::Data)?)),
            None => None,
        };
        let layer = tensor_parallel::VocabParallelProjection { weight: Param::initialized(self.weight.local.id, weight), bias };
        layer.forward(input, tensor, gather_output).map_err(HybridParallelError::Tensor)
    }
}
