use super::*;
use ruda_model::{module::ModuleDisplay, tensor::IntegerTensorCollective};
use crate::{Mhc, attention::{LearnedKVCompressor, LightningIndexer, CompressedAttention,
    CompressedAttentionParts, CompressedAttentionProjection, SparseRotaryEmbedding}};

/// Actual offset/gate/value/norm leaves, retaining the source's distinct overlapping paths.
#[derive(Module, Debug)]
pub struct FullyShardedKVCompressor<B: Backend, P: Module<B>> {
    pub value: P,
    pub gate: P,
    pub position_bias: ShardedParameter<B>,
    pub norm_weight: ShardedParameter<B>,
    pub overlap: bool,
    pub epsilon: f64,
}

/// Original indexer projections, optional token-key norm and discrete selection configuration.
#[derive(Module, Debug)]
pub struct FullyShardedLightningIndexer<B: Backend, P: Module<B>> {
    pub query: P,
    pub head_weight: P,
    pub key: Option<P>,
    pub key_norm: Option<FullyShardedLayerNorm<B>>,
    pub rotary: SparseRotaryEmbedding,
    pub topk: usize,
    pub query_chunk_size: usize,
    pub key_chunk_size: usize,
    pub detach_inputs: bool,
}

/// Every actual CSA/HCA leaf has local persistent storage; completed-block attention remains native.
#[derive(Module, Debug)]
pub struct FullyShardedCompressedAttention<B: Backend, P: Module<B>> {
    pub query_down: P,
    pub query_up: P,
    pub query_norm: ShardedParameter<B>,
    pub local_kv: P,
    pub local_norm: ShardedParameter<B>,
    pub compressor: FullyShardedKVCompressor<B, P>,
    pub output_down: Vec<P>,
    pub output_up: P,
    pub sink: Option<ShardedParameter<B>>,
    pub indexer: Option<FullyShardedLightningIndexer<B, P>>,
    pub index_compressor: Option<FullyShardedKVCompressor<B, P>>,
    pub rotary: SparseRotaryEmbedding,
    pub width: usize,
    pub window_size: usize,
    pub query_chunk_size: usize,
    pub epsilon: f64,
}

/// Original dynamic gain/logit/mapping parameters, all using the actual data-group shard topology.
#[derive(Module, Debug)]
pub struct FullyShardedMhc<B: Backend> {
    pub mapping: ShardedParameter<B>,
    pub alpha: ShardedParameter<B>,
    pub bias: ShardedParameter<B>,
    pub width: usize,
    pub streams: usize,
    pub sinkhorn_iterations: usize,
    pub epsilon: f64,
}

impl<B: Backend> ShardingContext<B> {
    pub fn kv_compressor<P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>>(&mut self,
        source: LearnedKVCompressor<B, P>) -> FullyShardedKVCompressor<B, P::Sharded> {
        FullyShardedKVCompressor { value: source.value.shard(self), gate: source.gate.shard(self),
            position_bias: self.parameter(source.position_bias), norm_weight: self.parameter(source.norm_weight),
            overlap: source.overlap, epsilon: source.epsilon }
    }

    pub fn lightning_indexer<P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>>(&mut self,
        source: LightningIndexer<B, P>) -> FullyShardedLightningIndexer<B, P::Sharded> {
        FullyShardedLightningIndexer { query: source.query.shard(self), head_weight: source.head_weight.shard(self),
            key: source.key.map(|key| key.shard(self)), key_norm: source.key_norm.map(|norm| self.layer_norm(norm)),
            rotary: source.rotary, topk: source.topk, query_chunk_size: source.query_chunk_size,
            key_chunk_size: source.key_chunk_size, detach_inputs: source.detach_inputs }
    }

    /// Reuse this context for mHC/FFN/table/head leaves too, so each source tie has one actual local leaf.
    pub fn compressed_attention<P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>>(&mut self,
        source: CompressedAttention<B, P>) -> FullyShardedCompressedAttention<B, P::Sharded> {
        let parts = source.parts;
        FullyShardedCompressedAttention { query_down: parts.query_down.shard(self), query_up: parts.query_up.shard(self),
            query_norm: self.parameter(parts.query_norm), local_kv: parts.local_kv.shard(self), local_norm: self.parameter(parts.local_norm),
            compressor: self.kv_compressor(parts.compressor), output_down: parts.output_down.into_iter().map(|value| value.shard(self)).collect(),
            output_up: parts.output_up.shard(self), sink: parts.sink.map(|value| self.parameter(value)),
            indexer: parts.indexer.map(|value| self.lightning_indexer(value)),
            index_compressor: parts.index_compressor.map(|value| self.kv_compressor(value)), rotary: source.rotary, width: source.width,
            window_size: source.window_size, query_chunk_size: source.query_chunk_size, epsilon: source.epsilon }
    }

    pub fn mhc(&mut self, source: Mhc<B>) -> FullyShardedMhc<B> {
        FullyShardedMhc { mapping: self.parameter(source.mapping), alpha: self.parameter(source.alpha), bias: self.parameter(source.bias),
            width: source.width, streams: source.streams, sinkhorn_iterations: source.sinkhorn_iterations, epsilon: source.epsilon }
    }
}

macro_rules! gather_compressed {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> FullyShardedKVCompressor<$backend, P>
        where P::Gathered: CompressedAttentionProjection<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<LearnedKVCompressor<$backend, P::Gathered>, C::Error> {
                Ok(LearnedKVCompressor::from_parts(self.value.gather_projection(communicator.clone())?, self.gate.gather_projection(communicator.clone())?,
                    Param::initialized(self.position_bias.local.id, self.position_bias.$gather::<C, 2>(communicator.clone())?),
                    Param::initialized(self.norm_weight.local.id, self.norm_weight.$gather::<C, 1>(communicator)?), self.overlap, self.epsilon))
            }
        }

        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> FullyShardedLightningIndexer<$backend, P>
        where P::Gathered: CompressedAttentionProjection<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<LightningIndexer<$backend, P::Gathered>, C::Error> {
                Ok(LightningIndexer::from_parts(self.query.gather_projection(communicator.clone())?, self.head_weight.gather_projection(communicator.clone())?,
                    self.key.as_ref().map(|key| key.gather_projection(communicator.clone())).transpose()?,
                    self.key_norm.as_ref().map(|norm| norm.$gather(communicator)).transpose()?, self.rotary.clone(), self.topk,
                    self.query_chunk_size, self.key_chunk_size, self.detach_inputs))
            }
        }

        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> FullyShardedCompressedAttention<$backend, P>
        where P::Gathered: CompressedAttentionProjection<$backend> {
            /// Gather only this actual attention's weights; the original forward/selection/VJP expressions are reused.
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<CompressedAttention<$backend, P::Gathered>, C::Error> {
                let parts = CompressedAttentionParts { query_down: self.query_down.gather_projection(communicator.clone())?,
                    query_up: self.query_up.gather_projection(communicator.clone())?,
                    query_norm: Param::initialized(self.query_norm.local.id, self.query_norm.$gather::<C, 1>(communicator.clone())?),
                    local_kv: self.local_kv.gather_projection(communicator.clone())?,
                    local_norm: Param::initialized(self.local_norm.local.id, self.local_norm.$gather::<C, 1>(communicator.clone())?),
                    compressor: self.compressor.$gather(communicator.clone())?,
                    output_down: self.output_down.iter().map(|value| value.gather_projection(communicator.clone())).collect::<Result<_, _>>()?,
                    output_up: self.output_up.gather_projection(communicator.clone())?,
                    sink: self.sink.as_ref().map(|value| value.$gather::<C, 1>(communicator.clone()).map(|full| Param::initialized(value.local.id, full))).transpose()?,
                    indexer: self.indexer.as_ref().map(|value| value.$gather(communicator.clone())).transpose()?,
                    index_compressor: self.index_compressor.as_ref().map(|value| value.$gather(communicator)).transpose()? };
                Ok(CompressedAttention::from_parts(parts, self.rotary.clone(), self.window_size, self.query_chunk_size, self.epsilon))
            }
        }

        impl<$($generics)*> FullyShardedMhc<$backend> {
            pub fn $gather<C: BroadcastTensorCollective<B>>(&self, communicator: C) -> Result<Mhc<$backend>, C::Error> {
                Ok(Mhc { mapping: Param::initialized(self.mapping.local.id, self.mapping.$gather::<C, 2>(communicator.clone())?),
                    alpha: Param::initialized(self.alpha.local.id, self.alpha.$gather::<C, 1>(communicator.clone())?),
                    bias: Param::initialized(self.bias.local.id, self.bias.$gather::<C, 1>(communicator)?), width: self.width, streams: self.streams,
                    sinkhorn_iterations: self.sinkhorn_iterations, epsilon: self.epsilon })
            }
        }
    };
}
gather_compressed!(B, [B: Backend], gather_inference);
gather_compressed!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

macro_rules! visit_compressed {
    ($module:ident, [$($field:ident),+]) => {
        impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for $module<B, P> {
            fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_shards(visitor);)+ }
            fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_packed_shards(visitor);)+ }
        }
    };
}
visit_compressed!(FullyShardedKVCompressor, [value, gate, position_bias, norm_weight]);
visit_compressed!(FullyShardedLightningIndexer, [query, head_weight, key, key_norm]);
visit_compressed!(FullyShardedCompressedAttention, [query_down, query_up, query_norm, local_kv, local_norm, compressor,
    output_down, output_up, sink, indexer, index_compressor]);
impl<B: Backend> FullyShardedModule<B> for FullyShardedMhc<B> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) {
        self.mapping.visit_shards(visitor); self.alpha.visit_shards(visitor); self.bias.visit_shards(visitor);
    }
}

macro_rules! compressed_adapters {
    ($module:ident, [$($field:ident),+]) => {
        impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for $module<B, P> {
            fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_adapter_shards(visitor);)+ }
        }
    };
}
compressed_adapters!(FullyShardedKVCompressor, [value, gate]);
compressed_adapters!(FullyShardedLightningIndexer, [query, head_weight, key]);
compressed_adapters!(FullyShardedCompressedAttention, [query_down, query_up, local_kv, compressor, output_down, output_up, indexer, index_compressor]);
