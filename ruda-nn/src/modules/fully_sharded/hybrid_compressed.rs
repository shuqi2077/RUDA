use super::*;
use ruda_model::{module::ModuleDisplay, tensor::{Bool, IntegerTensorCollective}};
use crate::{Embedding, attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout},
    transformer::{MhcTransformerBlock, HybridAttentionBackbone, HybridAttentionLanguageModel, HybridAttentionHead,
        HybridTiedEmbeddingAdapter, normalize_mhc, mhc_visible}};

#[derive(Module, Debug)]
pub struct FullyShardedMhcTransformerBlock<B: Backend, P: Module<B>> {
    pub attention_connection: FullyShardedMhc<B>,
    pub ffn_connection: FullyShardedMhc<B>,
    pub attention: FullyShardedCompressedAttention<B, P>,
    pub attention_norm: ShardedParameter<B>,
    pub ffn_norm: ShardedParameter<B>,
    pub gate: P,
    pub up: P,
    pub down: P,
    pub epsilon: f64,
}
#[derive(Module, Debug)]
pub struct FullyShardedHybridBackbone<B: Backend, P: Module<B>> {
    pub embedding: FullyShardedEmbedding<B>,
    pub layers: Vec<FullyShardedMhcTransformerBlock<B, P>>,
    pub final_norm: ShardedParameter<B>,
    pub epsilon: f64,
}
#[derive(Module, Debug)]
pub struct FullyShardedHybridTiedAdapter<B: Backend> {
    pub adapter_a: FullyShardedLinear<B>,
    pub adapter_b: FullyShardedLinear<B>,
    pub dropout: crate::Dropout,
    pub scale: f64,
}
#[derive(Module, Debug)]
pub enum FullyShardedHybridHead<B: Backend, P: Module<B>> {
    Linear(P),
    TiedEmbedding(core::marker::PhantomData<B>),
    TiedEmbeddingLoRA(FullyShardedHybridTiedAdapter<B>),
}
#[derive(Module, Debug)]
pub struct FullyShardedHybridLanguageModel<B: Backend, P: Module<B>> {
    pub backbone: FullyShardedHybridBackbone<B, P>,
    pub head: FullyShardedHybridHead<B, P>,
}

/// One transient actual output head; tied variants borrow the gathered table's original ID/layout.
#[derive(Debug)]
pub struct GatheredHybridHead<B: Backend, P: Module<B>> {
    head: HybridAttentionHead<B, P>,
    embedding: Option<Embedding<B>>,
}
impl<B: Backend, P: CompressedAttentionProjection<B>> GatheredHybridHead<B, P> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.head { HybridAttentionHead::Linear(value) => value.forward(hidden),
            _ => self.head.forward(hidden, self.embedding.as_ref().expect("actual tied head table")) }
    }
    pub fn project(&self, rows: Tensor<B, 2>) -> Tensor<B, 2> {
        let [count, width] = rows.dims();
        let logits = self.forward(rows.reshape([count, 1, width]));
        let vocab = logits.dims()[2];
        logits.reshape([count, vocab])
    }
}

impl<B: Backend> ShardingContext<B> {
    pub fn mhc_transformer_block<P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>>(&mut self,
        source: MhcTransformerBlock<B, P>) -> FullyShardedMhcTransformerBlock<B, P::Sharded> {
        FullyShardedMhcTransformerBlock { attention_connection: self.mhc(source.attention_connection), ffn_connection: self.mhc(source.ffn_connection),
            attention: self.compressed_attention(source.attention), attention_norm: self.parameter(source.attention_norm), ffn_norm: self.parameter(source.ffn_norm),
            gate: source.gate.shard(self), up: source.up.shard(self), down: source.down.shard(self), epsilon: source.epsilon }
    }
    pub fn hybrid_backbone<P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>>(&mut self,
        source: HybridAttentionBackbone<B, P>) -> FullyShardedHybridBackbone<B, P::Sharded> {
        FullyShardedHybridBackbone { embedding: self.embedding(source.embedding), layers: source.layers.into_iter().map(|value| self.mhc_transformer_block(value)).collect(),
            final_norm: self.parameter(source.final_norm), epsilon: source.epsilon }
    }
    pub fn hybrid_language_model<P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>>(&mut self,
        source: HybridAttentionLanguageModel<B, P>) -> FullyShardedHybridLanguageModel<B, P::Sharded> {
        let backbone = self.hybrid_backbone(source.backbone);
        let head = match source.head {
            HybridAttentionHead::Linear(value) => FullyShardedHybridHead::Linear(value.shard(self)),
            HybridAttentionHead::TiedEmbedding(_) => FullyShardedHybridHead::TiedEmbedding(core::marker::PhantomData),
            HybridAttentionHead::TiedEmbeddingLoRA(value) => FullyShardedHybridHead::TiedEmbeddingLoRA(FullyShardedHybridTiedAdapter {
                adapter_a: self.linear(value.adapter_a), adapter_b: self.linear(value.adapter_b), dropout: value.dropout, scale: value.scale }),
        };
        FullyShardedHybridLanguageModel { backbone, head }
    }
}
impl<B: Backend, P: Module<B>> FullyShardedHybridLanguageModel<B, P> {
    pub fn from_full<Q: CompressedAttentionProjection<B> + ShardTransformerProjection<B, Sharded = P>>(
        source: HybridAttentionLanguageModel<B, Q>, rank: usize, world: usize) -> Self { ShardingContext::new(rank, world).hybrid_language_model(source) }
}

macro_rules! execute_sharded_hybrid {
    ($backend:ty, [$($generics:tt)*], $gather:ident, $lookup:ident, $run:ident, $packed_run:ident,
        $forward:ident, $aux:ident, $packed:ident, $packed_aux:ident, $head:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> FullyShardedMhcTransformerBlock<$backend, P>
        where P::Gathered: CompressedAttentionProjection<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<MhcTransformerBlock<$backend, P::Gathered>, C::Error> {
                Ok(MhcTransformerBlock::from_parts(self.attention_connection.$gather(communicator.clone())?, self.ffn_connection.$gather(communicator.clone())?,
                    self.attention.$gather(communicator.clone())?,
                    Param::initialized(self.attention_norm.local.id, self.attention_norm.$gather::<C, 1>(communicator.clone())?),
                    Param::initialized(self.ffn_norm.local.id, self.ffn_norm.$gather::<C, 1>(communicator.clone())?),
                    self.gate.gather_projection(communicator.clone())?, self.up.gather_projection(communicator.clone())?,
                    self.down.gather_projection(communicator)?, self.epsilon))
            }
        }
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> FullyShardedHybridBackbone<$backend, P>
        where P::Gathered: CompressedAttentionProjection<$backend> {
            pub fn $forward<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>,
                communicator: C) -> Result<Tensor<$backend, 3>, C::Error> { self.$run(tokens, valid, false, false, communicator).map(|result| result.output) }
            pub fn $aux<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>,
                indexer_warmup: bool, communicator: C) -> Result<CompressedAttentionOutput<$backend>, C::Error> { self.$run(tokens, valid, true, indexer_warmup, communicator) }
            fn $run<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>,
                auxiliary: bool, warmup: bool, communicator: C) -> Result<CompressedAttentionOutput<$backend>, C::Error> {
                let [batch, length] = tokens.dims();
                assert!(batch > 0 && length > 0 && matches!(tokens.dtype(), DType::I32 | DType::I64), "sharded hybrid tokens must be nonempty I32/I64");
                assert_eq!(tokens.device(), self.embedding.weight.local.val().device(), "sharded hybrid token device differs");
                let hidden = self.embedding.$lookup(tokens, communicator.clone())?;
                let valid = mhc_visible(&hidden, valid);
                let storage = hidden.dtype();
                let device = hidden.device();
                let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
                let hidden = hidden * valid.clone().cast::<FloatDType>(storage.into()).reshape([batch, length, 1]);
                let mut state = self.layers[0].attention_connection.expand(hidden);
                let mut losses = Vec::with_capacity(self.layers.len());
                for sharded in &self.layers {
                    let layer = sharded.$gather(communicator.clone())?;
                    if auxiliary { let result = layer.forward_with_aux(state, Some(valid.clone()), warmup); state = result.state; losses.push(result.indexer_loss); }
                    else { state = layer.forward(state, Some(valid.clone())); }
                }
                let norm = Param::initialized(self.final_norm.local.id, self.final_norm.$gather::<C, 1>(communicator)?);
                Ok(CompressedAttentionOutput { output: normalize_mhc(self.layers.last().unwrap().ffn_connection.reduce(state), &norm, self.epsilon),
                    indexer_loss: if auxiliary { Tensor::cat(losses, 0).sum() } else { Tensor::zeros([1], (&device, compute)) } })
            }
            pub fn $packed<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C) -> Result<Tensor<$backend, 2>, C::Error> {
                self.$packed_run(tokens, layout, valid, false, false, communicator).map(|result| result.output)
            }
            pub fn $packed_aux<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, indexer_warmup: bool, communicator: C) -> Result<PackedCompressedAttentionOutput<$backend>, C::Error> {
                self.$packed_run(tokens, layout, valid, true, indexer_warmup, communicator)
            }
            fn $packed_run<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, auxiliary: bool, warmup: bool, communicator: C) -> Result<PackedCompressedAttentionOutput<$backend>, C::Error> {
                let count = layout.tokens();
                assert_eq!(tokens.dims(), [count], "sharded hybrid packed token/layout count differs");
                assert!(matches!(tokens.dtype(), DType::I32 | DType::I64), "sharded hybrid packed IDs must be I32/I64");
                assert_eq!(tokens.device(), self.embedding.weight.local.val().device(), "sharded hybrid packed token device differs");
                let weight = self.embedding.weight.$gather::<C, 2>(communicator.clone())?;
                let width = weight.dims()[1]; let device = weight.device(); let storage = weight.dtype();
                let hidden = if count == 0 { Tensor::zeros([1, 0, width], (&device, storage)) } else { embedding(weight, tokens.reshape([1, count])) };
                let valid = mhc_visible(&hidden, valid.map(|value| value.reshape([1, count])));
                let hidden = hidden * valid.clone().cast::<FloatDType>(storage.into()).reshape([1, count, 1]);
                let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
                let streams = self.layers[0].attention_connection.streams;
                let mut state = self.layers[0].attention_connection.expand(hidden);
                let mut document_losses: Vec<Vec<Tensor<$backend, 1>>> = (0..layout.documents()).map(|_| Vec::new()).collect();
                for sharded in &self.layers {
                    // Every rank gathers once per layer, independent of its actual local document count.
                    let layer = sharded.$gather(communicator.clone())?;
                    let mut outputs = Vec::new();
                    for (document, bounds) in layout.boundaries().windows(2).enumerate() {
                        let (start, end) = (bounds[0], bounds[1]); let length = end - start;
                        if length == 0 { continue; }
                        let source = state.clone().slice_dim(1, start..end);
                        let mask = Some(valid.clone().slice_dim(1, start..end));
                        if auxiliary { let result = layer.forward_with_aux(source, mask, warmup); outputs.push(result.state); document_losses[document].push(result.indexer_loss); }
                        else { outputs.push(layer.forward(source, mask)); }
                    }
                    if !outputs.is_empty() { state = Tensor::cat(outputs, 1); }
                }
                assert_eq!(state.dims(), [1, count, streams, width], "sharded hybrid packed residual geometry differs");
                let norm = Param::initialized(self.final_norm.local.id, self.final_norm.$gather::<C, 1>(communicator)?);
                let output = if count == 0 { state.sum_dim(2).squeeze_dim(2) } else { self.layers.last().unwrap().ffn_connection.reduce(state) };
                let losses = document_losses.into_iter().map(|losses| if losses.is_empty() { Tensor::zeros([1], (&device, compute)) } else { Tensor::cat(losses, 0).sum() }).collect::<Vec<_>>();
                Ok(PackedCompressedAttentionOutput { output: normalize_mhc(output, &norm, self.epsilon).reshape([count, width]),
                    document_indexer_losses: if losses.is_empty() { Tensor::zeros([0], (&device, compute)) } else { Tensor::cat(losses, 0) } })
            }
        }
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>> FullyShardedHybridLanguageModel<$backend, P>
        where P::Gathered: CompressedAttentionProjection<$backend> {
            pub fn $head<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<GatheredHybridHead<$backend, P::Gathered>, C::Error> {
                let (head, embedding) = match &self.head {
                    FullyShardedHybridHead::Linear(value) => (HybridAttentionHead::Linear(value.gather_projection(communicator)?), None),
                    FullyShardedHybridHead::TiedEmbedding(_) => { let weight = self.backbone.embedding.weight.$gather::<C, 2>(communicator)?;
                        (HybridAttentionHead::TiedEmbedding(core::marker::PhantomData), Some(Embedding { weight: Param::initialized(self.backbone.embedding.weight.local.id, weight) })) },
                    FullyShardedHybridHead::TiedEmbeddingLoRA(value) => {
                        let weight = self.backbone.embedding.weight.$gather::<C, 2>(communicator.clone())?;
                        let embedding = Embedding { weight: Param::initialized(self.backbone.embedding.weight.local.id, weight) };
                        let adapter = HybridTiedEmbeddingAdapter::from_adapters(&embedding, value.adapter_a.$gather(communicator.clone())?, value.adapter_b.$gather(communicator)?, value.dropout.clone(), value.scale);
                        (HybridAttentionHead::TiedEmbeddingLoRA(adapter), Some(embedding))
                    },
                };
                Ok(GatheredHybridHead { head, embedding })
            }
            pub fn $forward<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>, communicator: C)
                -> Result<Tensor<$backend, 3>, C::Error> {
                let hidden = self.backbone.$forward(tokens, valid, communicator.clone())?;
                Ok(self.$head(communicator)?.forward(hidden))
            }
            pub fn $aux<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>,
                indexer_warmup: bool, communicator: C) -> Result<CompressedAttentionOutput<$backend>, C::Error> {
                let result = self.backbone.$aux(tokens, valid, indexer_warmup, communicator.clone())?;
                Ok(CompressedAttentionOutput { output: self.$head(communicator)?.forward(result.output), indexer_loss: result.indexer_loss })
            }
            pub fn $packed<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C) -> Result<Tensor<$backend, 2>, C::Error> {
                let hidden = self.backbone.$packed(tokens, layout, valid, communicator.clone())?;
                Ok(self.$head(communicator)?.project(hidden))
            }
            pub fn $packed_aux<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, indexer_warmup: bool, communicator: C) -> Result<PackedCompressedAttentionOutput<$backend>, C::Error> {
                let result = self.backbone.$packed_aux(tokens, layout, valid, indexer_warmup, communicator.clone())?;
                Ok(PackedCompressedAttentionOutput { output: self.$head(communicator)?.project(result.output), document_indexer_losses: result.document_indexer_losses })
            }
        }
    };
}
execute_sharded_hybrid!(B, [B: Backend], gather_inference, forward_inference, run_inference, packed_run_inference,
    forward_inference, forward_with_aux_inference, forward_packed_inference, forward_packed_with_aux_inference, gather_head_inference);
execute_sharded_hybrid!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather, forward, run, packed_run,
    forward, forward_with_aux, forward_packed, forward_packed_with_aux, gather_head);

macro_rules! hybrid_fields {
    ($module:ident, [$($field:ident),+]) => { impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for $module<B, P> {
        fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_shards(visitor);)+ }
        fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_packed_shards(visitor);)+ }
    } };
}
hybrid_fields!(FullyShardedMhcTransformerBlock, [attention_connection, ffn_connection, attention, attention_norm, ffn_norm, gate, up, down]);
hybrid_fields!(FullyShardedHybridBackbone, [embedding, layers, final_norm]);
hybrid_fields!(FullyShardedHybridLanguageModel, [backbone, head]);
impl<B: Backend> FullyShardedModule<B> for FullyShardedHybridTiedAdapter<B> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.adapter_a.visit_shards(visitor); self.adapter_b.visit_shards(visitor); }
}
impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for FullyShardedHybridHead<B, P> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Linear(value) => value.visit_shards(visitor), Self::TiedEmbedding(_) => {}, Self::TiedEmbeddingLoRA(value) => value.visit_shards(visitor) } }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { if let Self::Linear(value) = self { value.visit_packed_shards(visitor); } }
}
macro_rules! hybrid_adapters {
    ($module:ident, [$($field:ident),+]) => { impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for $module<B, P> {
        fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_adapter_shards(visitor);)+ }
    } };
}
hybrid_adapters!(FullyShardedMhcTransformerBlock, [attention, gate, up, down]);
hybrid_adapters!(FullyShardedHybridBackbone, [layers]);
hybrid_adapters!(FullyShardedHybridLanguageModel, [backbone, head]);
impl<B: Backend> FullyShardedAdapterModule<B> for FullyShardedHybridTiedAdapter<B> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.adapter_a.visit_shards(visitor); self.adapter_b.visit_shards(visitor); }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for FullyShardedHybridHead<B, P> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Linear(value) => value.visit_adapter_shards(visitor), Self::TiedEmbedding(_) => {}, Self::TiedEmbeddingLoRA(value) => value.visit_adapter_shards(visitor) } }
}
