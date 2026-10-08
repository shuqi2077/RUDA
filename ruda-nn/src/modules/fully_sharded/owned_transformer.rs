use super::*;
use core::fmt;
use ruda_model::{module::ModuleDisplay, tensor::{Bool, IntegerTensorCollective, VariableTensorCollective, MoeDispatchOps, MoeReceivedOps}};
use crate::expert_parallel::ExpertParallelReceived;
use crate::transformer::{ExpertParallelTransformerBlock, ExpertParallelTransformerLayer, ExpertParallelTransformerModel,
    ExpertParallelTransformerError, NativeMoeTransformerLayer, TransformerProjection};
use crate::attention::{DenseAttentionMask, DenseAttentionOptions, PackedSequenceLayout, PackedAttentionOptions, PackedDocumentAttentionMask};
use crate::cache::TransformerKvCache;
use crate::{loss::{CausalLoss, CausalCrossEntropyConfig}, pool::SequencePooling, transformer::SequenceHeadOutput};

/// Exact loaded attention, owned routed/shared branches and residual policies.
#[derive(Module, Debug)]
pub struct FullyShardedExpertParallelTransformerBlock<B: Backend, P: Module<B>, E: Module<B>> {
    pub attention: FullyShardedProjectedAttention<B, P>,
    pub routed: FullyShardedExpertParallelMoeLayer<B, P, E>,
    pub shared: Option<FullyShardedProjectedFeedForward<B, P>>,
    pub attention_norm: FullyShardedTransformerNorm<B>,
    pub feed_forward_norm: FullyShardedTransformerNorm<B>,
    pub residual_dropout: crate::Dropout,
    pub norm_first: bool,
}
#[derive(Module, Debug)]
pub enum FullyShardedExpertParallelTransformerLayer<B: Backend, P: Module<B>, E: Module<B>> {
    Local(FullyShardedNativeMoeTransformerLayer<B, P>),
    Parallel(FullyShardedExpertParallelTransformerBlock<B, P, E>),
}
/// Native full model with rank-owned expert storage additionally data-sharded.
#[derive(Module, Debug)]
pub struct FullyShardedExpertParallelTransformerModel<B: Backend, P: Module<B>, E: Module<B>> {
    pub embeddings: FullyShardedTransformerEmbeddings<B>,
    pub layers: Vec<FullyShardedExpertParallelTransformerLayer<B, P, E>>,
    pub normalization: Option<FullyShardedTransformerNorm<B>>,
    pub head: FullyShardedProjectedTransformerHead<B, P>,
}

impl<B: Backend> ShardingContext<B> {
    pub fn expert_parallel_transformer_block<P: ShardTransformerProjection<B>, E: ShardOwnedExperts<B>>(&mut self,
        source: ExpertParallelTransformerBlock<B, P, E>) -> FullyShardedExpertParallelTransformerBlock<B, P::Sharded, E::Sharded> {
        source.validate();
        FullyShardedExpertParallelTransformerBlock { attention: self.awq_attention(source.attention), routed: self.expert_parallel_layer(source.routed),
            shared: source.shared.map(|value| self.awq_feed_forward(value)), attention_norm: self.normalization(source.attention_norm),
            feed_forward_norm: self.normalization(source.feed_forward_norm), residual_dropout: source.residual_dropout, norm_first: source.norm_first }
    }
    pub fn expert_parallel_transformer_layer<P: ShardTransformerProjection<B>, E: ShardOwnedExperts<B>>(&mut self,
        source: ExpertParallelTransformerLayer<B, P, E>) -> FullyShardedExpertParallelTransformerLayer<B, P::Sharded, E::Sharded> {
        match source {
            ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Dense(value)) =>
                FullyShardedExpertParallelTransformerLayer::Local(FullyShardedNativeMoeTransformerLayer::Dense(self.awq_transformer(value))),
            ExpertParallelTransformerLayer::Local(NativeMoeTransformerLayer::Routed(value)) =>
                FullyShardedExpertParallelTransformerLayer::Local(FullyShardedNativeMoeTransformerLayer::Routed(self.moe_transformer(value))),
            ExpertParallelTransformerLayer::Parallel(value) => FullyShardedExpertParallelTransformerLayer::Parallel(self.expert_parallel_transformer_block(value)),
        }
    }
    /// One canonical context spans real tables, attention/router/shared/expert
    /// leaves, norms and head. No initialized substitute model is constructed.
    pub fn expert_parallel_transformer_model<P: ShardTransformerProjection<B>, E: ShardOwnedExperts<B>>(&mut self,
        source: ExpertParallelTransformerModel<B, P, E>) -> FullyShardedExpertParallelTransformerModel<B, P::Sharded, E::Sharded> {
        FullyShardedExpertParallelTransformerModel { embeddings: self.transformer_embeddings(source.embeddings),
            layers: source.layers.into_iter().map(|value| self.expert_parallel_transformer_layer(value)).collect(),
            normalization: source.normalization.map(|value| self.normalization(value)), head: self.awq_transformer_head(source.head) }
    }
}
impl<B: Backend, P: Module<B>, E: Module<B>> FullyShardedExpertParallelTransformerModel<B, P, E> {
    pub fn from_owned<Q: ShardTransformerProjection<B, Sharded = P>, G: ShardOwnedExperts<B, Sharded = E>>(
        source: ExpertParallelTransformerModel<B, Q, G>, data_rank: usize, data_world: usize) -> Self {
        ShardingContext::new(data_rank, data_world).expert_parallel_transformer_model(source)
    }
    pub fn new_kv_cache(&self, capacity: usize) -> TransformerKvCache<B> { TransformerKvCache::new(self.layers.len(), capacity) }
}

#[derive(Debug)]
pub enum FullyShardedExpertParallelModelError<D: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> {
    Data(D),
    Native(ExpertParallelTransformerError<C, P, E>),
    Head(P),
}
impl<D: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> fmt::Display for FullyShardedExpertParallelModelError<D, C, P, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Data(error) => write!(f, "expert-owned model data gather: {error:?}"),
            Self::Native(error) => write!(f, "{error}"), Self::Head(error) => write!(f, "expert-owned model head: {error:?}") }
    }
}
impl<D: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> core::error::Error for FullyShardedExpertParallelModelError<D, C, P, E> {}
impl<D: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> From<FullyShardedAwqError<D, P>> for FullyShardedExpertParallelModelError<D, C, P, E> {
    fn from(error: FullyShardedAwqError<D, P>) -> Self {
        match error { FullyShardedAwqError::Collective(error) => Self::Data(error), FullyShardedAwqError::Projection(error) => Self::Head(error) }
    }
}

macro_rules! gather_owned_transformer {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            FullyShardedExpertParallelTransformerBlock<$backend, P, E> {
            pub fn $gather<D: IntegerTensorCollective<B>>(&self, data: D)
                -> Result<ExpertParallelTransformerBlock<$backend, P::Gathered, E::Gathered>, D::Error> {
                let block = ExpertParallelTransformerBlock { attention: self.attention.$gather(data.clone())?, routed: self.routed.$gather(data.clone())?,
                    shared: self.shared.as_ref().map(|value| value.$gather(data.clone())).transpose()?,
                    attention_norm: self.attention_norm.$gather(data.clone())?, feed_forward_norm: self.feed_forward_norm.$gather(data)?,
                    residual_dropout: self.residual_dropout.clone(), norm_first: self.norm_first };
                block.validate(); Ok(block)
            }
        }
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            FullyShardedExpertParallelTransformerLayer<$backend, P, E> {
            /// Gather only the current layer, keeping its local/parallel variant.
            pub fn $gather<D: IntegerTensorCollective<B>>(&self, data: D)
                -> Result<ExpertParallelTransformerLayer<$backend, P::Gathered, E::Gathered>, D::Error> {
                match self { Self::Local(value) => value.$gather(data).map(ExpertParallelTransformerLayer::Local),
                    Self::Parallel(value) => value.$gather(data).map(ExpertParallelTransformerLayer::Parallel) }
            }
        }
    };
}
gather_owned_transformer!(B, [B: Backend], gather_inference);
gather_owned_transformer!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

macro_rules! execute_owned_model {
    ($backend:ty, [$($generics:tt)*], $gather:ident, $embed:ident, $native:ident, $native_packed:ident,
        $hidden_with:ident, $packed_hidden_with:ident, $hidden:ident, $packed_hidden:ident, $forward:ident, $packed:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            FullyShardedExpertParallelTransformerModel<$backend, P, E>
        where P::Gathered: TransformerProjection<$backend>, E::Gathered: ExpertParallelReceived<$backend> {
            /// Per-layer callback receives the real gathered native layer and
            /// data transport; expert/position/mask policies remain caller-owned.
            pub fn $hidden_with<D, C, F>(&self, input: FullyShardedTransformerInput<$backend>, data: D, mut layer: F)
                -> Result<Tensor<$backend, 3>, FullyShardedExpertParallelModelError<D::Error, C,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: fmt::Debug,
                F: FnMut(usize, &ExpertParallelTransformerLayer<$backend, P::Gathered, E::Gathered>, Tensor<$backend, 3>, D)
                    -> Result<Tensor<$backend, 3>, ExpertParallelTransformerError<C,
                        <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                let rows = input.tokens.dims(); let width = self.embeddings.hidden_width();
                let mut hidden = self.embeddings.$embed(input, data.clone()).map_err(FullyShardedExpertParallelModelError::Data)?;
                for (index, stored) in self.layers.iter().enumerate() {
                    let native = stored.$gather(data.clone()).map_err(FullyShardedExpertParallelModelError::Data)?;
                    hidden = layer(index, &native, hidden, data.clone()).map_err(FullyShardedExpertParallelModelError::Native)?;
                    assert_eq!(hidden.dims(), [rows[0], rows[1], width], "owned sharded model layer changed token axes");
                }
                Ok(if let Some(norm) = &self.normalization { norm.$gather(data).map_err(FullyShardedExpertParallelModelError::Data)?.forward(hidden) } else { hidden })
            }
            pub fn $packed_hidden_with<D, C, F>(&self, input: FullyShardedTransformerInput<$backend, 1>, layout: &PackedSequenceLayout, data: D, mut layer: F)
                -> Result<Tensor<$backend, 2>, FullyShardedExpertParallelModelError<D::Error, C,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: fmt::Debug,
                F: FnMut(usize, &ExpertParallelTransformerLayer<$backend, P::Gathered, E::Gathered>, Tensor<$backend, 2>, D)
                    -> Result<Tensor<$backend, 2>, ExpertParallelTransformerError<C,
                        <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                let shape = [layout.tokens(), self.embeddings.hidden_width()];
                let input = super::model::packed_input(input, layout);
                let mut hidden = self.embeddings.$embed(input, data.clone()).map_err(FullyShardedExpertParallelModelError::Data)?.reshape(shape);
                for (index, stored) in self.layers.iter().enumerate() {
                    let native = stored.$gather(data.clone()).map_err(FullyShardedExpertParallelModelError::Data)?;
                    hidden = layer(index, &native, hidden, data.clone()).map_err(FullyShardedExpertParallelModelError::Native)?;
                    assert_eq!(hidden.dims(), shape, "owned sharded model layer changed packed document rows");
                }
                Ok(if let Some(norm) = &self.normalization { norm.$gather(data).map_err(FullyShardedExpertParallelModelError::Data)?.forward(hidden) } else { hidden })
            }

            pub fn $hidden<D, C, F>(&self, input: FullyShardedTransformerInput<$backend>, masks: DenseAttentionMask<$backend>,
                options: DenseAttentionOptions, data: D, expert: C, mut positions: F)
                -> Result<Tensor<$backend, 3>, FullyShardedExpertParallelModelError<D::Error, C::Error,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>,
                F: FnMut(usize, Tensor<$backend, 4>, Tensor<$backend, 4>) -> (Tensor<$backend, 4>, Tensor<$backend, 4>) {
                self.$hidden_with(input, data, |index, layer, hidden, _| layer.$native(hidden, masks.clone(), options, expert.clone(),
                    |query, key| positions(index, query, key)))
            }
            pub fn $forward<D, C, F>(&self, input: FullyShardedTransformerInput<$backend>, masks: DenseAttentionMask<$backend>,
                options: DenseAttentionOptions, data: D, expert: C, positions: F)
                -> Result<Tensor<$backend, 3>, FullyShardedExpertParallelModelError<D::Error, C::Error,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>,
                F: FnMut(usize, Tensor<$backend, 4>, Tensor<$backend, 4>) -> (Tensor<$backend, 4>, Tensor<$backend, 4>) {
                let hidden = self.$hidden(input, masks, options, data.clone(), expert, positions)?;
                self.head.$embed(hidden, data).map_err(Into::into)
            }
            pub fn $packed_hidden<D, C, F>(&self, input: FullyShardedTransformerInput<$backend, 1>, layout: &PackedSequenceLayout,
                masks: Option<&[PackedDocumentAttentionMask<$backend>]>, options: PackedAttentionOptions, data: D, expert: C, mut positions: F)
                -> Result<Tensor<$backend, 2>, FullyShardedExpertParallelModelError<D::Error, C::Error,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>,
                F: FnMut(usize, Tensor<$backend, 3>, Tensor<$backend, 3>) -> (Tensor<$backend, 3>, Tensor<$backend, 3>) {
                self.$packed_hidden_with(input, layout, data, |index, layer, hidden, _| layer.$native_packed(hidden, layout, masks, options, expert.clone(),
                    |query, key| positions(index, query, key)))
            }
            pub fn $packed<D, C, F>(&self, input: FullyShardedTransformerInput<$backend, 1>, layout: &PackedSequenceLayout,
                masks: Option<&[PackedDocumentAttentionMask<$backend>]>, options: PackedAttentionOptions, data: D, expert: C, positions: F)
                -> Result<Tensor<$backend, 2>, FullyShardedExpertParallelModelError<D::Error, C::Error,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>,
                F: FnMut(usize, Tensor<$backend, 3>, Tensor<$backend, 3>) -> (Tensor<$backend, 3>, Tensor<$backend, 3>) {
                let hidden = self.$packed_hidden(input, layout, masks, options, data.clone(), expert, positions)?;
                self.head.$embed(hidden, data).map_err(Into::into)
            }
        }
    };
}
execute_owned_model!(B, [B: MoeDispatchOps + MoeReceivedOps], gather_inference, forward_inference, forward_with_positions_inference, forward_packed_with_positions_inference,
    forward_hidden_with_inference, forward_packed_hidden_with_inference, forward_hidden_inference, forward_packed_hidden_inference, forward_inference, forward_packed_inference);
execute_owned_model!(Autodiff<B, S>, [B: MoeDispatchOps + MoeReceivedOps, S: CheckpointStrategy], gather, forward, forward_with_positions, forward_packed_with_positions,
    forward_hidden_with, forward_packed_hidden_with, forward_hidden, forward_packed_hidden, forward, forward_packed);

macro_rules! owned_model_objectives {
    ($backend:ty, [$($generics:tt)*], $gather:ident, $hidden:ident, $packed_hidden:ident, $head_loss:ident, $packed_head_loss:ident,
        $causal:ident, $packed_causal:ident, $sequence:ident, $packed_sequence:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            FullyShardedExpertParallelTransformerModel<$backend, P, E>
        where P::Gathered: TransformerProjection<$backend>, E::Gathered: ExpertParallelReceived<$backend> {
            /// Native full-vocabulary chunked loss sum and exact valid-token
            /// count. The caller's explicitly selected scopes and objective
            /// normalization group are not replaced by an inferred DP/EP mean.
            pub fn $causal<D, C, F>(&self, input: FullyShardedTransformerInput<$backend>, labels: Tensor<$backend, 2, Int>,
                criterion: &CausalCrossEntropyConfig, label_smoothing: f64, data: D, layer: F)
                -> Result<CausalLoss<$backend>, FullyShardedExpertParallelModelError<D::Error, C,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: fmt::Debug,
                F: FnMut(usize, &ExpertParallelTransformerLayer<$backend, P::Gathered, E::Gathered>, Tensor<$backend, 3>, D)
                    -> Result<Tensor<$backend, 3>, ExpertParallelTransformerError<C,
                        <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                let hidden = self.$hidden(input, data.clone(), layer)?;
                self.head.$head_loss(hidden, labels, criterion, label_smoothing, data).map_err(Into::into)
            }
            /// Document-local shifted labels, including true empty packed
            /// ranks, retaining the source criterion's native chunk/count path.
            pub fn $packed_causal<D, C, F>(&self, input: FullyShardedTransformerInput<$backend, 1>, labels: Tensor<$backend, 1, Int>,
                layout: &PackedSequenceLayout, criterion: &CausalCrossEntropyConfig, label_smoothing: f64, data: D, layer: F)
                -> Result<CausalLoss<$backend>, FullyShardedExpertParallelModelError<D::Error, C,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: fmt::Debug,
                F: FnMut(usize, &ExpertParallelTransformerLayer<$backend, P::Gathered, E::Gathered>, Tensor<$backend, 2>, D)
                    -> Result<Tensor<$backend, 2>, ExpertParallelTransformerError<C,
                        <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                let hidden = self.$packed_hidden(input, layout, data.clone(), layer)?;
                self.head.$packed_head_loss(hidden, labels, layout, criterion, label_smoothing, data).map_err(Into::into)
            }
            pub fn $sequence<D, C, F>(&self, input: FullyShardedTransformerInput<$backend>, visible: Tensor<$backend, 2, Bool>,
                pooling: SequencePooling, data: D, layer: F)
                -> Result<SequenceHeadOutput<$backend>, FullyShardedExpertParallelModelError<D::Error, C,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: fmt::Debug,
                F: FnMut(usize, &ExpertParallelTransformerLayer<$backend, P::Gathered, E::Gathered>, Tensor<$backend, 3>, D)
                    -> Result<Tensor<$backend, 3>, ExpertParallelTransformerError<C,
                        <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                let hidden = self.$hidden(input, data.clone(), layer)?;
                self.head.$gather(data).map_err(FullyShardedExpertParallelModelError::Data)?
                    .forward_sequence(hidden, visible, pooling).map_err(FullyShardedExpertParallelModelError::Head)
            }
            pub fn $packed_sequence<D, C, F>(&self, input: FullyShardedTransformerInput<$backend, 1>, layout: &PackedSequenceLayout,
                visible: Option<Tensor<$backend, 1, Bool>>, pooling: SequencePooling, data: D, layer: F)
                -> Result<SequenceHeadOutput<$backend>, FullyShardedExpertParallelModelError<D::Error, C,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: fmt::Debug,
                F: FnMut(usize, &ExpertParallelTransformerLayer<$backend, P::Gathered, E::Gathered>, Tensor<$backend, 2>, D)
                    -> Result<Tensor<$backend, 2>, ExpertParallelTransformerError<C,
                        <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                let hidden = self.$packed_hidden(input, layout, data.clone(), layer)?;
                self.head.$gather(data).map_err(FullyShardedExpertParallelModelError::Data)?
                    .forward_packed_sequences(hidden, layout, visible, pooling).map_err(FullyShardedExpertParallelModelError::Head)
            }
        }
    };
}
owned_model_objectives!(B, [B: MoeDispatchOps + MoeReceivedOps], gather_inference, forward_hidden_with_inference, forward_packed_hidden_with_inference,
    forward_causal_loss_inference, forward_packed_causal_loss_inference, forward_causal_terms_with_inference, forward_packed_causal_terms_with_inference,
    forward_sequence_with_inference, forward_packed_sequences_with_inference);
owned_model_objectives!(Autodiff<B, S>, [B: MoeDispatchOps + MoeReceivedOps, S: CheckpointStrategy], gather, forward_hidden_with, forward_packed_hidden_with,
    forward_causal_loss, forward_packed_causal_loss, forward_causal_terms_with, forward_packed_causal_terms_with, forward_sequence_with, forward_packed_sequences_with);

impl<B: MoeDispatchOps + MoeReceivedOps, P: GatherTransformerProjection<B, B>, E: GatherOwnedExperts<B, B>> FullyShardedExpertParallelTransformerModel<B, P, E>
where P::Gathered: TransformerProjection<B>, E::Gathered: ExpertParallelReceived<B> {
    /// Source cache chunk commit/recovery semantics are unchanged. Gather only
    /// the current native layer, execute its new rows, then gather the real head.
    pub fn forward_cached_inference<D, C, F>(&self, input: FullyShardedTransformerInput<B>, visible: Option<Tensor<B, 2, Bool>>,
        cache: &mut TransformerKvCache<B>, masks: DenseAttentionMask<B>, options: DenseAttentionOptions, data: D, expert: C, mut positions: F)
        -> Result<Tensor<B, 3>, FullyShardedExpertParallelModelError<D::Error, C::Error,
            <P::Gathered as TransformerProjection<B>>::Error, <E::Gathered as ExpertParallelReceived<B>>::Error>>
    where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>,
        F: FnMut(usize, Tensor<B, 4>, Tensor<B, 4>, usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        cache.validate_layers(self.layers.len()); let rows = input.tokens.dims();
        let next = cache.position().checked_add(rows[1]).expect("owned sharded model cache position overflows");
        let mut hidden = self.embeddings.forward_inference(input, data.clone()).map_err(FullyShardedExpertParallelModelError::Data)?;
        for (index, stored) in self.layers.iter().enumerate() {
            let layer = stored.gather_inference(data.clone()).map_err(FullyShardedExpertParallelModelError::Data)?;
            hidden = layer.forward_cached_with_positions_inference(hidden, visible.clone(), &mut cache.layers_mut()[index], masks.clone(), options,
                expert.clone(), |query, key, position| positions(index, query, key, position)).map_err(FullyShardedExpertParallelModelError::Native)?;
            assert_eq!((hidden.dims()[0], hidden.dims()[1]), (rows[0], rows[1]), "owned sharded cache changed new-token rows");
        }
        cache.finish_chunk(next);
        let hidden = if let Some(norm) = &self.normalization { norm.gather_inference(data.clone()).map_err(FullyShardedExpertParallelModelError::Data)?.forward(hidden) } else { hidden };
        self.head.forward_inference(hidden, data).map_err(Into::into)
    }
}

macro_rules! owned_model_visitors {
    ($target:ident, [$($field:ident),+], [$($adapter:ident),*]) => {
        impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for $target<B, P, E> {
            fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_shards(visitor);)+ }
            fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_packed_shards(visitor);)+ }
        }
        impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for $target<B, P, E> {
            fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { $(self.$adapter.visit_adapter_shards(visitor);)* }
        }
    };
}
owned_model_visitors!(FullyShardedExpertParallelTransformerBlock, [attention, routed, shared, attention_norm, feed_forward_norm], [attention, routed, shared]);
owned_model_visitors!(FullyShardedExpertParallelTransformerModel, [embeddings, layers, normalization, head], [layers, head]);
impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B>
    for FullyShardedExpertParallelTransformerLayer<B, P, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Local(value) => value.visit_shards(visitor), Self::Parallel(value) => value.visit_shards(visitor) } }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Local(value) => value.visit_packed_shards(visitor), Self::Parallel(value) => value.visit_packed_shards(visitor) } }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B>
    for FullyShardedExpertParallelTransformerLayer<B, P, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Local(value) => value.visit_adapter_shards(visitor), Self::Parallel(value) => value.visit_adapter_shards(visitor) } }
}
