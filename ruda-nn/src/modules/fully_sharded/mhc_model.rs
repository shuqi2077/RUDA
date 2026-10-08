use super::*;
use core::fmt;
use ruda_model::{module::ModuleDisplay, tensor::{Bool, IntegerTensorCollective}};
use crate::{attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout},
    transformer::{MhcResidualModel, MhcResidualBranch, TransformerProjection}, pool::SequencePooling};

/// Actual table, original ordered compressed/mHC layers and independent native task/vocabulary head.
#[derive(Module, Debug)]
pub struct FullyShardedMhcResidualModel<B: Backend, P: Module<B>, F: Module<B>, H: Module<B>> {
    pub embedding: FullyShardedEmbedding<B>,
    pub stack: FullyShardedMhcResidualStack<B, P, F>,
    pub head: FullyShardedProjectedTransformerHead<B, H>,
}

#[derive(Debug)]
pub enum FullyShardedMhcModelError<C: fmt::Debug, R: fmt::Debug, H: fmt::Debug> {
    Collective(C),
    Stack(FullyShardedMhcError<C, R>),
    Head(FullyShardedProjectedError<C, H>),
}
impl<C: fmt::Debug, R: fmt::Debug, H: fmt::Debug> fmt::Display for FullyShardedMhcModelError<C, R, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Collective(error) => write!(f, "mHC model parameter transport: {error:?}"),
            Self::Stack(error) => write!(f, "{error}"), Self::Head(error) => write!(f, "mHC model head: {error}") }
    }
}
impl<C: fmt::Debug, R: fmt::Debug, H: fmt::Debug> core::error::Error for FullyShardedMhcModelError<C, R, H> {}

impl<B: Backend> ShardingContext<B> {
    pub fn mhc_residual_model<P, F, H>(&mut self, source: MhcResidualModel<B, P, F, H>)
        -> FullyShardedMhcResidualModel<B, P::Sharded, F::Sharded, H::Sharded>
    where P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>, F: ShardMhcResidualBranch<B>, H: ShardTransformerProjection<B> {
        FullyShardedMhcResidualModel { embedding: self.embedding(source.embedding), stack: self.mhc_residual_stack(source.stack),
            head: self.awq_transformer_head(source.head) }
    }
}
impl<B: Backend, P: Module<B>, F: Module<B>, H: Module<B>> FullyShardedMhcResidualModel<B, P, F, H> {
    pub fn from_full<Q, G, R>(source: MhcResidualModel<B, Q, G, R>, rank: usize, world: usize) -> Self
    where Q: CompressedAttentionProjection<B> + ShardTransformerProjection<B, Sharded = P>,
        G: ShardMhcResidualBranch<B, Sharded = F>, R: ShardTransformerProjection<B, Sharded = H> {
        ShardingContext::new(rank, world).mhc_residual_model(source)
    }
}

macro_rules! mhc_model_execution {
    ($backend:ty, [$($generics:tt)*], $gather:ident, $embedding:ident, $lookup:ident, $packed_embedding:ident,
        $stack:ident, $stack_aux:ident, $packed_stack:ident, $packed_stack_aux:ident,
        $hidden:ident, $hidden_aux:ident, $forward:ident, $aux:ident, $sequence:ident,
        $packed_hidden:ident, $packed_hidden_aux:ident, $packed:ident, $packed_aux:ident, $packed_sequence:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, F: GatherMhcResidualBranch<$backend, B>, H: GatherTransformerProjection<$backend, B>>
            FullyShardedMhcResidualModel<$backend, P, F, H>
        where P::Gathered: CompressedAttentionProjection<$backend>, H::Gathered: TransformerProjection<$backend> {
            fn $embedding<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, communicator: C) -> Result<Tensor<$backend, 3>, C::Error> {
                let [batch, length] = tokens.dims();
                assert!(batch > 0 && length > 0, "sharded mHC model token rows must be nonempty");
                assert_eq!(tokens.device(), self.embedding.weight.local.val().device(), "sharded mHC token device differs");
                self.embedding.$lookup(tokens, communicator)
            }
            fn $packed_embedding<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                communicator: C) -> Result<Tensor<$backend, 2>, C::Error> {
                let length = tokens.dims()[0];
                assert_eq!(length, layout.tokens(), "sharded mHC packed token/layout count differs");
                assert_eq!(tokens.device(), self.embedding.weight.local.val().device(), "sharded mHC packed token device differs");
                let weight = self.embedding.weight.$gather::<C, 2>(communicator)?;
                let width = weight.dims()[1];
                // Every rank gathers, including zero-token ranks; the loss scope handles locally unused table paths.
                if length == 0 { return Ok(Tensor::zeros([0, width], (&weight.device(), weight.dtype()))); }
                Ok(embedding(weight, tokens.reshape([1, length])).reshape([length, width]))
            }

            pub fn $hidden<C, R, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>, communicator: C, branch: G)
                -> Result<Tensor<$backend, 3>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let input = self.$embedding(tokens, communicator.clone()).map_err(FullyShardedMhcModelError::Collective)?;
                self.stack.$stack(input, valid, communicator, branch).map_err(FullyShardedMhcModelError::Stack)
            }
            pub fn $hidden_aux<C, R, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>,
                indexer_warmup: bool, communicator: C, branch: G)
                -> Result<CompressedAttentionOutput<$backend>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let input = self.$embedding(tokens, communicator.clone()).map_err(FullyShardedMhcModelError::Collective)?;
                self.stack.$stack_aux(input, valid, indexer_warmup, communicator, branch).map_err(FullyShardedMhcModelError::Stack)
            }
            pub fn $forward<C, R, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>, communicator: C, branch: G)
                -> Result<Tensor<$backend, 3>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let hidden = self.$hidden(tokens, valid, communicator.clone(), branch)?;
                self.head.$gather(communicator).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error)))?
                    .forward(hidden).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error)))
            }
            pub fn $aux<C, R, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>,
                indexer_warmup: bool, communicator: C, branch: G)
                -> Result<CompressedAttentionOutput<$backend>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let hidden = self.$hidden_aux(tokens, valid, indexer_warmup, communicator.clone(), branch)?;
                let head = self.head.$gather(communicator).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error)))?;
                Ok(CompressedAttentionOutput { output: head.forward(hidden.output).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error)))?,
                    indexer_loss: hidden.indexer_loss })
            }
            pub fn $sequence<C, R, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Tensor<$backend, 2, Bool>,
                pooling: SequencePooling, communicator: C, branch: G)
                -> Result<crate::transformer::SequenceHeadOutput<$backend>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let hidden = self.$hidden(tokens, Some(valid.clone()), communicator.clone(), branch)?;
                self.head.$gather(communicator).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error)))?
                    .forward_sequence(hidden, valid, pooling).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error)))
            }

            pub fn $packed_hidden<C, R, G>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C, branch: G)
                -> Result<Tensor<$backend, 2>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let input = self.$packed_embedding(tokens, layout, communicator.clone()).map_err(FullyShardedMhcModelError::Collective)?;
                self.stack.$packed_stack(input, layout, valid, communicator, branch).map_err(FullyShardedMhcModelError::Stack)
            }
            pub fn $packed_hidden_aux<C, R, G>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, indexer_warmup: bool, communicator: C, branch: G)
                -> Result<PackedCompressedAttentionOutput<$backend>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let input = self.$packed_embedding(tokens, layout, communicator.clone()).map_err(FullyShardedMhcModelError::Collective)?;
                self.stack.$packed_stack_aux(input, layout, valid, indexer_warmup, communicator, branch).map_err(FullyShardedMhcModelError::Stack)
            }
            pub fn $packed<C, R, G>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C, branch: G)
                -> Result<Tensor<$backend, 2>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let hidden = self.$packed_hidden(tokens, layout, valid, communicator.clone(), branch)?;
                self.head.$gather(communicator).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error)))?
                    .forward(hidden).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error)))
            }
            pub fn $packed_aux<C, R, G>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, indexer_warmup: bool, communicator: C, branch: G)
                -> Result<PackedCompressedAttentionOutput<$backend>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let hidden = self.$packed_hidden_aux(tokens, layout, valid, indexer_warmup, communicator.clone(), branch)?;
                let head = self.head.$gather(communicator).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error)))?;
                Ok(PackedCompressedAttentionOutput { output: head.forward(hidden.output).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error)))?,
                    document_indexer_losses: hidden.document_indexer_losses })
            }
            pub fn $packed_sequence<C, R, G>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, pooling: SequencePooling, communicator: C, branch: G)
                -> Result<crate::transformer::SequenceHeadOutput<$backend>, FullyShardedMhcModelError<C::Error, R, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let hidden = self.$packed_hidden(tokens, layout, valid.clone(), communicator.clone(), branch)?;
                self.head.$gather(communicator).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Collective(error)))?
                    .forward_packed_sequences(hidden, layout, valid, pooling).map_err(|error| FullyShardedMhcModelError::Head(FullyShardedProjectedError::Projection(error)))
            }
        }
    };
}
mhc_model_execution!(B, [B: Backend], gather_inference, embed_inference, forward_inference, embed_packed_inference,
    try_forward_with_inference, try_forward_with_aux_inference, try_forward_packed_with_inference, try_forward_packed_with_aux_inference,
    try_forward_hidden_with_inference, try_forward_hidden_with_aux_inference, try_forward_with_inference, try_forward_with_aux_inference,
    try_forward_sequence_with_inference, try_forward_packed_hidden_with_inference, try_forward_packed_hidden_with_aux_inference,
    try_forward_packed_with_inference, try_forward_packed_with_aux_inference, try_forward_packed_sequences_with_inference);
mhc_model_execution!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather, embed, forward, embed_packed,
    try_forward_with, try_forward_with_aux, try_forward_packed_with, try_forward_packed_with_aux,
    try_forward_hidden_with, try_forward_hidden_with_aux, try_forward_with, try_forward_with_aux, try_forward_sequence_with,
    try_forward_packed_hidden_with, try_forward_packed_hidden_with_aux, try_forward_packed_with, try_forward_packed_with_aux, try_forward_packed_sequences_with);

macro_rules! default_mhc_model {
    ($backend:ty, [$($generics:tt)*], $with:ident, $packed_with:ident, $hidden_with:ident, $packed_hidden_with:ident,
        $forward:ident, $packed:ident, $hidden:ident, $packed_hidden:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, F: GatherMhcResidualBranch<$backend, B>, H: GatherTransformerProjection<$backend, B>>
            FullyShardedMhcResidualModel<$backend, P, F, H>
        where P::Gathered: CompressedAttentionProjection<$backend>, F::Gathered: MhcResidualBranch<$backend>, H::Gathered: TransformerProjection<$backend> {
            pub fn $forward<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>, communicator: C)
                -> Result<Tensor<$backend, 3>, FullyShardedMhcModelError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>> {
                self.$with(tokens, valid, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
            pub fn $packed<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C)
                -> Result<Tensor<$backend, 2>, FullyShardedMhcModelError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>> {
                self.$packed_with(tokens, layout, valid, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
            pub fn $hidden<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>, communicator: C)
                -> Result<Tensor<$backend, 3>, FullyShardedMhcModelError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>> {
                self.$hidden_with(tokens, valid, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
            pub fn $packed_hidden<C: IntegerTensorCollective<B>>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C)
                -> Result<Tensor<$backend, 2>, FullyShardedMhcModelError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>> {
                self.$packed_hidden_with(tokens, layout, valid, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
        }
    };
}
default_mhc_model!(B, [B: Backend], try_forward_with_inference, try_forward_packed_with_inference,
    try_forward_hidden_with_inference, try_forward_packed_hidden_with_inference, forward_inference, forward_packed_inference, forward_hidden_inference, forward_packed_hidden_inference);
default_mhc_model!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], try_forward_with, try_forward_packed_with,
    try_forward_hidden_with, try_forward_packed_hidden_with, forward, forward_packed, forward_hidden, forward_packed_hidden);

impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, F: FullyShardedModule<B> + ModuleDisplay, H: FullyShardedModule<B> + ModuleDisplay>
    FullyShardedModule<B> for FullyShardedMhcResidualModel<B, P, F, H> {
    fn visit_shards<G: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut G) {
        self.embedding.visit_shards(visitor); self.stack.visit_shards(visitor); self.head.visit_shards(visitor);
    }
    fn visit_packed_shards<G: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut G) { self.stack.visit_packed_shards(visitor); self.head.visit_packed_shards(visitor); }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, F: FullyShardedAdapterModule<B> + ModuleDisplay, H: FullyShardedAdapterModule<B> + ModuleDisplay>
    FullyShardedAdapterModule<B> for FullyShardedMhcResidualModel<B, P, F, H> {
    fn visit_adapter_shards<G: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut G) { self.stack.visit_adapter_shards(visitor); self.head.visit_adapter_shards(visitor); }
}
