use super::*;
use core::fmt;
use ruda_model::{module::ModuleDisplay, tensor::{Bool, IntegerTensorCollective}};
use crate::attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout};
use crate::transformer::{MhcResidualBlock, MhcResidualStack, MhcResidualBranch, normalize_mhc, mhc_visible};

/// Complete actual native residual block, retaining only local persistent mapping/attention/branch parameters.
#[derive(Module, Debug)]
pub struct FullyShardedMhcResidualBlock<B: Backend, P: Module<B>, F: Module<B>> {
    pub attention_connection: FullyShardedMhc<B>,
    pub ffn_connection: FullyShardedMhc<B>,
    pub attention: FullyShardedCompressedAttention<B, P>,
    pub attention_norm: ShardedParameter<B>,
    pub ffn_norm: ShardedParameter<B>,
    pub feed_forward: F,
    pub epsilon: f64,
}

#[derive(Module, Debug)]
pub struct FullyShardedMhcResidualStack<B: Backend, P: Module<B>, F: Module<B>> {
    pub layers: Vec<FullyShardedMhcResidualBlock<B, P, F>>,
    pub final_norm: ShardedParameter<B>,
    pub epsilon: f64,
}

#[derive(Debug)]
pub enum FullyShardedMhcError<C: fmt::Debug, R: fmt::Debug> { Collective(C), Branch(R) }
impl<C: fmt::Debug, R: fmt::Debug> fmt::Display for FullyShardedMhcError<C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Collective(error) => write!(f, "mHC parameter transport: {error:?}"), Self::Branch(error) => write!(f, "mHC native branch: {error:?}") }
    }
}
impl<C: fmt::Debug, R: fmt::Debug> core::error::Error for FullyShardedMhcError<C, R> {}

impl<B: Backend> FullyShardedMhc<B> {
    /// Parameter-free expansion uses the original source stream geometry, without any gather or fabricated mapping.
    pub fn expand(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
        assert_eq!(input.dims()[2], self.width, "sharded mHC expansion width differs");
        input.unsqueeze_dim(2).repeat_dim(2, self.streams)
    }
    pub fn reduce(&self, state: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, tokens, streams, width] = state.dims();
        assert!(batch > 0 && tokens > 0, "sharded mHC reduction rows must be nonempty");
        assert_eq!((streams, width), (self.streams, self.width), "sharded mHC reduction geometry differs");
        state.mean_dim(2).squeeze_dim(2)
    }
}

impl<B: Backend> ShardingContext<B> {
    pub fn mhc_residual_block<P, F>(&mut self, source: MhcResidualBlock<B, P, F>) -> FullyShardedMhcResidualBlock<B, P::Sharded, F::Sharded>
    where P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>, F: ShardMhcResidualBranch<B> {
        FullyShardedMhcResidualBlock { attention_connection: self.mhc(source.attention_connection), ffn_connection: self.mhc(source.ffn_connection),
            attention: self.compressed_attention(source.attention), attention_norm: self.parameter(source.attention_norm),
            ffn_norm: self.parameter(source.ffn_norm), feed_forward: source.feed_forward.shard_branch(self), epsilon: source.epsilon }
    }
    pub fn mhc_residual_stack<P, F>(&mut self, source: MhcResidualStack<B, P, F>) -> FullyShardedMhcResidualStack<B, P::Sharded, F::Sharded>
    where P: CompressedAttentionProjection<B> + ShardTransformerProjection<B>, F: ShardMhcResidualBranch<B> {
        assert!(!source.layers.is_empty(), "sharded mHC stack must contain actual loaded layers");
        FullyShardedMhcResidualStack { layers: source.layers.into_iter().map(|value| self.mhc_residual_block(value)).collect(),
            final_norm: self.parameter(source.final_norm), epsilon: source.epsilon }
    }
}

impl<B: Backend, P: Module<B>, F: Module<B>> FullyShardedMhcResidualStack<B, P, F> {
    pub fn from_full<Q, G>(source: MhcResidualStack<B, Q, G>, rank: usize, world: usize) -> Self
    where Q: CompressedAttentionProjection<B> + ShardTransformerProjection<B, Sharded = P>, G: ShardMhcResidualBranch<B, Sharded = F> {
        ShardingContext::new(rank, world).mhc_residual_stack(source)
    }
    pub fn width(&self) -> usize { self.layers[0].attention.width }
}

macro_rules! mhc_stack_execution {
    ($backend:ty, [$($generics:tt)*], $gather:ident, $run:ident, $packed_run:ident,
        $with:ident, $with_aux:ident, $packed_with:ident, $packed_aux:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, F: GatherMhcResidualBranch<$backend, B>>
            FullyShardedMhcResidualBlock<$backend, P, F> where P::Gathered: CompressedAttentionProjection<$backend> {
            /// Only one transient original block is materialized, using differentiable gathers on the AD path.
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<MhcResidualBlock<$backend, P::Gathered, F::Gathered>, C::Error> {
                Ok(MhcResidualBlock::from_parts(self.attention_connection.$gather(communicator.clone())?, self.ffn_connection.$gather(communicator.clone())?,
                    self.attention.$gather(communicator.clone())?,
                    Param::initialized(self.attention_norm.local.id, self.attention_norm.$gather::<C, 1>(communicator.clone())?),
                    Param::initialized(self.ffn_norm.local.id, self.ffn_norm.$gather::<C, 1>(communicator.clone())?),
                    self.feed_forward.gather_branch(communicator)?, self.epsilon))
            }
        }

        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, F: GatherMhcResidualBranch<$backend, B>>
            FullyShardedMhcResidualStack<$backend, P, F> where P::Gathered: CompressedAttentionProjection<$backend> {
            pub fn $with<C, R, G>(&self, input: Tensor<$backend, 3>, valid: Option<Tensor<$backend, 2, Bool>>, communicator: C, branch: G)
                -> Result<Tensor<$backend, 3>, FullyShardedMhcError<C::Error, R>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                self.$run(input, valid, false, false, communicator, branch).map(|result| result.output)
            }
            pub fn $with_aux<C, R, G>(&self, input: Tensor<$backend, 3>, valid: Option<Tensor<$backend, 2, Bool>>,
                indexer_warmup: bool, communicator: C, branch: G) -> Result<CompressedAttentionOutput<$backend>, FullyShardedMhcError<C::Error, R>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                self.$run(input, valid, true, indexer_warmup, communicator, branch)
            }
            fn $run<C, R, G>(&self, input: Tensor<$backend, 3>, valid: Option<Tensor<$backend, 2, Bool>>, auxiliary: bool, warmup: bool,
                communicator: C, mut branch: G) -> Result<CompressedAttentionOutput<$backend>, FullyShardedMhcError<C::Error, R>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let [batch, tokens, width] = input.dims();
                assert!(batch > 0 && tokens > 0 && !self.layers.is_empty(), "sharded mHC rows/layers must be nonempty");
                assert_eq!(width, self.width(), "sharded mHC hidden width differs");
                let valid = mhc_visible(&input, valid);
                let compute = if input.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
                let device = input.device();
                let mut losses = Vec::with_capacity(self.layers.len());
                let storage = input.dtype();
                let input = input * valid.clone().cast::<FloatDType>(storage.into()).reshape([batch, tokens, 1]);
                let mut state = self.layers[0].attention_connection.expand(input);
                for (index, sharded) in self.layers.iter().enumerate() {
                    let layer = sharded.$gather(communicator.clone()).map_err(FullyShardedMhcError::Collective)?;
                    if auxiliary {
                        let result = layer.try_forward_with_aux(state, Some(valid.clone()), warmup,
                            |feed, input| branch(index, feed, input, communicator.clone())).map_err(FullyShardedMhcError::Branch)?;
                        state = result.state; losses.push(result.indexer_loss);
                    } else {
                        state = layer.try_forward_with(state, Some(valid.clone()),
                            |feed, input| branch(index, feed, input, communicator.clone())).map_err(FullyShardedMhcError::Branch)?;
                    }
                }
                let norm = Param::initialized(self.final_norm.local.id, self.final_norm.$gather::<C, 1>(communicator).map_err(FullyShardedMhcError::Collective)?);
                let indexer_loss = if auxiliary { Tensor::cat(losses, 0).sum() } else { Tensor::zeros([1], (&device, compute)) };
                Ok(CompressedAttentionOutput { output: normalize_mhc(self.layers.last().unwrap().ffn_connection.reduce(state), &norm, self.epsilon), indexer_loss })
            }

            pub fn $packed_with<C, R, G>(&self, input: Tensor<$backend, 2>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C, branch: G) -> Result<Tensor<$backend, 2>, FullyShardedMhcError<C::Error, R>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                self.$packed_run(input, layout, valid, false, false, communicator, branch).map(|result| result.output)
            }
            pub fn $packed_aux<C, R, G>(&self, input: Tensor<$backend, 2>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, indexer_warmup: bool, communicator: C, branch: G)
                -> Result<PackedCompressedAttentionOutput<$backend>, FullyShardedMhcError<C::Error, R>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                self.$packed_run(input, layout, valid, true, indexer_warmup, communicator, branch)
            }
            fn $packed_run<C, R, G>(&self, input: Tensor<$backend, 2>, layout: &PackedSequenceLayout, valid: Option<Tensor<$backend, 1, Bool>>,
                auxiliary: bool, warmup: bool, communicator: C, mut branch: G) -> Result<PackedCompressedAttentionOutput<$backend>, FullyShardedMhcError<C::Error, R>>
            where C: IntegerTensorCollective<B>, R: fmt::Debug,
                G: FnMut(usize, &F::Gathered, Tensor<$backend, 3>, C) -> Result<Tensor<$backend, 3>, R> {
                let [tokens, width] = input.dims();
                assert_eq!((tokens, width), (layout.tokens(), self.width()), "sharded mHC packed layout/width differs");
                let compute = if input.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
                let mut losses = Tensor::zeros([layout.documents()], (&input.device(), compute));
                let input = input.reshape([1, tokens, width]);
                let valid = mhc_visible(&input, valid.map(|mask| mask.reshape([1, tokens])));
                let storage = input.dtype();
                let input = input * valid.clone().cast::<FloatDType>(storage.into()).reshape([1, tokens, 1]);
                let mut state = self.layers[0].attention_connection.expand(input);
                for (index, sharded) in self.layers.iter().enumerate() {
                    let layer = sharded.$gather(communicator.clone()).map_err(FullyShardedMhcError::Collective)?;
                    if auxiliary {
                        let result = layer.try_forward_packed_with_aux(state, layout, Some(valid.clone().reshape([tokens])), warmup,
                            |feed, input| branch(index, feed, input, communicator.clone())).map_err(FullyShardedMhcError::Branch)?;
                        state = result.state; losses = losses + result.document_indexer_losses;
                    } else {
                        state = layer.try_forward_packed_with(state, layout, Some(valid.clone().reshape([tokens])),
                            |feed, input| branch(index, feed, input, communicator.clone())).map_err(FullyShardedMhcError::Branch)?;
                    }
                }
                let norm = Param::initialized(self.final_norm.local.id, self.final_norm.$gather::<C, 1>(communicator).map_err(FullyShardedMhcError::Collective)?);
                let output = if tokens == 0 { state.sum_dim(2).squeeze_dim(2) } else { self.layers.last().unwrap().ffn_connection.reduce(state) };
                Ok(PackedCompressedAttentionOutput { output: normalize_mhc(output, &norm, self.epsilon).reshape([tokens, width]), document_indexer_losses: losses })
            }
        }
    };
}
mhc_stack_execution!(B, [B: Backend], gather_inference, run_inference, packed_run_inference,
    try_forward_with_inference, try_forward_with_aux_inference, try_forward_packed_with_inference, try_forward_packed_with_aux_inference);
mhc_stack_execution!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather, run, packed_run,
    try_forward_with, try_forward_with_aux, try_forward_packed_with, try_forward_packed_with_aux);

macro_rules! default_mhc_stack {
    ($backend:ty, [$($generics:tt)*], $with:ident, $aux:ident, $packed_with:ident, $packed_aux:ident,
        $forward:ident, $forward_aux:ident, $packed:ident, $forward_packed_aux:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, F: GatherMhcResidualBranch<$backend, B>>
            FullyShardedMhcResidualStack<$backend, P, F>
        where P::Gathered: CompressedAttentionProjection<$backend>, F::Gathered: MhcResidualBranch<$backend> {
            pub fn $forward<C: IntegerTensorCollective<B>>(&self, input: Tensor<$backend, 3>, valid: Option<Tensor<$backend, 2, Bool>>, communicator: C)
                -> Result<Tensor<$backend, 3>, FullyShardedMhcError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error>> {
                self.$with(input, valid, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
            pub fn $forward_aux<C: IntegerTensorCollective<B>>(&self, input: Tensor<$backend, 3>, valid: Option<Tensor<$backend, 2, Bool>>,
                indexer_warmup: bool, communicator: C) -> Result<CompressedAttentionOutput<$backend>, FullyShardedMhcError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error>> {
                self.$aux(input, valid, indexer_warmup, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
            pub fn $packed<C: IntegerTensorCollective<B>>(&self, input: Tensor<$backend, 2>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, communicator: C) -> Result<Tensor<$backend, 2>, FullyShardedMhcError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error>> {
                self.$packed_with(input, layout, valid, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
            pub fn $forward_packed_aux<C: IntegerTensorCollective<B>>(&self, input: Tensor<$backend, 2>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, indexer_warmup: bool, communicator: C)
                -> Result<PackedCompressedAttentionOutput<$backend>, FullyShardedMhcError<C::Error, <F::Gathered as MhcResidualBranch<$backend>>::Error>> {
                self.$packed_aux(input, layout, valid, indexer_warmup, communicator, |_, feed, input, _| feed.forward_branch(input))
            }
        }
    };
}
default_mhc_stack!(B, [B: Backend], try_forward_with_inference, try_forward_with_aux_inference, try_forward_packed_with_inference,
    try_forward_packed_with_aux_inference, forward_inference, forward_with_aux_inference, forward_packed_inference, forward_packed_with_aux_inference);
default_mhc_stack!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], try_forward_with, try_forward_with_aux, try_forward_packed_with,
    try_forward_packed_with_aux, forward, forward_with_aux, forward_packed, forward_packed_with_aux);

impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, F: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B>
    for FullyShardedMhcResidualBlock<B, P, F> {
    fn visit_shards<G: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut G) {
        self.attention_connection.visit_shards(visitor); self.ffn_connection.visit_shards(visitor); self.attention.visit_shards(visitor);
        self.attention_norm.visit_shards(visitor); self.ffn_norm.visit_shards(visitor); self.feed_forward.visit_shards(visitor);
    }
    fn visit_packed_shards<G: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut G) {
        self.attention.visit_packed_shards(visitor); self.feed_forward.visit_packed_shards(visitor);
    }
}
impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, F: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B>
    for FullyShardedMhcResidualStack<B, P, F> {
    fn visit_shards<G: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut G) { self.layers.visit_shards(visitor); self.final_norm.visit_shards(visitor); }
    fn visit_packed_shards<G: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut G) { self.layers.visit_packed_shards(visitor); }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, F: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B>
    for FullyShardedMhcResidualBlock<B, P, F> {
    fn visit_adapter_shards<G: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut G) {
        self.attention.visit_adapter_shards(visitor); self.feed_forward.visit_adapter_shards(visitor);
    }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, F: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B>
    for FullyShardedMhcResidualStack<B, P, F> {
    fn visit_adapter_shards<G: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut G) { self.layers.visit_adapter_shards(visitor); }
}
