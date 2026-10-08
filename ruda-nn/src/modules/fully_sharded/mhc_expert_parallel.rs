use super::*;
use ruda_model::{module::ModuleDisplay, tensor::{Bool, IntegerTensorCollective, VariableTensorCollective, MoeDispatchOps, MoeReceivedOps}};
use crate::expert_parallel::ExpertParallelReceived;
use crate::transformer::{ExpertParallelMhcFeedForward, ExpertParallelMhcError, MixedMhcFeedForward, MixedMhcBranchError,
    MhcResidualBranchShape, MhcResidualBranch, TransformerProjection};
use crate::attention::{CompressedAttentionProjection, CompressedAttentionOutput, PackedCompressedAttentionOutput, PackedSequenceLayout};

/// Original expert-owned mHC branch and explicitly present shared FFN, with all
/// persistent native weights kept as local data-axis slices.
#[derive(Module, Debug)]
pub struct FullyShardedExpertParallelMhcFeedForward<B: Backend, P: Module<B>, E: Module<B>> {
    pub routed: FullyShardedExpertParallelMoeLayer<B, P, E>,
    pub shared: Option<FullyShardedProjectedFeedForward<B, P>>,
}
#[derive(Module, Debug)]
pub enum FullyShardedMixedMhcFeedForward<B: Backend, L: Module<B>, P: Module<B>, E: Module<B>> {
    Local(L),
    Parallel(FullyShardedExpertParallelMhcFeedForward<B, P, E>),
}

impl<B: Backend> ShardingContext<B> {
    pub fn expert_parallel_mhc<P: ShardTransformerProjection<B>, E: ShardOwnedExperts<B>>(&mut self,
        source: ExpertParallelMhcFeedForward<B, P, E>) -> FullyShardedExpertParallelMhcFeedForward<B, P::Sharded, E::Sharded> {
        source.validate_branch();
        FullyShardedExpertParallelMhcFeedForward { routed: self.expert_parallel_layer(source.routed),
            shared: source.shared.map(|value| self.awq_feed_forward(value)) }
    }
}
impl<B: Backend, P: ShardTransformerProjection<B>, E: ShardOwnedExperts<B>> ShardMhcResidualBranch<B> for ExpertParallelMhcFeedForward<B, P, E> {
    type Sharded = FullyShardedExpertParallelMhcFeedForward<B, P::Sharded, E::Sharded>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.expert_parallel_mhc(self) }
}
impl<B: Backend, L: ShardMhcResidualBranch<B>, P: ShardTransformerProjection<B>, E: ShardOwnedExperts<B>> ShardMhcResidualBranch<B>
    for MixedMhcFeedForward<B, L, P, E> {
    type Sharded = FullyShardedMixedMhcFeedForward<B, L::Sharded, P::Sharded, E::Sharded>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded {
        match self { Self::Local(value) => FullyShardedMixedMhcFeedForward::Local(value.shard_branch(context)),
            Self::Parallel(value) => FullyShardedMixedMhcFeedForward::Parallel(value.shard_branch(context)) }
    }
}

macro_rules! gather_mhc_expert_parallel {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>> GatherMhcResidualBranch<$backend, B>
            for FullyShardedExpertParallelMhcFeedForward<$backend, P, E> {
            type Gathered = ExpertParallelMhcFeedForward<$backend, P::Gathered, E::Gathered>;
            fn gather_branch<C: IntegerTensorCollective<B>>(&self, data: C) -> Result<Self::Gathered, C::Error> {
                Ok(ExpertParallelMhcFeedForward::from_parts(self.routed.$gather(data.clone())?,
                    self.shared.as_ref().map(|value| value.$gather(data)).transpose()?))
            }
        }
        impl<$($generics)*, L: GatherMhcResidualBranch<$backend, B>, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            GatherMhcResidualBranch<$backend, B> for FullyShardedMixedMhcFeedForward<$backend, L, P, E> {
            type Gathered = MixedMhcFeedForward<$backend, L::Gathered, P::Gathered, E::Gathered>;
            fn gather_branch<C: IntegerTensorCollective<B>>(&self, data: C) -> Result<Self::Gathered, C::Error> {
                match self { Self::Local(value) => value.gather_branch(data).map(MixedMhcFeedForward::Local),
                    Self::Parallel(value) => value.gather_branch(data).map(MixedMhcFeedForward::Parallel) }
            }
        }
    };
}
gather_mhc_expert_parallel!(B, [B: Backend], gather_inference);
gather_mhc_expert_parallel!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

#[derive(Debug)]
pub enum FullyShardedExpertParallelMhcError<D: core::fmt::Debug, C: core::fmt::Debug, P: core::fmt::Debug, E: core::fmt::Debug> {
    Data(D),
    Branch(ExpertParallelMhcError<C, P, E>),
}
impl<D: core::fmt::Debug, C: core::fmt::Debug, P: core::fmt::Debug, E: core::fmt::Debug> core::fmt::Display for FullyShardedExpertParallelMhcError<D, C, P, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self { Self::Data(error) => write!(f, "mHC owned branch data gather: {error:?}"), Self::Branch(error) => write!(f, "{error}") }
    }
}
impl<D: core::fmt::Debug, C: core::fmt::Debug, P: core::fmt::Debug, E: core::fmt::Debug> core::error::Error for FullyShardedExpertParallelMhcError<D, C, P, E> {}

macro_rules! execute_sharded_mhc_experts {
    ($backend:ty, [$($generics:tt)*], $native:ident, $forward:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            FullyShardedExpertParallelMhcFeedForward<$backend, P, E>
        where P::Gathered: TransformerProjection<$backend>, E::Gathered: ExpertParallelReceived<$backend> {
            /// Separate caller-owned groups/scopes retain their own transport
            /// derivatives. Objective weighting/completion remains caller-selected.
            pub fn $forward<D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, const N: usize>(&self,
                input: Tensor<$backend, N>, data: D, expert: C)
                -> Result<Tensor<$backend, N>, FullyShardedExpertParallelMhcError<D::Error, C::Error,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                self.gather_branch(data).map_err(FullyShardedExpertParallelMhcError::Data)?
                    .$native(input, expert).map_err(FullyShardedExpertParallelMhcError::Branch)
            }
        }
    };
}
execute_sharded_mhc_experts!(B, [B: MoeDispatchOps + MoeReceivedOps], forward_inference, forward_inference);
execute_sharded_mhc_experts!(Autodiff<B, S>, [B: MoeDispatchOps + MoeReceivedOps, S: CheckpointStrategy], forward, forward);

/// Native branch errors retain local, router, owned-expert and expert-transport
/// categories, separately from data gathers and the actual task-head error.
pub type FullyShardedMixedMhcModelError<D, C, L, P, E, H> = FullyShardedMhcModelError<D, MixedMhcBranchError<L, C, P, E>, H>;

macro_rules! execute_mixed_mhc_model {
    ($backend:ty, [$($generics:tt)*], $native:ident, $with:ident, $hidden_with:ident, $aux_with:ident, $packed_with:ident, $packed_aux_with:ident,
        $forward:ident, $hidden:ident, $aux:ident, $packed:ident, $packed_aux:ident) => {
        impl<$($generics)*, A: GatherTransformerProjection<$backend, B>, L: GatherMhcResidualBranch<$backend, B>,
            P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>, H: GatherTransformerProjection<$backend, B>>
            FullyShardedMhcResidualModel<$backend, A, FullyShardedMixedMhcFeedForward<$backend, L, P, E>, H>
        where A::Gathered: CompressedAttentionProjection<$backend>, L::Gathered: MhcResidualBranch<$backend>,
            P::Gathered: TransformerProjection<$backend>, E::Gathered: ExpertParallelReceived<$backend>, H::Gathered: TransformerProjection<$backend> {
            /// The expert-group factory is evaluated only on actual parallel
            /// branches. It may return separately scope-bound per-layer groups;
            /// data gather order and original native branch graphs are unchanged.
            pub fn $forward<D, C, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>, data: D, mut experts: G)
                -> Result<Tensor<$backend, 3>, FullyShardedMixedMhcModelError<D::Error, C::Error,
                    <L::Gathered as MhcResidualBranch<$backend>>::Error, <P::Gathered as TransformerProjection<$backend>>::Error,
                    <E::Gathered as ExpertParallelReceived<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, G: FnMut(usize) -> C {
                self.$with(tokens, valid, data, |index, feed, input, _| feed.$native(input, || experts(index)))
            }
            pub fn $hidden<D, C, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>, data: D, mut experts: G)
                -> Result<Tensor<$backend, 3>, FullyShardedMixedMhcModelError<D::Error, C::Error,
                    <L::Gathered as MhcResidualBranch<$backend>>::Error, <P::Gathered as TransformerProjection<$backend>>::Error,
                    <E::Gathered as ExpertParallelReceived<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, G: FnMut(usize) -> C {
                self.$hidden_with(tokens, valid, data, |index, feed, input, _| feed.$native(input, || experts(index)))
            }
            pub fn $aux<D, C, G>(&self, tokens: Tensor<$backend, 2, Int>, valid: Option<Tensor<$backend, 2, Bool>>,
                indexer_warmup: bool, data: D, mut experts: G)
                -> Result<CompressedAttentionOutput<$backend>, FullyShardedMixedMhcModelError<D::Error, C::Error,
                    <L::Gathered as MhcResidualBranch<$backend>>::Error, <P::Gathered as TransformerProjection<$backend>>::Error,
                    <E::Gathered as ExpertParallelReceived<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, G: FnMut(usize) -> C {
                self.$aux_with(tokens, valid, indexer_warmup, data, |index, feed, input, _| feed.$native(input, || experts(index)))
            }
            pub fn $packed<D, C, G>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, data: D, mut experts: G)
                -> Result<Tensor<$backend, 2>, FullyShardedMixedMhcModelError<D::Error, C::Error,
                    <L::Gathered as MhcResidualBranch<$backend>>::Error, <P::Gathered as TransformerProjection<$backend>>::Error,
                    <E::Gathered as ExpertParallelReceived<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, G: FnMut(usize) -> C {
                self.$packed_with(tokens, layout, valid, data, |index, feed, input, _| feed.$native(input, || experts(index)))
            }
            pub fn $packed_aux<D, C, G>(&self, tokens: Tensor<$backend, 1, Int>, layout: &PackedSequenceLayout,
                valid: Option<Tensor<$backend, 1, Bool>>, indexer_warmup: bool, data: D, mut experts: G)
                -> Result<PackedCompressedAttentionOutput<$backend>, FullyShardedMixedMhcModelError<D::Error, C::Error,
                    <L::Gathered as MhcResidualBranch<$backend>>::Error, <P::Gathered as TransformerProjection<$backend>>::Error,
                    <E::Gathered as ExpertParallelReceived<$backend>>::Error, <H::Gathered as TransformerProjection<$backend>>::Error>>
            where D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, G: FnMut(usize) -> C {
                self.$packed_aux_with(tokens, layout, valid, indexer_warmup, data, |index, feed, input, _| feed.$native(input, || experts(index)))
            }
        }
    };
}
execute_mixed_mhc_model!(B, [B: MoeDispatchOps + MoeReceivedOps], forward_inference,
    try_forward_with_inference, try_forward_hidden_with_inference, try_forward_with_aux_inference, try_forward_packed_with_inference, try_forward_packed_with_aux_inference,
    forward_with_experts_inference, forward_hidden_with_experts_inference, forward_with_experts_aux_inference, forward_packed_with_experts_inference, forward_packed_with_experts_aux_inference);
execute_mixed_mhc_model!(Autodiff<B, S>, [B: MoeDispatchOps + MoeReceivedOps, S: CheckpointStrategy], forward,
    try_forward_with, try_forward_hidden_with, try_forward_with_aux, try_forward_packed_with, try_forward_packed_with_aux,
    forward_with_experts, forward_hidden_with_experts, forward_with_experts_aux, forward_packed_with_experts, forward_packed_with_experts_aux);

impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B>
    for FullyShardedExpertParallelMhcFeedForward<B, P, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.routed.visit_shards(visitor); self.shared.visit_shards(visitor); }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { self.routed.visit_packed_shards(visitor); self.shared.visit_packed_shards(visitor); }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B>
    for FullyShardedExpertParallelMhcFeedForward<B, P, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.routed.visit_adapter_shards(visitor); self.shared.visit_adapter_shards(visitor); }
}
impl<B: Backend, L: FullyShardedModule<B> + ModuleDisplay, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay>
    FullyShardedModule<B> for FullyShardedMixedMhcFeedForward<B, L, P, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Local(value) => value.visit_shards(visitor), Self::Parallel(value) => value.visit_shards(visitor) } }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Local(value) => value.visit_packed_shards(visitor), Self::Parallel(value) => value.visit_packed_shards(visitor) } }
}
impl<B: Backend, L: FullyShardedAdapterModule<B> + ModuleDisplay, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay>
    FullyShardedAdapterModule<B> for FullyShardedMixedMhcFeedForward<B, L, P, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Local(value) => value.visit_adapter_shards(visitor), Self::Parallel(value) => value.visit_adapter_shards(visitor) } }
}
