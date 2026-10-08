use super::*;
use core::fmt;
use ruda_model::{module::ModuleDisplay, tensor::{IntegerTensorCollective, VariableTensorCollective, MoeDispatchOps, MoeReceivedOps, MoeOptions}};
use crate::expert_parallel::{ExpertParallelMoeLayer, ExpertParallelMoeOutput, ExpertParallelMoeError, ExpertParallelReceived};
use crate::transformer::TransformerProjection;

/// Expert-owned routed graph whose router, actual owned experts and correction
/// bias are separately data-sharded. Expert ownership remains in the expert
/// representation; the data communicator never substitutes for expert exchange.
#[derive(Module, Debug)]
pub struct FullyShardedExpertParallelMoeLayer<B: Backend, P: Module<B>, E: Module<B>> {
    pub router: P,
    pub experts: E,
    pub correction_bias: Option<ShardedParameter<B>>,
    #[module(skip)] pub options: MoeOptions,
    #[module(skip)] pub router_input_dtype: Option<FloatDType>,
}

impl<B: Backend> ShardingContext<B> {
    pub fn expert_parallel_layer<P: ShardTransformerProjection<B>, E: ShardOwnedExperts<B>>(&mut self,
        source: ExpertParallelMoeLayer<B, P, E>) -> FullyShardedExpertParallelMoeLayer<B, P::Sharded, E::Sharded> {
        source.validate();
        FullyShardedExpertParallelMoeLayer { router: source.router.shard(self), experts: source.experts.shard_owned(self),
            correction_bias: source.correction_bias.map(|value| self.parameter(value)), options: source.options, router_input_dtype: source.router_input_dtype }
    }
}
impl<B: Backend, P: Module<B>, E: Module<B>> FullyShardedExpertParallelMoeLayer<B, P, E> {
    pub fn from_owned<Q: ShardTransformerProjection<B, Sharded = P>, G: ShardOwnedExperts<B, Sharded = E>>(
        source: ExpertParallelMoeLayer<B, Q, G>, data_rank: usize, data_world: usize) -> Self {
        ShardingContext::new(data_rank, data_world).expert_parallel_layer(source)
    }
}

#[derive(Debug)]
pub enum FullyShardedExpertParallelError<D: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> {
    Data(D),
    Expert(ExpertParallelMoeError<C, P, E>),
}
impl<D: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> fmt::Display for FullyShardedExpertParallelError<D, C, P, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Data(error) => write!(f, "owned expert data gather: {error:?}"), Self::Expert(error) => write!(f, "{error}") }
    }
}
impl<D: fmt::Debug, C: fmt::Debug, P: fmt::Debug, E: fmt::Debug> core::error::Error for FullyShardedExpertParallelError<D, C, P, E> {}

macro_rules! gather_expert_parallel {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            FullyShardedExpertParallelMoeLayer<$backend, P, E> {
            /// Gather only this owner's values over the explicit data group,
            /// retaining the original global router width, options and leaf IDs.
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, data: C)
                -> Result<ExpertParallelMoeLayer<$backend, P::Gathered, E::Gathered>, C::Error> {
                Ok(ExpertParallelMoeLayer::from_expert_parts(self.router.gather_projection(data.clone())?, self.experts.gather_owned(data.clone())?,
                    self.correction_bias.as_ref().map(|bias| bias.$gather::<C, 1>(data).map(|value| Param::initialized(bias.local.id, value))).transpose()?,
                    self.options, self.router_input_dtype))
            }
        }
    };
}
gather_expert_parallel!(B, [B: Backend], gather_inference);
gather_expert_parallel!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

macro_rules! execute_expert_parallel {
    ($backend:ty, [$($generics:tt)*], $gather:ident, $native:ident, $forward:ident, $detailed:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherOwnedExperts<$backend, B>>
            FullyShardedExpertParallelMoeLayer<$backend, P, E>
        where P::Gathered: TransformerProjection<$backend>, E::Gathered: ExpertParallelReceived<$backend> {
            pub fn $forward<D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, const N: usize>(&self,
                input: Tensor<$backend, N>, data: D, expert: C)
                -> Result<Tensor<$backend, N>, FullyShardedExpertParallelError<D::Error, C::Error,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                self.$detailed(input, data, expert).map(|result| result.output)
            }

            /// Execute one actual router pass. Returned logits, selections and
            /// exchange counts are those used by this native expert graph.
            pub fn $detailed<D: IntegerTensorCollective<B>, C: VariableTensorCollective<B>, const N: usize>(&self,
                input: Tensor<$backend, N>, data: D, expert: C)
                -> Result<ExpertParallelMoeOutput<$backend, N>, FullyShardedExpertParallelError<D::Error, C::Error,
                    <P::Gathered as TransformerProjection<$backend>>::Error, <E::Gathered as ExpertParallelReceived<$backend>>::Error>> {
                self.$gather(data).map_err(FullyShardedExpertParallelError::Data)?
                    .$native(input, expert).map_err(FullyShardedExpertParallelError::Expert)
            }
        }
    };
}
execute_expert_parallel!(B, [B: MoeDispatchOps + MoeReceivedOps], gather_inference, forward_detailed_inference, forward_inference, forward_detailed_inference);
execute_expert_parallel!(Autodiff<B, S>, [B: MoeDispatchOps + MoeReceivedOps, S: CheckpointStrategy], gather, forward_detailed, forward, forward_detailed);

impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B>
    for FullyShardedExpertParallelMoeLayer<B, P, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) {
        self.router.visit_shards(visitor); self.experts.visit_shards(visitor); self.correction_bias.visit_shards(visitor);
    }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) {
        self.router.visit_packed_shards(visitor); self.experts.visit_packed_shards(visitor);
    }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B>
    for FullyShardedExpertParallelMoeLayer<B, P, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) {
        self.router.visit_adapter_shards(visitor); self.experts.visit_adapter_shards(visitor);
    }
}
