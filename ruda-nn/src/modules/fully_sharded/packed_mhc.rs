use super::*;
use ruda_model::{module::ModuleDisplay, tensor::{IntegerTensorCollective, FloatDType}};
use crate::{Nf4MoeLayer, Nf4MoeRouting, transformer::{PackedMhcFeedForward, MhcFeedForward}};

/// Exact original packed/floating-adapted expert representation, without a gathered dense base shadow.
#[derive(Module, Debug)]
pub struct FullyShardedPackedMoeLayer<B: Backend, P: Module<B>, E: Module<B>> {
    pub router: P,
    pub experts: E,
    pub correction_bias: Option<ShardedParameter<B>>,
    #[module(skip)] pub routing: Nf4MoeRouting,
    #[module(skip)] pub router_input_dtype: Option<FloatDType>,
}
#[derive(Module, Debug)]
pub struct FullyShardedPackedMhcFeedForward<B: Backend, P: Module<B>, E: Module<B>> {
    pub routed: FullyShardedPackedMoeLayer<B, P, E>,
    pub shared: Option<FullyShardedProjectedFeedForward<B, P>>,
}
#[derive(Module, Debug)]
pub enum FullyShardedMhcFeedForward<B: Backend, P: Module<B>, E: Module<B>> {
    Dense(FullyShardedProjectedFeedForward<B, P>),
    Floating(FullyShardedNativeMoeFeedForward<B, P>),
    Packed(FullyShardedPackedMhcFeedForward<B, P, E>),
}

impl<B: Backend> ShardingContext<B> {
    pub fn packed_moe_layer<P: ShardTransformerProjection<B>, E: ShardFrozenExperts<B>>(&mut self,
        source: Nf4MoeLayer<B, P, E>) -> FullyShardedPackedMoeLayer<B, P::Sharded, E::Sharded> {
        source.validate(); FullyShardedPackedMoeLayer { router: source.router.shard(self), experts: source.experts.shard_experts(self),
            correction_bias: source.correction_bias.map(|value| self.parameter(value)), routing: source.routing, router_input_dtype: source.router_input_dtype }
    }
    pub fn packed_mhc_feed_forward<P: ShardTransformerProjection<B>, E: ShardFrozenExperts<B>>(&mut self,
        source: PackedMhcFeedForward<B, P, E>) -> FullyShardedPackedMhcFeedForward<B, P::Sharded, E::Sharded> {
        FullyShardedPackedMhcFeedForward { routed: self.packed_moe_layer(source.routed), shared: source.shared.map(|value| self.awq_feed_forward(value)) }
    }
}
impl<B: Backend, P: ShardTransformerProjection<B>, E: ShardFrozenExperts<B>> ShardMhcResidualBranch<B> for PackedMhcFeedForward<B, P, E> {
    type Sharded = FullyShardedPackedMhcFeedForward<B, P::Sharded, E::Sharded>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.packed_mhc_feed_forward(self) }
}
impl<B: Backend, P: ShardTransformerProjection<B>, E: ShardFrozenExperts<B>> ShardMhcResidualBranch<B> for MhcFeedForward<B, P, E> {
    type Sharded = FullyShardedMhcFeedForward<B, P::Sharded, E::Sharded>;
    fn shard_branch(self, context: &mut ShardingContext<B>) -> Self::Sharded {
        match self { Self::Dense(value) => FullyShardedMhcFeedForward::Dense(value.shard_branch(context)),
            Self::Floating(value) => FullyShardedMhcFeedForward::Floating(value.shard_branch(context)),
            Self::Packed(value) => FullyShardedMhcFeedForward::Packed(value.shard_branch(context)) }
    }
}

macro_rules! gather_packed_mhc {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherFrozenExperts<$backend, B>> FullyShardedPackedMoeLayer<$backend, P, E> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Nf4MoeLayer<$backend, P::Gathered, E::Gathered>, C::Error> {
                Ok(Nf4MoeLayer::from_parts(self.router.gather_projection(communicator.clone())?, self.experts.gather_experts(communicator.clone())?,
                    self.correction_bias.as_ref().map(|bias| bias.$gather::<C, 1>(communicator).map(|value| Param::initialized(bias.local.id, value))).transpose()?,
                    self.routing, self.router_input_dtype))
            }
        }
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherFrozenExperts<$backend, B>> GatherMhcResidualBranch<$backend, B>
            for FullyShardedPackedMhcFeedForward<$backend, P, E> {
            type Gathered = PackedMhcFeedForward<$backend, P::Gathered, E::Gathered>;
            fn gather_branch<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                Ok(PackedMhcFeedForward::from_parts(self.routed.$gather(communicator.clone())?,
                    self.shared.as_ref().map(|value| value.$gather(communicator)).transpose()?))
            }
        }
        impl<$($generics)*, P: GatherTransformerProjection<$backend, B>, E: GatherFrozenExperts<$backend, B>> GatherMhcResidualBranch<$backend, B>
            for FullyShardedMhcFeedForward<$backend, P, E> {
            type Gathered = MhcFeedForward<$backend, P::Gathered, E::Gathered>;
            fn gather_branch<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                match self { Self::Dense(value) => value.gather_branch(communicator).map(MhcFeedForward::Dense),
                    Self::Floating(value) => value.gather_branch(communicator).map(MhcFeedForward::Floating),
                    Self::Packed(value) => value.gather_branch(communicator).map(MhcFeedForward::Packed) }
            }
        }
    };
}
gather_packed_mhc!(B, [B: Backend], gather_inference);
gather_packed_mhc!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for FullyShardedPackedMoeLayer<B, P, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.router.visit_shards(visitor); self.experts.visit_shards(visitor); self.correction_bias.visit_shards(visitor); }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { self.router.visit_packed_shards(visitor); self.experts.visit_packed_shards(visitor); }
}
impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for FullyShardedPackedMhcFeedForward<B, P, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.routed.visit_shards(visitor); self.shared.visit_shards(visitor); }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { self.routed.visit_packed_shards(visitor); self.shared.visit_packed_shards(visitor); }
}
impl<B: Backend, P: FullyShardedModule<B> + ModuleDisplay, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for FullyShardedMhcFeedForward<B, P, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Dense(value) => value.visit_shards(visitor), Self::Floating(value) => value.visit_shards(visitor), Self::Packed(value) => value.visit_shards(visitor) } }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Dense(value) => value.visit_packed_shards(visitor), Self::Floating(value) => value.visit_packed_shards(visitor), Self::Packed(value) => value.visit_packed_shards(visitor) } }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for FullyShardedPackedMoeLayer<B, P, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.router.visit_adapter_shards(visitor); self.experts.visit_adapter_shards(visitor); }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for FullyShardedPackedMhcFeedForward<B, P, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.routed.visit_adapter_shards(visitor); self.shared.visit_adapter_shards(visitor); }
}
impl<B: Backend, P: FullyShardedAdapterModule<B> + ModuleDisplay, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for FullyShardedMhcFeedForward<B, P, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Dense(value) => value.visit_adapter_shards(visitor), Self::Floating(value) => value.visit_adapter_shards(visitor), Self::Packed(value) => value.visit_adapter_shards(visitor) } }
}
