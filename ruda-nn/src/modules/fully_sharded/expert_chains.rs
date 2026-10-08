use super::*;
use ruda_model::{module::ModuleDisplay, tensor::IntegerTensorCollective};
use crate::{FrozenExpertGeometry, FrozenNf4SwiGluExperts, FrozenPackedSwiGluExperts, AdaptedPackedSwiGluExperts, AdaptedFloatingSwiGluExperts,
    SelectablePackedExperts, MixedAdaptedExperts};

macro_rules! shard_expert_chain {
    ($source:ident, $target:ident, $projection:ident, $context:ident, $project:ident) => {
        #[derive(Module, Debug)]
        pub struct $target<B: Backend> { pub gate: $projection<B>, pub up: $projection<B>, pub down: $projection<B> }
        impl<B: Backend> ShardingContext<B> {
            pub fn $context(&mut self, source: $source<B>) -> $target<B> {
                source.validate(); $target { gate: self.$project(source.gate), up: self.$project(source.up), down: self.$project(source.down) }
            }
        }
        impl<B: Backend> ShardFrozenExperts<B> for $source<B> {
            type Sharded = $target<B>;
            fn shard_experts(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.$context(self) }
        }
        impl<B: Backend> FullyShardedModule<B> for $target<B> {
            fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.gate.visit_shards(visitor); self.up.visit_shards(visitor); self.down.visit_shards(visitor); }
            fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { self.gate.visit_packed_shards(visitor); self.up.visit_packed_shards(visitor); self.down.visit_packed_shards(visitor); }
        }
        impl<B: Backend> FullyShardedAdapterModule<B> for $target<B> {
            fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.gate.visit_adapter_shards(visitor); self.up.visit_adapter_shards(visitor); self.down.visit_adapter_shards(visitor); }
        }
    };
}
shard_expert_chain!(FrozenNf4SwiGluExperts, FullyShardedNf4SwiGluExperts, FullyShardedNf4ExpertProjection, nf4_experts, nf4_expert_projection);
shard_expert_chain!(FrozenPackedSwiGluExperts, FullyShardedPackedSwiGluExperts, FullyShardedPackedExpertProjection, packed_experts, packed_expert_projection);
shard_expert_chain!(AdaptedPackedSwiGluExperts, FullyShardedAdaptedPackedSwiGluExperts, FullyShardedAdaptedPackedExpertProjection, adapted_packed_experts, adapted_packed_expert_projection);
shard_expert_chain!(AdaptedFloatingSwiGluExperts, FullyShardedAdaptedFloatingSwiGluExperts, FullyShardedFloatingExpertProjection, adapted_floating_experts, floating_expert_projection);

#[derive(Module, Debug)]
pub enum FullyShardedSelectablePackedExperts<B: Backend> {
    Original(FullyShardedPackedSwiGluExperts<B>),
    Adapted(FullyShardedAdaptedPackedSwiGluExperts<B>),
}
#[derive(Module, Debug)]
pub enum FullyShardedMixedAdaptedExperts<B: Backend, E: Module<B>> {
    Original(E),
    Packed(FullyShardedAdaptedPackedSwiGluExperts<B>),
    Floating(FullyShardedAdaptedFloatingSwiGluExperts<B>),
}
impl<B: Backend> ShardFrozenExperts<B> for SelectablePackedExperts<B> {
    type Sharded = FullyShardedSelectablePackedExperts<B>;
    fn shard_experts(self, context: &mut ShardingContext<B>) -> Self::Sharded {
        match self { Self::Original(value) => FullyShardedSelectablePackedExperts::Original(value.shard_experts(context)),
            Self::Adapted(value) => FullyShardedSelectablePackedExperts::Adapted(value.shard_experts(context)) }
    }
}
impl<B: Backend, E: ShardFrozenExperts<B>> ShardFrozenExperts<B> for MixedAdaptedExperts<B, E> {
    type Sharded = FullyShardedMixedAdaptedExperts<B, E::Sharded>;
    fn shard_experts(self, context: &mut ShardingContext<B>) -> Self::Sharded {
        match self { Self::Original(value) => FullyShardedMixedAdaptedExperts::Original(value.shard_experts(context)),
            Self::Packed(value) => FullyShardedMixedAdaptedExperts::Packed(value.shard_experts(context)),
            Self::Floating(value) => FullyShardedMixedAdaptedExperts::Floating(value.shard_experts(context)) }
    }
}

macro_rules! gather_expert_chains {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*> GatherFrozenExperts<$backend, B> for FullyShardedNf4SwiGluExperts<$backend> {
            type Gathered = FrozenNf4SwiGluExperts<$backend>;
            fn gather_experts<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                Ok(FrozenNf4SwiGluExperts::from_parts(self.gate.$gather(communicator.clone())?, self.up.$gather(communicator.clone())?, self.down.$gather(communicator)?))
            }
        }
        impl<$($generics)*> GatherFrozenExperts<$backend, B> for FullyShardedPackedSwiGluExperts<$backend> {
            type Gathered = FrozenPackedSwiGluExperts<$backend>;
            fn gather_experts<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                Ok(FrozenPackedSwiGluExperts::from_parts(self.gate.gather_base(communicator.clone())?, self.up.gather_base(communicator.clone())?, self.down.gather_base(communicator)?))
            }
        }
        impl<$($generics)*> GatherFrozenExperts<$backend, B> for FullyShardedAdaptedPackedSwiGluExperts<$backend> {
            type Gathered = AdaptedPackedSwiGluExperts<$backend>;
            fn gather_experts<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                Ok(AdaptedPackedSwiGluExperts::from_parts(self.gate.$gather(communicator.clone())?, self.up.$gather(communicator.clone())?, self.down.$gather(communicator)?))
            }
        }
        impl<$($generics)*> GatherFrozenExperts<$backend, B> for FullyShardedAdaptedFloatingSwiGluExperts<$backend> {
            type Gathered = AdaptedFloatingSwiGluExperts<$backend>;
            fn gather_experts<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                Ok(AdaptedFloatingSwiGluExperts::from_parts(self.gate.$gather(communicator.clone())?, self.up.$gather(communicator.clone())?, self.down.$gather(communicator)?))
            }
        }
        impl<$($generics)*> GatherFrozenExperts<$backend, B> for FullyShardedSelectablePackedExperts<$backend> {
            type Gathered = SelectablePackedExperts<$backend>;
            fn gather_experts<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                match self { Self::Original(value) => value.gather_experts(communicator).map(SelectablePackedExperts::Original),
                    Self::Adapted(value) => value.gather_experts(communicator).map(SelectablePackedExperts::Adapted) }
            }
        }
        impl<$($generics)*, E: GatherFrozenExperts<$backend, B>> GatherFrozenExperts<$backend, B> for FullyShardedMixedAdaptedExperts<$backend, E> {
            type Gathered = MixedAdaptedExperts<$backend, E::Gathered>;
            fn gather_experts<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                match self { Self::Original(value) => value.gather_experts(communicator).map(MixedAdaptedExperts::Original),
                    Self::Packed(value) => value.gather_experts(communicator).map(MixedAdaptedExperts::Packed),
                    Self::Floating(value) => value.gather_experts(communicator).map(MixedAdaptedExperts::Floating) }
            }
        }
    };
}
gather_expert_chains!(B, [B: Backend], gather_inference);
gather_expert_chains!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

impl<B: Backend> FullyShardedModule<B> for FullyShardedSelectablePackedExperts<B> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Original(value) => value.visit_shards(visitor), Self::Adapted(value) => value.visit_shards(visitor) } }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Original(value) => value.visit_packed_shards(visitor), Self::Adapted(value) => value.visit_packed_shards(visitor) } }
}
impl<B: Backend> FullyShardedAdapterModule<B> for FullyShardedSelectablePackedExperts<B> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Original(value) => value.visit_adapter_shards(visitor), Self::Adapted(value) => value.visit_adapter_shards(visitor) } }
}
impl<B: Backend, E: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for FullyShardedMixedAdaptedExperts<B, E> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Original(value) => value.visit_shards(visitor), Self::Packed(value) => value.visit_shards(visitor), Self::Floating(value) => value.visit_shards(visitor) } }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Original(value) => value.visit_packed_shards(visitor), Self::Packed(value) => value.visit_packed_shards(visitor), Self::Floating(value) => value.visit_packed_shards(visitor) } }
}
impl<B: Backend, E: FullyShardedAdapterModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for FullyShardedMixedAdaptedExperts<B, E> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { Self::Original(value) => value.visit_adapter_shards(visitor), Self::Packed(value) => value.visit_adapter_shards(visitor), Self::Floating(value) => value.visit_adapter_shards(visitor) } }
}
