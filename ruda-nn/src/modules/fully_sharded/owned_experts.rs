use super::*;
use ruda_model::{module::ModuleDisplay, tensor::IntegerTensorCollective};
use crate::{OwnedFloatingExpertAdapters, OwnedAwqExperts, OwnedPackedExperts, SelectableOwnedExperts, MixedOwnedExperts};
use crate::expert_parallel::{ExpertOwnership, ExpertParallelGeometry, ExpertParallelSwiGluExperts};

/// Data-shard an already expert-owned source. The context rank/world belongs to
/// replicas of this owner, not to the independent expert transport group.
pub trait ShardOwnedExperts<B: Backend>: ExpertParallelGeometry<B> {
    type Sharded: FullyShardedModule<B> + ModuleDisplay;
    fn shard_owned(self, context: &mut ShardingContext<B>) -> Self::Sharded;
}

/// Reconstruct the original owner, including empty ownership and native packed
/// windows, from the caller's data transport. No expert exchange occurs here.
pub trait GatherOwnedExperts<AB: Backend, B: Backend>: FullyShardedModule<AB> + ModuleDisplay {
    type Gathered: ExpertParallelGeometry<AB>;
    fn gather_owned<C: IntegerTensorCollective<B>>(&self, data: C) -> Result<Self::Gathered, C::Error>;
}

/// Local data slices of the original whole native received-expert cubes.
#[derive(Module, Debug)]
pub struct FullyShardedOwnedSwiGluExperts<B: Backend> {
    pub gate: ShardedParameter<B>,
    pub up: ShardedParameter<B>,
    pub down: ShardedParameter<B>,
    #[module(skip)] pub ownership: ExpertOwnership,
    pub rank: usize,
}

impl<B: Backend> FullyShardedOwnedSwiGluExperts<B> {
    pub fn dimensions(&self) -> [usize; 3] {
        assert_eq!(self.gate.logical_shape.len(), 3, "owned expert gate logical rank differs");
        let shape = &self.gate.logical_shape;
        [shape[0], shape[2], shape[1]]
    }

    /// The expert rank and data rank are checked independently; zero owners
    /// keep their actual empty cubes rather than a fabricated expert row.
    pub fn validate(&self) {
        let [owned, hidden, inner] = self.dimensions();
        assert_eq!(owned, self.ownership.range(self.rank).len(), "owned expert interval differs");
        assert!(hidden > 0 && inner > 0, "owned expert feature axes must be positive");
        assert_eq!(self.up.logical_shape, self.gate.logical_shape, "owned gate/up shapes differ");
        assert_eq!(self.down.logical_shape, [owned, hidden, inner], "owned down shape differs");
        let source = self.gate.local.val();
        assert!(matches!(source.dtype(), DType::F16 | DType::BF16 | DType::F32), "unsupported owned expert storage");
        for shard in [&self.gate, &self.up, &self.down] {
            assert_eq!(shard.local.val().dtype(), source.dtype(), "owned expert storage differs");
            assert_eq!(shard.local.val().device(), source.device(), "owned expert devices differ");
            assert_eq!((shard.rank, shard.world_size), (self.gate.rank, self.gate.world_size), "owned expert data topology differs");
            let _ = ShardedParameter::from_local(shard.local.clone(), shard.logical_shape.clone(), shard.rank, shard.world_size);
        }
    }
}

impl<B: Backend> ShardingContext<B> {
    pub fn owned_experts<E: ShardOwnedExperts<B>>(&mut self, source: E) -> E::Sharded { source.shard_owned(self) }

    pub fn owned_swiglu_experts(&mut self, source: ExpertParallelSwiGluExperts<B>) -> FullyShardedOwnedSwiGluExperts<B> {
        source.validate();
        let value = FullyShardedOwnedSwiGluExperts { gate: self.parameter(source.gate), up: self.parameter(source.up),
            down: self.parameter(source.down), ownership: source.ownership, rank: source.rank };
        value.validate(); value
    }
}
impl<B: Backend> ShardOwnedExperts<B> for ExpertParallelSwiGluExperts<B> {
    type Sharded = FullyShardedOwnedSwiGluExperts<B>;
    fn shard_owned(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.owned_swiglu_experts(self) }
}
impl<B: Backend> FullyShardedModule<B> for FullyShardedOwnedSwiGluExperts<B> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { visitor(&self.gate); visitor(&self.up); visitor(&self.down); }
}
impl<B: Backend> FullyShardedAdapterModule<B> for FullyShardedOwnedSwiGluExperts<B> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, _: &mut F) {}
}

macro_rules! owned_source {
    ($source:ident, $target:ident, $experts:ident) => {
        #[derive(Module, Debug)]
        pub struct $target<B: Backend> {
            pub experts: $experts<B>,
            #[module(skip)] pub ownership: ExpertOwnership,
            pub rank: usize,
        }
        impl<B: Backend> ShardOwnedExperts<B> for $source<B> {
            type Sharded = $target<B>;
            fn shard_owned(self, context: &mut ShardingContext<B>) -> Self::Sharded {
                self.validate(); $target { experts: self.experts.shard_experts(context), ownership: self.ownership, rank: self.rank }
            }
        }
        impl<B: Backend> FullyShardedModule<B> for $target<B> {
            fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.experts.visit_shards(visitor); }
            fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { self.experts.visit_packed_shards(visitor); }
        }
        impl<B: Backend> FullyShardedAdapterModule<B> for $target<B> {
            fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.experts.visit_adapter_shards(visitor); }
        }
    };
}
owned_source!(OwnedFloatingExpertAdapters, FullyShardedOwnedFloatingExpertAdapters, FullyShardedAdaptedFloatingSwiGluExperts);
owned_source!(OwnedAwqExperts, FullyShardedOwnedAwqExperts, FullyShardedSelectablePackedExperts);
owned_source!(OwnedPackedExperts, FullyShardedOwnedPackedExperts, FullyShardedSelectablePackedExperts);

#[derive(Module, Debug)]
pub enum FullyShardedSelectableOwnedExperts<B: Backend> {
    Original(FullyShardedOwnedSwiGluExperts<B>),
    Adapted(FullyShardedOwnedFloatingExpertAdapters<B>),
}
#[derive(Module, Debug)]
pub enum FullyShardedMixedOwnedExperts<B: Backend> {
    Floating(FullyShardedSelectableOwnedExperts<B>),
    Packed(FullyShardedOwnedPackedExperts<B>),
}
impl<B: Backend> ShardOwnedExperts<B> for SelectableOwnedExperts<B> {
    type Sharded = FullyShardedSelectableOwnedExperts<B>;
    fn shard_owned(self, context: &mut ShardingContext<B>) -> Self::Sharded {
        match self { Self::Original(value) => FullyShardedSelectableOwnedExperts::Original(value.shard_owned(context)),
            Self::Adapted(value) => FullyShardedSelectableOwnedExperts::Adapted(value.shard_owned(context)) }
    }
}
impl<B: Backend> ShardOwnedExperts<B> for MixedOwnedExperts<B> {
    type Sharded = FullyShardedMixedOwnedExperts<B>;
    fn shard_owned(self, context: &mut ShardingContext<B>) -> Self::Sharded {
        match self { Self::Floating(value) => FullyShardedMixedOwnedExperts::Floating(value.shard_owned(context)),
            Self::Packed(value) => FullyShardedMixedOwnedExperts::Packed(value.shard_owned(context)) }
    }
}

macro_rules! gather_owned_source {
    ($backend:ty, [$($generics:tt)*], $source:ident, $target:ident) => {
        impl<$($generics)*> GatherOwnedExperts<$backend, B> for $target<$backend> {
            type Gathered = $source<$backend>;
            fn gather_owned<C: IntegerTensorCollective<B>>(&self, data: C) -> Result<Self::Gathered, C::Error> {
                Ok($source::from_parts(self.experts.gather_experts(data)?, self.ownership.clone(), self.rank))
            }
        }
    };
}
macro_rules! gather_owned {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*> GatherOwnedExperts<$backend, B> for FullyShardedOwnedSwiGluExperts<$backend> {
            type Gathered = ExpertParallelSwiGluExperts<$backend>;
            fn gather_owned<C: IntegerTensorCollective<B>>(&self, data: C) -> Result<Self::Gathered, C::Error> {
                self.validate();
                Ok(ExpertParallelSwiGluExperts::from_parameters(
                    Param::initialized(self.gate.local.id, self.gate.$gather::<C, 3>(data.clone())?),
                    Param::initialized(self.up.local.id, self.up.$gather::<C, 3>(data.clone())?),
                    Param::initialized(self.down.local.id, self.down.$gather::<C, 3>(data)?), self.ownership.clone(), self.rank))
            }
        }
        gather_owned_source!($backend, [$($generics)*], OwnedFloatingExpertAdapters, FullyShardedOwnedFloatingExpertAdapters);
        gather_owned_source!($backend, [$($generics)*], OwnedAwqExperts, FullyShardedOwnedAwqExperts);
        gather_owned_source!($backend, [$($generics)*], OwnedPackedExperts, FullyShardedOwnedPackedExperts);
        impl<$($generics)*> GatherOwnedExperts<$backend, B> for FullyShardedSelectableOwnedExperts<$backend> {
            type Gathered = SelectableOwnedExperts<$backend>;
            fn gather_owned<C: IntegerTensorCollective<B>>(&self, data: C) -> Result<Self::Gathered, C::Error> {
                match self { Self::Original(value) => value.gather_owned(data).map(SelectableOwnedExperts::Original),
                    Self::Adapted(value) => value.gather_owned(data).map(SelectableOwnedExperts::Adapted) }
            }
        }
        impl<$($generics)*> GatherOwnedExperts<$backend, B> for FullyShardedMixedOwnedExperts<$backend> {
            type Gathered = MixedOwnedExperts<$backend>;
            fn gather_owned<C: IntegerTensorCollective<B>>(&self, data: C) -> Result<Self::Gathered, C::Error> {
                match self { Self::Floating(value) => value.gather_owned(data).map(MixedOwnedExperts::Floating),
                    Self::Packed(value) => value.gather_owned(data).map(MixedOwnedExperts::Packed) }
            }
        }
    };
}
gather_owned!(B, [B: Backend], gather_inference);
gather_owned!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

macro_rules! owned_visitors {
    ($target:ident, $first:ident, $second:ident) => {
        impl<B: Backend> FullyShardedModule<B> for $target<B> {
            fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) {
                match self { Self::$first(value) => value.visit_shards(visitor), Self::$second(value) => value.visit_shards(visitor) }
            }
            fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) {
                match self { Self::$first(value) => value.visit_packed_shards(visitor), Self::$second(value) => value.visit_packed_shards(visitor) }
            }
        }
        impl<B: Backend> FullyShardedAdapterModule<B> for $target<B> {
            fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) {
                match self { Self::$first(value) => value.visit_adapter_shards(visitor), Self::$second(value) => value.visit_adapter_shards(visitor) }
            }
        }
    };
}
owned_visitors!(FullyShardedSelectableOwnedExperts, Original, Adapted);
owned_visitors!(FullyShardedMixedOwnedExperts, Floating, Packed);
