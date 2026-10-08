use super::*;
use ruda_model::{module::ModuleDisplay, tensor::{IntegerTensorCollective, MoeExpertStrategy}};
use crate::{FrozenExpertGeometry, ExpertLoRABase, FrozenNf4ExpertProjection, FrozenAwqExpertProjection, FrozenNf4ExpertWindow,
    FrozenPackedExpertProjection, ExpertLinear, PackedExpertLoRA, AdaptedExpertProjection, FloatingExpertProjection};

pub trait ShardExpertBase<B: Backend>: ExpertLoRABase<B> {
    type Sharded: FullyShardedModule<B> + ModuleDisplay;
    fn shard_base(self, context: &mut ShardingContext<B>) -> Self::Sharded;
}
pub trait GatherExpertBase<AB: Backend, B: Backend>: FullyShardedModule<AB> + ModuleDisplay {
    type Gathered: ExpertLoRABase<AB>;
    fn gather_base<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error>;
}
pub trait ShardFrozenExperts<B: Backend>: FrozenExpertGeometry<B> {
    type Sharded: FullyShardedModule<B> + ModuleDisplay;
    fn shard_experts(self, context: &mut ShardingContext<B>) -> Self::Sharded;
}
pub trait GatherFrozenExperts<AB: Backend, B: Backend>: FullyShardedModule<AB> + ModuleDisplay {
    type Gathered: FrozenExpertGeometry<AB>;
    fn gather_experts<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error>;
}

#[derive(Module, Debug)]
pub struct FullyShardedNf4ExpertProjection<B: Backend> {
    pub payload: FullyShardedNf4Linear<B>,
    pub experts: usize,
    pub output_features: usize,
}
#[derive(Module, Debug)]
pub struct FullyShardedAwqExpertProjection<B: Backend> {
    pub qweight: ShardedPackedParameter<B>,
    pub qzeros: ShardedPackedParameter<B>,
    pub scales: ShardedParameter<B>,
    pub bias: Option<ShardedParameter<B>>,
    pub group_size: usize,
}
/// Original flat-block byte window, including a zero-expert owner and its unchanged codebook.
#[derive(Module, Debug)]
pub struct FullyShardedNf4ExpertWindow<B: Backend> {
    pub packed: ShardedPackedParameter<B>,
    pub scales: ShardedParameter<B>,
    pub codebook: ShardedParameter<B>,
    pub experts: usize,
    pub input_features: usize,
    pub output_features: usize,
    pub block_size: usize,
    pub element_offset: usize,
    pub tile_rows: usize,
    pub use_tensor_core: bool,
}
#[derive(Module, Debug)]
pub enum FullyShardedPackedExpertProjection<B: Backend> {
    Nf4(FullyShardedNf4ExpertProjection<B>),
    Awq(FullyShardedAwqExpertProjection<B>),
    Nf4Window(FullyShardedNf4ExpertWindow<B>),
}
#[derive(Module, Debug)]
pub struct FullyShardedExpertLinear<B: Backend> {
    pub weight: ShardedParameter<B>,
    #[module(skip)] pub forward_strategy: MoeExpertStrategy,
    #[module(skip)] pub backward_strategy: MoeExpertStrategy,
}
#[derive(Module, Debug)]
pub struct FullyShardedExpertLoRA<B: Backend, Base: Module<B>> {
    pub base: Base,
    pub adapter_a: FullyShardedExpertLinear<B>,
    pub adapter_b: FullyShardedExpertLinear<B>,
    pub dropout: crate::Dropout,
    pub scale: f64,
}
#[derive(Module, Debug)]
pub enum FullyShardedAdaptedPackedExpertProjection<B: Backend> {
    Frozen(FullyShardedPackedExpertProjection<B>),
    LoRA(FullyShardedExpertLoRA<B, FullyShardedPackedExpertProjection<B>>),
}
#[derive(Module, Debug)]
pub enum FullyShardedFloatingExpertProjection<B: Backend> {
    Dense(FullyShardedExpertLinear<B>),
    LoRA(FullyShardedExpertLoRA<B, FullyShardedExpertLinear<B>>),
}

impl<B: Backend> ShardingContext<B> {
    pub fn nf4_expert_projection(&mut self, source: FrozenNf4ExpertProjection<B>) -> FullyShardedNf4ExpertProjection<B> {
        source.validate(); FullyShardedNf4ExpertProjection { payload: self.nf4(source.payload), experts: source.experts, output_features: source.output_features }
    }
    pub fn awq_expert_projection(&mut self, source: FrozenAwqExpertProjection<B>) -> FullyShardedAwqExpertProjection<B> {
        source.validate(); FullyShardedAwqExpertProjection { qweight: self.packed_parameter(source.qweight), qzeros: self.packed_parameter(source.qzeros),
            scales: self.parameter(source.scales), bias: source.bias.map(|value| self.parameter(value)), group_size: source.group_size }
    }
    pub fn nf4_expert_window(&mut self, source: FrozenNf4ExpertWindow<B>) -> FullyShardedNf4ExpertWindow<B> {
        source.validate(); FullyShardedNf4ExpertWindow { packed: self.packed_parameter(source.packed), scales: self.parameter(source.scales), codebook: self.parameter(source.codebook),
            experts: source.experts, input_features: source.input_features, output_features: source.output_features, block_size: source.block_size,
            element_offset: source.element_offset, tile_rows: source.tile_rows, use_tensor_core: source.use_tensor_core }
    }
    pub fn packed_expert_projection(&mut self, source: FrozenPackedExpertProjection<B>) -> FullyShardedPackedExpertProjection<B> {
        match source { FrozenPackedExpertProjection::Nf4(value) => FullyShardedPackedExpertProjection::Nf4(self.nf4_expert_projection(value)),
            FrozenPackedExpertProjection::Awq(value) => FullyShardedPackedExpertProjection::Awq(self.awq_expert_projection(value)),
            FrozenPackedExpertProjection::Nf4Window(value) => FullyShardedPackedExpertProjection::Nf4Window(self.nf4_expert_window(value)) }
    }
    pub fn expert_linear(&mut self, source: ExpertLinear<B>) -> FullyShardedExpertLinear<B> {
        source.validate(); FullyShardedExpertLinear { weight: self.parameter(source.weight), forward_strategy: source.forward_strategy, backward_strategy: source.backward_strategy }
    }
    pub fn expert_lora<Base: ShardExpertBase<B>>(&mut self, source: PackedExpertLoRA<B, Base>) -> FullyShardedExpertLoRA<B, Base::Sharded> {
        source.validate(); FullyShardedExpertLoRA { base: source.base.shard_base(self), adapter_a: self.expert_linear(source.adapter_a),
            adapter_b: self.expert_linear(source.adapter_b), dropout: source.dropout, scale: source.scale }
    }
    pub fn adapted_packed_expert_projection(&mut self, source: AdaptedExpertProjection<B>) -> FullyShardedAdaptedPackedExpertProjection<B> {
        match source { AdaptedExpertProjection::Frozen(value) => FullyShardedAdaptedPackedExpertProjection::Frozen(self.packed_expert_projection(value)),
            AdaptedExpertProjection::LoRA(value) => FullyShardedAdaptedPackedExpertProjection::LoRA(self.expert_lora(value)) }
    }
    pub fn floating_expert_projection(&mut self, source: FloatingExpertProjection<B>) -> FullyShardedFloatingExpertProjection<B> {
        match source { FloatingExpertProjection::Dense(value) => FullyShardedFloatingExpertProjection::Dense(self.expert_linear(value)),
            FloatingExpertProjection::LoRA(value) => FullyShardedFloatingExpertProjection::LoRA(self.expert_lora(value)) }
    }
}
impl<B: Backend> ShardExpertBase<B> for FrozenPackedExpertProjection<B> {
    type Sharded = FullyShardedPackedExpertProjection<B>;
    fn shard_base(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.packed_expert_projection(self) }
}
impl<B: Backend> ShardExpertBase<B> for ExpertLinear<B> {
    type Sharded = FullyShardedExpertLinear<B>;
    fn shard_base(self, context: &mut ShardingContext<B>) -> Self::Sharded { context.expert_linear(self) }
}

macro_rules! gather_expert_projections {
    ($backend:ty, [$($generics:tt)*], $gather:ident) => {
        impl<$($generics)*> FullyShardedNf4ExpertProjection<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<FrozenNf4ExpertProjection<$backend>, C::Error> {
                Ok(FrozenNf4ExpertProjection::from_flat_payload(self.payload.$gather(communicator)?, self.experts, self.output_features))
            }
        }
        impl<$($generics)*> FullyShardedAwqExpertProjection<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<FrozenAwqExpertProjection<$backend>, C::Error> {
                let value = FrozenAwqExpertProjection { qweight: Param::initialized(self.qweight.local.id, self.qweight.$gather::<C, 3>(communicator.clone())?),
                    qzeros: Param::initialized(self.qzeros.local.id, self.qzeros.$gather::<C, 3>(communicator.clone())?),
                    scales: Param::initialized(self.scales.local.id, self.scales.$gather::<C, 3>(communicator.clone())?),
                    bias: self.bias.as_ref().map(|bias| bias.$gather::<C, 2>(communicator).map(|value| Param::initialized(bias.local.id, value))).transpose()?, group_size: self.group_size };
                value.validate(); Ok(value)
            }
        }
        impl<$($generics)*> FullyShardedNf4ExpertWindow<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<FrozenNf4ExpertWindow<$backend>, C::Error> {
                let value = FrozenNf4ExpertWindow { packed: Param::initialized(self.packed.local.id, self.packed.$gather::<C, 1>(communicator.clone())?),
                    scales: Param::initialized(self.scales.local.id, self.scales.$gather::<C, 1>(communicator.clone())?),
                    codebook: Param::initialized(self.codebook.local.id, self.codebook.$gather::<C, 1>(communicator)?), experts: self.experts,
                    input_features: self.input_features, output_features: self.output_features, block_size: self.block_size, element_offset: self.element_offset,
                    tile_rows: self.tile_rows, use_tensor_core: self.use_tensor_core };
                value.validate(); Ok(value)
            }
        }
        impl<$($generics)*> GatherExpertBase<$backend, B> for FullyShardedPackedExpertProjection<$backend> {
            type Gathered = FrozenPackedExpertProjection<$backend>;
            fn gather_base<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                match self { Self::Nf4(value) => value.$gather(communicator).map(FrozenPackedExpertProjection::Nf4),
                    Self::Awq(value) => value.$gather(communicator).map(FrozenPackedExpertProjection::Awq),
                    Self::Nf4Window(value) => value.$gather(communicator).map(FrozenPackedExpertProjection::Nf4Window) }
            }
        }
        impl<$($generics)*> GatherExpertBase<$backend, B> for FullyShardedExpertLinear<$backend> {
            type Gathered = ExpertLinear<$backend>;
            fn gather_base<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<Self::Gathered, C::Error> {
                Ok(ExpertLinear::from_parameters(Param::initialized(self.weight.local.id, self.weight.$gather::<C, 3>(communicator)?), self.forward_strategy, self.backward_strategy))
            }
        }
        impl<$($generics)*, Base: GatherExpertBase<$backend, B>> FullyShardedExpertLoRA<$backend, Base> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<PackedExpertLoRA<$backend, Base::Gathered>, C::Error> {
                let value = PackedExpertLoRA { base: self.base.gather_base(communicator.clone())?, adapter_a: self.adapter_a.gather_base(communicator.clone())?,
                    adapter_b: self.adapter_b.gather_base(communicator)?, dropout: self.dropout.clone(), scale: self.scale };
                value.validate(); Ok(value)
            }
        }
        impl<$($generics)*> FullyShardedAdaptedPackedExpertProjection<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<AdaptedExpertProjection<$backend>, C::Error> {
                match self { Self::Frozen(value) => value.gather_base(communicator).map(AdaptedExpertProjection::Frozen),
                    Self::LoRA(value) => value.$gather(communicator).map(AdaptedExpertProjection::LoRA) }
            }
        }
        impl<$($generics)*> FullyShardedFloatingExpertProjection<$backend> {
            pub fn $gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<FloatingExpertProjection<$backend>, C::Error> {
                match self { Self::Dense(value) => value.gather_base(communicator).map(FloatingExpertProjection::Dense),
                    Self::LoRA(value) => value.$gather(communicator).map(FloatingExpertProjection::LoRA) }
            }
        }
    };
}
gather_expert_projections!(B, [B: Backend], gather_inference);
gather_expert_projections!(Autodiff<B, S>, [B: Backend, S: CheckpointStrategy], gather);

macro_rules! expert_fields {
    ($module:ident, [$($field:ident),+]) => {
        impl<B: Backend> FullyShardedModule<B> for $module<B> {
            fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_shards(visitor);)+ }
            fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { $(self.$field.visit_packed_shards(visitor);)+ }
        }
    };
}
expert_fields!(FullyShardedNf4ExpertProjection, [payload]);
expert_fields!(FullyShardedAwqExpertProjection, [qweight, qzeros, scales, bias]);
expert_fields!(FullyShardedNf4ExpertWindow, [packed, scales, codebook]);
expert_fields!(FullyShardedExpertLinear, [weight]);
macro_rules! expert_variants {
    ($module:ident, [$($variant:ident),+]) => {
        impl<B: Backend> FullyShardedModule<B> for $module<B> {
            fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { $(Self::$variant(value) => value.visit_shards(visitor),)+ } }
            fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { match self { $(Self::$variant(value) => value.visit_packed_shards(visitor),)+ } }
        }
        impl<B: Backend> FullyShardedAdapterModule<B> for $module<B> {
            fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { match self { $(Self::$variant(value) => value.visit_adapter_shards(visitor),)+ } }
        }
    };
}
expert_variants!(FullyShardedPackedExpertProjection, [Nf4, Awq, Nf4Window]);
expert_variants!(FullyShardedAdaptedPackedExpertProjection, [Frozen, LoRA]);
expert_variants!(FullyShardedFloatingExpertProjection, [Dense, LoRA]);
macro_rules! frozen_projection_adapters {
    ($($module:ident),+) => { $(impl<B: Backend> FullyShardedAdapterModule<B> for $module<B> {
        fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, _: &mut F) {}
    })+ };
}
frozen_projection_adapters!(FullyShardedNf4ExpertProjection, FullyShardedAwqExpertProjection, FullyShardedNf4ExpertWindow, FullyShardedExpertLinear);
impl<B: Backend, Base: FullyShardedModule<B> + ModuleDisplay> FullyShardedModule<B> for FullyShardedExpertLoRA<B, Base> {
    fn visit_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.base.visit_shards(visitor); self.adapter_a.visit_shards(visitor); self.adapter_b.visit_shards(visitor); }
    fn visit_packed_shards<F: FnMut(&ShardedPackedParameter<B>)>(&self, visitor: &mut F) { self.base.visit_packed_shards(visitor); }
}
impl<B: Backend, Base: FullyShardedModule<B> + ModuleDisplay> FullyShardedAdapterModule<B> for FullyShardedExpertLoRA<B, Base> {
    fn visit_adapter_shards<F: FnMut(&ShardedParameter<B>)>(&self, visitor: &mut F) { self.adapter_a.visit_shards(visitor); self.adapter_b.visit_shards(visitor); }
}
