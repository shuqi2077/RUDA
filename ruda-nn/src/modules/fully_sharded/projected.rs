use super::*;
use ruda_model::{module::ModuleDisplay,tensor::IntegerTensorCollective};
use crate::transformer::{TransformerProjectionShape,Nf4TransformerProjection,MixedTransformerProjection,
    AwqTransformerProjection,ProjectedTransformerStack,ProjectedTransformerModel};

/// Partition actual projection leaves through the caller's canonical shared-ID context.
pub trait ShardTransformerProjection<B:Backend>:TransformerProjectionShape<B> {
    /// Original selected projection with only local persistent integer/floating leaves.
    type Sharded:FullyShardedModule<B>+ModuleDisplay;
    /// Slice the original values without decoding, merging or requantizing a base.
    fn shard(self,context:&mut ShardingContext<B>) -> Self::Sharded;
}

/// Gather an actual projection on the execution backend from the original transport
/// backend. Distinguishing the two keeps inference and AD collectives explicit.
pub trait GatherTransformerProjection<AB:Backend,B:Backend>:FullyShardedModule<AB>+ModuleDisplay {
    /// Transient original native projection, not a dense surrogate for packed storage.
    type Gathered:TransformerProjectionShape<AB>;
    /// Retain exact integer bytes and original floating storage/parameter identities.
    fn gather_projection<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<Self::Gathered,C::Error>;
}

impl<B:Backend> ShardTransformerProjection<B> for AwqTransformerProjection<B> {
    type Sharded=FullyShardedAwqProjection<B>;
    fn shard(self,context:&mut ShardingContext<B>) -> Self::Sharded {
        match self {Self::Dense(layer)=>FullyShardedAwqProjection::Dense(context.linear(layer)),
            Self::LoRA(layer)=>FullyShardedAwqProjection::LoRA(context.lora(layer)),
            Self::Awq(layer)=>FullyShardedAwqProjection::Awq(context.awq(layer)),
            Self::AwqLoRA(layer)=>FullyShardedAwqProjection::AwqLoRA(context.awq_lora(layer))}
    }
}
impl<B:Backend> GatherTransformerProjection<B,B> for FullyShardedAwqProjection<B> {
    type Gathered=AwqTransformerProjection<B>;
    fn gather_projection<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<Self::Gathered,C::Error> {self.gather_inference(communicator)}
}
impl<B:Backend,S:CheckpointStrategy> GatherTransformerProjection<Autodiff<B,S>,B> for FullyShardedAwqProjection<Autodiff<B,S>> {
    type Gathered=AwqTransformerProjection<Autodiff<B,S>>;
    fn gather_projection<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<Self::Gathered,C::Error> {self.gather(communicator)}
}

/// Exact native NF4 projection choice with every persistent base/adapter leaf locally sharded.
#[derive(Module,Debug)]
pub enum FullyShardedNf4Projection<B:Backend> {
    /// Actual original floating values and trainability.
    Dense(FullyShardedLinear<B>),
    /// Actual original floating-base LoRA leaves.
    LoRA(FullyShardedLoRALinear<B>),
    /// Original high-nibble-first U8 base plus original FP32 metadata.
    Nf4(FullyShardedNf4Linear<B>),
    /// Original packed base plus actual floating A/B leaves.
    Nf4LoRA(FullyShardedNf4LoRALinear<B>),
}
impl<B:Backend> ShardTransformerProjection<B> for Nf4TransformerProjection<B> {
    type Sharded=FullyShardedNf4Projection<B>;
    fn shard(self,context:&mut ShardingContext<B>) -> Self::Sharded {
        match self {Self::Dense(layer)=>FullyShardedNf4Projection::Dense(context.linear(layer)),
            Self::LoRA(layer)=>FullyShardedNf4Projection::LoRA(context.lora(layer)),
            Self::Nf4(layer)=>FullyShardedNf4Projection::Nf4(context.nf4(layer)),
            Self::Nf4LoRA(layer)=>FullyShardedNf4Projection::Nf4LoRA(context.nf4_lora(layer))}
    }
}

/// Explicit mixed-storage native projection, preserving each actual original format.
#[derive(Module,Debug)]
pub enum FullyShardedMixedProjection<B:Backend> {
    /// Actual original dense local leaves.
    Dense(FullyShardedLinear<B>),
    /// Actual original floating-base adapter local leaves.
    LoRA(FullyShardedLoRALinear<B>),
    /// Actual original AWQ word/zero/scale local leaves.
    Awq(FullyShardedAwqLinear<B>),
    /// Actual original AWQ base and independent adapter local leaves.
    AwqLoRA(FullyShardedAwqLoRALinear<B>),
    /// Actual original NF4 byte/FP32 metadata local leaves.
    Nf4(FullyShardedNf4Linear<B>),
    /// Actual original NF4 base and independent adapter local leaves.
    Nf4LoRA(FullyShardedNf4LoRALinear<B>),
}
impl<B:Backend> ShardTransformerProjection<B> for MixedTransformerProjection<B> {
    type Sharded=FullyShardedMixedProjection<B>;
    fn shard(self,context:&mut ShardingContext<B>) -> Self::Sharded {
        match self {Self::Dense(layer)=>FullyShardedMixedProjection::Dense(context.linear(layer)),
            Self::LoRA(layer)=>FullyShardedMixedProjection::LoRA(context.lora(layer)),
            Self::Awq(layer)=>FullyShardedMixedProjection::Awq(context.awq(layer)),
            Self::AwqLoRA(layer)=>FullyShardedMixedProjection::AwqLoRA(context.awq_lora(layer)),
            Self::Nf4(layer)=>FullyShardedMixedProjection::Nf4(context.nf4(layer)),
            Self::Nf4LoRA(layer)=>FullyShardedMixedProjection::Nf4LoRA(context.nf4_lora(layer))}
    }
}

macro_rules! gather_selected_projection {
    ($source:ident,$target:ident,[$($variant:ident),+],$backend:ty,[$($generics:tt)*],$gather:ident) => {
        impl<$($generics)*> GatherTransformerProjection<$backend,B> for $source<$backend> {
            type Gathered=$target<$backend>;
            fn gather_projection<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<Self::Gathered,C::Error> {
                match self {$(Self::$variant(layer)=>layer.$gather(communicator).map($target::$variant),)+}
            }
        }
    };
}
gather_selected_projection!(FullyShardedNf4Projection,Nf4TransformerProjection,[Dense,LoRA,Nf4,Nf4LoRA],B,[B:Backend],gather_inference);
gather_selected_projection!(FullyShardedNf4Projection,Nf4TransformerProjection,[Dense,LoRA,Nf4,Nf4LoRA],Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);
gather_selected_projection!(FullyShardedMixedProjection,MixedTransformerProjection,[Dense,LoRA,Awq,AwqLoRA,Nf4,Nf4LoRA],B,[B:Backend],gather_inference);
gather_selected_projection!(FullyShardedMixedProjection,MixedTransformerProjection,[Dense,LoRA,Awq,AwqLoRA,Nf4,Nf4LoRA],Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);

/// Shared native sharded attention graph over original caller-selected local storage.
pub type FullyShardedProjectedAttention<B,P> = FullyShardedAwqAttention<B,P>;
/// Shared native sharded ordinary/gated FFN graph.
pub type FullyShardedProjectedFeedForward<B,P> = FullyShardedAwqFeedForward<B,P>;
/// Shared native sharded dense/packed/cached block graph.
pub type FullyShardedProjectedTransformerBlock<B,P> = FullyShardedAwqTransformerBlock<B,P>;
/// Shared original block-order execution, gathering only the current block.
pub type FullyShardedProjectedTransformerStack<B,P> = FullyShardedAwqTransformerStack<B,P>;
/// Shared exact native sharded normalized output head and chunked objective.
pub type FullyShardedProjectedTransformerHead<B,P> = FullyShardedAwqTransformerHead<B,P>;
/// Shared complete native model and scoped globally normalized causal training graph.
pub type FullyShardedProjectedTransformerModel<B,P> = FullyShardedAwqTransformerModel<B,P>;
/// NF4 complete-model training without requiring an AWQ backend extension.
pub type FullyShardedNf4TransformerModel<B> = FullyShardedProjectedTransformerModel<B,FullyShardedNf4Projection<B>>;
/// Complete native mixed dense/AWQ/NF4 model with explicit independent projection roles.
pub type FullyShardedMixedTransformerModel<B> = FullyShardedProjectedTransformerModel<B,FullyShardedMixedProjection<B>>;
/// Original transport failure or actual selected native projection failure.
pub type FullyShardedProjectedError<C,Q> = FullyShardedAwqError<C,Q>;
/// Original complete-model failure or original distributed loss-completion failure.
pub type FullyShardedProjectedTrainingError<C,Q> = FullyShardedAwqTrainingError<C,Q>;

impl<B:Backend> ShardingContext<B> {
    /// Partition a complete actual native stack with the same context used for tied tables/head leaves.
    pub fn projected_transformer_stack<P:ShardTransformerProjection<B>>(&mut self,stack:ProjectedTransformerStack<B,P>)
        -> FullyShardedProjectedTransformerStack<B,P::Sharded> {self.awq_transformer_stack(stack)}
    /// Partition a complete actual native model, preserving shared IDs across all original roles.
    pub fn projected_transformer_model<P:ShardTransformerProjection<B>>(&mut self,model:ProjectedTransformerModel<B,P>)
        -> FullyShardedProjectedTransformerModel<B,P::Sharded> {self.awq_transformer_model(model)}
}
