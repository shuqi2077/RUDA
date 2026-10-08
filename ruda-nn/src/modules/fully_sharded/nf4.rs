use super::*;
use ruda_model::tensor::{FrozenNf4Ops,IntegerTensorCollective,Nf4ProjectionOptions};
use core::fmt;
use crate::{loss::{CausalCrossEntropyConfig,CausalLoss},attention::PackedSequenceLayout};
use ruda_autodiff::collective::{CollectiveScope,ScopedCollectiveError};

/// Actual fully data-sharded native NF4: packed bytes, FP32 metadata and bias
/// persist only as local slices. Forward/input backward retain packed payload,
/// not an expanded complete floating base. Decoded scratch follows the explicit tile policy.
#[derive(Module,Debug)]
pub struct FullyShardedNf4Linear<B:Backend> {
    /// Original high-nibble-first U8 bytes, sharded without repacking nibbles.
    pub packed:ShardedPackedParameter<B>,
    /// Original frozen FP32 flat-block scale slices.
    pub scales:ShardedParameter<B>,
    /// Original frozen FP32 codebook slices.
    pub codebook:ShardedParameter<B>,
    /// Actual optional frozen original bias slices.
    pub bias:Option<ShardedParameter<B>>,
    /// Original logical input width.
    pub input_features:usize,
    /// Original logical output width.
    pub output_features:usize,
    /// Original even block size.
    pub block_size:usize,
    /// Explicit bounded decoded row tile.
    pub tile_rows:usize,
    /// Original fused half/BF16 execution selection, with no failed-call retries.
    pub use_tensor_core:bool,
}
impl<B:Backend> FullyShardedNf4Linear<B> {
    /// Assemble real caller-loaded local checkpoint slices with explicit original geometry.
    pub fn from_shards(packed:ShardedPackedParameter<B>,scales:ShardedParameter<B>,codebook:ShardedParameter<B>,bias:Option<ShardedParameter<B>>,
        input_features:usize,output_features:usize,block_size:usize,tile_rows:usize,use_tensor_core:bool) -> Self {
        let layer=Self {packed,scales,codebook,bias,input_features,output_features,block_size,tile_rows,use_tensor_core};layer.validate();layer
    }
    /// Slice actual loaded native payload on the original backend/device.
    pub fn from_full(layer:crate::FrozenNf4Linear<B>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).nf4(layer)}
    /// Original actual geometry and execution choices, independent of storage topology.
    pub fn options(&self) -> Nf4ProjectionOptions {Nf4ProjectionOptions {input_features:self.input_features,output_features:self.output_features,
        block_size:self.block_size,tile_rows:self.tile_rows,use_tensor_core:self.use_tensor_core}}
    /// Validate actual original byte/float metadata without decoding quantized weights.
    pub fn validate(&self) {
        assert!(self.input_features>0 && self.output_features>0 && self.block_size>0 && self.block_size%2==0
            && self.block_size<=u32::MAX as usize && self.tile_rows>0,"invalid sharded NF4 dimensions/block/tile");
        let size=self.input_features.checked_mul(self.output_features).expect("NF4 logical size overflows");assert!(size<=u32::MAX as usize,"NF4 indexing overflows");
        assert_eq!(self.packed.logical_shape,[size.div_ceil(2)],"NF4 logical byte count differs");
        assert_eq!(self.packed.local.val().dtype(),DType::U8,"NF4 bytes must retain U8 storage");
        assert_eq!(self.scales.logical_shape,[size.div_ceil(self.block_size)],"NF4 original scale count differs");
        assert_eq!(self.codebook.logical_shape,[16],"NF4 original codebook shape differs");
        let device=self.packed.local.val().device();let topology=(self.packed.rank,self.packed.world_size);
        let _=ShardedPackedParameter::from_local(self.packed.local.clone(),self.packed.logical_shape.clone(),self.packed.rank,self.packed.world_size);
        for parameter in [&self.scales,&self.codebook] {
            assert_eq!(parameter.local.val().dtype(),DType::F32,"NF4 quantization metadata must retain FP32");
        }
        for parameter in [&self.scales,&self.codebook].into_iter().chain(self.bias.iter()) {
            let value=parameter.local.val();assert_eq!(value.device(),device,"NF4 local devices differ");
            assert_eq!((parameter.rank,parameter.world_size),topology,"NF4 local topologies differ");
            assert!(!value.is_require_grad(),"NF4 base metadata/bias must remain frozen");
            let _=ShardedParameter::from_local(parameter.local.clone(),parameter.logical_shape.clone(),parameter.rank,parameter.world_size);
        }
        if let Some(bias)=&self.bias {assert_eq!(bias.logical_shape,[self.output_features],"NF4 original bias width differs");}
    }
}

/// Original packed NF4 base and native floating LoRA leaves, all locally sharded.
#[derive(Module,Debug)]
pub struct FullyShardedNf4LoRALinear<B:Backend> {
    /// Actual original locally sharded frozen byte/scales/codebook/bias payload.
    pub base:FullyShardedNf4Linear<B>,
    /// Original native trainable A leaf slices.
    pub adapter_a:FullyShardedLinear<B>,
    /// Original native trainable B leaf slices.
    pub adapter_b:FullyShardedLinear<B>,
    /// Original adapter-only input dropout.
    pub dropout:crate::Dropout,
    /// Original LoRA or rsLoRA multiplier.
    pub scale:f64,
}
impl<B:Backend> FullyShardedNf4LoRALinear<B> {
    /// Preserve actual loaded base/A/B values and identities, without dense expansion.
    pub fn from_full(layer:crate::Nf4LoRALinear<B>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).nf4_lora(layer)}
    /// Validate exact adapter widths, actual storage/trainability and original data topology.
    pub fn validate(&self) {
        self.base.validate();let shape=&self.adapter_a.weight.logical_shape;
        assert_eq!(shape.len(),2,"NF4 adapter A must be a matrix");let rank=shape[1];
        assert!(rank>0 && self.scale.is_finite(),"invalid NF4 adapter rank/scale");assert_eq!(shape[0],self.base.input_features,"NF4 adapter A input width differs");
        assert_eq!(self.adapter_b.weight.logical_shape,[rank,self.base.output_features],"NF4 adapter B output width differs");
        assert!(self.adapter_a.bias.is_none() && self.adapter_b.bias.is_none(),"NF4 adapters must remain bias-free");
        let device=self.base.packed.local.val().device();let topology=(self.base.packed.rank,self.base.packed.world_size);
        for parameter in [&self.adapter_a.weight,&self.adapter_b.weight] {
            let value=parameter.local.val();assert_eq!(value.device(),device,"NF4 adapter/base devices differ");
            assert_eq!((parameter.rank,parameter.world_size),topology,"NF4 adapter/base topologies differ");
            assert!(!B::ad_enabled(&device) || value.is_require_grad(),"NF4 adapters must be trainable on the AD backend");
            let _=ShardedParameter::from_local(parameter.local.clone(),parameter.logical_shape.clone(),parameter.rank,parameter.world_size);
        }
    }
}

macro_rules! gather_nf4 {
    ($backend:ty,[$($generics:tt)*],$gather:ident) => {
        impl<$($generics)*> FullyShardedNf4Linear<$backend> {
            /// Transient actual original packed bytes and frozen quantization metadata.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<crate::FrozenNf4Linear<$backend>,C::Error> {
                self.validate();
                let packed=self.packed.$gather::<C,1>(communicator.clone())?;
                let scales=self.scales.$gather::<C,1>(communicator.clone())?;
                let book=self.codebook.$gather::<C,1>(communicator.clone())?;
                let bias=self.bias.as_ref().map(|bias|bias.$gather::<C,1>(communicator).map(|value|Param::initialized(bias.local.id,value))).transpose()?;
                Ok(crate::FrozenNf4Linear::from_parameters(Param::initialized(self.packed.local.id,packed),Param::initialized(self.scales.local.id,scales),
                    Param::initialized(self.codebook.local.id,book),bias,self.input_features,self.output_features,self.block_size,self.tile_rows,self.use_tensor_core))
            }
        }
        impl<$($generics)*> FullyShardedNf4LoRALinear<$backend> {
            /// Original immutable packed base and actual differentiable/inference native A/B gathers.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<crate::Nf4LoRALinear<$backend>,C::Error> {
                self.validate();Ok(crate::Nf4LoRALinear {base:self.base.$gather(communicator.clone())?,
                    adapter_a:self.adapter_a.$gather(communicator.clone())?,adapter_b:self.adapter_b.$gather(communicator)?,dropout:self.dropout.clone(),scale:self.scale})
            }
        }
    };
}
gather_nf4!(B,[B:Backend],gather_inference);
gather_nf4!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);

/// Actual original data-transport or native NF4 projection failure.
#[derive(Debug)]
pub enum FullyShardedNf4Error<C:fmt::Debug,Q:fmt::Debug> {
    /// Original native integer/floating collective error.
    Collective(C),
    /// Original native packed NF4 validation/execution error.
    Projection(Q),
}
impl<C:fmt::Debug,Q:fmt::Debug> fmt::Display for FullyShardedNf4Error<C,Q> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Collective(error)=>write!(f,"NF4 data transport: {error:?}"),Self::Projection(error)=>write!(f,"NF4 projection: {error:?}")}
    }
}
impl<C:fmt::Debug,Q:fmt::Debug> core::error::Error for FullyShardedNf4Error<C,Q> {}
macro_rules! nf4_forward {
    ($module:ident) => {
        impl<B:FrozenNf4Ops> $module<B> {
            /// Original native packed projection after exact byte/float data gathers.
            pub fn forward_inference<C:IntegerTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
                -> Result<Tensor<B,D>,FullyShardedNf4Error<C::Error,B::Nf4Error>> {
                self.gather_inference(communicator).map_err(FullyShardedNf4Error::Collective)?.forward(input).map_err(FullyShardedNf4Error::Projection)
            }
        }
        impl<B:FrozenNf4Ops,S:CheckpointStrategy> $module<Autodiff<B,S>> {
            /// Actual first-order input/A/B graph with existing SUM/reduce-scatter adapter derivatives.
            pub fn forward<C:IntegerTensorCollective<B>,const D:usize>(&self,input:Tensor<Autodiff<B,S>,D>,communicator:C)
                -> Result<Tensor<Autodiff<B,S>,D>,FullyShardedNf4Error<C::Error,<Autodiff<B,S> as FrozenNf4Ops>::Nf4Error>> {
                self.gather(communicator).map_err(FullyShardedNf4Error::Collective)?.forward(input).map_err(FullyShardedNf4Error::Projection)
            }
        }
    };
}
nf4_forward!(FullyShardedNf4Linear);
nf4_forward!(FullyShardedNf4LoRALinear);

macro_rules! nf4_causal_loss {
    ($module:ident,$backend:ty,[$($generics:tt)*],$gather:ident,$causal:ident,$packed:ident) => {
        impl<$($generics)*> $module<$backend> {
            /// Gather actual packed/floating head values once for every original full-vocabulary token chunk.
            pub fn $causal<C:IntegerTensorCollective<B>>(&self,hidden:Tensor<$backend,3>,labels:Tensor<$backend,2,Int>,
                criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C)
                -> Result<CausalLoss<$backend>,FullyShardedNf4Error<C::Error,<$backend as FrozenNf4Ops>::Nf4Error>> {
                let head=self.$gather(communicator).map_err(FullyShardedNf4Error::Collective)?;
                criterion.try_forward_hidden_with_smoothing(hidden,labels,|rows|head.forward(rows),label_smoothing).map_err(FullyShardedNf4Error::Projection)
            }
            /// Original flat-document label boundaries/ignore rules/smoothing, without full token-wide logits.
            pub fn $packed<C:IntegerTensorCollective<B>>(&self,hidden:Tensor<$backend,2>,labels:Tensor<$backend,1,Int>,layout:&PackedSequenceLayout,
                criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C)
                -> Result<CausalLoss<$backend>,FullyShardedNf4Error<C::Error,<$backend as FrozenNf4Ops>::Nf4Error>> {
                let head=self.$gather(communicator).map_err(FullyShardedNf4Error::Collective)?;
                criterion.try_forward_packed_hidden_with_smoothing(hidden,labels,layout,|rows|head.forward(rows),label_smoothing).map_err(FullyShardedNf4Error::Projection)
            }
        }
    };
}
nf4_causal_loss!(FullyShardedNf4Linear,B,[B:FrozenNf4Ops],gather_inference,forward_causal_loss_inference,forward_packed_causal_loss_inference);
nf4_causal_loss!(FullyShardedNf4LoRALinear,B,[B:FrozenNf4Ops],gather_inference,forward_causal_loss_inference,forward_packed_causal_loss_inference);
nf4_causal_loss!(FullyShardedNf4Linear,Autodiff<B,S>,[B:FrozenNf4Ops,S:CheckpointStrategy],gather,forward_causal_loss,forward_packed_causal_loss);
nf4_causal_loss!(FullyShardedNf4LoRALinear,Autodiff<B,S>,[B:FrozenNf4Ops,S:CheckpointStrategy],gather,forward_causal_loss,forward_packed_causal_loss);

/// Actual NF4 projection/transport or native distributed loss-completion error.
#[derive(Debug)]
pub enum FullyShardedNf4TrainingError<C:fmt::Debug,Q:fmt::Debug> {
    /// Original actual packed projection/data-gather failure.
    Projection(FullyShardedNf4Error<C,Q>),
    /// Original actual loss scope/count/transport failure.
    Loss(ScopedCollectiveError<C>),
}
impl<C:fmt::Debug,Q:fmt::Debug> fmt::Display for FullyShardedNf4TrainingError<C,Q> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Projection(error)=>write!(f,"NF4 training projection: {error}"),Self::Loss(error)=>write!(f,"NF4 training loss: {error}")}
    }
}
impl<C:fmt::Debug,Q:fmt::Debug> core::error::Error for FullyShardedNf4TrainingError<C,Q> {}

macro_rules! nf4_distributed_loss {
    ($module:ident) => {
        impl<B:FrozenNf4Ops,S:CheckpointStrategy> $module<Autodiff<B,S>> {
            /// Complete a real caller-owned native sharded loss window. Run the backbone
            /// with this same scope's bound transport before passing its hidden states.
            /// Existing scope coordination keeps empty ranks participating; no second gradient SUM.
            pub fn forward_distributed_causal_loss<C:IntegerTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,3>,labels:Tensor<Autodiff<B,S>,2,Int>,
                criterion:&CausalCrossEntropyConfig,label_smoothing:f64,scope:&CollectiveScope<B,S>,communicator:C)
                -> Result<FullyShardedLoss<B,S>,FullyShardedNf4TrainingError<C::Error,<Autodiff<B,S> as FrozenNf4Ops>::Nf4Error>> {
                let loss=self.forward_causal_loss(hidden,labels,criterion,label_smoothing,scope.bind(communicator.clone())).map_err(FullyShardedNf4TrainingError::Projection)?;
                complete_fully_sharded_loss(scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedNf4TrainingError::Loss)
            }
            /// Original packed shifted loss and exact global token count with actual backbone graph scope.
            pub fn forward_distributed_packed_causal_loss<C:IntegerTensorCollective<B>>(&self,hidden:Tensor<Autodiff<B,S>,2>,labels:Tensor<Autodiff<B,S>,1,Int>,
                layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,scope:&CollectiveScope<B,S>,communicator:C)
                -> Result<FullyShardedLoss<B,S>,FullyShardedNf4TrainingError<C::Error,<Autodiff<B,S> as FrozenNf4Ops>::Nf4Error>> {
                let loss=self.forward_packed_causal_loss(hidden,labels,layout,criterion,label_smoothing,scope.bind(communicator.clone())).map_err(FullyShardedNf4TrainingError::Projection)?;
                complete_fully_sharded_loss(scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedNf4TrainingError::Loss)
            }
        }
    };
}
nf4_distributed_loss!(FullyShardedNf4Linear);
nf4_distributed_loss!(FullyShardedNf4LoRALinear);
