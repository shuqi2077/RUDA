use super::{FrozenPackedExpertProjection,FrozenPackedSwiGluExperts,FrozenExpertGeometry,FrozenSelectedExperts,
    LoRALinearConfig,LinearConfig,Dropout,DropoutConfig,Nf4MoeLayer};
use ruda_model::{module::{Module,Param,Initializer},tensor::{Tensor,Int,DType,TensorPrimitive,ExpertProjectionOps,ExpertProjectionOptions,ExpertProjectionSelection,
    MoeExpertStrategy,FrozenPackedExpertOps,NativeSwiGluOps,backend::Backend}};
use alloc::vec::Vec;
use core::fmt;
#[cfg(not(feature="std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Actual original native floating matrices, selecting one expert per incoming row.
#[derive(Module,Debug)]
pub struct ExpertLinear<B:Backend> {
    /// Actual original `[experts,output,input]` weights and their supplied parameter identity/flags.
    pub weight:Param<Tensor<B,3>>,
    /// Explicit original floating grouped forward policy.
    #[module(skip)]
    pub forward_strategy:MoeExpertStrategy,
    /// Explicit original independent floating grouped backward policy.
    #[module(skip)]
    pub backward_strategy:MoeExpertStrategy,
}
impl<B:Backend> ExpertLinear<B> {
    /// Connect actual original bias-free cube, without initialization, transposition or altered trainability.
    pub fn from_parameters(weight:Param<Tensor<B,3>>,forward_strategy:MoeExpertStrategy,backward_strategy:MoeExpertStrategy) -> Self {
        let layer=Self {weight,forward_strategy,backward_strategy};layer.validate();layer
    }
    /// Original logical `[experts,input,output]` widths.
    pub fn dimensions(&self) -> [usize;3] {let [e,n,k]=self.weight.val().dims();[e,k,n]}
    /// Validate actual original native storage/geometry without numeric readback.
    pub fn validate(&self) {
        let weight=self.weight.val();let [e,k,n]=self.dimensions();assert!(e>0 && e<u32::MAX as usize && k>0 && n>0,"native expert linear axes must be positive");
        assert!(e.checked_mul(k).and_then(|size|size.checked_mul(n)).is_some_and(|size|size<=u32::MAX as usize),"native expert linear exceeds U32 indexing");
        assert!(matches!(weight.dtype(),DType::F16|DType::BF16|DType::F32),"native expert linear storage must be FP16/BF16/FP32");
    }
}
/// Actual optional original native derivatives; native cube gradients retain FP32.
#[derive(Debug)]
pub struct ExpertLinearBackward<B:Backend> {pub input:Option<Tensor<B,2>>,pub weights:Option<Tensor<B,3>>}
impl<B:ExpertProjectionOps> ExpertLinear<B> {
    /// Actual selected expert projection, with real first-order input and cube gradients on AD.
    pub fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,B::ExpertProjectionError> {
        self.forward_with_state(input,ids,expert_start).map(|(output,_)|output)
    }
    /// Preserve actual original native operands/row mapping for an explicit selected first-order VJP.
    pub fn forward_with_state(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize) -> Result<(Tensor<B,2>,B::ExpertProjectionState),B::ExpertProjectionError> {
        self.validate();let (output,state)=B::expert_projection_forward(input.into_primitive().tensor(),ids.into_primitive(),self.weight.val().into_primitive().tensor(),
            ExpertProjectionOptions {expert_start,forward:self.forward_strategy,backward:self.backward_strategy})?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Original native requested input/cube derivatives, without omitted derivative allocations.
    pub fn backward_selected(&self,state:B::ExpertProjectionState,gradient:Tensor<B,2>,selection:ExpertProjectionSelection)
        -> Result<ExpertLinearBackward<B>,B::ExpertProjectionError> {
        let result=B::expert_projection_backward(state,gradient.into_primitive().tensor(),selection)?;
        Ok(ExpertLinearBackward {input:result.input.map(|value|Tensor::from_primitive(TensorPrimitive::Float(value))),
            weights:result.weights.map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))})
    }
}
/// Original frozen packed projection or actual native floating adapter projection failure.
#[derive(Debug)]
pub enum ExpertLoRAError<P:fmt::Debug,G:fmt::Debug> {Packed(P),Adapter(G)}
impl<P:fmt::Debug,G:fmt::Debug> fmt::Display for ExpertLoRAError<P,G> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Packed(error)=>write!(f,"packed expert base: {error:?}"),Self::Adapter(error)=>write!(f,"native expert adapter: {error:?}")}}
}
impl<P:fmt::Debug,G:fmt::Debug> core::error::Error for ExpertLoRAError<P,G> {}
/// Actual independent low-rank residual on each original immutable AWQ/NF4 expert matrix.
#[derive(Module,Debug)]
pub struct PackedExpertLoRA<B:Backend> {
    /// Actual original immutable packed base, never merged or requantized.
    pub base:FrozenPackedExpertProjection<B>,
    /// Actual bias-free `[experts,rank,input]` trainable A matrices.
    pub adapter_a:ExpertLinear<B>,
    /// Actual bias-free `[experts,output,rank]` trainable B matrices.
    pub adapter_b:ExpertLinear<B>,
    /// Original adapter-only input dropout on actual selected assignment rows.
    pub dropout:Dropout,
    /// Original explicit alpha/rank or alpha/sqrt(rank) multiplier.
    pub scale:f64,
}
impl<B:Backend> PackedExpertLoRA<B> {
    /// Validate source geometry/storage and supplied adapter leaves without altering parameter identities.
    pub fn validate(&self) {
        self.base.validate();self.adapter_a.validate();self.adapter_b.validate();let [e,k,n]=self.base.dimensions();
        let [ae,ak,rank]=self.adapter_a.dimensions();assert_eq!([ae,ak],[e,k],"expert adapter A geometry differs");
        assert_eq!(self.adapter_b.dimensions(),[e,rank,n],"expert adapter B geometry differs");assert!(self.scale.is_finite(),"expert adapter multiplier must be finite");
        let device=self.base.device();for value in [&self.adapter_a,&self.adapter_b] {
            let weight=value.weight.val();assert_eq!(weight.device(),device,"expert adapter/base device differs");}
    }
}
impl LoRALinearConfig {
    fn validate_expert_initialization<B:Backend>(&self,base:&FrozenPackedExpertProjection<B>,adapter_dtype:DType) {
        base.validate();assert!(self.rank>0 && self.alpha.is_finite(),"invalid original expert adapter rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),"original expert adapter dropout must be in [0,1)");
        assert!(matches!(adapter_dtype,DType::F16|DType::BF16|DType::F32),"expert adapter storage must be native floating");
        let [e,k,n]=base.dimensions();for width in [k,n] {assert!(e.checked_mul(self.rank).and_then(|size|size.checked_mul(width))
            .is_some_and(|size|size<=u32::MAX as usize),"expert adapter geometry exceeds native U32 indexing");}
    }
    /// Connect explicitly loaded original expert A/B values/IDs, using original LoRA or rsLoRA scaling.
    pub fn from_expert_adapters<B:Backend>(&self,base:FrozenPackedExpertProjection<B>,adapter_a:ExpertLinear<B>,adapter_b:ExpertLinear<B>,use_rslora:bool) -> PackedExpertLoRA<B> {
        assert!(self.rank>0 && self.alpha.is_finite(),"invalid original expert LoRA rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),"original expert adapter dropout must be in [0,1)");
        assert_eq!(adapter_a.dimensions()[2],self.rank,"loaded expert adapter rank differs from actual explicit configuration");
        let denominator=if use_rslora {(self.rank as f64).sqrt()}else {self.rank as f64};
        let layer=PackedExpertLoRA {base,adapter_a,adapter_b,dropout:DropoutConfig::new(self.dropout).init(),scale:self.alpha/denominator};layer.validate();
        assert!(!B::ad_enabled(&layer.base.device()) || (layer.adapter_a.weight.val().is_require_grad() && layer.adapter_b.weight.val().is_require_grad()),
            "loaded expert adapter leaves must be trainable when attaching with AD enabled");layer
    }
    /// Allocate actual new expert adapter leaves with the existing linear initializer and zero B.
    /// Dtype/rsLoRA and native forward/backward execution are explicitly caller-selected.
    pub fn init_expert_adapters<B:Backend>(&self,base:FrozenPackedExpertProjection<B>,adapter_dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> PackedExpertLoRA<B> {
        self.validate_expert_initialization(&base,adapter_dtype);
        let [e,k,n]=base.dimensions();let device=base.device();
        let a=LinearConfig::new(k,self.rank).initializer.init_with::<B,3,_>([e,self.rank,k],Some(k),Some(self.rank),&device)
            .map(|value|value.cast(adapter_dtype).detach().require_grad());
        let b=Initializer::Zeros.init_with::<B,3,_>([e,n,self.rank],Some(self.rank),Some(n),&device)
            .map(|value|value.cast(adapter_dtype).detach().require_grad());
        self.from_expert_adapters(base,ExpertLinear::from_parameters(a,forward,backward),ExpertLinear::from_parameters(b,forward,backward),use_rslora)
    }
}
impl<B:FrozenPackedExpertOps+ExpertProjectionOps> PackedExpertLoRA<B> {
    /// Original packed base plus scaled actual native per-expert A/B residual.
    /// Input/A/B receive real first-order derivatives; packed words and quantization metadata stay frozen.
    pub fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize)
        -> Result<Tensor<B,2>,ExpertLoRAError<B::PackedExpertError,B::ExpertProjectionError>> {
        self.validate();let base=self.base.forward(input.clone(),ids.clone(),expert_start).map_err(ExpertLoRAError::Packed)?;
        let adapted=self.dropout.forward(input.cast(self.adapter_a.weight.val().dtype()));
        let hidden=self.adapter_a.forward(adapted,ids.clone(),expert_start).map_err(ExpertLoRAError::Adapter)?.cast(self.adapter_b.weight.val().dtype());
        let update=self.adapter_b.forward(hidden,ids,expert_start).map_err(ExpertLoRAError::Adapter)?.mul_scalar(self.scale);
        let dtype=base.dtype();Ok(base+update.cast(dtype))
    }
}
/// Actual original frozen or explicitly adapted expert projection; no fake-quantization path.
#[derive(Module,Debug)]
pub enum AdaptedExpertProjection<B:Backend> {Frozen(FrozenPackedExpertProjection<B>),LoRA(PackedExpertLoRA<B>)}
impl<B:Backend> AdaptedExpertProjection<B> {
    /// Actual original `[experts,input,output]` geometry, independent of adapter presence.
    pub fn dimensions(&self) -> [usize;3] {match self {Self::Frozen(value)=>value.dimensions(),Self::LoRA(value)=>value.base.dimensions()}}
    /// Original actual resident base/adapter device.
    pub fn device(&self) -> B::Device {match self {Self::Frozen(value)=>value.device(),Self::LoRA(value)=>value.base.device()}}
    /// Validate original source geometry and real loaded/trainable adapter leaves.
    pub fn validate(&self) {match self {Self::Frozen(value)=>value.validate(),Self::LoRA(value)=>value.validate()}}
}
impl<B:FrozenPackedExpertOps+ExpertProjectionOps> AdaptedExpertProjection<B> {
    /// Execute only the actual caller-selected original frozen or adapted projection.
    pub fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize)
        -> Result<Tensor<B,2>,ExpertLoRAError<B::PackedExpertError,B::ExpertProjectionError>> {
        match self {Self::Frozen(value)=>value.forward(input,ids,expert_start).map_err(ExpertLoRAError::Packed),Self::LoRA(value)=>value.forward(input,ids,expert_start)}
    }
}
/// Actual packed/adapted projection or original native activation failure.
#[derive(Debug)]
pub enum AdaptedExpertError<P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> {Projection(ExpertLoRAError<P,G>),Activation(S)}
impl<P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> fmt::Display for AdaptedExpertError<P,G,S> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Projection(error)=>write!(f,"{error}"),Self::Activation(error)=>write!(f,"native expert SwiGLU: {error:?}")}}
}
impl<P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> core::error::Error for AdaptedExpertError<P,G,S> {}
/// Explicit source expert projection roles; absent roles retain the original frozen payload.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum ExpertAdapterTarget {Gate,Up,Down}
/// Actual independently adapted gate/up/down and original native storage-rounded activation.
#[derive(Module,Debug)]
pub struct AdaptedPackedSwiGluExperts<B:Backend> {
    /// Actual original frozen or adapted gate `[E,I,H]`.
    pub gate:AdaptedExpertProjection<B>,
    /// Actual independent original frozen or adapted up `[E,I,H]`.
    pub up:AdaptedExpertProjection<B>,
    /// Actual original frozen or adapted down `[E,H,I]`.
    pub down:AdaptedExpertProjection<B>,
}
impl<B:Backend> AdaptedPackedSwiGluExperts<B> {
    /// Connect explicitly loaded original projection/adapters without constructing missing leaves.
    pub fn from_parts(gate:AdaptedExpertProjection<B>,up:AdaptedExpertProjection<B>,down:AdaptedExpertProjection<B>) -> Self {
        let experts=Self {gate,up,down};experts.validate();experts
    }
    /// Preserve all actual original packed payload identities and storage before explicit adaptation.
    pub fn from_frozen(source:FrozenPackedSwiGluExperts<B>) -> Self {
        Self::from_parts(AdaptedExpertProjection::Frozen(source.gate),AdaptedExpertProjection::Frozen(source.up),AdaptedExpertProjection::Frozen(source.down))
    }
    /// Attach new actual A/B leaves only to explicit roles, rejecting duplicate or already adapted roles before allocation.
    pub fn with_adapters(mut self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        self.validate_adapter_targets(config,targets,dtype);
        let adapt=|value|match value {AdaptedExpertProjection::Frozen(base)=>AdaptedExpertProjection::LoRA(config.init_expert_adapters(base,dtype,use_rslora,forward,backward)),
            AdaptedExpertProjection::LoRA(_)=>unreachable!("validated original expert role is not adapted")};
        if targets.contains(&ExpertAdapterTarget::Gate) {self.gate=adapt(self.gate);}
        if targets.contains(&ExpertAdapterTarget::Up) {self.up=adapt(self.up);}
        if targets.contains(&ExpertAdapterTarget::Down) {self.down=adapt(self.down);}self.validate();self
    }
    pub(super) fn validate_adapter_targets(&self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType) {
        assert!(!targets.is_empty(),"expert adapter selection must contain an actual source projection role");
        for (index,target) in targets.iter().enumerate() {assert!(!targets[..index].contains(target),"duplicate expert adapter role");
            let role=match target {ExpertAdapterTarget::Gate=>&self.gate,ExpertAdapterTarget::Up=>&self.up,ExpertAdapterTarget::Down=>&self.down};
            let AdaptedExpertProjection::Frozen(base)=role else {panic!("selected original expert projection is already adapted")};
            config.validate_expert_initialization(base,dtype);}
    }
    /// Original actual selected trainable A/B parameter IDs, excluding frozen quantization metadata.
    pub fn adapter_parameter_ids(&self) -> Vec<ruda_model::module::ParamId> {
        let mut ids=Vec::new();for projection in [&self.gate,&self.up,&self.down] {if let AdaptedExpertProjection::LoRA(layer)=projection {
            for id in [layer.adapter_a.weight.id.clone(),layer.adapter_b.weight.id.clone()] {if !ids.contains(&id) {ids.push(id);}}}}ids
    }
}
impl<B:Backend> FrozenExpertGeometry<B> for AdaptedPackedSwiGluExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.gate.dimensions()}
    fn validate(&self) {
        self.gate.validate();self.up.validate();self.down.validate();let [e,h,i]=self.dimensions();
        assert_eq!(self.up.dimensions(),[e,h,i],"adapted gate/up original expert geometry differs");assert_eq!(self.down.dimensions(),[e,i,h],"adapted down expert geometry differs");
        assert_eq!(self.up.device(),self.gate.device(),"adapted up expert device differs");assert_eq!(self.down.device(),self.gate.device(),"adapted down expert device differs");
    }
    fn device(&self) -> B::Device {self.gate.device()}
}
impl<B:FrozenPackedExpertOps+ExpertProjectionOps+NativeSwiGluOps> FrozenSelectedExperts<B> for AdaptedPackedSwiGluExperts<B> {
    type Error=AdaptedExpertError<B::PackedExpertError,B::ExpertProjectionError,B::SwiGluError>;
    fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,Self::Error> {
        self.validate();let gate=self.gate.forward(input.clone(),ids.clone(),expert_start).map_err(AdaptedExpertError::Projection)?;
        let up=self.up.forward(input,ids.clone(),expert_start).map_err(AdaptedExpertError::Projection)?;
        let activated=B::native_swiglu(gate.into_primitive().tensor(),up.into_primitive().tensor()).map_err(AdaptedExpertError::Activation)?;
        self.down.forward(Tensor::from_primitive(TensorPrimitive::Float(activated)),ids,expert_start).map_err(AdaptedExpertError::Projection)
    }
}
/// Original full native routed layer with actual per-expert A/B training on immutable packed bases.
pub type AdaptedPackedMoeLayer<B,P> = Nf4MoeLayer<B,P,AdaptedPackedSwiGluExperts<B>>;
/// Original full native token-to-logit training/cache graph with actual selected expert adapters.
pub type AdaptedPackedMoeTransformerModel<B,P> = super::transformer::Nf4MoeTransformerModel<B,P,AdaptedPackedSwiGluExperts<B>>;
/// Original actual native block with selected expert adapters, without a proxy model graph.
pub type AdaptedPackedMoeTransformerBlock<B,P> = super::transformer::Nf4MoeTransformerBlock<B,P,AdaptedPackedSwiGluExperts<B>>;
/// Original loaded ordinary/floating/packed layer choice with actual per-expert adapters.
pub type AdaptedPackedMoeTransformerLayer<B,P> = super::transformer::Nf4MoeTransformerLayer<B,P,AdaptedPackedSwiGluExperts<B>>;

/// Preserve original whole packed-expert execution on unselected layers; only explicitly adapted layers use A/B.
#[derive(Module,Debug)]
pub enum SelectablePackedExperts<B:Backend> {
    /// Actual unmodified original whole native AWQ/NF4 expert chain and its original VJP.
    Original(FrozenPackedSwiGluExperts<B>),
    /// Actual explicitly selected expert adapters and their source-native first-order graph.
    Adapted(AdaptedPackedSwiGluExperts<B>),
}
impl<B:Backend> SelectablePackedExperts<B> {
    /// Actual explicitly present expert adapter IDs; original frozen chains contain none.
    pub fn adapter_parameter_ids(&self) -> Vec<ruda_model::module::ParamId> {match self {Self::Original(_)=>Vec::new(),Self::Adapted(value)=>value.adapter_parameter_ids()}}
}
impl<B:Backend> FrozenExpertGeometry<B> for SelectablePackedExperts<B> {
    fn dimensions(&self) -> [usize;3] {match self {Self::Original(value)=>value.dimensions(),Self::Adapted(value)=>value.dimensions()}}
    fn validate(&self) {match self {Self::Original(value)=>value.validate(),Self::Adapted(value)=>value.validate()}}
    fn device(&self) -> B::Device {match self {Self::Original(value)=>value.device(),Self::Adapted(value)=>value.device()}}
}
impl<B:FrozenPackedExpertOps+ExpertProjectionOps+NativeSwiGluOps> FrozenSelectedExperts<B> for SelectablePackedExperts<B> {
    type Error=AdaptedExpertError<B::PackedExpertError,B::ExpertProjectionError,B::SwiGluError>;
    fn forward(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,Self::Error> {
        match self {Self::Original(value)=>value.forward(input,ids,expert_start).map_err(|error|AdaptedExpertError::Projection(ExpertLoRAError::Packed(error))),
            Self::Adapted(value)=>value.forward(input,ids,expert_start)}
    }
}
/// Complete native routed layer preserving original execution for unselected packed expert layers.
pub type SelectablePackedMoeLayer<B,P> = Nf4MoeLayer<B,P,SelectablePackedExperts<B>>;
/// Complete original native Transformer with only explicitly selected packed expert layers adapted.
pub type SelectablePackedMoeTransformerModel<B,P> = super::transformer::Nf4MoeTransformerModel<B,P,SelectablePackedExperts<B>>;
