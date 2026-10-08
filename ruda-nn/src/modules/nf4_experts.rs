use super::FrozenNf4Linear;
use ruda_model::{module::{Module,Param},tensor::{Tensor,Int,TensorPrimitive,DType,FloatDType,FrozenNf4GroupedOps,FrozenNf4SwiGluOps,Nf4ExpertPayload,Nf4GroupedOptions,
    MoeDispatchOps,MoeOptions,MoeSelectionOptions,MoeRouterWeightOptions,MoeExpertStrategy,MoeCombineGradientStrategy,dispatch_moe,combine_moe,backend::Backend}};
use super::transformer::{TransformerProjectionShape,TransformerProjection};
use core::fmt;

/// Frozen original NF4 expert cube, retaining source parameter IDs and global flat blocks.
#[derive(Module,Debug)]
pub struct FrozenNf4ExpertProjection<B:Backend> {
    /// Actual original `[experts*output,input]` packed bytes/scales/book; no bias.
    pub payload:FrozenNf4Linear<B>,
    /// Number of actual resident original expert matrices.
    pub experts:usize,
    /// Original output width of one expert, before flattening the cube for storage.
    pub output_features:usize,
}
impl<B:Backend> FrozenNf4ExpertProjection<B> {
    /// Connect an actual source flat packed cube; no model family, quantization or reblocking is inferred.
    pub fn from_flat_payload(payload:FrozenNf4Linear<B>,experts:usize,output_features:usize) -> Self {
        let layer=Self {payload,experts,output_features};layer.validate();layer
    }
    /// Validate original source payload and per-expert logical geometry without numeric readback.
    pub fn validate(&self) {
        self.payload.validate();assert!(self.experts>0 && self.output_features>0,"NF4 expert axes must be positive");
        assert_eq!(self.experts.checked_mul(self.output_features),Some(self.payload.output_features),"NF4 flattened expert geometry differs");
        assert!(self.payload.bias.is_none(),"native NF4 expert projections are bias-free");
    }
    /// Actual native per-expert options and explicitly selected global expert range.
    pub fn options(&self,expert_start:usize) -> Nf4GroupedOptions {
        self.validate();assert!(expert_start.checked_add(self.experts).is_some_and(|end|end<=u32::MAX as usize),"NF4 global expert range exceeds U32");
        let mut projection=self.payload.options();projection.output_features=self.output_features;
        Nf4GroupedOptions {experts:self.experts,expert_start,projection}
    }
    /// Actual immutable original byte/scale/codebook primitives.
    pub fn primitives(&self,expert_start:usize) -> Nf4ExpertPayload<B> {
        Nf4ExpertPayload {packed:self.payload.packed.val().into_primitive(),scales:self.payload.scales.val().into_primitive().tensor(),
            codebook:self.payload.codebook.val().into_primitive().tensor(),options:self.options(expert_start)}
    }
}
impl<B:FrozenNf4GroupedOps> FrozenNf4ExpertProjection<B> {
    /// Evaluate only the actual assigned expert for each row, restoring exact original row order.
    pub fn forward(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,B::Nf4GroupedError> {
        self.forward_with_state(input,global_ids,expert_start).map(|(output,_)|output)
    }
    /// Retain original native row permutation and operands for an explicit first-order VJP.
    pub fn forward_with_state(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize)
        -> Result<(Tensor<B,2>,B::Nf4GroupedState),B::Nf4GroupedError> {
        let (output,state)=B::frozen_nf4_grouped_forward(input.into_primitive().tensor(),global_ids.into_primitive(),self.primitives(expert_start))?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Original first-order input gradient with no quantized base or integer-selection derivative.
    pub fn input_backward(&self,state:B::Nf4GroupedState,gradient:Tensor<B,2>) -> Result<Tensor<B,2>,B::Nf4GroupedError> {
        B::frozen_nf4_grouped_input_backward(state,gradient.into_primitive().tensor()).map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
    }
}

/// Native selected packed NF4 gate/up/down with original ruDNN SwiGLU arithmetic.
#[derive(Module,Debug)]
pub struct FrozenNf4SwiGluExperts<B:Backend> {
    /// Original frozen gate cube `[E,I,H]`.
    pub gate:FrozenNf4ExpertProjection<B>,
    /// Original independent frozen up cube `[E,I,H]`.
    pub up:FrozenNf4ExpertProjection<B>,
    /// Original frozen down cube `[E,H,I]`.
    pub down:FrozenNf4ExpertProjection<B>,
}
impl<B:Backend> FrozenNf4SwiGluExperts<B> {
    /// Connect actual loaded source payloads with independent block sizes/execution choices.
    pub fn from_parts(gate:FrozenNf4ExpertProjection<B>,up:FrozenNf4ExpertProjection<B>,down:FrozenNf4ExpertProjection<B>) -> Self {
        let experts=Self {gate,up,down};experts.validate();experts
    }
    /// Original `[experts,hidden,intermediate]` logical dimensions.
    pub fn dimensions(&self) -> [usize;3] {[self.gate.experts,self.gate.payload.input_features,self.gate.output_features]}
    /// Validate source geometry, device and original independently frozen quantization metadata.
    pub fn validate(&self) {
        self.gate.validate();self.up.validate();self.down.validate();let [e,h,i]=self.dimensions();
        assert_eq!([self.up.experts,self.up.payload.input_features,self.up.output_features],[e,h,i],"NF4 gate/up cube geometry differs");
        assert_eq!([self.down.experts,self.down.payload.input_features,self.down.output_features],[e,i,h],"NF4 down cube geometry differs");
        let device=self.gate.payload.packed.val().device();
        assert_eq!(self.up.payload.packed.val().device(),device,"NF4 up expert device differs");assert_eq!(self.down.payload.packed.val().device(),device,"NF4 down expert device differs");
    }
}
impl<B:FrozenNf4SwiGluOps> FrozenNf4SwiGluExperts<B> {
    /// Complete selected native frozen expert chain; AD retains only required actual input VJP state.
    pub fn forward(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,B::Nf4GroupedError> {
        self.forward_with_state(input,global_ids,expert_start,false).map(|(output,_)|output)
    }
    /// Explicit original first-order native cache, without a floating base or learned QAT metadata.
    pub fn forward_with_state(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize,retain_input:bool)
        -> Result<(Tensor<B,2>,B::Nf4SwiGluState),B::Nf4GroupedError> {
        self.validate();let (output,state)=B::frozen_nf4_swiglu_forward(input.into_primitive().tensor(),global_ids.into_primitive(),
            self.gate.primitives(expert_start),self.up.primitives(expert_start),self.down.primitives(expert_start),retain_input)?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Source-native first-order input gradient through all three actual packed projections.
    pub fn input_backward(&self,state:B::Nf4SwiGluState,gradient:Tensor<B,2>) -> Result<Tensor<B,2>,B::Nf4GroupedError> {
        B::frozen_nf4_swiglu_input_backward(state,gradient.into_primitive().tensor()).map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
    }
}

/// Original discrete/continuous routing choices independent of each packed projection's execution.
#[derive(Clone,Copy,Debug,PartialEq)]
pub struct Nf4MoeRouting {
    /// Actual original softmax top-k or group-limited sigmoid selection.
    pub selection:MoeSelectionOptions,
    /// Actual original FP32 continuous selected-weight policy.
    pub weights:MoeRouterWeightOptions,
    /// Original actual combine weight-gradient reduction order.
    pub combine_backward:MoeCombineGradientStrategy,
}
/// Actual router, source-native routing or packed expert failure.
#[derive(Debug)]
pub enum Nf4MoeError<P:fmt::Debug,M:fmt::Debug,N:fmt::Debug> {
    /// Original caller-selected router projection failure.
    Router(P),
    /// Original native discrete/continuous routing, dispatch or combine failure.
    Routing(M),
    /// Original native selected packed NF4 expert failure.
    Experts(N),
}
impl<P:fmt::Debug,M:fmt::Debug,N:fmt::Debug> fmt::Display for Nf4MoeError<P,M,N> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Router(error)=>write!(f,"NF4 MoE router: {error:?}"),
        Self::Routing(error)=>write!(f,"NF4 MoE routing: {error:?}"),Self::Experts(error)=>write!(f,"NF4 MoE experts: {error:?}")}}
}
impl<P:fmt::Debug,M:fmt::Debug,N:fmt::Debug> core::error::Error for Nf4MoeError<P,M,N> {}
/// Complete local packed-expert routed branch; router storage/adapter selection remains explicit.
#[derive(Module,Debug)]
pub struct Nf4MoeLayer<B:Backend,P:Module<B>> {
    /// Original actual dense/LoRA/AWQ/NF4 router projection.
    pub router:P,
    /// Three actual packed expert cubes with their original independent execution options.
    pub experts:FrozenNf4SwiGluExperts<B>,
    /// Original optional FP32 selection-only correction bias.
    pub correction_bias:Option<Param<Tensor<B,1>>>,
    /// Explicit original source routing policy, not a floating expert strategy.
    #[module(skip)]
    pub routing:Nf4MoeRouting,
    /// Explicit optional router-input cast, independent of actual expert activation storage.
    #[module(skip)]
    pub router_input_dtype:Option<FloatDType>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> Nf4MoeLayer<B,P> {
    /// Connect actual loaded source modules without quantizing, merging adapters or selecting a model family.
    pub fn from_parts(router:P,experts:FrozenNf4SwiGluExperts<B>,correction_bias:Option<Param<Tensor<B,1>>>,routing:Nf4MoeRouting,router_input_dtype:Option<FloatDType>) -> Self {
        let layer=Self {router,experts,correction_bias,routing,router_input_dtype};layer.validate();layer
    }
    /// Actual original residual/input width.
    pub fn width(&self) -> usize {self.experts.dimensions()[1]}
    /// Validate original source router/cube geometry and independent correction-bias storage.
    pub fn validate(&self) {
        self.experts.validate();let [e,h,_]=self.experts.dimensions();assert_eq!(self.router.dimensions(),[h,e],"NF4 router/expert geometry differs");
        if let Some(bias)=&self.correction_bias {let bias=bias.val();assert_eq!(bias.dims(),[e],"NF4 correction bias width differs");
            assert_eq!(bias.dtype(),DType::F32,"NF4 correction bias must retain FP32");assert_eq!(bias.device(),self.experts.gate.payload.packed.val().device(),"NF4 correction bias device differs");}
        if let Some(dtype)=self.router_input_dtype {assert!(matches!(DType::from(dtype),DType::F16|DType::BF16|DType::F32),"unsupported NF4 router input storage");}
    }
}
/// Actual output and the same original tracked routing tensors used for that output.
#[derive(Debug)]
pub struct Nf4MoeOutput<B:Backend,const D:usize> {
    /// Complete actual local routed output, retaining original token axes.
    pub output:Tensor<B,D>,
    /// Actual original router logits, without a second dropout-bearing router pass.
    pub router_logits:Tensor<B,2>,
    /// Actual native U32 selected expert IDs for auxiliary objectives.
    pub selected_experts:Tensor<B,2,Int>,
}
impl<B:MoeDispatchOps+FrozenNf4SwiGluOps,P:TransformerProjection<B>> Nf4MoeLayer<B,P> {
    /// Complete native route -> selected packed experts -> original ordered combine.
    /// Shared experts, residuals and model-level normalization remain caller-owned.
    pub fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Nf4MoeError<P::Error,B::MoeError,B::Nf4GroupedError>> {
        self.forward_detailed(input).map(|output|output.output)
    }
    /// Real input/router first-order graph with immutable packed bases and original selection metadata.
    pub fn forward_detailed<const D:usize>(&self,input:Tensor<B,D>) -> Result<Nf4MoeOutput<B,D>,Nf4MoeError<P::Error,B::MoeError,B::Nf4GroupedError>> {
        self.validate();assert!(D>0,"NF4 MoE requires an actual feature axis");let shape=input.dims();assert_eq!(shape[D-1],self.width(),"NF4 MoE input width differs");
        let rows=shape[..D-1].iter().try_fold(1usize,|count,&axis|count.checked_mul(axis)).expect("NF4 MoE token count overflows");
        let input=input.reshape([rows,self.width()]);let router_input=if let Some(dtype)=self.router_input_dtype {input.clone().cast(dtype)} else {input.clone()};
        let logits=self.router.forward(router_input).map_err(Nf4MoeError::Router)?;
        // Dispatch uses only selection/weights. Packed expert execution is defined
        // independently by its original NF4 options, never these inactive float strategies.
        let options=MoeOptions {selection:self.routing.selection,weights:self.routing.weights,combine_backward:self.routing.combine_backward,
            forward:MoeExpertStrategy::Scalar,backward:MoeExpertStrategy::Scalar};
        let dispatch=dispatch_moe(input,logits.clone(),self.correction_bias.as_ref().map(Param::val),options).map_err(Nf4MoeError::Routing)?;
        let expert_values=self.experts.forward(dispatch.values,dispatch.row_experts,0).map_err(Nf4MoeError::Experts)?;
        let output=combine_moe(dispatch.state,expert_values,dispatch.weights,self.routing.combine_backward).map_err(Nf4MoeError::Routing)?;
        Ok(Nf4MoeOutput {output:output.reshape(shape),router_logits:logits,selected_experts:dispatch.selected_experts})
    }
}
