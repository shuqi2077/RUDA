use super::{FrozenNf4ExpertProjection,FrozenNf4SwiGluExperts,Nf4MoeLayer};
use ruda_model::{module::{Module,ModuleDisplay,Param},tensor::{Tensor,Int,DType,TensorPrimitive,FrozenNf4SwiGluOps,FrozenPackedExpertOps,
    AwqExpertOptions,AwqExpertPayload,PackedExpertPayload,backend::Backend}};
use core::fmt;

/// Geometry/storage of selected native experts, independent of floating or frozen packed representation.
pub trait FrozenExpertGeometry<B:Backend>:Module<B>+ModuleDisplay {
    /// Original `[experts,hidden,intermediate]` logical dimensions.
    fn dimensions(&self) -> [usize;3];
    /// Validate actual original source storage without downloading numeric values.
    fn validate(&self);
    /// Original resident expert payload device.
    fn device(&self) -> B::Device;
}
/// Actual selected native chain; routing, residuals and shared experts remain architecture-owned.
pub trait FrozenSelectedExperts<B:Backend>:FrozenExpertGeometry<B> {
    /// Original actual expert representation or first-order contract failure.
    type Error:fmt::Debug;
    /// Evaluate only actual U32 assigned experts, returning original incoming row order.
    fn forward(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,Self::Error>;
}
impl<B:Backend> FrozenExpertGeometry<B> for FrozenNf4SwiGluExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn validate(&self) {self.validate();}
    fn device(&self) -> B::Device {self.gate.payload.packed.val().device()}
}
impl<B:FrozenNf4SwiGluOps> FrozenSelectedExperts<B> for FrozenNf4SwiGluExperts<B> {
    type Error=B::Nf4GroupedError;
    fn forward(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,Self::Error> {self.forward(input,global_ids,expert_start)}
}

/// Actual original immutable AWQ expert words/zeros/scales and explicitly present per-expert bias.
#[derive(Module,Debug)]
pub struct FrozenAwqExpertProjection<B:Backend> {
    /// Original permuted packed words `[experts,input,output/8]`.
    pub qweight:Param<Tensor<B,3,Int>>,
    /// Original packed zero-point words `[experts,input/group,output/8]`.
    pub qzeros:Param<Tensor<B,3,Int>>,
    /// Original scale storage `[experts,input/group,output]`, never an inferred activation dtype.
    pub scales:Param<Tensor<B,3>>,
    /// Actual optional original frozen scale-dtype output bias `[experts,output]`.
    pub bias:Option<Param<Tensor<B,2>>>,
    /// Actual complete source input-channel group size.
    pub group_size:usize,
}
impl<B:Backend> FrozenAwqExpertProjection<B> {
    /// Connect actual loaded source values/IDs; no quantization or unpacked base is created.
    pub fn from_parameters(qweight:Param<Tensor<B,3,Int>>,qzeros:Param<Tensor<B,3,Int>>,scales:Param<Tensor<B,3>>,bias:Option<Param<Tensor<B,2>>>,group_size:usize) -> Self {
        let layer=Self {qweight,qzeros,scales,bias,group_size}.no_grad();layer.validate();layer
    }
    /// Actual original `[experts,input,output]` geometry from source words/scales.
    pub fn dimensions(&self) -> [usize;3] {let [e,k,_]=self.qweight.val().dims();[e,k,self.scales.val().dims()[2]]}
    /// Validate native original packed shape/dtype/device and frozen metadata without numeric readback.
    pub fn validate(&self) {
        let weight=self.qweight.val();let zeros=self.qzeros.val();let scales=self.scales.val();let [e,k,n]=self.dimensions();
        assert!(e<u32::MAX as usize && k>0 && n>0 && n%8==0 && self.group_size>0 && k%self.group_size==0,"invalid original AWQ expert geometry/groups");
        assert!(k.checked_mul(n).is_some_and(|size|size<=u32::MAX as usize),"AWQ per-expert matrix exceeds U32 indexing");
        assert!(e.checked_mul(k).and_then(|size|size.checked_mul(n)).is_some_and(|size|size<=u32::MAX as usize),"AWQ expert indexing exceeds U32");
        assert_eq!(weight.dims(),[e,k,n/8],"AWQ expert word shape differs");assert_eq!(zeros.dims(),[e,k/self.group_size,n/8],"AWQ expert zero shape differs");
        assert_eq!(scales.dims(),[e,k/self.group_size,n],"AWQ expert scale shape differs");assert_eq!(weight.dtype(),DType::I32,"AWQ expert words must retain I32");
        assert_eq!(zeros.dtype(),DType::I32,"AWQ expert zeros must retain I32");assert!(matches!(scales.dtype(),DType::F16|DType::BF16|DType::F32),"unsupported AWQ expert scale dtype");
        assert_eq!(zeros.device(),weight.device(),"AWQ expert zero device differs");assert_eq!(scales.device(),weight.device(),"AWQ expert scale device differs");
        assert!(!scales.is_require_grad(),"AWQ expert scales must remain frozen");
        if let Some(bias)=&self.bias {let bias=bias.val();assert_eq!(bias.dims(),[e,n],"AWQ expert bias geometry differs");assert_eq!(bias.dtype(),scales.dtype(),"AWQ expert bias storage differs");
            assert_eq!(bias.device(),weight.device(),"AWQ expert bias device differs");assert!(!bias.is_require_grad(),"AWQ expert bias must remain frozen");}
    }
    /// Original native payload and explicit global expert range.
    pub fn primitives(&self,expert_start:usize) -> AwqExpertPayload<B> {
        self.validate();let [experts,input_features,output_features]=self.dimensions();
        assert!(expert_start.checked_add(experts).is_some_and(|end|end<=u32::MAX as usize),"AWQ global expert range exceeds U32");
        AwqExpertPayload {qweight:self.qweight.val().into_primitive(),qzeros:self.qzeros.val().into_primitive(),scales:self.scales.val().into_primitive().tensor(),
            bias:self.bias.as_ref().map(|value|value.val().into_primitive().tensor()),options:AwqExpertOptions {experts,expert_start,input_features,output_features,group_size:self.group_size}}
    }
}
/// Actual caller-selected original expert projection; packed formats are never converted implicitly.
#[derive(Module,Debug)]
pub enum FrozenPackedExpertProjection<B:Backend> {
    /// Original high-nibble-first bytes with global flat-block FP32 scales/book.
    Nf4(FrozenNf4ExpertProjection<B>),
    /// Original input-group AWQ I32 words/zero points with independent scale storage.
    Awq(FrozenAwqExpertProjection<B>),
}
impl<B:Backend> From<FrozenNf4ExpertProjection<B>> for FrozenPackedExpertProjection<B> {fn from(value:FrozenNf4ExpertProjection<B>) -> Self {Self::Nf4(value)}}
impl<B:Backend> From<FrozenAwqExpertProjection<B>> for FrozenPackedExpertProjection<B> {fn from(value:FrozenAwqExpertProjection<B>) -> Self {Self::Awq(value)}}
impl<B:Backend> FrozenPackedExpertProjection<B> {
    /// Actual original `[experts,input,output]` dimensions, without decoding.
    pub fn dimensions(&self) -> [usize;3] {match self {Self::Nf4(value)=>[value.experts,value.payload.input_features,value.output_features],Self::Awq(value)=>value.dimensions()}}
    /// Validate original immutable source geometry/storage for only the actual selected representation.
    pub fn validate(&self) {match self {Self::Nf4(value)=>value.validate(),Self::Awq(value)=>value.validate()}}
    /// Original actual packed payload device.
    pub fn device(&self) -> B::Device {match self {Self::Nf4(value)=>value.payload.packed.val().device(),Self::Awq(value)=>value.qweight.val().device()}}
    /// Actual native operands for the explicitly declared global expert range.
    pub fn primitives(&self,expert_start:usize) -> PackedExpertPayload<B> {match self {Self::Nf4(value)=>PackedExpertPayload::Nf4(value.primitives(expert_start)),Self::Awq(value)=>PackedExpertPayload::Awq(value.primitives(expert_start))}}
}
impl<B:FrozenPackedExpertOps> FrozenPackedExpertProjection<B> {
    /// Native selected original packed projection, preserving activation storage and row order.
    pub fn forward(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,B::PackedExpertError> {
        self.forward_with_state(input,global_ids,expert_start).map(|(output,_)|output)
    }
    /// Original native first-order row state, without packed base gradients.
    pub fn forward_with_state(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<(Tensor<B,2>,B::PackedProjectionState),B::PackedExpertError> {
        let (output,state)=B::packed_expert_forward(input.into_primitive().tensor(),global_ids.into_primitive(),self.primitives(expert_start))?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Original actual first-order input gradient, not a higher-order differentiable surrogate.
    pub fn input_backward(&self,state:B::PackedProjectionState,gradient:Tensor<B,2>) -> Result<Tensor<B,2>,B::PackedExpertError> {
        B::packed_expert_input_backward(state,gradient.into_primitive().tensor()).map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
    }
}
impl<B:FrozenPackedExpertOps> FrozenAwqExpertProjection<B> {
    /// Direct native original AWQ selected expert projection with real input derivatives.
    pub fn forward(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,B::PackedExpertError> {
        self.forward_with_state(input,global_ids,expert_start).map(|(output,_)|output)
    }
    /// Preserve actual native row mapping and original packed coefficients for an explicit first-order VJP.
    pub fn forward_with_state(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<(Tensor<B,2>,B::PackedProjectionState),B::PackedExpertError> {
        let (output,state)=B::packed_expert_forward(input.into_primitive().tensor(),global_ids.into_primitive(),PackedExpertPayload::Awq(self.primitives(expert_start)))?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Original rounded-AWQ-coefficient transpose derivative, retaining original input activation storage.
    pub fn input_backward(&self,state:B::PackedProjectionState,gradient:Tensor<B,2>) -> Result<Tensor<B,2>,B::PackedExpertError> {
        B::packed_expert_input_backward(state,gradient.into_primitive().tensor()).map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
    }
}
/// Three actual independently selected AWQ/NF4 gate/up/down projections and original native SwiGLU.
#[derive(Module,Debug)]
pub struct FrozenPackedSwiGluExperts<B:Backend> {
    /// Actual source gate cube `[E,I,H]`.
    pub gate:FrozenPackedExpertProjection<B>,
    /// Actual independent source up cube `[E,I,H]`.
    pub up:FrozenPackedExpertProjection<B>,
    /// Actual source down cube `[E,H,I]`.
    pub down:FrozenPackedExpertProjection<B>,
}
impl<B:Backend> FrozenPackedSwiGluExperts<B> {
    /// Connect actual source payloads, without inferring formats or a model family.
    pub fn from_parts(gate:FrozenPackedExpertProjection<B>,up:FrozenPackedExpertProjection<B>,down:FrozenPackedExpertProjection<B>) -> Self {
        let experts=Self {gate,up,down};experts.validate();experts
    }
}
impl<B:Backend> FrozenExpertGeometry<B> for FrozenPackedSwiGluExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.gate.dimensions()}
    fn validate(&self) {
        self.gate.validate();self.up.validate();self.down.validate();let [e,h,i]=self.dimensions();
        assert_eq!(self.up.dimensions(),[e,h,i],"packed gate/up expert geometry differs");assert_eq!(self.down.dimensions(),[e,i,h],"packed down expert geometry differs");
        assert_eq!(self.up.device(),self.gate.device(),"packed up expert device differs");assert_eq!(self.down.device(),self.gate.device(),"packed down expert device differs");
    }
    fn device(&self) -> B::Device {self.gate.device()}
}
impl<B:FrozenPackedExpertOps> FrozenSelectedExperts<B> for FrozenPackedSwiGluExperts<B> {
    type Error=B::PackedExpertError;
    fn forward(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize) -> Result<Tensor<B,2>,Self::Error> {
        self.forward_with_state(input,global_ids,expert_start,false).map(|(output,_)|output)
    }
}
impl<B:FrozenPackedExpertOps> FrozenPackedSwiGluExperts<B> {
    /// Preserve actual native first-order input intermediates; taped AD retains them when required.
    pub fn forward_with_state(&self,input:Tensor<B,2>,global_ids:Tensor<B,1,Int>,expert_start:usize,retain_input:bool)
        -> Result<(Tensor<B,2>,B::PackedSwiGluState),B::PackedExpertError> {
        self.validate();let (output,state)=B::packed_swiglu_forward(input.into_primitive().tensor(),global_ids.into_primitive(),self.gate.primitives(expert_start),
            self.up.primitives(expert_start),self.down.primitives(expert_start),retain_input)?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Real source-native input VJP through all actual selected original packed projections.
    pub fn input_backward(&self,state:B::PackedSwiGluState,gradient:Tensor<B,2>) -> Result<Tensor<B,2>,B::PackedExpertError> {
        B::packed_swiglu_input_backward(state,gradient.into_primitive().tensor()).map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))
    }
}
/// Complete original local routed layer with independent AWQ/NF4 expert projection choices.
pub type PackedMoeLayer<B,P> = Nf4MoeLayer<B,P,FrozenPackedSwiGluExperts<B>>;
