use core::fmt;
use ruda_model::{module::{Module,Param},tensor::{Tensor,Int,DType,FloatDType,TensorPrimitive,MoeOps,MoeOptions,MoeGradientSelection,backend::Backend}};
use crate::transformer::{TransformerProjectionShape,TransformerProjection};

/// Actual original native bias-free expert weights, retaining loaded cube layout and parameter IDs.
#[derive(Module,Debug)]
pub struct NativeSwiGluExperts<B:Backend> {
    /// Original `[experts,intermediate,hidden]` gate weights.
    pub gate:Param<Tensor<B,3>>,
    /// Original `[experts,intermediate,hidden]` independent up weights.
    pub up:Param<Tensor<B,3>>,
    /// Original `[experts,hidden,intermediate]` down weights.
    pub down:Param<Tensor<B,3>>,
}
impl<B:Backend> NativeSwiGluExperts<B> {
    /// Connect actual loaded values and trainability, without random expert weights or transposed guesses.
    pub fn from_parameters(gate:Param<Tensor<B,3>>,up:Param<Tensor<B,3>>,down:Param<Tensor<B,3>>) -> Self {
        let experts=Self {gate,up,down};experts.validate();experts
    }
    /// Original actual `[experts,hidden,intermediate]` geometry, without tensor value readback.
    pub fn dimensions(&self) -> [usize;3] {let [experts,inner,hidden]=self.gate.val().dims();[experts,hidden,inner]}
    /// Validate original source geometry/storage/device metadata, leaving every original flag intact.
    pub fn validate(&self) {
        let gate=self.gate.val();let up=self.up.val();let down=self.down.val();let [experts,inner,hidden]=gate.dims();
        assert!(experts>0 && inner>0 && hidden>0,"native MoE expert axes must be positive");
        assert_eq!(up.dims(),gate.dims(),"native gate/up expert geometry differs");assert_eq!(down.dims(),[experts,hidden,inner],"native down expert geometry differs");
        assert!(matches!(gate.dtype(),DType::F16|DType::BF16|DType::F32),"unsupported native expert storage");
        for value in [&up,&down] {assert_eq!(value.dtype(),gate.dtype(),"native expert storage differs");assert_eq!(value.device(),gate.device(),"native expert devices differ");}
    }
}
impl<B:MoeOps> NativeSwiGluExperts<B> {
    /// Execute actual native routed experts. AD uses the original training state only
    /// when a real input/logit/expert parent is tracked; untracked inference retains no backward cache.
    pub fn forward(&self,input:Tensor<B,2>,logits:Tensor<B,2>,correction_bias:Option<Tensor<B,1>>,options:MoeOptions) -> Result<Tensor<B,2>,B::MoeError> {
        self.validate();B::moe_inference(input.into_primitive().tensor(),logits.into_primitive().tensor(),correction_bias.map(|bias|bias.into_primitive().tensor()),
            self.gate.val().into_primitive().tensor(),self.up.val().into_primitive().tensor(),self.down.val().into_primitive().tensor(),options)
            .map(|output|Tensor::from_primitive(TensorPrimitive::Float(output)))
    }
    /// Retain actual original device dispatch/router/expert state for explicit VJP or routing metadata.
    pub fn forward_with_state(&self,input:Tensor<B,2>,logits:Tensor<B,2>,correction_bias:Option<Tensor<B,1>>,options:MoeOptions)
        -> Result<(Tensor<B,2>,B::MoeState),B::MoeError> {
        self.validate();let (output,state)=B::moe_forward(input.into_primitive().tensor(),logits.into_primitive().tensor(),correction_bias.map(|bias|bias.into_primitive().tensor()),
            self.gate.val().into_primitive().tensor(),self.up.val().into_primitive().tensor(),self.down.val().into_primitive().tensor(),options)?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Native forward with explicit derivative requirements. Router-only training
    /// can retain original expert outputs without retaining expert activation VJP caches.
    /// Ordinary taped AD always preserves the derivatives required by its tracked parents.
    pub fn forward_selected_with_state(&self,input:Tensor<B,2>,logits:Tensor<B,2>,correction_bias:Option<Tensor<B,1>>,options:MoeOptions,selection:MoeGradientSelection)
        -> Result<(Tensor<B,2>,B::MoeState),B::MoeError> {
        self.validate();let (output,state)=B::moe_forward_selected(input.into_primitive().tensor(),logits.into_primitive().tensor(),correction_bias.map(|bias|bias.into_primitive().tensor()),
            self.gate.val().into_primitive().tensor(),self.up.val().into_primitive().tensor(),self.down.val().into_primitive().tensor(),options,selection)?;
        Ok((Tensor::from_primitive(TensorPrimitive::Float(output)),state))
    }
    /// Actual discrete original U32 expert selections, usable with explicit native continuous-weight objectives.
    pub fn routing_indices(&self,state:&B::MoeState) -> Tensor<B,2,Int> {Tensor::from_primitive(B::moe_route_indices(state))}
    /// Original native first-order derivatives, including FP32 expert weight gradients.
    /// Native calls retain FP32 tensors; tracked AD primals reject a manual higher-order VJP.
    pub fn backward_with_state(&self,state:B::MoeState,gradient:Tensor<B,2>) -> Result<NativeMoeBackward<B>,B::MoeError> {
        let result=B::moe_backward(state,gradient.into_primitive().tensor())?;
        Ok(NativeMoeBackward {input:Tensor::from_primitive(TensorPrimitive::Float(result.input)),logits:Tensor::from_primitive(TensorPrimitive::Float(result.logits)),
            gate:Tensor::from_primitive(TensorPrimitive::Float(result.gate)),up:Tensor::from_primitive(TensorPrimitive::Float(result.up)),down:Tensor::from_primitive(TensorPrimitive::Float(result.down))})
    }
    /// Original explicit VJP with only requested outputs. Frozen expert matrices can
    /// propagate real upstream input derivatives without allocating FP32 weight-gradient cubes.
    pub fn backward_selected_with_state(&self,state:B::MoeState,gradient:Tensor<B,2>,selection:MoeGradientSelection)
        -> Result<NativeMoeBackwardSelected<B>,B::MoeError> {
        let result=B::moe_backward_selected(state,gradient.into_primitive().tensor(),selection)?;
        Ok(NativeMoeBackwardSelected {input:result.input.map(|value|Tensor::from_primitive(TensorPrimitive::Float(value))),
            logits:result.logits.map(|value|Tensor::from_primitive(TensorPrimitive::Float(value))),gate:result.gate.map(|value|Tensor::from_primitive(TensorPrimitive::Float(value))),
            up:result.up.map(|value|Tensor::from_primitive(TensorPrimitive::Float(value))),down:result.down.map(|value|Tensor::from_primitive(TensorPrimitive::Float(value)))})
    }
}
/// Actual original native expert branch VJP; no optimizer or router-selection policy is inferred.
#[derive(Debug)]
pub struct NativeMoeBackward<B:Backend> {
    /// Original source-token input derivative.
    pub input:Tensor<B,2>,
    /// Original source-logit derivative.
    pub logits:Tensor<B,2>,
    /// Original FP32 gate cube derivative.
    pub gate:Tensor<B,3>,
    /// Original FP32 up cube derivative.
    pub up:Tensor<B,3>,
    /// Original FP32 down cube derivative.
    pub down:Tensor<B,3>,
}
/// Actual optional native derivatives; original expert cube outputs retain FP32.
#[derive(Debug)]
pub struct NativeMoeBackwardSelected<B:Backend> {
    /// Requested source-token derivative.
    pub input:Option<Tensor<B,2>>,
    /// Requested source-logit derivative.
    pub logits:Option<Tensor<B,2>>,
    /// Requested original FP32 gate cube derivative.
    pub gate:Option<Tensor<B,3>>,
    /// Requested original FP32 up cube derivative.
    pub up:Option<Tensor<B,3>>,
    /// Requested original FP32 down cube derivative.
    pub down:Option<Tensor<B,3>>,
}
/// Original router projection or routed native expert execution failure.
#[derive(Debug)]
pub enum NativeMoeLayerError<P:fmt::Debug,M:fmt::Debug> {
    /// Original caller-selected dense/adapter/packed router projection failure.
    Router(P),
    /// Original native routed-expert failure.
    Experts(M),
}
impl<P:fmt::Debug,M:fmt::Debug> fmt::Display for NativeMoeLayerError<P,M> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Router(error)=>write!(f,"native router projection: {error:?}"),Self::Experts(error)=>write!(f,"native routed experts: {error:?}")}
    }
}
impl<P:fmt::Debug,M:fmt::Debug> core::error::Error for NativeMoeLayerError<P,M> {}

/// Complete actual local routed branch with caller-selected router projection and original expert weights.
#[derive(Module,Debug)]
pub struct NativeMoeLayer<B:Backend,P:Module<B>> {
    /// Original explicit dense, adapted, AWQ or NF4 router projection.
    pub router:P,
    /// Original actual native expert cubes and trainability.
    pub experts:NativeSwiGluExperts<B>,
    /// Original optional FP32 selection-only correction bias; no bias gradient is inferred from top-k.
    pub correction_bias:Option<Param<Tensor<B,1>>>,
    /// Actual original discrete/continuous routing and forward/backward execution choices.
    #[module(skip)]
    pub options:MoeOptions,
    /// Explicit optional router-input cast; expert input and stored weights are not cast by this option.
    #[module(skip)]
    pub router_input_dtype:Option<FloatDType>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> NativeMoeLayer<B,P> {
    /// Connect actual original components without an inferred architecture or router precision.
    pub fn from_parts(router:P,experts:NativeSwiGluExperts<B>,correction_bias:Option<Param<Tensor<B,1>>>,options:MoeOptions,router_input_dtype:Option<FloatDType>) -> Self {
        let layer=Self {router,experts,correction_bias,options,router_input_dtype};layer.validate();layer
    }
    /// Original logical branch width, independent of its router's output expert count.
    pub fn width(&self) -> usize {self.experts.dimensions()[1]}
    /// Validate actual original router/expert geometry and independent FP32 bias metadata.
    pub fn validate(&self) {
        self.experts.validate();let [experts,hidden,_]=self.experts.dimensions();assert_eq!(self.router.dimensions(),[hidden,experts],"native router/expert geometry differs");
        if let Some(bias)=&self.correction_bias {let bias=bias.val();assert_eq!(bias.dims(),[experts],"native correction bias width differs");
            assert_eq!(bias.dtype(),DType::F32,"native correction bias must retain FP32");assert_eq!(bias.device(),self.experts.gate.val().device(),"native correction bias device differs");}
        if let Some(dtype)=self.router_input_dtype {assert!(matches!(DType::from(dtype),DType::F16|DType::BF16|DType::F32),"unsupported native router compute storage");}
    }
}
/// Actual routed branch output and the same original logits/selections used for that output.
/// Auxiliary objectives can use these tensors without recomputing a dropout-bearing router.
#[derive(Debug)]
pub struct NativeMoeLayerOutput<B:MoeOps,const D:usize> {
    /// Native branch output retaining every actual input leading axis.
    pub output:Tensor<B,D>,
    /// Actual original tracked router logits used in this forward.
    pub router_logits:Tensor<B,2>,
    /// Actual original nondifferentiable U32 expert selections.
    pub selected_experts:Tensor<B,2,Int>,
    /// Actual original native dispatch/router/expert state.
    pub state:B::MoeState,
}
impl<B:MoeOps,P:TransformerProjection<B>> NativeMoeLayer<B,P> {
    fn project<const D:usize>(&self,input:Tensor<B,D>) -> Result<(Tensor<B,2>,Tensor<B,2>,[usize;D]),P::Error> {
        self.validate();assert!(D>0,"native MoE input must have a feature axis");let shape=input.dims();assert_eq!(shape[D-1],self.width(),"native branch input width differs");
        let rows=shape[..D-1].iter().try_fold(1usize,|count,&axis|count.checked_mul(axis)).expect("native token count overflows");
        let input=input.reshape([rows,self.width()]);let routed=if let Some(dtype)=self.router_input_dtype {input.clone().cast(dtype)} else {input.clone()};
        let logits=self.router.forward(routed)?;Ok((input,logits,shape))
    }
    /// Actual complete routed branch, without a residual/shared expert or model-family default.
    pub fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,NativeMoeLayerError<P::Error,B::MoeError>> {
        let (input,logits,shape)=self.project(input).map_err(NativeMoeLayerError::Router)?;
        self.experts.forward(input,logits,self.correction_bias.as_ref().map(Param::val),self.options).map(|output|output.reshape(shape)).map_err(NativeMoeLayerError::Experts)
    }
    /// Retain actual original logits/selections for explicit auxiliary objectives and exact native state.
    pub fn forward_detailed<const D:usize>(&self,input:Tensor<B,D>) -> Result<NativeMoeLayerOutput<B,D>,NativeMoeLayerError<P::Error,B::MoeError>> {
        let (input,logits,shape)=self.project(input).map_err(NativeMoeLayerError::Router)?;
        let (output,state)=self.experts.forward_with_state(input,logits.clone(),self.correction_bias.as_ref().map(Param::val),self.options).map_err(NativeMoeLayerError::Experts)?;
        Ok(NativeMoeLayerOutput {output:output.reshape(shape),router_logits:logits,selected_experts:self.experts.routing_indices(&state),state})
    }
}
