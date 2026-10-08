use super::*;
use crate::{NativeSwiGluExperts,NativeMoeLayer,NativeMoeLayerOutput,NativeMoeLayerError};
use crate::transformer::TransformerProjection;
use ruda_model::{module::ModuleDisplay,tensor::{MoeOps,MoeOptions,IntegerTensorCollective}};

/// Actual native expert cubes with only rank-local persistent floating leaves.
#[derive(Module,Debug)]
pub struct FullyShardedNativeSwiGluExperts<B:Backend> {
    /// Original `[experts,intermediate,hidden]` logical gate cube.
    pub gate:ShardedParameter<B>,
    /// Original independent up cube.
    pub up:ShardedParameter<B>,
    /// Original `[experts,hidden,intermediate]` logical down cube.
    pub down:ShardedParameter<B>,
}
impl<B:Backend> FullyShardedNativeSwiGluExperts<B> {
    /// Slice actual original source cubes without replacement values or changed parameter IDs.
    pub fn from_full(experts:NativeSwiGluExperts<B>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).moe_experts(experts)}
    /// Original actual `[experts,hidden,intermediate]` logical geometry.
    pub fn dimensions(&self) -> [usize;3] {
        assert_eq!(self.gate.logical_shape.len(),3,"native expert logical rank differs");let shape=&self.gate.logical_shape;[shape[0],shape[2],shape[1]]
    }
    /// Validate actual original local intervals, cube geometry, storage and data topology.
    pub fn validate(&self) {
        let [experts,hidden,inner]=self.dimensions();assert!(experts>0 && hidden>0 && inner>0,"native expert axes must be positive");
        assert_eq!(self.up.logical_shape,self.gate.logical_shape,"native sharded gate/up geometry differs");
        assert_eq!(self.down.logical_shape,[experts,hidden,inner],"native sharded down geometry differs");
        let source=self.gate.local.val();assert!(matches!(source.dtype(),DType::F16|DType::BF16|DType::F32),"unsupported native expert storage");
        for parameter in [&self.gate,&self.up,&self.down] {
            let value=parameter.local.val();assert_eq!(value.dtype(),source.dtype(),"native expert storage differs");assert_eq!(value.device(),source.device(),"native expert devices differ");
            assert_eq!((parameter.rank,parameter.world_size),(self.gate.rank,self.gate.world_size),"native expert topology differs");
            let _=ShardedParameter::from_local(parameter.local.clone(),parameter.logical_shape.clone(),parameter.rank,parameter.world_size);
        }
    }
}
/// Complete actual local routed branch with both packed router and floating expert leaves data-sharded.
#[derive(Module,Debug)]
pub struct FullyShardedNativeMoeLayer<B:Backend,P:Module<B>> {
    /// Original explicitly selected router projection's local leaves.
    pub router:P,
    /// Original actual expert cube local leaves, not independent fake experts per rank.
    pub experts:FullyShardedNativeSwiGluExperts<B>,
    /// Original optional independent FP32 selection-only correction bias slices.
    pub correction_bias:Option<ShardedParameter<B>>,
    /// Original explicit routing and expert execution choices.
    #[module(skip)]
    pub options:MoeOptions,
    /// Original explicit optional router-input work dtype.
    #[module(skip)]
    pub router_input_dtype:Option<FloatDType>,
}
impl<B:Backend> ShardingContext<B> {
    /// Preserve actual original expert cubes through the same shared-ID context as the full model.
    pub fn moe_experts(&mut self,experts:NativeSwiGluExperts<B>) -> FullyShardedNativeSwiGluExperts<B> {
        experts.validate();let experts=FullyShardedNativeSwiGluExperts {gate:self.parameter(experts.gate),up:self.parameter(experts.up),down:self.parameter(experts.down)};
        experts.validate();experts
    }
    /// Partition the original native branch without inventing source topology or quantizing its experts.
    pub fn moe_layer<P:ShardTransformerProjection<B>>(&mut self,layer:NativeMoeLayer<B,P>) -> FullyShardedNativeMoeLayer<B,P::Sharded> {
        layer.validate();FullyShardedNativeMoeLayer {router:self.awq_projection(layer.router),experts:self.moe_experts(layer.experts),
            correction_bias:layer.correction_bias.map(|bias|self.parameter(bias)),options:layer.options,router_input_dtype:layer.router_input_dtype}
    }
}
impl<B:Backend,P:Module<B>> FullyShardedNativeMoeLayer<B,P> {
    /// Partition actual loaded native components; reuse an external context for cross-component parameter ties.
    pub fn from_full<Q:ShardTransformerProjection<B,Sharded=P>>(layer:NativeMoeLayer<B,Q>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).moe_layer(layer)}
}
impl<B:Backend,P:FullyShardedModule<B>+ModuleDisplay> FullyShardedNativeMoeLayer<B,P> {
    /// Original logical native branch width, independent of router expert-count output width.
    pub fn width(&self) -> usize {self.experts.dimensions()[1]}
    /// Validate exact original devices/topology for all actual floating and packed local roles.
    pub fn validate(&self) {
        self.experts.validate();let device=self.experts.gate.local.val().device();let topology=(self.experts.gate.rank,self.experts.gate.world_size);
        self.router.visit_shards(&mut |parameter| {assert_eq!(parameter.local.val().device(),device,"router/expert devices differ");
            assert_eq!((parameter.rank,parameter.world_size),topology,"router/expert topologies differ");});
        self.router.visit_packed_shards(&mut |parameter| {assert_eq!(parameter.local.val().device(),device,"packed router/expert devices differ");
            assert_eq!((parameter.rank,parameter.world_size),topology,"packed router/expert topologies differ");});
        if let Some(bias)=&self.correction_bias {assert_eq!(bias.logical_shape,[self.experts.dimensions()[0]],"native correction bias logical width differs");
            assert_eq!(bias.local.val().dtype(),DType::F32,"native correction bias must retain FP32");assert_eq!(bias.local.val().device(),device,"native correction bias device differs");
            assert_eq!((bias.rank,bias.world_size),topology,"native correction bias topology differs");}
    }
}
/// Original integer/floating transport failure or original router/expert native execution failure.
#[derive(Debug)]
pub enum FullyShardedNativeMoeError<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> {
    /// Original actual native data-collective error.
    Collective(C),
    /// Original actual native routed branch error.
    Layer(NativeMoeLayerError<P,M>),
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::fmt::Display for FullyShardedNativeMoeError<C,P,M> {
    fn fmt(&self,f:&mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {Self::Collective(error)=>write!(f,"native MoE data transport: {error:?}"),Self::Layer(error)=>write!(f,"native MoE branch: {error}")}
    }
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::error::Error for FullyShardedNativeMoeError<C,P,M> {}

macro_rules! native_moe_gather {
    ($backend:ty,[$($generics:tt)*],$gather:ident) => {
        impl<$($generics)*> FullyShardedNativeSwiGluExperts<$backend> {
            /// Transient original expert cubes with actual native inference/AD gathers.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<NativeSwiGluExperts<$backend>,C::Error> {
                self.validate();Ok(NativeSwiGluExperts::from_parameters(Param::initialized(self.gate.local.id,self.gate.$gather::<C,3>(communicator.clone())?),
                    Param::initialized(self.up.local.id,self.up.$gather::<C,3>(communicator.clone())?),Param::initialized(self.down.local.id,self.down.$gather::<C,3>(communicator)?)))
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedNativeMoeLayer<$backend,P> {
            /// Original packed/dense router and expert values, without full persistent model replicas.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<NativeMoeLayer<$backend,P::Gathered>,C::Error> {
                self.validate();Ok(NativeMoeLayer::from_parts(self.router.gather_projection(communicator.clone())?,self.experts.$gather(communicator.clone())?,
                    self.correction_bias.as_ref().map(|bias|bias.$gather::<C,1>(communicator).map(|value|Param::initialized(bias.local.id,value))).transpose()?,self.options,self.router_input_dtype))
            }
        }
    };
}
native_moe_gather!(B,[B:Backend],gather_inference);
native_moe_gather!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);
macro_rules! native_moe_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$forward:ident,$detailed:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedNativeMoeLayer<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Original routed branch with native first-order input/router/expert derivatives on AD.
            pub fn $forward<C:IntegerTensorCollective<B>,const D:usize>(&self,input:Tensor<$backend,D>,communicator:C)
                -> Result<Tensor<$backend,D>,FullyShardedNativeMoeError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>> {
                self.$gather(communicator).map_err(FullyShardedNativeMoeError::Collective)?.forward(input).map_err(FullyShardedNativeMoeError::Layer)
            }
            /// Retain the same actual gathered router logits/selections, avoiding a second dropout-bearing router pass.
            pub fn $detailed<C:IntegerTensorCollective<B>,const D:usize>(&self,input:Tensor<$backend,D>,communicator:C)
                -> Result<NativeMoeLayerOutput<$backend,D>,FullyShardedNativeMoeError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>> {
                self.$gather(communicator).map_err(FullyShardedNativeMoeError::Collective)?.forward_detailed(input).map_err(FullyShardedNativeMoeError::Layer)
            }
        }
    };
}
native_moe_execution!(B,[B:MoeOps],gather_inference,forward_inference,forward_detailed_inference);
native_moe_execution!(Autodiff<B,S>,[B:MoeOps,S:CheckpointStrategy],gather,forward,forward_detailed);
