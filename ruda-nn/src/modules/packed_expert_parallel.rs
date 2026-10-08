use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use core::{fmt,ops::Range};
use ruda_model::{module::{Module,ModuleVisitor,Param,ParamId},record::RecorderError,tensor::{Tensor,Int,DType,
    MoeOptions,MoeExpertStrategy,MoeReceivedOps,FrozenPackedExpertOps,ExpertProjectionOps,NativeSwiGluOps,backend::Backend}};
use super::{FrozenAwqExpertProjection,FrozenPackedExpertProjection,FrozenPackedSwiGluExperts,AdaptedExpertProjection,
    AdaptedPackedSwiGluExperts,SelectablePackedExperts,AdaptedExpertError,LoRALinearConfig,ExpertAdapterTarget,FrozenNf4ExpertWindow,
    FrozenExpertGeometry,FrozenSelectedExperts,ExpertAdapterProjections,ExpertAdapterProjectionRef,ExpertAdapterMapper};
use super::expert_parallel::{ExpertOwnership,ExpertPartitionContext,ExpertParallelGeometry,ExpertParallelReceived,ExpertParallelMoeLayer};

struct WordEntry<B:Backend> {shape:[usize;3],range:Range<usize>,local:Param<Tensor<B,3,Int>>}
struct BiasEntry<B:Backend> {shape:[usize;2],range:Range<usize>,local:Param<Tensor<B,2>>}
/// Canonical actual expert-owned AWQ words/zeros/scales/bias and floating A/B.
/// Retains only owned copies; preserves original packed bits and complete input groups.
pub struct AwqExpertPartitionContext<B:Backend> {
    words:BTreeMap<ParamId,WordEntry<B>>,
    bias:BTreeMap<ParamId,BiasEntry<B>>,
    floating:ExpertPartitionContext<B>,
}
impl<B:Backend> Default for AwqExpertPartitionContext<B> {fn default() -> Self {Self::new()}}
impl<B:Backend> AwqExpertPartitionContext<B> {
    /// Start a source-ID context without inferring rank, expert split, storage or adapter parameters.
    pub fn new() -> Self {Self {words:BTreeMap::new(),bias:BTreeMap::new(),floating:ExpertPartitionContext::new()}}
    fn words(&mut self,parameter:Param<Tensor<B,3,Int>>,ownership:&ExpertOwnership,rank:usize) -> Param<Tensor<B,3,Int>> {
        let value=parameter.val();let shape=value.dims();let [e,rows,columns]=shape;let range=ownership.range(rank);
        assert_eq!(e,ownership.experts(),"original AWQ word cube count differs from declared expert ownership");
        assert_eq!(value.dtype(),DType::I32,"owned AWQ word/zero storage must retain actual I32 bits");
        if let Some(entry)=self.words.get(&parameter.id) {
            assert_eq!(entry.shape,shape,"tied AWQ word source geometry differs");assert_eq!(entry.range,range,"tied AWQ words have different owned intervals");
            let local=entry.local.val();assert_eq!(local.device(),value.device(),"tied AWQ word source devices differ");
            let (id,_,mapper)=parameter.consume();return Param::from_mapped_value(id,local,mapper);
        }
        let count=range.len();let mut local=Tensor::<B,3,Int>::empty([count,rows,columns],(&value.device(),DType::I32));
        if count!=0 {local=local.slice_assign([0..count,0..rows,0..columns],value.slice([range.clone(),0..rows,0..columns]));}
        let local=parameter.map(|_|local);self.words.insert(local.id,WordEntry {shape,range,local:local.clone()});local
    }
    fn bias(&mut self,parameter:Param<Tensor<B,2>>,ownership:&ExpertOwnership,rank:usize) -> Param<Tensor<B,2>> {
        let value=parameter.val();let shape=value.dims();let range=ownership.range(rank);assert_eq!(shape[0],ownership.experts(),"original AWQ bias count differs from declared ownership");
        assert!(!value.is_require_grad(),"original AWQ expert bias must remain frozen");
        if let Some(entry)=self.bias.get(&parameter.id) {
            assert_eq!(entry.shape,shape,"tied AWQ bias geometry differs");assert_eq!(entry.range,range,"tied AWQ bias has different owned intervals");
            let local=entry.local.val();assert_eq!(local.dtype(),value.dtype(),"tied AWQ bias storage differs");assert_eq!(local.device(),value.device(),"tied AWQ bias devices differ");
            let (id,_,mapper)=parameter.consume();return Param::from_mapped_value(id,local,mapper);
        }
        let count=range.len();let width=shape[1];let mut local=Tensor::<B,2>::empty([count,width],(&value.device(),value.dtype()));
        if count!=0 {local=local.slice_assign([0..count,0..width],value.slice([range.clone(),0..width]));}
        let local=parameter.map(|_|local.detach().set_require_grad(false));self.bias.insert(local.id,BiasEntry {shape,range,local:local.clone()});local
    }
    /// Copy only the actual owned expert-axis interval, without reordering lanes or changing any scale/bias storage.
    pub fn projection(&mut self,source:FrozenAwqExpertProjection<B>,ownership:&ExpertOwnership,rank:usize) -> FrozenAwqExpertProjection<B> {
        source.validate();assert_eq!(source.dimensions()[0],ownership.experts(),"original AWQ expert count differs from declared ownership");
        let local=FrozenAwqExpertProjection {qweight:self.words(source.qweight,ownership,rank),qzeros:self.words(source.qzeros,ownership,rank),
            scales:self.floating.parameter(source.scales,ownership,rank),bias:source.bias.map(|value|self.bias(value,ownership,rank)),group_size:source.group_size};
        local.validate();local
    }
    fn packed(&mut self,source:FrozenPackedExpertProjection<B>,ownership:&ExpertOwnership,rank:usize) -> FrozenPackedExpertProjection<B> {
        match source {FrozenPackedExpertProjection::Awq(source)=>self.projection(source,ownership,rank).into(),
            FrozenPackedExpertProjection::Nf4(_)|FrozenPackedExpertProjection::Nf4Window(_)=>unreachable!("validated original source projections are AWQ")}
    }
    fn adapted(&mut self,source:AdaptedExpertProjection<B>,ownership:&ExpertOwnership,rank:usize) -> AdaptedExpertProjection<B> {
        match source {
            AdaptedExpertProjection::Frozen(source)=>AdaptedExpertProjection::Frozen(self.packed(source,ownership,rank)),
            AdaptedExpertProjection::LoRA(mut layer)=>{
                layer.base=self.packed(layer.base,ownership,rank);
                layer.adapter_a.weight=self.floating.parameter(layer.adapter_a.weight,ownership,rank);
                layer.adapter_b.weight=self.floating.parameter(layer.adapter_b.weight,ownership,rank);layer.validate();AdaptedExpertProjection::LoRA(layer)
            },
        }
    }
    /// Validate all actual source formats first, then copy only this rank's packed bases and present A/B.
    pub fn experts(&mut self,source:SelectablePackedExperts<B>,ownership:ExpertOwnership,rank:usize) -> OwnedAwqExperts<B> {
        validate_awq(&source);assert_eq!(source.dimensions()[0],ownership.experts(),"original packed expert count differs from declared ownership");ownership.range(rank);
        let experts=match source {
            SelectablePackedExperts::Original(source)=>SelectablePackedExperts::Original(FrozenPackedSwiGluExperts::from_parts(
                self.packed(source.gate,&ownership,rank),self.packed(source.up,&ownership,rank),self.packed(source.down,&ownership,rank))),
            SelectablePackedExperts::Adapted(source)=>SelectablePackedExperts::Adapted(AdaptedPackedSwiGluExperts::from_parts(
                self.adapted(source.gate,&ownership,rank),self.adapted(source.up,&ownership,rank),self.adapted(source.down,&ownership,rank))),
        };OwnedAwqExperts::from_parts(experts,ownership,rank)
    }
}
fn validate_awq<B:Backend>(experts:&SelectablePackedExperts<B>) {
    experts.validate();let base=|projection:&FrozenPackedExpertProjection<B>| {
        assert!(matches!(projection,FrozenPackedExpertProjection::Awq(_)),"AWQ expert ownership requires actual AWQ source projections; formats are not converted");
    };
    match experts {
        SelectablePackedExperts::Original(value)=>{for projection in [&value.gate,&value.up,&value.down] {base(projection);}},
        SelectablePackedExperts::Adapted(value)=>{for projection in [&value.gate,&value.up,&value.down] {match projection {
            AdaptedExpertProjection::Frozen(value)=>base(value),AdaptedExpertProjection::LoRA(value)=>base(&value.base)}}},
    }
}
/// Actual rank-owned original AWQ words/zeros/scales/bias and explicitly present per-expert A/B.
#[derive(Module,Debug)]
pub struct OwnedAwqExperts<B:Backend> {
    /// Only the actual local original packed expert chain or explicitly adapted chain.
    pub experts:SelectablePackedExperts<B>,
    /// Complete original declared expert-world ownership, including empty owners.
    #[module(skip)]
    pub ownership:ExpertOwnership,
    /// Explicit original expert-world rank.
    pub rank:usize,
}
impl<B:Backend> OwnedAwqExperts<B> {
    /// Connect actual caller-loaded local packed values/A/B, without constructing missing projections.
    pub fn from_parts(experts:SelectablePackedExperts<B>,ownership:ExpertOwnership,rank:usize) -> Self {
        let owned=Self {experts,ownership,rank};owned.validate();owned
    }
    /// Actual `[owned_experts,hidden,intermediate]` widths.
    pub fn dimensions(&self) -> [usize;3] {self.experts.dimensions()}
    /// Actual original resident packed source device.
    pub fn device(&self) -> B::Device {self.experts.device()}
    /// Validate actual owned count/storage without decoding or downloading packed coefficients.
    pub fn validate(&self) {validate_awq(&self.experts);assert_eq!(self.dimensions()[0],self.ownership.range(self.rank).len(),"actual owned AWQ count differs from declared interval");}
    /// Copy only actual owned source expert values with a caller-shared canonical ID context.
    pub fn from_full(source:SelectablePackedExperts<B>,ownership:ExpertOwnership,rank:usize,context:&mut AwqExpertPartitionContext<B>) -> Self {
        context.experts(source,ownership,rank)
    }
    /// Attach new A/B only to explicit locally owned AWQ roles; original words/metadata remain frozen.
    pub fn with_adapters(mut self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        let experts=match self.experts {SelectablePackedExperts::Original(value)=>AdaptedPackedSwiGluExperts::from_frozen(value),SelectablePackedExperts::Adapted(value)=>value};
        self.experts=SelectablePackedExperts::Adapted(experts.with_adapters(config,targets,dtype,use_rslora,forward,backward));self.validate();self
    }
    /// Canonical actual owned A/B identities only, excluding original quantization metadata.
    pub fn adapter_parameter_ids(&self) -> Vec<ParamId> {self.experts.adapter_parameter_ids()}
}
struct FloatingIds(BTreeSet<ParamId>);
impl<B:Backend> ModuleVisitor<B> for FloatingIds {
    fn visit_float<const D:usize>(&mut self,parameter:&Param<Tensor<B,D>>) {self.0.insert(parameter.id);}
}
impl<B:Backend> ExpertParallelGeometry<B> for OwnedAwqExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn ownership(&self) -> &ExpertOwnership {&self.ownership}
    fn rank(&self) -> usize {self.rank}
    fn device(&self) -> B::Device {self.device()}
    fn validate(&self) {self.validate();}
    fn parameter_ids(&self) -> Vec<ParamId> {let mut ids=FloatingIds(BTreeSet::new());self.experts.visit(&mut ids);ids.0.into_iter().collect()}
}
/// Original routing error or original packed/A/B/activation error, without loss of source error types.
#[derive(Debug)]
pub enum AwqExpertParallelError<M:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> {
    /// Original native dispatch/routing/combine failure.
    Routing(M),
    /// Actual source packed projection, native A/B or original storage-rounded SwiGLU failure.
    Experts(AdaptedExpertError<P,G,S>),
}
impl<M:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> fmt::Display for AwqExpertParallelError<M,P,G,S> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Routing(error)=>write!(f,"native expert routing: {error:?}"),Self::Experts(error)=>write!(f,"{error}")}}
}
impl<M:fmt::Debug,P:fmt::Debug,G:fmt::Debug,S:fmt::Debug> core::error::Error for AwqExpertParallelError<M,P,G,S> {}
impl<B:MoeReceivedOps+FrozenPackedExpertOps+ExpertProjectionOps+NativeSwiGluOps> ExpertParallelReceived<B> for OwnedAwqExperts<B> {
    type Error=AwqExpertParallelError<B::MoeError,B::PackedExpertError,B::ExpertProjectionError,B::SwiGluError>;
    fn routing_error(error:B::MoeError) -> Self::Error {AwqExpertParallelError::Routing(error)}
    fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,_options:MoeOptions) -> Result<Tensor<B,2>,Self::Error> {
        self.validate();self.experts.forward(input,ids,self.ownership.range(self.rank).start).map_err(AwqExpertParallelError::Experts)
    }
}
impl<B:Backend> FrozenExpertGeometry<B> for OwnedAwqExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn validate(&self) {self.validate();}
    fn device(&self) -> B::Device {self.device()}
}
impl<B:Backend> ExpertAdapterProjections<B> for OwnedAwqExperts<B> {
    fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {self.experts.expert_adapter_projections()}
    fn map_expert_adapters<M:ExpertAdapterMapper<B>>(mut self,mapper:&mut M) -> Result<Self,RecorderError> {
        self.experts=self.experts.map_expert_adapters(mapper)?;self.validate();Ok(self)
    }
}
/// Complete original expert-owned routing/transport/combine with actual AWQ source coefficients and optional local A/B.
pub type AwqExpertParallelMoeLayer<B,P> = ExpertParallelMoeLayer<B,P,OwnedAwqExperts<B>>;
/// Complete original dense/packed/cache/causal-loss model execution with actual rank-owned AWQ experts.
pub type AwqExpertParallelTransformerModel<B,P> = crate::transformer::ExpertParallelTransformerModel<B,P,OwnedAwqExperts<B>>;

struct ByteWindowEntry<B:Backend> {shape:[usize;1],range:Range<usize>,local:Param<Tensor<B,1,Int>>}
struct ScaleWindowEntry<B:Backend> {shape:[usize;1],range:Range<usize>,local:Param<Tensor<B,1>>}
/// Shared exact-source ownership for independent AWQ/NF4 projections and already loaded expert adapters.
/// Retains only actual owned packed windows, intersecting original scale blocks and canonical A/B leaves.
pub struct PackedExpertPartitionContext<B:Backend> {
    awq:AwqExpertPartitionContext<B>,
    bytes:BTreeMap<ParamId,ByteWindowEntry<B>>,
    scales:BTreeMap<ParamId,ScaleWindowEntry<B>>,
}
impl<B:Backend> Default for PackedExpertPartitionContext<B> {fn default() -> Self {Self::new()}}
impl<B:Backend> PackedExpertPartitionContext<B> {
    /// Begin an explicit source-ID context, without selecting a topology or quantization policy.
    pub fn new() -> Self {Self {awq:AwqExpertPartitionContext::new(),bytes:BTreeMap::new(),scales:BTreeMap::new()}}
    fn bytes(&mut self,source:Param<Tensor<B,1,Int>>,range:Range<usize>) -> Param<Tensor<B,1,Int>> {
        let value=source.val();let shape=value.dims();assert_eq!(value.dtype(),DType::U8,"NF4 ownership retains original U8 bytes");
        assert!(range.end<=shape[0] && range.start<=range.end,"NF4 owned byte interval exceeds actual source storage");
        if let Some(entry)=self.bytes.get(&source.id) {
            assert_eq!(entry.shape,shape,"tied NF4 source byte lengths differ");assert_eq!(entry.range,range,"tied NF4 source byte windows differ");
            let local=entry.local.val();assert_eq!(local.device(),value.device(),"tied NF4 source byte devices differ");
            let (id,_,mapper)=source.consume();return Param::from_mapped_value(id,local,mapper);
        }
        let mut local=Tensor::<B,1,Int>::empty([range.len()],(&value.device(),DType::U8));
        if !range.is_empty() {local=local.slice_assign([0..range.len()],value.slice([range.clone()]));}
        let local=source.map(|_|local);self.bytes.insert(local.id,ByteWindowEntry {shape,range,local:local.clone()});local
    }
    fn scales(&mut self,source:Param<Tensor<B,1>>,range:Range<usize>) -> Param<Tensor<B,1>> {
        let value=source.val();let shape=value.dims();assert_eq!(value.dtype(),DType::F32,"NF4 ownership retains original FP32 scales");
        assert!(!value.is_require_grad(),"NF4 source scales remain frozen");
        assert!(range.end<=shape[0] && range.start<=range.end,"NF4 owned scale interval exceeds actual source storage");
        if let Some(entry)=self.scales.get(&source.id) {
            assert_eq!(entry.shape,shape,"tied NF4 source scale lengths differ");assert_eq!(entry.range,range,"tied NF4 source scale windows differ");
            let local=entry.local.val();assert_eq!(local.device(),value.device(),"tied NF4 source scale devices differ");
            let (id,_,mapper)=source.consume();return Param::from_mapped_value(id,local,mapper);
        }
        let mut local=Tensor::<B,1>::empty([range.len()],(&value.device(),DType::F32));
        if !range.is_empty() {local=local.slice_assign([0..range.len()],value.slice([range.clone()]));}
        let local=source.map(|_|local.detach().set_require_grad(false));self.scales.insert(local.id,ScaleWindowEntry {shape,range,local:local.clone()});local
    }
    fn nf4(&mut self,mut source:FrozenNf4ExpertWindow<B>,ownership:&ExpertOwnership,rank:usize) -> FrozenNf4ExpertWindow<B> {
        source.validate();assert_eq!(source.experts,ownership.experts(),"actual NF4 expert count differs from declared ownership");
        let range=ownership.range(rank);let matrix=source.input_features*source.output_features;
        let start=source.element_offset+range.start*matrix;let end=source.element_offset+range.end*matrix;
        let (bytes,scales,offset)=if range.is_empty() {(0..0,0..0,0)} else {
            let block_start=start/source.block_size;
            (block_start*(source.block_size/2)..end.div_ceil(2),block_start..end.div_ceil(source.block_size),start%source.block_size)
        };
        source.packed=self.bytes(source.packed,bytes);source.scales=self.scales(source.scales,scales);
        source.codebook=self.scales(source.codebook,0..16);
        source.experts=range.len();source.element_offset=offset;source.validate();source
    }
    /// Copy only actual source-owned coefficients, retaining leading partial blocks and original nibble order.
    pub fn projection(&mut self,source:FrozenPackedExpertProjection<B>,ownership:&ExpertOwnership,rank:usize) -> FrozenPackedExpertProjection<B> {
        source.validate();match source {
            FrozenPackedExpertProjection::Awq(value)=>self.awq.projection(value,ownership,rank).into(),
            FrozenPackedExpertProjection::Nf4Window(value)=>self.nf4(value,ownership,rank).into(),
            FrozenPackedExpertProjection::Nf4(value)=>{
                let payload=value.payload;
                self.nf4(FrozenNf4ExpertWindow {packed:payload.packed,scales:payload.scales,codebook:payload.codebook,experts:value.experts,
                    input_features:payload.input_features,output_features:value.output_features,block_size:payload.block_size,element_offset:0,
                    tile_rows:payload.tile_rows,use_tensor_core:payload.use_tensor_core},ownership,rank).into()
            },
        }
    }
    fn adapted(&mut self,source:AdaptedExpertProjection<B>,ownership:&ExpertOwnership,rank:usize) -> AdaptedExpertProjection<B> {
        match source {
            AdaptedExpertProjection::Frozen(value)=>AdaptedExpertProjection::Frozen(self.projection(value,ownership,rank)),
            AdaptedExpertProjection::LoRA(mut value)=>{
                value.base=self.projection(value.base,ownership,rank);
                value.adapter_a.weight=self.awq.floating.parameter(value.adapter_a.weight,ownership,rank);
                value.adapter_b.weight=self.awq.floating.parameter(value.adapter_b.weight,ownership,rank);
                value.validate();AdaptedExpertProjection::LoRA(value)
            },
        }
    }
    /// Preserve each actual projection's format/execution and copy only owned original bases and present A/B.
    pub fn experts(&mut self,source:SelectablePackedExperts<B>,ownership:ExpertOwnership,rank:usize) -> OwnedPackedExperts<B> {
        source.validate();assert_eq!(source.dimensions()[0],ownership.experts(),"actual source count differs from declared packed expert ownership");ownership.range(rank);
        let experts=match source {
            SelectablePackedExperts::Original(value)=>SelectablePackedExperts::Original(FrozenPackedSwiGluExperts::from_parts(
                self.projection(value.gate,&ownership,rank),self.projection(value.up,&ownership,rank),self.projection(value.down,&ownership,rank))),
            SelectablePackedExperts::Adapted(value)=>SelectablePackedExperts::Adapted(AdaptedPackedSwiGluExperts::from_parts(
                self.adapted(value.gate,&ownership,rank),self.adapted(value.up,&ownership,rank),self.adapted(value.down,&ownership,rank))),
        };OwnedPackedExperts::from_parts(experts,ownership,rank)
    }
}
/// Actual rank-owned independent original NF4/AWQ gate/up/down and optional native expert adapters.
#[derive(Module,Debug)]
pub struct OwnedPackedExperts<B:Backend> {
    /// Actual original owned packed projections, including exact NF4 source-block windows.
    pub experts:SelectablePackedExperts<B>,
    /// Original complete explicit expert-world partition.
    #[module(skip)]
    pub ownership:ExpertOwnership,
    /// Actual rank within the explicit expert world.
    pub rank:usize,
}
impl<B:Backend> OwnedPackedExperts<B> {
    /// Connect actual loaded owned values, without initializing missing bases or A/B.
    pub fn from_parts(experts:SelectablePackedExperts<B>,ownership:ExpertOwnership,rank:usize) -> Self {
        let owned=Self {experts,ownership,rank};owned.validate();owned
    }
    /// Copy only actual owned source intervals with a shared canonical source-ID context.
    pub fn from_full(source:SelectablePackedExperts<B>,ownership:ExpertOwnership,rank:usize,context:&mut PackedExpertPartitionContext<B>) -> Self {
        context.experts(source,ownership,rank)
    }
    /// Actual `[owned_experts,hidden,intermediate]` logical widths.
    pub fn dimensions(&self) -> [usize;3] {self.experts.dimensions()}
    /// Actual resident original source payload device.
    pub fn device(&self) -> B::Device {self.experts.device()}
    /// Validate resident packed windows against the actual declared owner interval.
    pub fn validate(&self) {self.experts.validate();assert_eq!(self.dimensions()[0],self.ownership.range(self.rank).len(),"actual owned packed count differs from declared interval");}
    /// Attach new local A/B to explicit roles without changing any original quantization metadata.
    pub fn with_adapters(mut self,config:&LoRALinearConfig,targets:&[ExpertAdapterTarget],dtype:DType,use_rslora:bool,
        forward:MoeExpertStrategy,backward:MoeExpertStrategy) -> Self {
        let experts=match self.experts {SelectablePackedExperts::Original(value)=>AdaptedPackedSwiGluExperts::from_frozen(value),SelectablePackedExperts::Adapted(value)=>value};
        self.experts=SelectablePackedExperts::Adapted(experts.with_adapters(config,targets,dtype,use_rslora,forward,backward));self.validate();self
    }
    /// Canonical actual owned A/B identities only.
    pub fn adapter_parameter_ids(&self) -> Vec<ParamId> {self.experts.adapter_parameter_ids()}
}
impl<B:Backend> ExpertParallelGeometry<B> for OwnedPackedExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn ownership(&self) -> &ExpertOwnership {&self.ownership}
    fn rank(&self) -> usize {self.rank}
    fn device(&self) -> B::Device {self.device()}
    fn validate(&self) {self.validate();}
    fn parameter_ids(&self) -> Vec<ParamId> {let mut ids=FloatingIds(BTreeSet::new());self.experts.visit(&mut ids);ids.0.into_iter().collect()}
}
/// Original routing, packed projection, native A/B or source activation error.
pub type PackedExpertParallelError<M,P,G,S> = AwqExpertParallelError<M,P,G,S>;
impl<B:MoeReceivedOps+FrozenPackedExpertOps+ExpertProjectionOps+NativeSwiGluOps> ExpertParallelReceived<B> for OwnedPackedExperts<B> {
    type Error=PackedExpertParallelError<B::MoeError,B::PackedExpertError,B::ExpertProjectionError,B::SwiGluError>;
    fn routing_error(error:B::MoeError) -> Self::Error {PackedExpertParallelError::Routing(error)}
    fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,_options:MoeOptions) -> Result<Tensor<B,2>,Self::Error> {
        self.validate();self.experts.forward(input,ids,self.ownership.range(self.rank).start).map_err(PackedExpertParallelError::Experts)
    }
}
impl<B:Backend> FrozenExpertGeometry<B> for OwnedPackedExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn validate(&self) {self.validate();}
    fn device(&self) -> B::Device {self.device()}
}
impl<B:Backend> ExpertAdapterProjections<B> for OwnedPackedExperts<B> {
    fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {self.experts.expert_adapter_projections()}
    fn map_expert_adapters<M:ExpertAdapterMapper<B>>(mut self,mapper:&mut M) -> Result<Self,RecorderError> {
        self.experts=self.experts.map_expert_adapters(mapper)?;self.validate();Ok(self)
    }
}
/// Complete original expert-owned transport/routing with independent original NF4/AWQ projections.
pub type PackedExpertParallelMoeLayer<B,P> = ExpertParallelMoeLayer<B,P,OwnedPackedExperts<B>>;
/// Complete dense/packed/cache/causal-loss graph with exact source-owned quantized experts.
pub type PackedExpertParallelTransformerModel<B,P> = crate::transformer::ExpertParallelTransformerModel<B,P,OwnedPackedExperts<B>>;
