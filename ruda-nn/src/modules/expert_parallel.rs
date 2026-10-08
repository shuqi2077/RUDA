//! Actual expert-owned native cubes and differentiable assignment exchanges; transport is explicit.
use alloc::{vec::Vec,collections::BTreeMap};
use core::{fmt,ops::Range};
use ruda_model::{module::{Module,ModuleDisplay,Param,ParamId},tensor::{Tensor,Int,DType,FloatDType,MoeOptions,MoeDispatchOps,MoeReceivedOps,
    MoeReceivedOptions,dispatch_moe,combine_moe,received_moe_experts,VariableTensorCollective,VariableTensorExchange,backend::Backend}};
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,collective::{all_to_all_v_coordinated,ScopedCollectiveError}};
use crate::{NativeSwiGluExperts,transformer::{TransformerProjectionShape,TransformerProjection}};

/// Explicit global contiguous expert ownership, including actual zero-expert ranks.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct ExpertOwnership {prefix:Vec<usize>}
impl ExpertOwnership {
    /// Declare every rank's original expert interval. No even split or model-family policy is inferred.
    pub fn new(prefix:Vec<usize>) -> Self {
        assert!(prefix.len()>=2 && prefix.len()-1<=u32::MAX as usize,"expert ownership requires an actual finite world");
        assert_eq!(prefix[0],0,"global expert ownership must begin at zero");
        assert!(prefix.windows(2).all(|pair|pair[0]<=pair[1]),"expert ownership prefix must be nondecreasing");
        assert!(*prefix.last().unwrap()>0 && *prefix.last().unwrap()<=u32::MAX as usize,"global expert count must fit native U32 IDs");Self {prefix}
    }
    /// Original declared expert transport world, not the data-parallel world.
    pub fn world_size(&self) -> usize {self.prefix.len()-1}
    /// Original full global expert count.
    pub fn experts(&self) -> usize {*self.prefix.last().unwrap()}
    /// Original owned global range for an explicitly selected rank.
    pub fn range(&self,rank:usize) -> Range<usize> {assert!(rank<self.world_size(),"expert rank is outside the declared original world");self.prefix[rank]..self.prefix[rank+1]}
    /// Actual original immutable complete expert prefix.
    pub fn prefix(&self) -> &[usize] {&self.prefix}
}
/// Only the actual expert cubes owned by this rank, not a full-expert replica behind slice views.
#[derive(Module,Debug)]
pub struct ExpertParallelSwiGluExperts<B:Backend> {
    /// Original local `[owned_experts,intermediate,hidden]` gate values and ID.
    pub gate:Param<Tensor<B,3>>,
    /// Original independent local up cube.
    pub up:Param<Tensor<B,3>>,
    /// Original local `[owned_experts,hidden,intermediate]` down cube.
    pub down:Param<Tensor<B,3>>,
    /// Explicit original whole expert-world ownership.
    #[module(skip)]
    pub ownership:ExpertOwnership,
    /// Actual original expert-world rank, independent of any data-axis rank.
    pub rank:usize,
}
/// Actual rank-owned expert geometry, independent of original or adapted floating execution.
pub trait ExpertParallelGeometry<B:Backend>:Module<B>+ModuleDisplay {
    /// Actual local `[owned_experts,hidden,intermediate]` geometry, including empty owners.
    fn dimensions(&self) -> [usize;3];
    /// Original complete expert-world ownership.
    fn ownership(&self) -> &ExpertOwnership;
    /// Explicit original expert-world rank.
    fn rank(&self) -> usize;
    /// Actual original resident expert device.
    fn device(&self) -> B::Device;
    /// Validate actual local geometry, storage and declared ownership.
    fn validate(&self);
    /// Actual rank-owned floating parameter identities, including present expert A/B.
    fn parameter_ids(&self) -> Vec<ParamId>;
}
/// Native execution on actual received rows; routing weights are applied once by the original source combine.
pub trait ExpertParallelReceived<B:MoeReceivedOps>:ExpertParallelGeometry<B> {
    /// Execute the actual local experts in original receive order, without a full-expert replica.
    fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,options:MoeOptions) -> Result<Tensor<B,2>,B::MoeError>;
}
struct ExpertEntry<B:Backend> {source_shape:[usize;3],range:Range<usize>,local:Param<Tensor<B,3>>}
/// Canonical original shared expert IDs across the complete loaded model.
/// Keeps only rank-owned native copies, never full source cubes or their graphs.
pub struct ExpertPartitionContext<B:Backend> {shared:BTreeMap<(ParamId,bool),ExpertEntry<B>>}
impl<B:Backend> Default for ExpertPartitionContext<B> {fn default() -> Self {Self::new()}}
impl<B:Backend> ExpertPartitionContext<B> {
    /// Start an explicit shared-ID expert partition context, without choosing rank or ownership.
    pub fn new() -> Self {Self {shared:BTreeMap::new()}}
    /// Copy original caller-selected expert intervals, reusing one actual leaf for every tied source ID.
    pub fn experts(&mut self,experts:NativeSwiGluExperts<B>,ownership:ExpertOwnership,rank:usize) -> ExpertParallelSwiGluExperts<B> {
        experts.validate();assert_eq!(experts.dimensions()[0],ownership.experts(),"global source cube count differs from actual declared ownership");
        let range=ownership.range(rank);ExpertParallelSwiGluExperts::from_parameters(local_cube(experts.gate,range.clone(),&mut self.shared),
            local_cube(experts.up,range.clone(),&mut self.shared),local_cube(experts.down,range,&mut self.shared),ownership,rank)
    }
    /// Copy an actual floating base or adapter cube using the same canonical owned-leaf context.
    /// Original values/IDs/dtypes/flags and parameter mappers remain intact; only the selected interval is retained.
    pub fn parameter(&mut self,parameter:Param<Tensor<B,3>>,ownership:&ExpertOwnership,rank:usize) -> Param<Tensor<B,3>> {
        assert_eq!(parameter.val().dims()[0],ownership.experts(),"original floating expert/adapter cube differs from declared ownership");
        local_cube(parameter,ownership.range(rank),&mut self.shared)
    }
}
fn local_cube<B:Backend>(parameter:Param<Tensor<B,3>>,range:Range<usize>,shared:&mut BTreeMap<(ParamId,bool),ExpertEntry<B>>) -> Param<Tensor<B,3>> {
    let source=parameter.val();let [_,rows,columns]=source.dims();let count=range.len();
    let key=(parameter.id,source.is_require_grad());
    if let Some(entry)=shared.get(&key) {let value=entry.local.val();assert_eq!(entry.source_shape,source.dims(),"tied expert source cube shapes differ");
        assert_eq!(entry.range,range,"one tied expert source ID cannot have different rank-owned intervals");
        assert_eq!(value.dtype(),source.dtype(),"tied expert source storage differs");assert_eq!(value.device(),source.device(),"tied expert source devices differ");
        assert_eq!(value.is_require_grad(),source.is_require_grad(),"tied expert source trainability differs");
        let (id,_,mapper)=parameter.consume();return Param::from_mapped_value(id,value,mapper);}
    let mut local=Tensor::<B,3>::empty([count,rows,columns],(&source.device(),source.dtype()));
    if count!=0 {local=local.slice_assign([0..count,0..rows,0..columns],source.clone().slice([range.clone(),0..rows,0..columns]));}
    // Preserve source identity/flags while making only the owned GPU copy a new optimizer leaf.
    let local=parameter.map(|_|local.detach().set_require_grad(source.is_require_grad()));
    shared.insert(key,ExpertEntry {source_shape:source.dims(),range,local:local.clone()});local
}
impl<B:Backend> ExpertParallelSwiGluExperts<B> {
    /// Connect caller-loaded actual local values, retaining all original source IDs and flags.
    pub fn from_parameters(gate:Param<Tensor<B,3>>,up:Param<Tensor<B,3>>,down:Param<Tensor<B,3>>,ownership:ExpertOwnership,rank:usize) -> Self {
        let experts=Self {gate,up,down,ownership,rank};experts.validate();experts
    }
    /// Copy only this rank's actual original expert rows on the source backend, then release full source graphs.
    /// No model values are downloaded or quantized, and no rank ownership is guessed.
    pub fn from_full(experts:NativeSwiGluExperts<B>,ownership:ExpertOwnership,rank:usize) -> Self {
        ExpertPartitionContext::new().experts(experts,ownership,rank)
    }
    /// Actual original `[owned_experts,hidden,intermediate]` local geometry.
    pub fn dimensions(&self) -> [usize;3] {let [experts,inner,hidden]=self.gate.val().dims();[experts,hidden,inner]}
    /// Validate actual local ownership/shape/storage/device metadata; zero-expert owners are retained.
    pub fn validate(&self) {
        let gate=self.gate.val();let [experts,inner,hidden]=gate.dims();
        assert_eq!(experts,self.ownership.range(self.rank).len(),"local expert count differs from actual owned global interval");
        assert!(inner>0 && hidden>0,"native expert hidden/intermediate widths must be positive");
        assert_eq!(self.up.val().dims(),gate.dims(),"local gate/up geometry differs");assert_eq!(self.down.val().dims(),[experts,hidden,inner],"local down geometry differs");
        assert!(matches!(gate.dtype(),DType::F16|DType::BF16|DType::F32),"unsupported original local expert storage");
        for value in [self.up.val(),self.down.val()] {assert_eq!(value.dtype(),gate.dtype(),"local expert storage differs");assert_eq!(value.device(),gate.device(),"local expert devices differ");}
    }
}
impl<B:MoeReceivedOps> ExpertParallelSwiGluExperts<B> {
    /// Compute only this rank's original experts on actual received assignments. Router weights are not applied here.
    pub fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,options:MoeOptions) -> Result<Tensor<B,2>,B::MoeError> {
        self.validate();received_moe_experts(input,ids,self.gate.val(),self.up.val(),self.down.val(),MoeReceivedOptions {
            expert_start:self.ownership.range(self.rank).start,forward:options.forward,backward:options.backward})
    }
}
impl<B:Backend> ExpertParallelGeometry<B> for ExpertParallelSwiGluExperts<B> {
    fn dimensions(&self) -> [usize;3] {self.dimensions()}
    fn ownership(&self) -> &ExpertOwnership {&self.ownership}
    fn rank(&self) -> usize {self.rank}
    fn device(&self) -> B::Device {self.gate.val().device()}
    fn validate(&self) {self.validate();}
    fn parameter_ids(&self) -> Vec<ParamId> {let mut ids=Vec::new();for id in [self.gate.id,self.up.id,self.down.id] {if !ids.contains(&id) {ids.push(id);}}ids}
}
impl<B:MoeReceivedOps> ExpertParallelReceived<B> for ExpertParallelSwiGluExperts<B> {
    fn forward_received(&self,input:Tensor<B,2>,ids:Tensor<B,1,Int>,options:MoeOptions) -> Result<Tensor<B,2>,B::MoeError> {
        self.forward_received(input,ids,options)
    }
}
/// Actual router projection plus only rank-owned original floating expert cubes.
#[derive(Module,Debug)]
pub struct ExpertParallelMoeLayer<B:Backend,P:Module<B>,E:Module<B> =ExpertParallelSwiGluExperts<B>> {
    /// Original actual full-logit router projection; its replication/sharding is caller-owned.
    pub router:P,
    /// Actual original local expert cubes and their explicit expert-world ownership.
    pub experts:E,
    /// Original optional FP32 selection-only correction bias.
    pub correction_bias:Option<Param<Tensor<B,1>>>,
    /// Original explicit native routing/GEMM/VJP options.
    #[module(skip)]
    pub options:MoeOptions,
    /// Original explicit router-input work dtype, independent of expert storage.
    #[module(skip)]
    pub router_input_dtype:Option<FloatDType>,
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>> ExpertParallelMoeLayer<B,P,E> {
    /// Connect actual loaded components without replicating expert weights or selecting a gradient-reduction group.
    pub fn from_expert_parts(router:P,experts:E,correction_bias:Option<Param<Tensor<B,1>>>,options:MoeOptions,router_input_dtype:Option<FloatDType>) -> Self {
        let layer=Self {router,experts,correction_bias,options,router_input_dtype};layer.validate();layer
    }
    /// Actual original residual/input width.
    pub fn width(&self) -> usize {self.experts.dimensions()[1]}
    /// Validate the actual original global router and local expert geometry.
    pub fn validate(&self) {
        self.experts.validate();assert_eq!(self.router.dimensions(),[self.width(),self.experts.ownership().experts()],"actual router/global expert geometry differs");
        if let Some(bias)=&self.correction_bias {let bias=bias.val();assert_eq!(bias.dims(),[self.experts.ownership().experts()],"global correction bias shape differs");
            assert_eq!(bias.dtype(),DType::F32,"selection-only correction bias must retain FP32");assert_eq!(bias.device(),self.experts.device(),"correction bias/expert device differs");}
        if let Some(dtype)=self.router_input_dtype {assert!(matches!(DType::from(dtype),DType::F16|DType::BF16|DType::F32),"unsupported original router work dtype");}
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>> ExpertParallelMoeLayer<B,P> {
    /// Preserve original cube-only constructor inference and loaded-source behavior.
    pub fn from_parts(router:P,experts:ExpertParallelSwiGluExperts<B>,correction_bias:Option<Param<Tensor<B,1>>>,options:MoeOptions,router_input_dtype:Option<FloatDType>) -> Self {
        Self::from_expert_parts(router,experts,correction_bias,options,router_input_dtype)
    }
}
/// Original projection, actual expert-world transport or actual native row/expert execution failure.
#[derive(Debug)]
pub enum ExpertParallelMoeError<C:fmt::Debug,P:fmt::Debug,M:fmt::Debug> {
    /// Actual original expert-world transport failure.
    Collective(C),
    /// Original actual router projection failure.
    Router(P),
    /// Original actual native dispatch/expert/combine failure.
    Native(M),
    /// Actual mismatched native transport row-count or expert ownership metadata.
    Protocol(&'static str),
}
impl<C:fmt::Debug,P:fmt::Debug,M:fmt::Debug> fmt::Display for ExpertParallelMoeError<C,P,M> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {
        Self::Collective(error)=>write!(f,"expert transport: {error:?}"),Self::Router(error)=>write!(f,"expert router: {error:?}"),
        Self::Native(error)=>write!(f,"native owned experts: {error:?}"),Self::Protocol(error)=>f.write_str(error)}}
}
impl<C:fmt::Debug,P:fmt::Debug,M:fmt::Debug> core::error::Error for ExpertParallelMoeError<C,P,M> {}
/// Actual output and the same original tracked router logits/selections, not a second router pass.
#[derive(Debug)]
pub struct ExpertParallelMoeOutput<B:Backend,const D:usize> {
    /// Actual original local source token axes and routed output.
    pub output:Tensor<B,D>,
    /// Original tracked logits used for this exact routing decision.
    pub router_logits:Tensor<B,2>,
    /// Original native U32 selected expert IDs.
    pub selected_experts:Tensor<B,2,Int>,
    /// Actual assignment row counts sent to each expert owner.
    pub sent_rows:Vec<usize>,
    /// Actual assignment row counts received from each source rank.
    pub received_rows:Vec<usize>,
}
fn native_exchange<B:Backend,C:VariableTensorCollective<B>,const D:usize>(input:Tensor<B,D>,communicator:C,counts:&[usize])
    -> Result<VariableTensorExchange<Tensor<B,D>>,ScopedCollectiveError<C::Error>> {input.all_to_all_v(communicator,counts).map_err(ScopedCollectiveError::Collective)}
fn exchange_error<C:fmt::Debug,P:fmt::Debug,M:fmt::Debug>(error:ScopedCollectiveError<C>) -> ExpertParallelMoeError<C,P,M> {
    match error {ScopedCollectiveError::Collective(error)=>ExpertParallelMoeError::Collective(error),ScopedCollectiveError::Protocol(message)=>ExpertParallelMoeError::Protocol(message)}
}
macro_rules! execute_expert_parallel {
    ($backend:ty,[$($generics:tt)*],$exchange:path,$forward:ident,$detailed:ident) => {
        impl<$($generics)*,P:TransformerProjection<$backend>,E:ExpertParallelReceived<$backend>> ExpertParallelMoeLayer<$backend,P,E> {
            /// Complete native source dispatch -> actual variable exchanges -> owned experts -> original ordered combine.
            /// Every expert-world rank enters even when it owns no experts or has no source token rows.
            pub fn $forward<C:VariableTensorCollective<B>,const D:usize>(&self,input:Tensor<$backend,D>,communicator:C)
                -> Result<Tensor<$backend,D>,ExpertParallelMoeError<C::Error,P::Error,<$backend as ruda_model::tensor::MoeOps>::MoeError>> {
                self.$detailed(input,communicator).map(|output|output.output)
            }
            /// Retain original router objectives and actual assignment metadata without re-running dropout-bearing projections.
            pub fn $detailed<C:VariableTensorCollective<B>,const D:usize>(&self,input:Tensor<$backend,D>,communicator:C)
                -> Result<ExpertParallelMoeOutput<$backend,D>,ExpertParallelMoeError<C::Error,P::Error,<$backend as ruda_model::tensor::MoeOps>::MoeError>> {
                self.validate();assert!(D>0,"expert-parallel input requires an actual feature axis");
                if communicator.rank() as usize!=self.experts.rank() || communicator.world_size() as usize!=self.experts.ownership().world_size() {
                    return Err(ExpertParallelMoeError::Protocol("actual expert transport differs from the declared original ownership"));}
                let shape=input.dims();assert_eq!(shape[D-1],self.width(),"actual source token feature width differs");
                let rows=shape[..D-1].iter().try_fold(1usize,|count,&axis|count.checked_mul(axis)).expect("actual expert source token count overflows");
                let input=input.reshape([rows,self.width()]);let router_input=if let Some(dtype)=self.router_input_dtype {input.clone().cast(dtype)} else {input.clone()};
                let logits=self.router.forward(router_input).map_err(ExpertParallelMoeError::Router)?;
                let dispatch=dispatch_moe(input,logits.clone(),self.correction_bias.as_ref().map(Param::val),self.options).map_err(ExpertParallelMoeError::Native)?;
                let sent_rows=<$backend as MoeDispatchOps>::moe_dispatch_counts(&dispatch.state,self.experts.ownership().prefix()).map_err(ExpertParallelMoeError::Native)?;
                let incoming=$exchange(dispatch.values,communicator.clone(),&sent_rows).map_err(exchange_error)?;
                let original_ids=Tensor::<B,1,Int>::from_primitive(dispatch.row_experts.into_primitive());
                let ids=original_ids.all_to_all_v_int(communicator.clone(),&sent_rows).map_err(ExpertParallelMoeError::Collective)?;
                if incoming.receive_counts!=ids.receive_counts {return Err(ExpertParallelMoeError::Protocol("transported assignment rows and original U32 IDs differ"));}
                let received_rows=incoming.receive_counts;let original_ids=Tensor::<$backend,1,Int>::from_primitive(ids.value.into_primitive());
                let output=self.experts.forward_received(incoming.value,original_ids,self.options).map_err(ExpertParallelMoeError::Native)?;
                let returned=$exchange(output,communicator,&received_rows).map_err(exchange_error)?;
                if returned.receive_counts!=sent_rows {return Err(ExpertParallelMoeError::Protocol("returned expert rows differ from the original source assignments"));}
                let output=combine_moe(dispatch.state,returned.value,dispatch.weights,self.options.combine_backward).map_err(ExpertParallelMoeError::Native)?.reshape(shape);
                Ok(ExpertParallelMoeOutput {output,router_logits:logits,selected_experts:dispatch.selected_experts,sent_rows,received_rows})
            }
        }
    };
}
execute_expert_parallel!(B,[B:MoeDispatchOps+MoeReceivedOps],native_exchange,forward_inference,forward_detailed_inference);
execute_expert_parallel!(Autodiff<B,S>,[B:MoeDispatchOps+MoeReceivedOps,S:CheckpointStrategy],all_to_all_v_coordinated,forward,forward_detailed);
