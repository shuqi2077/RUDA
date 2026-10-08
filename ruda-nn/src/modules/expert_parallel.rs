//! Actual expert-owned native cubes and differentiable assignment exchanges; transport is explicit.
use alloc::vec::Vec;
use core::{fmt,ops::Range};
use ruda_model::{module::{Module,Param},tensor::{Tensor,Int,DType,FloatDType,MoeOptions,MoeDispatchOps,MoeReceivedOps,
    MoeReceivedOptions,dispatch_moe,combine_moe,received_moe_experts,VariableTensorCollective,VariableTensorExchange,backend::Backend}};
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,collective::all_to_all_v_ordered};
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
fn local_cube<B:Backend>(parameter:Param<Tensor<B,3>>,range:Range<usize>) -> Param<Tensor<B,3>> {
    let source=parameter.val();let [_,rows,columns]=source.dims();let count=range.len();
    let mut local=Tensor::<B,3>::empty([count,rows,columns],(&source.device(),source.dtype()));
    if count!=0 {local=local.slice_assign([0..count,0..rows,0..columns],source.clone().slice([range,0..rows,0..columns]));}
    // Preserve source identity/flags while making only the owned GPU copy a new optimizer leaf.
    Param::initialized(parameter.id,local.detach().set_require_grad(source.is_require_grad()))
}
impl<B:Backend> ExpertParallelSwiGluExperts<B> {
    /// Connect caller-loaded actual local values, retaining all original source IDs and flags.
    pub fn from_parameters(gate:Param<Tensor<B,3>>,up:Param<Tensor<B,3>>,down:Param<Tensor<B,3>>,ownership:ExpertOwnership,rank:usize) -> Self {
        let experts=Self {gate,up,down,ownership,rank};experts.validate();experts
    }
    /// Copy only this rank's actual original expert rows on the source backend, then release full source graphs.
    /// No model values are downloaded or quantized, and no rank ownership is guessed.
    pub fn from_full(experts:NativeSwiGluExperts<B>,ownership:ExpertOwnership,rank:usize) -> Self {
        experts.validate();assert_eq!(experts.dimensions()[0],ownership.experts(),"global source cube count differs from actual declared ownership");
        let range=ownership.range(rank);Self::from_parameters(local_cube(experts.gate,range.clone()),local_cube(experts.up,range.clone()),local_cube(experts.down,range),ownership,rank)
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
/// Actual router projection plus only rank-owned original floating expert cubes.
#[derive(Module,Debug)]
pub struct ExpertParallelMoeLayer<B:Backend,P:Module<B>> {
    /// Original actual full-logit router projection; its replication/sharding is caller-owned.
    pub router:P,
    /// Actual original local expert cubes and their explicit expert-world ownership.
    pub experts:ExpertParallelSwiGluExperts<B>,
    /// Original optional FP32 selection-only correction bias.
    pub correction_bias:Option<Param<Tensor<B,1>>>,
    /// Original explicit native routing/GEMM/VJP options.
    #[module(skip)]
    pub options:MoeOptions,
    /// Original explicit router-input work dtype, independent of expert storage.
    #[module(skip)]
    pub router_input_dtype:Option<FloatDType>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> ExpertParallelMoeLayer<B,P> {
    /// Connect actual loaded components without replicating expert weights or selecting a gradient-reduction group.
    pub fn from_parts(router:P,experts:ExpertParallelSwiGluExperts<B>,correction_bias:Option<Param<Tensor<B,1>>>,options:MoeOptions,router_input_dtype:Option<FloatDType>) -> Self {
        let layer=Self {router,experts,correction_bias,options,router_input_dtype};layer.validate();layer
    }
    /// Actual original residual/input width.
    pub fn width(&self) -> usize {self.experts.dimensions()[1]}
    /// Validate the actual original global router and local expert geometry.
    pub fn validate(&self) {
        self.experts.validate();assert_eq!(self.router.dimensions(),[self.width(),self.experts.ownership.experts()],"actual router/global expert geometry differs");
        if let Some(bias)=&self.correction_bias {let bias=bias.val();assert_eq!(bias.dims(),[self.experts.ownership.experts()],"global correction bias shape differs");
            assert_eq!(bias.dtype(),DType::F32,"selection-only correction bias must retain FP32");assert_eq!(bias.device(),self.experts.gate.val().device(),"correction bias/expert device differs");}
        if let Some(dtype)=self.router_input_dtype {assert!(matches!(DType::from(dtype),DType::F16|DType::BF16|DType::F32),"unsupported original router work dtype");}
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
    -> Result<VariableTensorExchange<Tensor<B,D>>,C::Error> {input.all_to_all_v(communicator,counts)}
macro_rules! execute_expert_parallel {
    ($backend:ty,[$($generics:tt)*],$exchange:path,$forward:ident,$detailed:ident) => {
        impl<$($generics)*,P:TransformerProjection<$backend>> ExpertParallelMoeLayer<$backend,P> {
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
                if communicator.rank() as usize!=self.experts.rank || communicator.world_size() as usize!=self.experts.ownership.world_size() {
                    return Err(ExpertParallelMoeError::Protocol("actual expert transport differs from the declared original ownership"));}
                let shape=input.dims();assert_eq!(shape[D-1],self.width(),"actual source token feature width differs");
                let rows=shape[..D-1].iter().try_fold(1usize,|count,&axis|count.checked_mul(axis)).expect("actual expert source token count overflows");
                let input=input.reshape([rows,self.width()]);let router_input=if let Some(dtype)=self.router_input_dtype {input.clone().cast(dtype)} else {input.clone()};
                let logits=self.router.forward(router_input).map_err(ExpertParallelMoeError::Router)?;
                let dispatch=dispatch_moe(input,logits.clone(),self.correction_bias.as_ref().map(Param::val),self.options).map_err(ExpertParallelMoeError::Native)?;
                let sent_rows=<$backend as MoeDispatchOps>::moe_dispatch_counts(&dispatch.state,self.experts.ownership.prefix()).map_err(ExpertParallelMoeError::Native)?;
                let incoming=$exchange(dispatch.values,communicator.clone(),&sent_rows).map_err(ExpertParallelMoeError::Collective)?;
                let original_ids=Tensor::<B,1,Int>::from_primitive(dispatch.row_experts.into_primitive());
                let ids=original_ids.all_to_all_v_int(communicator.clone(),&sent_rows).map_err(ExpertParallelMoeError::Collective)?;
                if incoming.receive_counts!=ids.receive_counts {return Err(ExpertParallelMoeError::Protocol("transported assignment rows and original U32 IDs differ"));}
                let received_rows=incoming.receive_counts;let original_ids=Tensor::<$backend,1,Int>::from_primitive(ids.value.into_primitive());
                let output=self.experts.forward_received(incoming.value,original_ids,self.options).map_err(ExpertParallelMoeError::Native)?;
                let returned=$exchange(output,communicator,&received_rows).map_err(ExpertParallelMoeError::Collective)?;
                if returned.receive_counts!=sent_rows {return Err(ExpertParallelMoeError::Protocol("returned expert rows differ from the original source assignments"));}
                let output=combine_moe(dispatch.state,returned.value,dispatch.weights,self.options.combine_backward).map_err(ExpertParallelMoeError::Native)?.reshape(shape);
                Ok(ExpertParallelMoeOutput {output,router_logits:logits,selected_experts:dispatch.selected_experts,sent_rows,received_rows})
            }
        }
    };
}
execute_expert_parallel!(B,[B:MoeDispatchOps+MoeReceivedOps],native_exchange,forward_inference,forward_detailed_inference);
execute_expert_parallel!(Autodiff<B,S>,[B:MoeDispatchOps+MoeReceivedOps,S:CheckpointStrategy],all_to_all_v_ordered,forward,forward_detailed);
