use super::*;
use crate::LearningRate;
use ruda_model::{record::{Record,PrecisionSettings},tensor::{Tensor,DType,BroadcastTensorCollective,backend::Backend}};
use serde::{Serialize,Deserialize};

macro_rules! coordinate_optimizer_type {
    (Sgd,$b:ident,native)=>{Sgd<$b>};
    ($optimizer:ident,$b:ident,native)=>{$optimizer};
    (Sgd,$b:ident,master)=>{Fp32MasterOptimizer<Sgd<$b>>};
    ($optimizer:ident,$b:ident,master)=>{Fp32MasterOptimizer<$optimizer>};
}
macro_rules! native_coordinate {
    ($(($variant:ident,$optimizer:ident,$kind:ident,$scalar:ty,$wrap:ident)),+ $(,)?) => {
        /// Explicit original native coordinate algorithm choice for heterogeneous parameter groups.
        /// Matrix optimizers such as Muon are not flattened into independent fragments by this type.
        #[derive(Clone)]
        pub enum NativeCoordinateOptimizer<B:Backend> {$(
            #[doc=concat!("Original configured ",stringify!($variant)," numerical implementation.")]
            $variant(coordinate_optimizer_type!($optimizer,B,$kind)),
        )+}
        /// Actual corresponding original history, retaining its algorithm identity and native scalar clocks/options.
        #[derive(Clone)]
        pub enum NativeCoordinateState<B:Backend,const D:usize> {$(
            #[doc=concat!("Actual original ",stringify!($variant)," state, not reinitialized or reinterpreted.")]
            $variant(<coordinate_optimizer_type!($optimizer,B,$kind) as SimpleOptimizer<B>>::State<D>),
        )+}
        /// Typed native history serialization preserving original algorithm choice and concrete inner record format.
        #[derive(Serialize,Deserialize,Clone)]
        #[serde(bound="")]
        pub enum NativeCoordinateStateItem<B:Backend,P:PrecisionSettings,const D:usize> {$(
            #[doc=concat!("Serialized original ",stringify!($variant)," history.")]
            $variant(<<coordinate_optimizer_type!($optimizer,B,$kind) as SimpleOptimizer<B>>::State<D> as Record<B>>::Item<P>),
        )+}
        /// Actual original native scalar counters/optional branches tagged with the original algorithm identity.
        #[derive(Clone,Debug,PartialEq,Eq)]
        pub enum NativeCoordinateScalars {$(
            #[doc=concat!("Original ",stringify!($variant)," clocks/optional-state structure.")]
            $variant($scalar),
        )+}
        impl<B:Backend> NativeCoordinateOptimizer<B> {$(
            #[doc=concat!("Tag an actual existing ",stringify!($variant)," full native history for heterogeneous group import; all original ranks/clocks/buffers remain unchanged.")]
            pub fn $wrap<A:ruda_model::tensor::backend::AutodiffBackend<InnerBackend=B>>(record:crate::record::AdaptorRecord<coordinate_optimizer_type!($optimizer,B,$kind),A>)
                -> crate::record::AdaptorRecord<Self,A> {
                use crate::record::{AdaptorRecord,AdaptorRecordV1};
                let AdaptorRecord::V1(record)=record;
                AdaptorRecord::V1(match record {
                    AdaptorRecordV1::Rank0(state)=>AdaptorRecordV1::Rank0(NativeCoordinateState::<B,0>::$variant(state)),
                    AdaptorRecordV1::Rank1(state)=>AdaptorRecordV1::Rank1(NativeCoordinateState::<B,1>::$variant(state)),
                    AdaptorRecordV1::Rank2(state)=>AdaptorRecordV1::Rank2(NativeCoordinateState::<B,2>::$variant(state)),
                    AdaptorRecordV1::Rank3(state)=>AdaptorRecordV1::Rank3(NativeCoordinateState::<B,3>::$variant(state)),
                    AdaptorRecordV1::Rank4(state)=>AdaptorRecordV1::Rank4(NativeCoordinateState::<B,4>::$variant(state)),
                    AdaptorRecordV1::Rank5(state)=>AdaptorRecordV1::Rank5(NativeCoordinateState::<B,5>::$variant(state)),
                    AdaptorRecordV1::Rank6(state)=>AdaptorRecordV1::Rank6(NativeCoordinateState::<B,6>::$variant(state)),
                    AdaptorRecordV1::Rank7(state)=>AdaptorRecordV1::Rank7(NativeCoordinateState::<B,7>::$variant(state)),
                    AdaptorRecordV1::Rank8(state)=>AdaptorRecordV1::Rank8(NativeCoordinateState::<B,8>::$variant(state)),
                })
            }
        )+}
        impl<B:Backend,const D:usize> Record<B> for NativeCoordinateState<B,D> {
            type Item<P:PrecisionSettings>=NativeCoordinateStateItem<B,P,D>;
            fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {match self {$(Self::$variant(state)=>NativeCoordinateStateItem::$variant(state.into_item::<P>()),)+}}
            fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
                match item {$(NativeCoordinateStateItem::$variant(state)=>Self::$variant(Record::<B>::from_item::<P>(state,device)),)+}
            }
        }
        impl<B:Backend,const D:usize> OptimizerCheckpointBuffers<B,D> for NativeCoordinateState<B,D> {
            fn visit_checkpoint_buffers<F:FnMut(&Tensor<B,D>)>(&self,visit:&mut F) {match self {$(Self::$variant(state)=>state.visit_checkpoint_buffers(visit),)+}}
            fn map_checkpoint_buffers<F:FnMut(Tensor<B,D>)->Tensor<B,D>>(self,map:&mut F) -> Self {match self {$(Self::$variant(state)=>Self::$variant(state.map_checkpoint_buffers(map)),)+}}
        }
        impl<B:Backend,const D:usize> FlatOptimizerCheckpointState<B,D> for NativeCoordinateState<B,D> {
            type FlatState=NativeCoordinateState<B,1>;
            fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
                match self {$(Self::$variant(state)=>Ok(NativeCoordinateState::$variant(state.into_flat_shard(shard)?)),)+}
            }
        }
        impl<B:Backend,const D:usize> OptimizerCheckpointScalars for NativeCoordinateState<B,D> {
            type Scalars=NativeCoordinateScalars;
            fn checkpoint_scalars(&self) -> Self::Scalars {match self {$(Self::$variant(state)=>NativeCoordinateScalars::$variant(state.checkpoint_scalars()),)+}}
        }
        impl<B:Backend> SimpleOptimizer<B> for NativeCoordinateOptimizer<B> {
            type State<const D:usize>=NativeCoordinateState<B,D>;
            fn step<const D:usize>(&self,lr:LearningRate,tensor:Tensor<B,D>,gradient:Tensor<B,D>,state:Option<Self::State<D>>)
                -> (Tensor<B,D>,Option<Self::State<D>>) {
                match self {$(Self::$variant(optimizer)=>{
                    let state=match state {None=>None,Some(NativeCoordinateState::$variant(state))=>Some(state),Some(_)=>panic!("original native coordinate optimizer/history identity differs")};
                    let (tensor,state)=optimizer.step(lr,tensor,gradient,state);(tensor,state.map(NativeCoordinateState::$variant))
                },)+}
            }
            fn to_device<const D:usize>(state:Self::State<D>,device:&B::Device) -> Self::State<D> {
                match state {$(NativeCoordinateState::$variant(state)=>NativeCoordinateState::$variant(
                    <coordinate_optimizer_type!($optimizer,B,$kind) as SimpleOptimizer<B>>::to_device(state,device)),)+}
            }
        }
        impl<B:Backend> ElementwiseShardOptimizer<B> for NativeCoordinateOptimizer<B> {
            fn validate_element_sharding(&self) -> Result<(),&'static str> {match self {$(Self::$variant(optimizer)=>ElementwiseShardOptimizer::<B>::validate_element_sharding(optimizer),)+}}
            fn shard_gradient_dtype(&self,storage:DType) -> DType {match self {$(Self::$variant(optimizer)=>ElementwiseShardOptimizer::<B>::shard_gradient_dtype(optimizer,storage),)+}}
            fn validate_fully_sharded_execution(&self) -> Result<(),&'static str> {match self {$(Self::$variant(optimizer)=>ElementwiseShardOptimizer::<B>::validate_fully_sharded_execution(optimizer),)+}}
            fn validate_fully_sharded_history(&self,state:&Self::State<1>) -> Result<(),&'static str> {
                match (self,state) {$((Self::$variant(optimizer),NativeCoordinateState::$variant(state))=>ElementwiseShardOptimizer::<B>::validate_fully_sharded_history(optimizer,state),)+
                    _=>Err("original native coordinate optimizer/history identity differs")}
            }
            fn step_fully_sharded<C:BroadcastTensorCollective<B>>(&self,lr:LearningRate,tensor:Tensor<B,1>,gradient:Tensor<B,1>,
                state:Option<Self::State<1>>,binding:&FullyShardedOptimizerParameter<C>)
                -> Result<(Tensor<B,1>,Option<Self::State<1>>),FullyShardedElementwiseError<C::Error>> {
                match self {$(Self::$variant(optimizer)=>{
                    let state=match state {None=>None,Some(NativeCoordinateState::$variant(state))=>Some(state),Some(_)=>return Err(FullyShardedElementwiseError::State("original native coordinate optimizer/history identity differs"))};
                    let (tensor,state)=optimizer.step_fully_sharded(lr,tensor,gradient,state,binding)?;Ok((tensor,state.map(NativeCoordinateState::$variant)))
                },)+}
            }
        }
    };
}
native_coordinate!(
    (Sgd,Sgd,native,bool,wrap_sgd_record),
    (Adam,Adam,native,(usize,bool),wrap_adam_record),
    (AdamW,AdamW,native,(usize,bool),wrap_adamw_record),
    (AdaGrad,AdaGrad,native,usize,wrap_adagrad_record),
    (RmsProp,RmsProp,native,(bool,bool),wrap_rmsprop_record),
    (Adan,Adan,native,usize,wrap_adan_record),
    (MasterSgd,Sgd,master,Option<bool>,wrap_master_sgd_record),
    (MasterAdam,Adam,master,Option<(usize,bool)>,wrap_master_adam_record),
    (MasterAdamW,AdamW,master,Option<(usize,bool)>,wrap_master_adamw_record),
    (MasterAdaGrad,AdaGrad,master,Option<usize>,wrap_master_adagrad_record),
    (MasterRmsProp,RmsProp,master,Option<(bool,bool)>,wrap_master_rmsprop_record),
    (MasterAdan,Adan,master,Option<usize>,wrap_master_adan_record),
);
