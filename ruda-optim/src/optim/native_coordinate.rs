use super::*;
use crate::LearningRate;
use ruda_model::{record::{Record,PrecisionSettings},tensor::{Tensor,DType,BroadcastTensorCollective,backend::Backend}};
use serde::{Serialize,Deserialize};

macro_rules! native_coordinate {
    ($b:ident,$d:ident,$p:ident; $(($variant:ident,$optimizer:ty,$scalar:ty,$wrap:ident)),+ $(,)?) => {
        /// Explicit original native coordinate algorithm choice for heterogeneous parameter groups.
        /// Matrix optimizers such as Muon are not flattened into independent fragments by this type.
        #[derive(Clone)]
        pub enum NativeCoordinateOptimizer<$b:Backend> {$(
            #[doc=concat!("Original configured ",stringify!($variant)," numerical implementation.")]
            $variant($optimizer),
        )+}
        /// Actual corresponding original history, retaining its algorithm identity and native scalar clocks/options.
        #[derive(Clone)]
        pub enum NativeCoordinateState<$b:Backend,const $d:usize> {$(
            #[doc=concat!("Actual original ",stringify!($variant)," state, not reinitialized or reinterpreted.")]
            $variant(<$optimizer as SimpleOptimizer<$b>>::State<$d>),
        )+}
        /// Typed native history serialization preserving original algorithm choice and concrete inner record format.
        #[derive(Serialize,Deserialize,Clone)]
        #[serde(bound="")]
        pub enum NativeCoordinateStateItem<$b:Backend,$p:PrecisionSettings,const $d:usize> {$(
            #[doc=concat!("Serialized original ",stringify!($variant)," history.")]
            $variant(<<$optimizer as SimpleOptimizer<$b>>::State<$d> as Record<$b>>::Item<$p>),
        )+}
        /// Actual original native scalar counters/optional branches tagged with the original algorithm identity.
        #[derive(Clone,Debug,PartialEq,Eq)]
        pub enum NativeCoordinateScalars {$(
            #[doc=concat!("Original ",stringify!($variant)," clocks/optional-state structure.")]
            $variant($scalar),
        )+}
        impl<$b:Backend> NativeCoordinateOptimizer<$b> {$(
            #[doc=concat!("Tag an actual existing ",stringify!($variant)," full native history for heterogeneous group import; all original ranks/clocks/buffers remain unchanged.")]
            pub fn $wrap<A:ruda_model::tensor::backend::AutodiffBackend<InnerBackend=$b>>(record:crate::record::AdaptorRecord<$optimizer,A>)
                -> crate::record::AdaptorRecord<Self,A> {
                use crate::record::{AdaptorRecord,AdaptorRecordV1};
                let AdaptorRecord::V1(record)=record;
                AdaptorRecord::V1(match record {
                    AdaptorRecordV1::Rank0(state)=>AdaptorRecordV1::Rank0(NativeCoordinateState::<$b,0>::$variant(state)),
                    AdaptorRecordV1::Rank1(state)=>AdaptorRecordV1::Rank1(NativeCoordinateState::<$b,1>::$variant(state)),
                    AdaptorRecordV1::Rank2(state)=>AdaptorRecordV1::Rank2(NativeCoordinateState::<$b,2>::$variant(state)),
                    AdaptorRecordV1::Rank3(state)=>AdaptorRecordV1::Rank3(NativeCoordinateState::<$b,3>::$variant(state)),
                    AdaptorRecordV1::Rank4(state)=>AdaptorRecordV1::Rank4(NativeCoordinateState::<$b,4>::$variant(state)),
                    AdaptorRecordV1::Rank5(state)=>AdaptorRecordV1::Rank5(NativeCoordinateState::<$b,5>::$variant(state)),
                    AdaptorRecordV1::Rank6(state)=>AdaptorRecordV1::Rank6(NativeCoordinateState::<$b,6>::$variant(state)),
                    AdaptorRecordV1::Rank7(state)=>AdaptorRecordV1::Rank7(NativeCoordinateState::<$b,7>::$variant(state)),
                    AdaptorRecordV1::Rank8(state)=>AdaptorRecordV1::Rank8(NativeCoordinateState::<$b,8>::$variant(state)),
                })
            }
        )+}
        impl<$b:Backend,const $d:usize> Record<$b> for NativeCoordinateState<$b,$d> {
            type Item<$p:PrecisionSettings>=NativeCoordinateStateItem<$b,$p,$d>;
            fn into_item<$p:PrecisionSettings>(self) -> Self::Item<$p> {match self {$(Self::$variant(state)=>NativeCoordinateStateItem::$variant(state.into_item::<$p>()),)+}}
            fn from_item<$p:PrecisionSettings>(item:Self::Item<$p>,device:&$b::Device) -> Self {
                match item {$(NativeCoordinateStateItem::$variant(state)=>Self::$variant(Record::<$b>::from_item::<$p>(state,device)),)+}
            }
        }
        impl<$b:Backend,const $d:usize> OptimizerCheckpointBuffers<$b,$d> for NativeCoordinateState<$b,$d> {
            fn visit_checkpoint_buffers<F:FnMut(&Tensor<$b,$d>)>(&self,visit:&mut F) {match self {$(Self::$variant(state)=>state.visit_checkpoint_buffers(visit),)+}}
            fn map_checkpoint_buffers<F:FnMut(Tensor<$b,$d>)->Tensor<$b,$d>>(self,map:&mut F) -> Self {match self {$(Self::$variant(state)=>Self::$variant(state.map_checkpoint_buffers(map)),)+}}
        }
        impl<$b:Backend,const $d:usize> FlatOptimizerCheckpointState<$b,$d> for NativeCoordinateState<$b,$d> {
            type FlatState=NativeCoordinateState<$b,1>;
            fn into_flat_shard(self,shard:&FlatOptimizerTensorShard) -> Result<Self::FlatState,OptimizerShardError> {
                match self {$(Self::$variant(state)=>Ok(NativeCoordinateState::$variant(state.into_flat_shard(shard)?)),)+}
            }
        }
        impl<$b:Backend,const $d:usize> OptimizerCheckpointScalars for NativeCoordinateState<$b,$d> {
            type Scalars=NativeCoordinateScalars;
            fn checkpoint_scalars(&self) -> Self::Scalars {match self {$(Self::$variant(state)=>NativeCoordinateScalars::$variant(state.checkpoint_scalars()),)+}}
        }
        impl<$b:Backend> SimpleOptimizer<$b> for NativeCoordinateOptimizer<$b> {
            type State<const $d:usize>=NativeCoordinateState<$b,$d>;
            fn step<const $d:usize>(&self,lr:LearningRate,tensor:Tensor<$b,$d>,gradient:Tensor<$b,$d>,state:Option<Self::State<$d>>)
                -> (Tensor<$b,$d>,Option<Self::State<$d>>) {
                match self {$(Self::$variant(optimizer)=>{
                    let state=match state {None=>None,Some(NativeCoordinateState::$variant(state))=>Some(state),Some(_)=>panic!("original native coordinate optimizer/history identity differs")};
                    let (tensor,state)=optimizer.step(lr,tensor,gradient,state);(tensor,state.map(NativeCoordinateState::$variant))
                },)+}
            }
            fn to_device<const $d:usize>(state:Self::State<$d>,device:&$b::Device) -> Self::State<$d> {
                match state {$(NativeCoordinateState::$variant(state)=>NativeCoordinateState::$variant(
                    <$optimizer as SimpleOptimizer<$b>>::to_device(state,device)),)+}
            }
        }
        impl<$b:Backend> ElementwiseShardOptimizer<$b> for NativeCoordinateOptimizer<$b> {
            fn validate_element_sharding(&self) -> Result<(),&'static str> {match self {$(Self::$variant(optimizer)=>ElementwiseShardOptimizer::<$b>::validate_element_sharding(optimizer),)+}}
            fn shard_gradient_dtype(&self,storage:DType) -> DType {match self {$(Self::$variant(optimizer)=>ElementwiseShardOptimizer::<$b>::shard_gradient_dtype(optimizer,storage),)+}}
            fn validate_fully_sharded_execution(&self) -> Result<(),&'static str> {match self {$(Self::$variant(optimizer)=>ElementwiseShardOptimizer::<$b>::validate_fully_sharded_execution(optimizer),)+}}
            fn validate_fully_sharded_history(&self,state:&Self::State<1>) -> Result<(),&'static str> {
                match (self,state) {$((Self::$variant(optimizer),NativeCoordinateState::$variant(state))=>ElementwiseShardOptimizer::<$b>::validate_fully_sharded_history(optimizer,state),)+
                    _=>Err("original native coordinate optimizer/history identity differs")}
            }
            fn step_fully_sharded<C:BroadcastTensorCollective<$b>>(&self,lr:LearningRate,tensor:Tensor<$b,1>,gradient:Tensor<$b,1>,
                state:Option<Self::State<1>>,binding:&FullyShardedOptimizerParameter<C>)
                -> Result<(Tensor<$b,1>,Option<Self::State<1>>),FullyShardedElementwiseError<C::Error>> {
                match self {$(Self::$variant(optimizer)=>{
                    let state=match state {None=>None,Some(NativeCoordinateState::$variant(state))=>Some(state),Some(_)=>return Err(FullyShardedElementwiseError::State("original native coordinate optimizer/history identity differs"))};
                    let (tensor,state)=optimizer.step_fully_sharded(lr,tensor,gradient,state,binding)?;Ok((tensor,state.map(NativeCoordinateState::$variant)))
                },)+}
            }
        }
    };
}
native_coordinate!(B,D,P;
    (Sgd,Sgd<B>,bool,wrap_sgd_record),
    (Adam,Adam,(usize,bool),wrap_adam_record),
    (AdamW,AdamW,(usize,bool),wrap_adamw_record),
    (AdaGrad,AdaGrad,usize,wrap_adagrad_record),
    (RmsProp,RmsProp,(bool,bool),wrap_rmsprop_record),
    (Adan,Adan,usize,wrap_adan_record),
    (MasterSgd,Fp32MasterOptimizer<Sgd<B>>,Option<bool>,wrap_master_sgd_record),
    (MasterAdam,Fp32MasterOptimizer<Adam>,Option<(usize,bool)>,wrap_master_adam_record),
    (MasterAdamW,Fp32MasterOptimizer<AdamW>,Option<(usize,bool)>,wrap_master_adamw_record),
    (MasterAdaGrad,Fp32MasterOptimizer<AdaGrad>,Option<usize>,wrap_master_adagrad_record),
    (MasterRmsProp,Fp32MasterOptimizer<RmsProp>,Option<(bool,bool)>,wrap_master_rmsprop_record),
    (MasterAdan,Fp32MasterOptimizer<Adan>,Option<usize>,wrap_master_adan_record),
);
