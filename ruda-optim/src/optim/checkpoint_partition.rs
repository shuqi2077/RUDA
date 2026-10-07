use alloc::vec::Vec;
use core::{fmt,ops::Range};
use hashbrown::HashMap;
use ruda_model::{module::ParamId,tensor::{Tensor,DType,backend::{Backend,AutodiffBackend}}};
use super::{AdaptiveMomentumState,AdamState,AdamWState,Adam,AdamW,Fp32MasterOptimizer,Fp32MasterState,
    record::{AdaptorRecord,AdaptorRecordV1}};

/// Explicit original tensor geometry and its actual contiguous optimizer-state shard.
/// Use the same axis/interval as the corresponding loaded native model parameter.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct OptimizerTensorShard {
    /// Exact original complete parameter shape, not an inferred head/embedding layout.
    pub global_shape:Vec<usize>,
    /// Actual stored parameter axis to partition.
    pub axis:usize,
    /// Actual nonempty global interval, excluding transport padding.
    pub interval:Range<usize>,
}

/// Explicit checkpoint geometry/storage inconsistency, before optimizer updates occur.
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum OptimizerShardError {
    /// Axis/interval cannot represent the declared actual parameter geometry.
    Placement(&'static str),
    /// Loaded native state has another original geometry.
    Shape(&'static str),
    /// Native buffers or explicit FP32 master precision differ.
    DType(&'static str),
    /// Native buffers belonging to one parameter are on different devices.
    Device(&'static str),
}
impl fmt::Display for OptimizerShardError {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Placement(value)=>write!(f,"invalid optimizer tensor shard: {value}"),Self::Shape(value)=>write!(f,"optimizer shard shape differs: {value}"),
            Self::DType(value)=>write!(f,"optimizer shard precision differs: {value}"),Self::Device(value)=>write!(f,"optimizer shard device differs: {value}")}
    }
}
impl core::error::Error for OptimizerShardError {}

impl OptimizerTensorShard {
    /// Declare exact source shape and slice, without guessing matrix roles or rank assignments.
    pub fn new(global_shape:Vec<usize>,axis:usize,interval:Range<usize>) -> Result<Self,OptimizerShardError> {
        let shard = Self {global_shape,axis,interval};shard.validate()?;Ok(shard)
    }
    /// Inspect placement metadata without launching any tensor operation.
    pub fn validate(&self) -> Result<(),OptimizerShardError> {
        if self.axis >= self.global_shape.len() || self.interval.start >= self.interval.end || self.interval.end > self.global_shape[self.axis] {
            return Err(OptimizerShardError::Placement("axis/interval must be a nonempty actual source slice"));
        }
        Ok(())
    }
    /// Actual resulting local parameter/state dimensions.
    pub fn local_shape(&self) -> Result<Vec<usize>,OptimizerShardError> {
        self.validate()?;let mut shape = self.global_shape.clone();shape[self.axis] = self.interval.len();Ok(shape)
    }
    fn check<B:Backend,const D:usize>(&self,value:&Tensor<B,D>,name:&'static str) -> Result<(),OptimizerShardError> {
        self.validate()?;
        if value.dims().as_slice() != self.global_shape.as_slice() {return Err(OptimizerShardError::Shape(name));}
        if !value.dtype().is_float() {return Err(OptimizerShardError::DType("native moment/master storage must remain floating"));}
        Ok(())
    }
    fn slice<B:Backend,const D:usize>(&self,value:Tensor<B,D>) -> Tensor<B,D> {
        if self.interval == (0..self.global_shape[self.axis]) {value} else {value.slice_dim(self.axis,self.interval.clone())}
    }
}

impl<B:Backend,const D:usize> AdaptiveMomentumState<B,D> {
    /// Slice original Adam/AdamW first/second moments and optional AMSGrad maximum together.
    /// Preserves time, actual precision/device and the original optional maximum; no state is reset.
    pub fn try_into_shard(self,shard:&OptimizerTensorShard) -> Result<Self,OptimizerShardError> {
        shard.check(&self.moment_1,"first moment")?;shard.check(&self.moment_2,"second moment")?;
        check_buffer(&self.moment_2,&self.moment_1,"second moment")?;
        if let Some(maximum) = &self.max_moment_2 {shard.check(maximum,"AMSGrad maximum")?;check_buffer(maximum,&self.moment_1,"AMSGrad maximum")?;}
        Ok(Self {time:self.time,moment_1:shard.slice(self.moment_1),moment_2:shard.slice(self.moment_2),max_moment_2:self.max_moment_2.map(|value|shard.slice(value))})
    }
}

fn check_buffer<B:Backend,const D:usize>(value:&Tensor<B,D>,reference:&Tensor<B,D>,name:&'static str) -> Result<(),OptimizerShardError> {
    if value.dtype() != reference.dtype() {return Err(OptimizerShardError::DType(name));}
    if value.device() != reference.device() {return Err(OptimizerShardError::Device(name));}
    Ok(())
}

macro_rules! adaptive_state {
    ($state:ident) => {
        impl<B:Backend,const D:usize> $state<B,D> {
            /// Explicitly partition original adaptive moments without changing algorithm, time or configuration.
            pub fn try_into_shard(self,shard:&OptimizerTensorShard) -> Result<Self,OptimizerShardError> {
                Ok(Self {momentum:self.momentum.try_into_shard(shard)?})
            }
        }
        impl<B:Backend,const D:usize> Fp32MasterState<B,D,$state<B,D>> {
            /// Slice original FP32 master and matching native optimizer buffers as one continuation state.
            /// A genuinely absent inner state stays absent; no zero moments or new step counter are invented.
            pub fn try_into_shard(self,shard:&OptimizerTensorShard) -> Result<Self,OptimizerShardError> {
                shard.check(&self.master,"FP32 master")?;
                if self.master.dtype() != DType::F32 {return Err(OptimizerShardError::DType("master must remain FP32"));}
                if let Some(inner) = &self.inner {
                    check_buffer(&inner.momentum.moment_1,&self.master,"first moment/master")?;
                    check_buffer(&inner.momentum.moment_2,&self.master,"second moment/master")?;
                    if let Some(maximum) = &inner.momentum.max_moment_2 {check_buffer(maximum,&self.master,"AMSGrad maximum/master")?;}
                }
                Ok(Self {master:shard.slice(self.master),inner:self.inner.map(|state|state.try_into_shard(shard)).transpose()?})
            }
        }
    };
}
adaptive_state!(AdamState);
adaptive_state!(AdamWState);

macro_rules! adaptor_partition {
    ($name:ident,$map:ident,$optimizer:ty) => {
        /// Partition an actual native adaptive optimizer record, retaining its original version/rank/time/precision.
        pub fn $name<B:AutodiffBackend>(record:AdaptorRecord<$optimizer,B>,shard:&OptimizerTensorShard)
            -> Result<AdaptorRecord<$optimizer,B>,OptimizerShardError> {
            Ok(AdaptorRecord::V1(match record {AdaptorRecord::V1(record)=>match record {
                AdaptorRecordV1::Rank0(value)=>AdaptorRecordV1::Rank0(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank1(value)=>AdaptorRecordV1::Rank1(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank2(value)=>AdaptorRecordV1::Rank2(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank3(value)=>AdaptorRecordV1::Rank3(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank4(value)=>AdaptorRecordV1::Rank4(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank5(value)=>AdaptorRecordV1::Rank5(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank6(value)=>AdaptorRecordV1::Rank6(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank7(value)=>AdaptorRecordV1::Rank7(value.try_into_shard(shard)?),
                AdaptorRecordV1::Rank8(value)=>AdaptorRecordV1::Rank8(value.try_into_shard(shard)?),
            }}))
        }
        /// Partition only declared existing native states; unchanged parameters keep their original records/IDs.
        /// Never-used parameters with no original state remain absent rather than receiving synthetic buffers.
        pub fn $map<B:AutodiffBackend>(records:HashMap<ParamId,AdaptorRecord<$optimizer,B>>,shards:&HashMap<ParamId,OptimizerTensorShard>)
            -> Result<HashMap<ParamId,AdaptorRecord<$optimizer,B>>,OptimizerShardError> {
            for shard in shards.values() {shard.validate()?;}
            records.into_iter().map(|(id,record)| {
                if let Some(shard) = shards.get(&id) {$name::<B>(record,shard).map(|record|(id,record))} else {Ok((id,record))}
            }).collect()
        }
    };
}
adaptor_partition!(partition_adam_record,partition_adam_records,Adam);
adaptor_partition!(partition_adamw_record,partition_adamw_records,AdamW);
adaptor_partition!(partition_adam_master_record,partition_adam_master_records,Fp32MasterOptimizer<Adam>);
adaptor_partition!(partition_adamw_master_record,partition_adamw_master_records,Fp32MasterOptimizer<AdamW>);
