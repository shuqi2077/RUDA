use super::*;
use core::fmt;
use alloc::vec::Vec;
use ruda_tensor::{DType,TensorData};

/// Native transport errors remain attached to the caller's actual group.
#[derive(Debug)]
pub enum GroupedCollectiveError<A:fmt::Debug,C:fmt::Debug,G:fmt::Debug> {
    First(ScopedCollectiveError<A>),
    Second(ScopedCollectiveError<C>),
    Indexed {index:usize,error:ScopedCollectiveError<A>},
    Coordinator(ScopedCollectiveError<G>),
}
impl<A:fmt::Debug,C:fmt::Debug,G:fmt::Debug> fmt::Display for GroupedCollectiveError<A,C,G> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::First(error)=>write!(f,"first loss group: {error}"),Self::Second(error)=>write!(f,"second loss group: {error}"),
            Self::Indexed {index,error}=>write!(f,"loss group slot {index}: {error}"),Self::Coordinator(error)=>write!(f,"loss graph coordinator: {error}")}
    }
}
impl<A:fmt::Debug,C:fmt::Debug,G:fmt::Debug> core::error::Error for GroupedCollectiveError<A,C,G> {}

/// Actual completed scalar and native graph-work counters. No gradient mean,
/// gradient all-reduce, optimizer decision or model policy is imposed.
pub struct GroupedCollectiveLoss<B:Backend,S:CheckpointStrategy> {
    pub loss:Tensor<Autodiff<B,S>,1>,
    pub rounds:usize,
    pub local_anchors_added:usize,
}

fn any_changed<B:Backend,C:BroadcastTensorCollective<B>>(changed:bool,device:&B::Device,coordinator:&C)
    -> Result<bool,ScopedCollectiveError<C::Error>> {
    let world=coordinator.world_size();
    if world==0 || coordinator.rank()>=world {return Err(ScopedCollectiveError::Protocol("invalid explicit completion coordinator"));}
    if world==1 {return Ok(changed);}
    let flag=Tensor::<B,1>::from_data(TensorData::new(alloc::vec![if changed {1.0f32} else {0.0}],[1]),(device,DType::F32));
    let flags=Tensor::<B,1>::from_primitive(TensorPrimitive::Float(coordinator.all_gather_float(flag.into_primitive().tensor())
        .map_err(ScopedCollectiveError::Collective)?));
    if flags.dims()!=[world as usize] || flags.dtype()!=DType::F32 || flags.device()!=*device {
        return Err(ScopedCollectiveError::Protocol("completion coordinator changed flag shape/storage/device"));
    }
    let flags=flags.into_data().to_vec::<f32>().map_err(|_|ScopedCollectiveError::Protocol("completion convergence metadata cannot be decoded"))?;
    if flags.iter().any(|&flag|flag!=0.0 && flag!=1.0) {return Err(ScopedCollectiveError::Protocol("invalid exact completion convergence flag"));}
    Ok(flags.iter().any(|&flag|flag==1.0))
}

/// Close two independently scoped groups to a fixed point. Every coordinator
/// participant must call the same two group slots in order and participate in
/// its own corresponding subgroups; the explicit coordinator must contain the
/// union of both groups' participants. Membership is not inferred from ranks.
/// Forward calls/tracking/order still must match inside each original group.
///
/// A newly attached zero anchor can expose another group's communication path.
/// Retaining both snapshots and globally voting after each propagation round
/// prevents either group from closing against a stale local loss graph.
/// Finite original captured graphs converge monotonically; there is no guessed
/// iteration cap, fabricated loss, added numerical gradient or optimizer skip.
pub fn complete_collective_scope_pair<B,S,A,C,G>(first:&CollectiveScope<B,S>,first_transport:A,
    second:&CollectiveScope<B,S>,second_transport:C,mut loss:Tensor<Autodiff<B,S>,1>,coordinator:G)
    -> Result<GroupedCollectiveLoss<B,S>,GroupedCollectiveError<A::Error,C::Error,G::Error>>
where B:Backend,S:CheckpointStrategy,A:BroadcastTensorCollective<B>,C:BroadcastTensorCollective<B>,G:BroadcastTensorCollective<B> {
    if first.same_window(second) {return Err(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("two original groups require distinct forward windows")));}
    if coordinator.world_size()==0 || coordinator.rank()>=coordinator.world_size()
        || first_transport.world_size()>coordinator.world_size() || second_transport.world_size()>coordinator.world_size() {
        return Err(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("explicit coordinator must cover both original groups")));
    }
    let first=first.prepare_completion(&loss,first_transport).map_err(GroupedCollectiveError::First)?;
    let second=second.prepare_completion(&loss,second_transport).map_err(GroupedCollectiveError::Second)?;
    let device=loss.device();let mut rounds=0usize;let mut local_anchors_added=0usize;
    loop {
        let step=first.propagate(loss).map_err(GroupedCollectiveError::First)?;loss=step.loss;let added=step.added_anchors;
        let step=second.propagate(loss).map_err(GroupedCollectiveError::Second)?;loss=step.loss;
        let added=added.checked_add(step.added_anchors).ok_or(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("completion anchor count overflows")))?;
        local_anchors_added=local_anchors_added.checked_add(added).ok_or(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("completion total anchor count overflows")))?;
        rounds=rounds.checked_add(1).ok_or(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("completion round count overflows")))?;
        if !any_changed(added!=0,&device,&coordinator).map_err(GroupedCollectiveError::Coordinator)? {break;}
    }
    Ok(GroupedCollectiveLoss {loss,rounds,local_anchors_added})
}

/// Same fixed-point closure for any explicit ordered group slots sharing one
/// transport type. A slot may contain different subgroup members on each rank,
/// but all coordinator ranks must supply the same number and order of slots.
pub fn complete_collective_scopes<B,S,C,G>(groups:&[(CollectiveScope<B,S>,C)],mut loss:Tensor<Autodiff<B,S>,1>,coordinator:G)
    -> Result<GroupedCollectiveLoss<B,S>,GroupedCollectiveError<C::Error,C::Error,G::Error>>
where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>,G:BroadcastTensorCollective<B> {
    if loss.dims()!=[1] || coordinator.world_size()==0 || coordinator.rank()>=coordinator.world_size() {
        return Err(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("valid scalar and explicit coordinator are required")));
    }
    // Exactly represent usize slot counts as four 16-bit coordination words.
    let count=u64::try_from(groups.len()).map_err(|_|GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("group slot count overflows")))?;
    let words=(0..4).map(|word|((count>>(word*16))&65535) as f32).collect::<Vec<_>>();
    let counts=Tensor::<B,1>::from_data(TensorData::new(words,[4]),(&loss.device(),DType::F32));
    let counts=if coordinator.world_size()==1 {counts} else {Tensor::from_primitive(TensorPrimitive::Float(
        coordinator.all_gather_float(counts.into_primitive().tensor()).map_err(|error|GroupedCollectiveError::Coordinator(ScopedCollectiveError::Collective(error)))?))};
    let expected=(coordinator.world_size() as usize).checked_mul(4).ok_or(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("group slot metadata size overflows")))?;
    if counts.dims()!=[expected] || counts.dtype()!=DType::F32 || counts.device()!=loss.device() {
        return Err(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("group slot metadata changed shape/storage/device")));
    }
    let words=counts.into_data().to_vec::<f32>().map_err(|_|GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("group slot metadata cannot be decoded")))?;
    for rank in 0..coordinator.world_size() as usize {
        let mut other=0u64;
        for word in 0..4 {let part=words[rank*4+word];
            if !part.is_finite() || part<0.0 || part>65535.0 || part as u32 as f32!=part {return Err(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("invalid exact group slot word")));}
            other|=(part as u64)<<(word*16);
        }
        if other!=count {return Err(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("coordinator group slot counts differ")));}
    }
    let mut windows=Vec::with_capacity(groups.len());
    for (index,(scope,transport)) in groups.iter().enumerate() {
        if groups[..index].iter().any(|(other,_)|scope.same_window(other)) || transport.world_size()>coordinator.world_size() {
            return Err(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("distinct covered original group windows are required")));
        }
        windows.push(scope.prepare_completion(&loss,transport.clone()).map_err(|error|GroupedCollectiveError::Indexed {index,error})?);
    }
    let device=loss.device();let mut rounds=0usize;let mut local_anchors_added=0usize;
    loop {
        let mut added=0usize;
        for (index,window) in windows.iter().enumerate() {
            let step=window.propagate(loss).map_err(|error|GroupedCollectiveError::Indexed {index,error})?;loss=step.loss;
            added=added.checked_add(step.added_anchors).ok_or(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("completion anchor count overflows")))?;
        }
        local_anchors_added=local_anchors_added.checked_add(added).ok_or(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("completion total anchor count overflows")))?;
        rounds=rounds.checked_add(1).ok_or(GroupedCollectiveError::Coordinator(ScopedCollectiveError::Protocol("completion round count overflows")))?;
        if !any_changed(added!=0,&device,&coordinator).map_err(GroupedCollectiveError::Coordinator)? {break;}
    }
    Ok(GroupedCollectiveLoss {loss,rounds,local_anchors_added})
}
