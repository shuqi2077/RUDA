use super::*;
use alloc::{sync::Arc,vec::Vec};
use core::fmt;
use ruda_tensor::{DType,TensorData,collective::{IntegerTensorCollective,VariableTensorCollective,VariableTensorExchange},tensor::{Bool,FloatTensor,IntTensor}};
use crate::NodeId;
#[cfg(feature="std")]
use parking_lot::Mutex;
#[cfg(not(feature="std"))]
use spin::Mutex;

struct Entry<B:Backend,S:CheckpointStrategy> {
    node:NodeId,
    anchor:Tensor<Autodiff<B,S>,1>,
}

/// One explicit data-sharded forward/loss window, retaining scalar graph anchors and original collective order.
/// Captures actual differentiable collectives using its bound transport, not synthetic model tokens or weights.
/// Complete the loss on every rank before backward. Use one scope per actual transport group,
/// with matching forward calls, tracking and topology; complete each independently selected group explicitly.
/// this coordinates rank-local unused gather paths, not arbitrary mismatched collective-bearing architectures.
pub struct CollectiveScope<B:Backend,S:CheckpointStrategy> {entries:Arc<Mutex<Vec<Entry<B,S>>>>}
impl<B:Backend,S:CheckpointStrategy> Clone for CollectiveScope<B,S> {
    fn clone(&self) -> Self {Self {entries:self.entries.clone()}}
}
impl<B:Backend,S:CheckpointStrategy> fmt::Debug for CollectiveScope<B,S> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {f.debug_struct("CollectiveScope").field("captured_collectives",&self.entries.lock().len()).finish()}
}
impl<B:Backend,S:CheckpointStrategy> Default for CollectiveScope<B,S> {fn default() -> Self {Self::new()}}

/// Actual transport/protocol errors while completing the native sharded loss graph.
#[derive(Debug)]
pub enum ScopedCollectiveError<E:fmt::Debug> {
    /// Underlying original tensor transport error.
    Collective(E),
    /// Actual scope/topology/transport metadata mismatch.
    Protocol(&'static str),
}
impl<E:fmt::Debug> fmt::Display for ScopedCollectiveError<E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Collective(error)=>write!(f,"scoped collective transport: {error:?}"),Self::Protocol(message)=>write!(f,"scoped collective protocol: {message}")}
    }
}
impl<E:fmt::Debug> core::error::Error for ScopedCollectiveError<E> {}

/// The actual original transport with one explicitly bound native AD loss-window context.
pub struct ScopedTensorCollective<C,B:Backend,S:CheckpointStrategy> {inner:C,scope:CollectiveScope<B,S>}

/// One consumed forward window whose original graph anchors remain available
/// while independently selected transport groups propagate loss reachability.
pub struct CollectiveScopeCompletion<B:Backend,S:CheckpointStrategy,C> {
    entries:Vec<Entry<B,S>>,
    communicator:C,
    device:B::Device,
    world:u32,
}

/// One native graph propagation pass, not a claim that other groups have closed.
pub struct CollectiveCompletionStep<B:Backend,S:CheckpointStrategy> {
    pub loss:Tensor<Autodiff<B,S>,1>,
    pub added_anchors:usize,
}
impl<C:Clone,B:Backend,S:CheckpointStrategy> Clone for ScopedTensorCollective<C,B,S> {
    fn clone(&self) -> Self {Self {inner:self.inner.clone(),scope:self.scope.clone()}}
}
impl<C:fmt::Debug,B:Backend,S:CheckpointStrategy> fmt::Debug for ScopedTensorCollective<C,B,S> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {f.debug_struct("ScopedTensorCollective").field("transport",&self.inner).field("scope",&self.scope).finish()}
}
impl<C:TensorCollective<B>,B:Backend,S:CheckpointStrategy> TensorCollective<B> for ScopedTensorCollective<C,B,S> {
    type Error=C::Error;
    fn world_size(&self) -> u32 {self.inner.world_size()}
    fn autodiff_context(&self) -> Option<&(dyn core::any::Any+Send+Sync)> {Some(&self.scope)}
    fn all_gather_float(&self,value:FloatTensor<B>) -> Result<FloatTensor<B>,Self::Error> {self.inner.all_gather_float(value)}
    fn reduce_scatter_sum(&self,value:FloatTensor<B>) -> Result<FloatTensor<B>,Self::Error> {self.inner.reduce_scatter_sum(value)}
}
impl<C:ReplicatedTensorCollective<B>,B:Backend,S:CheckpointStrategy> ReplicatedTensorCollective<B> for ScopedTensorCollective<C,B,S> {
    fn all_reduce_sum(&self,value:FloatTensor<B>) -> Result<FloatTensor<B>,Self::Error> {self.inner.all_reduce_sum(value)}
}
impl<C:BroadcastTensorCollective<B>,B:Backend,S:CheckpointStrategy> BroadcastTensorCollective<B> for ScopedTensorCollective<C,B,S> {
    fn rank(&self) -> u32 {self.inner.rank()}
    fn broadcast_float(&self,value:FloatTensor<B>,root:u32) -> Result<FloatTensor<B>,Self::Error> {self.inner.broadcast_float(value,root)}
}
impl<C:IntegerTensorCollective<B>,B:Backend,S:CheckpointStrategy> IntegerTensorCollective<B> for ScopedTensorCollective<C,B,S> {
    fn all_gather_int(&self,value:IntTensor<B>) -> Result<IntTensor<B>,Self::Error> {self.inner.all_gather_int(value)}
}
impl<C:VariableTensorCollective<B>,B:Backend,S:CheckpointStrategy> VariableTensorCollective<B> for ScopedTensorCollective<C,B,S> {
    fn all_to_all_v_float(&self,value:FloatTensor<B>,counts:&[usize]) -> Result<VariableTensorExchange<FloatTensor<B>>,Self::Error> {
        self.inner.all_to_all_v_float(value,counts)
    }
    fn all_to_all_v_int(&self,value:IntTensor<B>,counts:&[usize]) -> Result<VariableTensorExchange<IntTensor<B>>,Self::Error> {
        self.inner.all_to_all_v_int(value,counts)
    }
}

impl<B:Backend,S:CheckpointStrategy> CollectiveScope<B,S> {
    /// Create an independent actual loss window. No process-global tracer or model state is modified.
    pub fn new() -> Self {Self {entries:Arc::new(Mutex::new(Vec::new()))}}
    /// Bind only the caller-selected actual parameter/data transport to this loss window.
    pub fn bind<C:TensorCollective<B>>(&self,communicator:C) -> ScopedTensorCollective<C,B,S> {
        ScopedTensorCollective {inner:communicator,scope:self.clone()}
    }
    pub(super) fn capture<const D:usize>(&self,value:Tensor<Autodiff<B,S>,D>) {
        let primitive=value.clone().into_primitive().tensor();
        if !primitive.is_tracked() {return;}
        let node=primitive.node.id;
        let dims=value.dims();
        let ranges:[core::ops::Range<usize>;D]=core::array::from_fn(|_|0..1);
        let value=if dims.contains(&0) {value} else {value.slice(ranges)};
        // True-mask replacement gives a safe zero and zero derivative even if the original
        // weight coordinate is Inf/NaN; multiplying an arbitrary value by zero would not.
        let mask=Tensor::<Autodiff<B,S>,D,Bool>::zeros(value.dims(),&value.device()).bool_not();
        let anchor=value.cast(DType::F32).mask_fill(mask,0).sum();
        self.entries.lock().push(Entry {node,anchor});
    }
    /// Complete a real scalar loss by coordinating actual graph reachability across the original data group.
    /// Locally unused but globally used collectives receive only a safe zero dependency; globally unused
    /// operations remain absent from backward/optimizer gradients. Existing nonzero loss values are unchanged.
    /// Boolean path flags/counts are host coordination metadata, not host numerical gradient/model fallbacks.
    pub fn complete<C:BroadcastTensorCollective<B>>(&self,loss:Tensor<Autodiff<B,S>,1>,communicator:C)
        -> Result<Tensor<Autodiff<B,S>,1>,ScopedCollectiveError<C::Error>> {
        self.prepare_completion(&loss,communicator)?.propagate(loss).map(|step|step.loss)
    }

    /// Consume exactly this forward window, validating actual captured counts
    /// on its own group. No loss normalization or backward is performed here.
    pub fn prepare_completion<C:BroadcastTensorCollective<B>>(&self,loss:&Tensor<Autodiff<B,S>,1>,communicator:C)
        -> Result<CollectiveScopeCompletion<B,S,C>,ScopedCollectiveError<C::Error>> {
        if loss.dims()!=[1] {return Err(ScopedCollectiveError::Protocol("one actual scalar loss is required"));}
        let world=communicator.world_size();
        if world==0 || communicator.rank()>=world {return Err(ScopedCollectiveError::Protocol("invalid original data topology"));}
        let entries=core::mem::take(&mut *self.entries.lock());
        if world==1 {return Ok(CollectiveScopeCompletion {entries,communicator,device:loss.device(),world});}
        let count=u64::try_from(entries.len()).map_err(|_|ScopedCollectiveError::Protocol("scope count overflows"))?;
        let words=(0..4).map(|word|((count>>(word*16))&65535) as f32).collect::<Vec<_>>();
        let counts=Tensor::<B,1>::from_data(TensorData::new(words,[4]),(&loss.device(),DType::F32));
        let counts=if world==1 {counts} else {Tensor::from_primitive(TensorPrimitive::Float(communicator.all_gather_float(counts.into_primitive().tensor())
            .map_err(ScopedCollectiveError::Collective)?))};
        if counts.dims()!=[world as usize*4] || counts.dtype()!=DType::F32 || counts.device()!=loss.device() {
            return Err(ScopedCollectiveError::Protocol("scope count transport changed original shape/storage/device"));
        }
        let words=counts.into_data().to_vec::<f32>().map_err(|_|ScopedCollectiveError::Protocol("scope count metadata cannot be decoded"))?;
        for rank in 0..world as usize {
            let mut other=0u64;
            for word in 0..4 {
                let part=words[rank*4+word];
                if !part.is_finite() || part<0.0 || part>65535.0 || part as u32 as f32!=part {return Err(ScopedCollectiveError::Protocol("invalid exact scope count word"));}
                other|=(part as u64)<<(word*16);
            }
            if other!=count {return Err(ScopedCollectiveError::Protocol("rank forward collective count/tracking differs"));}
        }
        Ok(CollectiveScopeCompletion {entries,communicator,device:loss.device(),world})
    }

    /// Whether two handles refer to the same original forward window.
    pub fn same_window(&self,other:&Self) -> bool {Arc::ptr_eq(&self.entries,&other.entries)}
}

impl<B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>> CollectiveScopeCompletion<B,S,C> {
    /// Recompute real gradient-path flags from the current loss. Added zero
    /// dependencies may expose a path on another group, so mixed-group callers
    /// propagate repeatedly until their explicit common coordinator converges.
    pub fn propagate(&self,mut loss:Tensor<Autodiff<B,S>,1>)
        -> Result<CollectiveCompletionStep<B,S>,ScopedCollectiveError<C::Error>> {
        if loss.dims()!=[1] || loss.device()!=self.device {return Err(ScopedCollectiveError::Protocol("scope completion loss shape/device differs"));}
        let entries=&self.entries;let communicator=&self.communicator;let world=self.world;
        if communicator.world_size()!=world || communicator.rank()>=world {return Err(ScopedCollectiveError::Protocol("scope completion transport topology changed"));}
        if world==1 || entries.is_empty() {return Ok(CollectiveCompletionStep {loss,added_anchors:0});}
        let root=loss.clone().into_primitive().tensor();
        let ids=entries.iter().map(|entry|entry.node).collect::<Vec<_>>();
        let local=root.node.client.gradient_paths(root.node.id,&ids);
        let flags=local.iter().map(|present|if *present {1.0f32} else {0.0}).collect::<Vec<_>>();
        let flags=Tensor::<B,1>::from_data(TensorData::new(flags,[entries.len()]),(&loss.device(),DType::F32));
        let flags=if world==1 {flags} else {Tensor::from_primitive(TensorPrimitive::Float(communicator.all_gather_float(flags.into_primitive().tensor())
            .map_err(ScopedCollectiveError::Collective)?))};
        let total=entries.len().checked_mul(world as usize).ok_or(ScopedCollectiveError::Protocol("scope vote size overflows"))?;
        if flags.dims()!=[total] || flags.dtype()!=DType::F32 || flags.device()!=loss.device() {return Err(ScopedCollectiveError::Protocol("scope vote transport changed original shape/storage/device"));}
        let flags=flags.into_data().to_vec::<f32>().map_err(|_|ScopedCollectiveError::Protocol("scope path metadata cannot be decoded"))?;
        if flags.iter().any(|&flag|flag!=0.0 && flag!=1.0) {return Err(ScopedCollectiveError::Protocol("invalid exact scope path flag"));}
        let mut added_anchors=0;
        for (index,entry) in entries.iter().enumerate() {
            if !local[index] && (0..world as usize).any(|rank|flags[rank*entries.len()+index]==1.0) {
                let dtype=loss.dtype();loss=loss+entry.anchor.clone().cast(dtype);added_anchors+=1;
            }
        }
        Ok(CollectiveCompletionStep {loss,added_anchors})
    }
}
