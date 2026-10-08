use super::*;
use core::fmt;
use crate::loss::LossTerms;
use ruda_autodiff::collective::{CollectiveScope,ScopedTensorCollective,ScopedCollectiveError,GroupedCollectiveError,complete_collective_scope_pair};

/// Original DP, EP, graph-coordinator and objective-statistics transports.
/// The coordinator covers all DP/EP participants. The separate statistics
/// group selects which actual independent losses enter the denominator/metric;
/// neither an EP world size nor replicated loss multiplicity is guessed.
pub struct HybridCollectiveLossContext<B:Backend,S:CheckpointStrategy,D,E,G,N> {
    data_scope:CollectiveScope<B,S>,
    expert_scope:CollectiveScope<B,S>,
    data:D,
    expert:E,
    coordinator:G,
    statistics:N,
}
impl<B:Backend,S:CheckpointStrategy,D,E,G,N> HybridCollectiveLossContext<B,S,D,E,G,N>
where D:BroadcastTensorCollective<B>,E:BroadcastTensorCollective<B>,G:BroadcastTensorCollective<B>,N:BroadcastTensorCollective<B> {
    /// All participants supply the same DP/EP slot order and their original
    /// matching per-subgroup forward/tracking schedule for this loss window.
    pub fn new(data:D,expert:E,coordinator:G,statistics:N) -> Self {
        Self {data_scope:CollectiveScope::new(),expert_scope:CollectiveScope::new(),data,expert,coordinator,statistics}
    }
    pub fn data(&self) -> ScopedTensorCollective<D,B,S> {self.data_scope.bind(self.data.clone())}
    pub fn expert(&self) -> ScopedTensorCollective<E,B,S> {self.expert_scope.bind(self.expert.clone())}

    /// Close both native communication graphs first, then gather exact I64
    /// counts and detached F32/F64 metrics on the selected statistics group.
    pub fn complete(self,loss_sum:Tensor<Autodiff<B,S>,1>,local_count:Tensor<Autodiff<B,S>,1,Int>)
        -> Result<HybridFullyShardedLoss<B,S>,HybridLossError<D::Error,E::Error,G::Error,N::Error>> {
        let graph=complete_collective_scope_pair(&self.data_scope,self.data,&self.expert_scope,self.expert,loss_sum,self.coordinator)
            .map_err(HybridLossError::Graph)?;
        let statistics=super::training::loss_statistics(graph.loss,local_count,self.statistics).map_err(HybridLossError::Statistics)?;
        Ok(HybridFullyShardedLoss {statistics,completion_rounds:graph.rounds,local_anchors_added:graph.local_anchors_added})
    }
    /// Original fractional/class/sample weights and valid-element statistics
    /// keep their native work precision and zero-denominator semantics.
    pub fn complete_terms<const K:usize>(self,terms:LossTerms<Autodiff<B,S>,K>)
        -> Result<HybridFullyShardedWeightedLoss<B,S>,HybridLossError<D::Error,E::Error,G::Error,N::Error>> {
        let work=super::training::loss_terms_work(&terms).map_err(|error|HybridLossError::Statistics(ScopedCollectiveError::Protocol(error)))?;
        let graph=complete_collective_scope_pair(&self.data_scope,self.data,&self.expert_scope,self.expert,terms.values.clone().cast(work).sum(),self.coordinator)
            .map_err(HybridLossError::Graph)?;
        let statistics=super::training::weighted_statistics(terms,graph.loss,self.statistics).map_err(HybridLossError::Statistics)?;
        Ok(HybridFullyShardedWeightedLoss {statistics,completion_rounds:graph.rounds,local_anchors_added:graph.local_anchors_added})
    }
}

pub struct HybridFullyShardedLoss<B:Backend,S:CheckpointStrategy> {
    pub statistics:FullyShardedLoss<B,S>,
    pub completion_rounds:usize,
    pub local_anchors_added:usize,
}
impl<B:Backend,S:CheckpointStrategy> HybridFullyShardedLoss<B,S> {
    pub fn mean(&self) -> Tensor<Autodiff<B,S>,1> {self.statistics.mean()}
    pub fn normalized(&self,global_window_count:u64) -> Tensor<Autodiff<B,S>,1> {self.statistics.normalized(global_window_count)}
    pub fn global_mean(&self) -> Tensor<B,1> {self.statistics.global_mean()}
}
pub struct HybridFullyShardedWeightedLoss<B:Backend,S:CheckpointStrategy> {
    pub statistics:FullyShardedWeightedLoss<B,S>,
    pub completion_rounds:usize,
    pub local_anchors_added:usize,
}
impl<B:Backend,S:CheckpointStrategy> HybridFullyShardedWeightedLoss<B,S> {
    pub fn mean(&self) -> Tensor<Autodiff<B,S>,1> {self.statistics.mean()}
    pub fn normalized(&self,global_window_weight:Tensor<B,1>) -> Tensor<Autodiff<B,S>,1> {self.statistics.normalized(global_window_weight)}
    pub fn global_mean(&self) -> Tensor<B,1> {self.statistics.global_mean()}
}

#[derive(Debug)]
pub enum HybridLossError<D:fmt::Debug,E:fmt::Debug,G:fmt::Debug,N:fmt::Debug> {
    Graph(GroupedCollectiveError<D,E,G>),
    Statistics(ScopedCollectiveError<N>),
}
impl<D:fmt::Debug,E:fmt::Debug,G:fmt::Debug,N:fmt::Debug> fmt::Display for HybridLossError<D,E,G,N> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Graph(error)=>write!(f,"{error}"),Self::Statistics(error)=>write!(f,"hybrid objective statistics: {error}")}
    }
}
impl<D:fmt::Debug,E:fmt::Debug,G:fmt::Debug,N:fmt::Debug> core::error::Error for HybridLossError<D,E,G,N> {}

/// Native model/objective failures are independent of graph completion and
/// loss-statistics failures, retaining all original transport error types.
#[derive(Debug)]
pub enum HybridModelTrainingError<M:fmt::Debug,O:fmt::Debug,D:fmt::Debug,E:fmt::Debug,G:fmt::Debug,N:fmt::Debug> {
    Model(M),
    Objective(O),
    Loss(HybridLossError<D,E,G,N>),
}
impl<M:fmt::Debug,O:fmt::Debug,D:fmt::Debug,E:fmt::Debug,G:fmt::Debug,N:fmt::Debug> fmt::Display for HybridModelTrainingError<M,O,D,E,G,N> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Model(error)=>write!(f,"hybrid native model: {error:?}"),Self::Objective(error)=>write!(f,"hybrid native objective: {error:?}"),Self::Loss(error)=>write!(f,"{error}")}
    }
}
impl<M:fmt::Debug,O:fmt::Debug,D:fmt::Debug,E:fmt::Debug,G:fmt::Debug,N:fmt::Debug> core::error::Error for HybridModelTrainingError<M,O,D,E,G,N> {}
