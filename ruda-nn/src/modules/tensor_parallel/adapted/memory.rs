use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Dropout,Tensor,column};
use super::TensorParallelAdaptedGroupedQueryAttention;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},cache::ProjectedKvCache,transformer::AttentionAdapterTarget};
use super::super::attention::cached::masks;

impl<B: Backend> TensorParallelAdaptedGroupedQueryAttention<B> {
    /// Native query-only adapter projection without evaluating unused encoder projections.
    pub fn project_query_inference(&self,input: Tensor<B,3>) -> Tensor<B,4> {
        let [batch,tokens,_] = input.dims();
        self.local.query.forward(input).reshape([batch,tokens,self.local.query_heads,self.local.head_dimension]).swap_dims(1,2)
    }

    /// Native selected K/V adapters on actual source rows; query weights are not touched.
    pub fn project_memory_inference(&self,input: Tensor<B,3>) -> (Tensor<B,4>,Tensor<B,4>) {
        let [batch,tokens,_] = input.dims();
        (self.local.key.forward(input.clone()).reshape([batch,tokens,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2),
            self.local.value.forward(input).reshape([batch,tokens,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2))
    }

    /// Native inference over retained actual adapted source K/V and persistent source visibility.
    pub fn forward_cached_memory_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,4>,memory: &ProjectedKvCache<B>,
        mask: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<B,3>,C::Error> {
        let (key,value,visible) = memory.prefix().expect("prepare actual adapted local encoder K/V first");
        self.forward_projected_inference(query,key,value,masks(mask,visible),options,communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedGroupedQueryAttention<Autodiff<B,S>> {
    /// Actual query-only column projection with original adapter input dropout and A/B storage.
    pub fn project_query<C: BroadcastTensorCollective<B>>(&self,input: Tensor<Autodiff<B,S>,3>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,4>,C::Error> {
        self.project_query_with_adapter_dropout(input,communicator,|module,input|module.forward(input))
    }

    /// Explicit replicated query adapter dropout at the original A-input dtype.
    pub fn project_query_with_adapter_dropout<C,F>(&self,input: Tensor<Autodiff<B,S>,3>,communicator: C,dropout: F)
        -> Result<Tensor<Autodiff<B,S>,4>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let [batch,tokens,_] = input.dims();
        let query = column(&self.local.query,input,&communicator,None::<&C>,dropout)?;
        Ok(query.reshape([batch,tokens,self.local.query_heads,self.local.head_dimension]).swap_dims(1,2))
    }

    /// Actual source-only K/V projections with explicitly corresponding KV replica gradients.
    pub fn project_memory<C,K>(&self,input: Tensor<Autodiff<B,S>,3>,groups: &AttentionParallelGroups<C,K>)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.project_memory_with_adapter_dropout(input,groups,|_,module,input|module.forward(input))
    }

    /// Explicit per-source-projection adapter dropout, preserving independent K/V target configuration.
    pub fn project_memory_with_adapter_dropout<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,groups: &AttentionParallelGroups<C,K>,mut dropout: F)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnMut(AttentionAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let [batch,tokens,_] = input.dims();
        let key = column(&self.local.key,input.clone(),&groups.heads,groups.kv_replicas.as_ref(),|module,input|dropout(AttentionAdapterTarget::Key,module,input))?;
        let value = column(&self.local.value,input,&groups.heads,groups.kv_replicas.as_ref(),|module,input|dropout(AttentionAdapterTarget::Value,module,input))?;
        Ok((key.reshape([batch,tokens,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2),
            value.reshape([batch,tokens,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2)))
    }

    /// Inference-only detached source history with actual selected dense/LoRA output rows.
    pub fn forward_cached_memory<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,4>,memory: &ProjectedKvCache<Autodiff<B,S>>,
        mask: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let (key,value,visible) = memory.prefix().expect("prepare actual adapted parallel encoder K/V first");
        self.forward_projected(query,key,value,masks(mask,visible),options,communicator)
    }
}
