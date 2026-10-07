use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Dropout,Module,Tensor,column,row,row_inference,geometry};
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,dense_scaled_dot_product_attention},
    transformer::{AdaptedGroupedQueryAttention,AttentionAdapterTarget}};
use crate::cache::ProjectedKvCache;
use ruda_model::tensor::Bool;

/// Actual selected native LoRA/dense head and output-row projection partitions.
/// Column adapter A is replicated across head ranks; local B belongs to its
/// output columns, or is replicated only within the explicit KV replica group.
/// Row adapter A is local, while B is applied once after its full rank activation SUM.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedGroupedQueryAttention<B: Backend> {
    /// Original selected/unselected projection flags, base IDs, A/B dtypes and scales.
    pub local: AdaptedGroupedQueryAttention<B>,
}

impl<B: Backend> TensorParallelAdaptedGroupedQueryAttention<B> {
    /// Connect actual already partitioned adapters; never initialize global A/B or merge them.
    pub fn from_shard(local: AdaptedGroupedQueryAttention<B>) -> Self {
        assert!(local.query_heads > 0 && local.kv_heads > 0 && local.head_dimension > 0
            && local.query_heads.is_multiple_of(local.kv_heads),"invalid adapted parallel head geometry");
        let query = geometry(&local.query);let key = geometry(&local.key);
        assert_eq!(query[1],local.query_heads.checked_mul(local.head_dimension).expect("adapted parallel query width overflow"),"adapted query columns differ");
        assert_eq!(key[1],local.kv_heads.checked_mul(local.head_dimension).expect("adapted parallel KV width overflow"),"adapted KV columns differ");
        assert_eq!(geometry(&local.value),key,"adapted parallel K/V geometry differs");
        assert_eq!(geometry(&local.output),[query[1],query[0]],"adapted parallel output rows/residual differ");
        Self {local}
    }

    fn context(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        let [batch,heads,tokens,width] = query.dims();
        assert_eq!((heads,width),(self.local.query_heads,self.local.head_dimension),"adapted parallel projected query differs");
        assert_eq!((key.dims()[1],key.dims()[3]),(self.local.kv_heads,width),"adapted parallel projected keys differ");
        assert_eq!((value.dims()[1],value.dims()[3]),(self.local.kv_heads,width),"adapted parallel projected values differ");
        dense_scaled_dot_product_attention(query,key,value,masks,options,Some(&self.local.dropout)).swap_dims(1,2).reshape([batch,tokens,heads*width])
    }

    /// Native non-autodiff inference using actual selected adapters without merging.
    /// Original backend dropout mode is retained, so a native inference backend
    /// skips configured training dropout just as the original LoRA module does.
    pub fn forward_projected_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<B,3>,C::Error> {
        row_inference(&self.local.output,self.context(query,key,value,masks,options),&communicator)
    }

    /// Native local projection and row-reduced attention output, with no adapter repacking.
    pub fn forward_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<B,3>,C::Error> {
        let (query,key,value) = self.local.project(query,key,value);
        self.forward_projected_inference(query,key,value,masks,options,communicator)
    }

    /// Actual native adapter KV history with original inference dropout behavior.
    pub fn forward_cached_inference<C,F>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let (query,key,value) = self.local.project(input.clone(),input.clone(),input);
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key,cache.position());
        assert_eq!((query.dims(),key.dims()),geometry,"adapted native cached positions changed actual local heads");
        let (key,value,masks) = super::super::attention::cached::append(&query,key,value,visible,cache,masks,
            (self.local.query_heads,self.local.kv_heads,self.local.head_dimension));
        self.forward_projected_inference(query,key,value,masks,options,communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedGroupedQueryAttention<Autodiff<B,S>> {
    /// Actual adapted Q/K/V heads with explicit replica groups and original input dropout.
    /// Replicated adapter inputs/dropout draws must match across corresponding head ranks.
    pub fn project<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,groups: &AttentionParallelGroups<C,K>)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.project_with_adapter_dropout(query,key,value,groups,|_,dropout,input|dropout.forward(input))
    }

    /// Explicit shared adapter-dropout transform at each actual A-input dtype.
    /// Callback runs only for selected adapters, retaining per-projection configuration.
    pub fn project_with_adapter_dropout<C,K,F>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        groups: &AttentionParallelGroups<C,K>,mut dropout: F)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnMut(AttentionAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let [batch,queries,_] = query.dims();let [key_batch,keys,_] = key.dims();
        assert_eq!((key_batch,keys),(value.dims()[0],value.dims()[1]),"adapted parallel K/V rows differ");
        assert_eq!(batch,key_batch,"adapted parallel query/source batch differs");
        let query = column(&self.local.query,query,&groups.heads,None::<&K>,|module,input|dropout(AttentionAdapterTarget::Query,module,input))?;
        let key = column(&self.local.key,key,&groups.heads,groups.kv_replicas.as_ref(),|module,input|dropout(AttentionAdapterTarget::Key,module,input))?;
        let value = column(&self.local.value,value,&groups.heads,groups.kv_replicas.as_ref(),|module,input|dropout(AttentionAdapterTarget::Value,module,input))?;
        Ok((query.reshape([batch,queries,self.local.query_heads,self.local.head_dimension]).swap_dims(1,2),
            key.reshape([batch,keys,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2),
            value.reshape([batch,keys,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2)))
    }

    /// Native local-head attention plus row-sharded dense/LoRA output. Base and
    /// adapter rank activations are SUMmed separately at their actual stored dtypes.
    pub fn forward_projected<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,4>,key: Tensor<Autodiff<B,S>,4>,value: Tensor<Autodiff<B,S>,4>,
        masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        self.forward_projected_with_adapter_dropout(query,key,value,masks,options,communicator,|module,input|module.forward(input))
    }

    /// Explicit local-context adapter dropout before the actual row-sharded A projection.
    pub fn forward_projected_with_adapter_dropout<C,F>(&self,query: Tensor<Autodiff<B,S>,4>,key: Tensor<Autodiff<B,S>,4>,value: Tensor<Autodiff<B,S>,4>,
        masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,communicator: C,dropout: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let context = self.context(query,key,value,masks,options);
        row(&self.local.output,context,&communicator,dropout)
    }

    /// Adapted generic self/cross attention with original flags, actual groups and no merge.
    pub fn forward<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,groups: &AttentionParallelGroups<C,K>) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let (query,key,value) = self.project(query,key,value,groups)?;
        self.forward_projected(query,key,value,masks,options,groups.heads.clone())
    }

    /// Inference-only actual adapted local head cache, with persistent selected visibility.
    /// The original adapter scale/dtype is applied before positioned K/V are detached.
    pub fn forward_cached_projected<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,4>,key: Tensor<Autodiff<B,S>,4>,value: Tensor<Autodiff<B,S>,4>,
        visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,cache: &mut ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,
        options: DenseAttentionOptions,communicator: C) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let (key,value,masks) = super::super::attention::cached::append(&query,key,value,visible,cache,masks,
            (self.local.query_heads,self.local.kv_heads,self.local.head_dimension));
        self.forward_projected(query,key,value,masks,options,communicator)
    }
}
