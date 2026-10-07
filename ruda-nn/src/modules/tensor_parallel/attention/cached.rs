use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,TensorParallelGroupedQueryAttention,AttentionParallelGroups,Tensor,
    DenseAttentionMask,DenseAttentionOptions,region};
use ruda_model::tensor::Bool;
use crate::cache::ProjectedKvCache;

fn masks<B: Backend>(mut masks: DenseAttentionMask<B>,visible: Tensor<B,2,Bool>) -> DenseAttentionMask<B> {
    masks.key_valid = Some(if let Some(additional) = masks.key_valid {
        assert_eq!(additional.dims(),visible.dims(),"parallel cached visibility must describe the complete retained prefix");
        assert_eq!(additional.device(),visible.device(),"parallel cached visibility device differs");
        visible.bool_and(additional)
    } else {visible});
    masks
}

pub(in crate::modules::tensor_parallel) fn append<B: Backend>(query: &Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,new_visible: Option<Tensor<B,2,Bool>>,
    cache: &mut ProjectedKvCache<B>,mask: DenseAttentionMask<B>,geometry: (usize,usize,usize)) -> (Tensor<B,4>,Tensor<B,4>,DenseAttentionMask<B>) {
    assert_eq!((query.dims()[1],query.dims()[3]),(geometry.0,geometry.2),"parallel cached query geometry differs from actual head shards");
    assert_eq!((key.dims()[1],key.dims()[3]),(geometry.1,geometry.2),"parallel cached key geometry differs from actual head shards");
    assert_eq!((value.dims()[1],value.dims()[3]),(geometry.1,geometry.2),"parallel cached value geometry differs from actual head shards");
    assert_eq!(query.device(),key.device(),"parallel cached query/key devices differ");
    assert_eq!(query.dtype(),key.dtype(),"parallel cached query/key storage differs");
    assert_eq!(query.dims()[0],key.dims()[0],"parallel cached query/key batches differ");
    cache.validate_append(&key,&value,new_visible.as_ref());
    if let Some(additional) = &mask.key_valid {
        let length = cache.len().checked_add(key.dims()[2]).expect("parallel cached prefix length overflow");
        assert_eq!(additional.dims(),[query.dims()[0],length],"parallel cache mask must include old and new slots");
        assert_eq!(additional.device(),query.device(),"parallel cached additional mask device differs");
    }
    cache.append(key,value,new_visible);
    let (key,value,visible) = cache.prefix().expect("complete positioned parallel KV history was appended");
    (key,value,masks(mask,visible))
}

impl<B: Backend> TensorParallelGroupedQueryAttention<B> {
    /// Native local-shard cached inference; query/output are reduced without head gathers.
    /// Only actual new projected K/V are appended, with persistent explicit visibility.
    pub fn forward_cached_projected_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,
        new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,mask: DenseAttentionMask<B>,
        options: DenseAttentionOptions,communicator: C) -> Result<Tensor<B,3>,C::Error> {
        let (key,value,mask) = append(&query,key,value,new_visible,cache,mask,(self.local.query_heads,self.local.kv_heads,self.local.head_dimension));
        self.forward_projected_inference(query,key,value,mask,options,communicator)
    }

    /// Explicit positions transform only new heads and receive the absolute slot offset.
    /// Old positioned keys are never projected or transformed a second time.
    pub fn forward_cached_inference<C,F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        mask: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let (query,key,value) = self.local.project(input.clone(),input.clone(),input);
        let geometry = (query.dims(),key.dims());
        let (query,key) = positions(query,key,cache.position());
        assert_eq!((query.dims(),key.dims()),geometry,"parallel cached positions changed actual head geometry");
        self.forward_cached_projected_inference(query,key,value,new_visible,cache,mask,options,communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelGroupedQueryAttention<Autodiff<B,S>> {
    /// Query-only head projection for immutable already-prepared encoder memory.
    pub fn project_query<C: BroadcastTensorCollective<B>>(&self,input: Tensor<Autodiff<B,S>,3>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,4>,C::Error> {
        let [batch,tokens,_] = input.dims();
        let input = region::copy_to_region(input,communicator)?;
        Ok(self.local.query.forward(input).reshape([batch,tokens,self.local.query_heads,self.local.head_dimension]).swap_dims(1,2))
    }

    /// Project one actual encoder memory input into local K/V, including KV replica derivatives.
    /// A single copied memory node sums its combined key/value input contributions.
    pub fn project_memory<C,K>(&self,input: Tensor<Autodiff<B,S>,3>,groups: &AttentionParallelGroups<C,K>)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let [batch,tokens,_] = input.dims();
        let input = region::copy_to_region(input,groups.heads.clone())?;
        let key = self.kv_projection(&self.local.key,input.clone(),groups,None)?;
        let value = self.kv_projection(&self.local.value,input,groups,None)?;
        Ok((key.reshape([batch,tokens,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2),
            value.reshape([batch,tokens,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2)))
    }

    /// Inference-only positioned K/V append; cached history is intentionally detached.
    /// Causal alignment, retained windows and additional full-prefix masks stay explicit.
    pub fn forward_cached_projected<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,4>,key: Tensor<Autodiff<B,S>,4>,value: Tensor<Autodiff<B,S>,4>,
        new_visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,cache: &mut ProjectedKvCache<Autodiff<B,S>>,mask: DenseAttentionMask<Autodiff<B,S>>,
        options: DenseAttentionOptions,communicator: C) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let (key,value,mask) = append(&query,key,value,new_visible,cache,mask,(self.local.query_heads,self.local.kv_heads,self.local.head_dimension));
        self.forward_projected(query,key,value,mask,options,communicator)
    }

    /// Process the whole actual new chunk on this head shard, without old-history recompute.
    pub fn forward_cached<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,new_visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut ProjectedKvCache<Autodiff<B,S>>,mask: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let (query,key,value) = self.project_self(input,groups)?;
        let geometry = (query.dims(),key.dims());
        let (query,key) = positions(query,key,cache.position());
        assert_eq!((query.dims(),key.dims()),geometry,"parallel cached positions changed projected geometry");
        self.forward_cached_projected(query,key,value,new_visible,cache,mask,options,groups.heads.clone())
    }

    /// Attend actual immutable projected encoder K/V; no query or memory is inferred.
    pub fn forward_cached_memory<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,4>,memory: &ProjectedKvCache<Autodiff<B,S>>,
        mask: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let (key,value,visible) = memory.prefix().expect("prepare this rank's actual encoder K/V first");
        self.forward_projected(query,key,value,masks(mask,visible),options,communicator)
    }
}
