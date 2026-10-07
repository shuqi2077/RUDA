use ruda_model::tensor::{Bool,Tensor,backend::Backend};
use crate::{cache::ProjectedKvCache,transformer::AdaptedGroupedQueryAttention};
use super::{GroupedQueryAttention,DenseAttentionMask,DenseAttentionOptions};

fn cached_masks<B: Backend>(mut masks: DenseAttentionMask<B>,visible: Tensor<B,2,Bool>) -> DenseAttentionMask<B> {
    masks.key_valid = Some(if let Some(additional) = masks.key_valid {
        assert_eq!(additional.dims(),visible.dims(),"cached additional key validity must describe the complete retained prefix");
        assert_eq!(additional.device(),visible.device(),"cached additional visibility device differs");
        visible.bool_and(additional)
    } else { visible });
    masks
}

fn append<B: Backend>(query: &Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,
    new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,
    geometry: (usize,usize,usize)) -> (Tensor<B,4>,Tensor<B,4>,DenseAttentionMask<B>) {
    let [batch,heads,_,width] = query.dims();
    assert_eq!((heads,width),(geometry.0,geometry.2),"cached projected query head geometry differs");
    assert_eq!((key.dims()[0],key.dims()[1],key.dims()[3]),(batch,geometry.1,geometry.2),"cached projected key geometry differs");
    assert_eq!((value.dims()[0],value.dims()[1],value.dims()[3]),(batch,geometry.1,geometry.2),"cached projected value geometry differs");
    assert_eq!(query.dtype(),key.dtype(),"cached Q/K storage differs");
    assert_eq!(query.device(),key.device(),"cached Q/K device differs");
    cache.validate_append(&key,&value,new_visible.as_ref());
    if let Some(additional) = &masks.key_valid {
        let length = cache.len().checked_add(key.dims()[2]).expect("cached score length overflow");
        assert_eq!(additional.dims(),[batch,length],"cached additional mask must cover old and new key slots");
        assert_eq!(additional.device(),query.device(),"cached additional mask device differs");
    }
    let (key,value,visible) = cache.append(key,value,new_visible);
    (key,value,cached_masks(masks,visible))
}

impl<B: Backend> GroupedQueryAttention<B> {
    /// Project only actual query rows, without reprojecting cached encoder memory.
    pub fn project_query(&self,input: Tensor<B,3>) -> Tensor<B,4> {
        let [batch,tokens,_] = input.dims();
        self.query.forward(input).reshape([batch,tokens,self.query_heads,self.head_dimension]).swap_dims(1,2)
    }

    /// Project actual K/V independently of query payloads and query input width.
    pub fn project_key_value(&self,key: Tensor<B,3>,value: Tensor<B,3>) -> (Tensor<B,4>,Tensor<B,4>) {
        let [batch,tokens,_] = key.dims();
        assert_eq!((value.dims()[0],value.dims()[1]),(batch,tokens),"memory K/V input rows differ");
        (self.key.forward(key).reshape([batch,tokens,self.kv_heads,self.head_dimension]).swap_dims(1,2),
            self.value.forward(value).reshape([batch,tokens,self.kv_heads,self.head_dimension]).swap_dims(1,2))
    }

    /// Append positioned native K/V and attend only the supplied query rows.
    /// New visibility is stored; masks.key_valid is an additional full-prefix mask.
    /// Causal alignment/window/score bias remain exactly the caller's supplied policy.
    pub fn forward_cached_projected(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,
        new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions) -> Tensor<B,3> {
        let (key,value,masks) = append(&query,key,value,new_visible,cache,masks,(self.query_heads,self.kv_heads,self.head_dimension));
        self.forward_projected(query,key,value,masks,options)
    }

    /// Process the complete actual new chunk, without recomputing old projections or FFNs.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with_positions(input,new_visible,cache,masks,options,|query,key,_|(query,key))
    }

    /// Transform only new Q/K; the closure receives the actual absolute new-slot position.
    pub fn forward_cached_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let (query,key,value) = self.project(input.clone(),input.clone(),input);
        let shape = (query.dims(),key.dims());
        let (query,key) = positions(query,key,cache.position());
        assert_eq!((query.dims(),key.dims()),shape,"cached position transform changed new head geometry");
        self.forward_cached_projected(query,key,value,new_visible,cache,masks,options)
    }

    /// Attend immutable already-projected actual memory; only query/output weights run.
    pub fn forward_cached_memory(&self,query: Tensor<B,4>,memory: &ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        let (key,value,visible) = memory.prefix().expect("prepare actual projected memory before cached cross-attention");
        self.forward_projected(query,key,value,cached_masks(masks,visible),options)
    }
}

impl<B: Backend> AdaptedGroupedQueryAttention<B> {
    /// Apply the actual query projection including its selected adapter, independently of memory.
    pub fn project_query(&self,input: Tensor<B,3>) -> Tensor<B,4> {
        let [batch,tokens,_] = input.dims();
        self.query.forward(input).reshape([batch,tokens,self.query_heads,self.head_dimension]).swap_dims(1,2)
    }

    /// Prepare actual dense/adapted memory K/V without running a query projection.
    pub fn project_key_value(&self,key: Tensor<B,3>,value: Tensor<B,3>) -> (Tensor<B,4>,Tensor<B,4>) {
        let [batch,tokens,_] = key.dims();
        assert_eq!((value.dims()[0],value.dims()[1]),(batch,tokens),"adapted memory K/V rows differ");
        (self.key.forward(key).reshape([batch,tokens,self.kv_heads,self.head_dimension]).swap_dims(1,2),
            self.value.forward(value).reshape([batch,tokens,self.kv_heads,self.head_dimension]).swap_dims(1,2))
    }

    /// Explicit incremental inference with real selected adapters and detached cached K/V.
    pub fn forward_cached_projected(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,
        new_visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions) -> Tensor<B,3> {
        let (key,value,masks) = append(&query,key,value,new_visible,cache,masks,(self.query_heads,self.kv_heads,self.head_dimension));
        self.forward_projected(query,key,value,masks,options)
    }

    /// Process every actual new token with unchanged original base and adapter parameters.
    pub fn forward_cached(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_cached_with_positions(input,new_visible,cache,masks,options,|query,key,_|(query,key))
    }

    /// Caller-owned Q/K transforms see absolute new positions, never retransforming old keys.
    pub fn forward_cached_with_positions<F>(&self,input: Tensor<B,3>,new_visible: Option<Tensor<B,2,Bool>>,
        cache: &mut ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let (query,key,value) = self.project(input.clone(),input.clone(),input);
        let shape = (query.dims(),key.dims());
        let (query,key) = positions(query,key,cache.position());
        assert_eq!((query.dims(),key.dims()),shape,"adapted cached positions changed new head geometry");
        self.forward_cached_projected(query,key,value,new_visible,cache,masks,options)
    }

    /// Reuse actual already-positioned encoder K/V while applying dense/adapted Q/output.
    pub fn forward_cached_memory(&self,query: Tensor<B,4>,memory: &ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        let (key,value,visible) = memory.prefix().expect("prepare actual adapted projected memory before cached cross-attention");
        self.forward_projected(query,key,value,cached_masks(masks,visible),options)
    }
}
