use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::{module::Module,tensor::{FloatDType,Tensor,backend::Backend,module::linear}};
use crate::{Linear,attention::{GroupedQueryAttention,DenseAttentionMask,DenseAttentionOptions,dense_scaled_dot_product_attention}};
use region::BroadcastTensorCollective;

pub(super) mod cached;
mod packed;
mod masks;
pub use masks::*;

/// Explicit head-parallel group and optional group sharing this exact KV shard.
/// KV replica groups contain only ranks holding the same K/V parameter values;
/// different KV shards must never be reduced as replicas. Group membership,
/// replicated rows and collective ordering remain caller-owned.
#[derive(Clone,Debug)]
pub struct AttentionParallelGroups<C,K=C> {
    /// All ranks contributing query-head/output-row shards to one logical attention.
    pub heads: C,
    /// Ranks sharing this actual KV shard; None means locally owned KV parameters.
    pub kv_replicas: Option<K>,
}

impl<C> AttentionParallelGroups<C,C> {
    /// Explicit fully sharded KV parameters; no KV weight collective is introduced.
    pub fn sharded(heads: C) -> Self {Self {heads,kv_replicas:None}}
}

impl<C,K> AttentionParallelGroups<C,K> {
    /// Explicit replica group for MQA or repeated KV shards in head-parallel GQA.
    pub fn replicated_kv(heads: C,kv_replicas: K) -> Self {Self {heads,kv_replicas:Some(kv_replicas)}}
}

/// Actual local Q/K/V columns and output rows of native MHA/GQA/MQA.
/// Input/output residuals are replicated; attention heads and probabilities stay
/// local. The caller supplies geometrically valid corresponding head shards;
/// no global head placement, positional scheme or checkpoint slicing is inferred.
#[derive(Module,Debug)]
pub struct TensorParallelGroupedQueryAttention<B: Backend> {
    /// Original local projection IDs, head geometry and native probability dropout.
    pub local: GroupedQueryAttention<B>,
}

fn projection<B: Backend,const D: usize>(input: Tensor<B,D>,weight: Tensor<B,2>,bias: Option<Tensor<B,1>>,
    compute: Option<FloatDType>) -> Tensor<B,D> {
    if let Some(dtype) = compute {linear(input.cast(dtype),weight.cast(dtype),bias.map(|bias|bias.cast(dtype)))}
    else {linear(input,weight,bias)}
}

impl<B: Backend> TensorParallelGroupedQueryAttention<B> {
    /// Connect already loaded local head shards without recreating or copying parameters.
    /// Output bias is a replicated full residual vector, not a head-shard bias.
    pub fn from_shard(local: GroupedQueryAttention<B>) -> Self {
        let query = local.query.weight.val().dims();
        let key = local.key.weight.val().dims();
        assert!(local.query_heads > 0 && local.kv_heads > 0 && local.head_dimension > 0
            && local.query_heads.is_multiple_of(local.kv_heads),"invalid local parallel attention head geometry");
        assert_eq!(query[1],local.query_heads.checked_mul(local.head_dimension).expect("parallel query width overflow"),"query columns differ from local heads");
        assert_eq!(key[1],local.kv_heads.checked_mul(local.head_dimension).expect("parallel KV width overflow"),"KV columns differ from local heads");
        assert_eq!(local.value.weight.val().dims(),key,"parallel key/value projection geometry differs");
        assert_eq!(local.output.weight.val().dims(),[query[1],query[0]],"parallel output rows/residual width differ");
        for layer in [&local.query,&local.key,&local.value,&local.output] {
            if let Some(bias) = &layer.bias {assert_eq!(bias.val().dims(),[layer.weight.val().dims()[1]],"parallel projection bias differs from actual output width");}
        }
        Self {local}
    }

    fn partial(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,mut masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,compute: Option<FloatDType>) -> Tensor<B,3> {
        let [batch,heads,tokens,width] = query.dims();
        assert_eq!((heads,width),(self.local.query_heads,self.local.head_dimension),"parallel projected query heads differ");
        assert_eq!((key.dims()[1],key.dims()[3]),(self.local.kv_heads,width),"parallel projected key heads differ");
        assert_eq!((value.dims()[1],value.dims()[3]),(self.local.kv_heads,width),"parallel projected value heads differ");
        if let Some(dtype) = compute {masks.bias = masks.bias.map(|bias|bias.cast(dtype));}
        let context = dense_scaled_dot_product_attention(query,key,value,masks,options,Some(&self.local.dropout))
            .swap_dims(1,2).reshape([batch,tokens,heads*width]);
        projection(context,self.local.output.weight.val(),None,compute)
    }

    fn bias<const D: usize>(&self,output: Tensor<B,D>,compute: Option<FloatDType>) -> Tensor<B,D> {
        if let Some(bias) = &self.local.output.bias {
            let mut shape = [1;D];
            shape[D-1] = bias.val().dims()[0];
            let bias = if let Some(dtype) = compute {bias.val().cast(dtype)} else {bias.val()};
            output+bias.reshape(shape)
        } else {output}
    }

    /// Native inference on actual positioned local heads, with one output SUM.
    /// No autodiff wrapper, CPU activation transfer or full-head gathering occurs.
    /// Use the Autodiff project/forward methods when parameter gradients are required.
    pub fn forward_projected_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<B,3>,C::Error> {
        let partial = self.partial(query,key,value,masks,options,None);
        let output = communicator.all_reduce_sum(partial.into_primitive().tensor())?;
        Ok(self.bias(Tensor::from_primitive(ruda_model::tensor::TensorPrimitive::Float(output)),None))
    }

    /// Native self/cross inference with actual query and memory input widths.
    pub fn forward_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C) -> Result<Tensor<B,3>,C::Error> {
        let (query,key,value) = self.local.project(query,key,value);
        self.forward_projected_inference(query,key,value,masks,options,communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelGroupedQueryAttention<Autodiff<B,S>> {
    fn kv_projection<C,K>(&self,layer: &Linear<Autodiff<B,S>>,input: Tensor<Autodiff<B,S>,3>,
        groups: &AttentionParallelGroups<C,K>,compute: Option<FloatDType>) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let mut weight = layer.weight.val();
        let mut bias = layer.bias.as_ref().map(|bias|bias.val());
        if let Some(dtype) = compute {
            weight = weight.cast(dtype);
            bias = bias.map(|bias|bias.cast(dtype));
        }
        if let Some(replicas) = &groups.kv_replicas {
            if weight.is_require_grad() {weight = region::copy_to_region(weight,replicas.clone())?;}
            bias = bias.map(|bias|if bias.is_require_grad() {region::copy_to_region(bias,replicas.clone())} else {Ok(bias)}).transpose()?;
        }
        Ok(projection(input,weight,bias,compute))
    }

    fn project_copied<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        groups: &AttentionParallelGroups<C,K>,compute: Option<FloatDType>)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let [batch,queries,_] = query.dims();
        let [key_batch,keys,_] = key.dims();
        assert_eq!((key_batch,keys),(value.dims()[0],value.dims()[1]),"parallel projected K/V rows differ");
        assert_eq!(batch,key_batch,"parallel query/memory batch differs");
        let query = projection(query,self.local.query.weight.val(),self.local.query.bias.as_ref().map(|bias|bias.val()),compute);
        let key = self.kv_projection(&self.local.key,key,groups,compute)?;
        let value = self.kv_projection(&self.local.value,value,groups,compute)?;
        Ok((query.reshape([batch,queries,self.local.query_heads,self.local.head_dimension]).swap_dims(1,2),
            key.reshape([batch,keys,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2),
            value.reshape([batch,keys,self.local.kv_heads,self.local.head_dimension]).swap_dims(1,2)))
    }

    /// Actual local head projections; each independent input receives global SUM gradients.
    /// Optional KV replicas SUM only their shared weight/bias gradients, not other heads.
    pub fn project<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        groups: &AttentionParallelGroups<C,K>)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.project_with_compute_dtype_inner(query,key,value,groups,None)
    }

    fn project_with_compute_dtype_inner<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        groups: &AttentionParallelGroups<C,K>,compute: Option<FloatDType>)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let query = if let Some(dtype) = compute {query.cast(dtype)} else {query};
        let key = if let Some(dtype) = compute {key.cast(dtype)} else {key};
        let value = if let Some(dtype) = compute {value.cast(dtype)} else {value};
        let query = region::copy_to_region(query,groups.heads.clone())?;
        let key = region::copy_to_region(key,groups.heads.clone())?;
        let value = region::copy_to_region(value,groups.heads.clone())?;
        self.project_copied(query,key,value,groups,compute)
    }

    /// Explicit projection/input/KV-replica work dtype, with gradients converted back
    /// to original storage only after the corresponding SUM derivatives.
    pub fn project_with_compute_dtype<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        groups: &AttentionParallelGroups<C,K>,dtype: FloatDType)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.project_with_compute_dtype_inner(query,key,value,groups,Some(dtype))
    }

    /// Self-attention shares one copied input node, so its backward reduces the
    /// combined Q/K/V input contribution once rather than three independent times.
    pub fn project_self<C,K>(&self,input: Tensor<Autodiff<B,S>,3>,groups: &AttentionParallelGroups<C,K>)
        -> Result<(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let input = region::copy_to_region(input,groups.heads.clone())?;
        self.project_copied(input.clone(),input.clone(),input,groups,None)
    }

    /// Local masked GQA, output-row SUM with identity backward, then bias exactly once.
    /// Masks and additive bias describe local query heads; use the explicit head-mask
    /// slicing helpers when supplying a replicated full-head parameter tensor.
    pub fn forward_projected<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,4>,key: Tensor<Autodiff<B,S>,4>,value: Tensor<Autodiff<B,S>,4>,
        masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let partial = self.partial(query,key,value,masks,options,None);
        Ok(self.bias(region::reduce_from_region(partial,communicator)?,None))
    }

    /// Generic self/cross attention on caller-owned actual head and KV replica groups.
    pub fn forward<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,groups: &AttentionParallelGroups<C,K>)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let (query,key,value) = self.project(query,key,value,groups)?;
        self.forward_projected(query,key,value,masks,options,groups.heads.clone())
    }

    /// Self attention with a single input-gradient SUM and no global head gather.
    pub fn forward_self<C,K>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,
        options: DenseAttentionOptions,groups: &AttentionParallelGroups<C,K>) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let (query,key,value) = self.project_self(input,groups)?;
        self.forward_projected(query,key,value,masks,options,groups.heads.clone())
    }

    /// Explicit QKV/score/output work storage; final output returns to query storage.
    /// Parameters are not detached, converted into new leaves or silently merged.
    pub fn forward_with_compute_dtype<C,K>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,groups: &AttentionParallelGroups<C,K>,dtype: FloatDType)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let storage = query.dtype();
        let (query,key,value) = self.project_with_compute_dtype(query,key,value,groups,dtype)?;
        let partial = self.partial(query,key,value,masks,options,Some(dtype));
        Ok(self.bias(region::reduce_from_region(partial,groups.heads.clone())?,Some(dtype)).cast(storage))
    }
}
