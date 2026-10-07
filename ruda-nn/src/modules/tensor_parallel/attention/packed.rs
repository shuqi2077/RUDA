use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,TensorParallelGroupedQueryAttention,AttentionParallelGroups,Tensor,projection,region};
use crate::attention::{PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask,
    packed_scaled_dot_product_attention,packed_scaled_dot_product_attention_masked};

fn flat<B: Backend>(query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>) -> (Tensor<B,3>,Tensor<B,3>,Tensor<B,3>) {
    let [batch,heads,tokens,width] = query.dims();
    let [key_batch,kv_heads,keys,key_width] = key.dims();
    assert_eq!((batch,key_batch),(1,1),"packed parallel projections must use the actual flat token axes");
    (query.reshape([heads,tokens,width]).swap_dims(0,1),key.reshape([kv_heads,keys,key_width]).swap_dims(0,1),
        value.reshape([kv_heads,keys,key_width]).swap_dims(0,1))
}

impl<B: Backend> TensorParallelGroupedQueryAttention<B> {
    fn packed_partial(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,query_layout: &PackedSequenceLayout,
        key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions) -> Tensor<B,2> {
        let [tokens,heads,width] = query.dims();
        assert_eq!((heads,width),(self.local.query_heads,self.local.head_dimension),"packed parallel query heads differ");
        assert_eq!((key.dims()[1],key.dims()[2]),(self.local.kv_heads,width),"packed parallel KV head geometry differs");
        let context = if let Some(masks) = masks {
            packed_scaled_dot_product_attention_masked(query,key,value,query_layout,key_layout,masks,options,Some(&self.local.dropout))
        } else {packed_scaled_dot_product_attention(query,key,value,query_layout,key_layout,options,Some(&self.local.dropout))};
        projection(context.reshape([tokens,heads*width]),self.local.output.weight.val(),None,None)
    }

    /// Native packed inference with actual corresponding query/source document boundaries.
    /// None selects the existing unmasked packed primitive; explicit masks preserve local heads.
    pub fn forward_packed_projected_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<B>]>,
        options: PackedAttentionOptions,communicator: C) -> Result<Tensor<B,2>,C::Error> {
        let partial = self.packed_partial(query,key,value,query_layout,key_layout,masks,options);
        let output = communicator.all_reduce_sum(partial.into_primitive().tensor())?;
        Ok(self.bias(Tensor::from_primitive(ruda_model::tensor::TensorPrimitive::Float(output)),None))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelGroupedQueryAttention<Autodiff<B,S>> {
    /// Actual independent flat query/key/value projections into local packed heads.
    pub fn project_packed<C,K>(&self,query: Tensor<Autodiff<B,S>,2>,key: Tensor<Autodiff<B,S>,2>,value: Tensor<Autodiff<B,S>,2>,
        groups: &AttentionParallelGroups<C,K>) -> Result<(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let (queries,query_width) = (query.dims()[0],query.dims()[1]);
        let (keys,key_width) = (key.dims()[0],key.dims()[1]);
        let (values,value_width) = (value.dims()[0],value.dims()[1]);
        let (query,key,value) = self.project(query.reshape([1,queries,query_width]),key.reshape([1,keys,key_width]),value.reshape([1,values,value_width]),groups)?;
        Ok(flat(query,key,value))
    }

    /// Packed self-attention copies the actual common token input only once.
    pub fn project_packed_self<C,K>(&self,input: Tensor<Autodiff<B,S>,2>,groups: &AttentionParallelGroups<C,K>)
        -> Result<(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let [tokens,width] = input.dims();
        let (query,key,value) = self.project_self(input.reshape([1,tokens,width]),groups)?;
        Ok(flat(query,key,value))
    }

    /// Per-document local-head attention followed by output-row SUM and one replicated bias.
    /// Packed boundaries and explicit masks are not replaced with a global dense mask.
    pub fn forward_packed_projected<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,
        options: PackedAttentionOptions,communicator: C) -> Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        let partial = self.packed_partial(query,key,value,query_layout,key_layout,masks,options);
        Ok(self.bias(region::reduce_from_region(partial,communicator)?,None))
    }

    /// Packed generic self/cross attention retaining exact independent document layouts.
    pub fn forward_packed<C,K>(&self,query: Tensor<Autodiff<B,S>,2>,key: Tensor<Autodiff<B,S>,2>,value: Tensor<Autodiff<B,S>,2>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,
        options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let (query,key,value) = self.project_packed(query,key,value,groups)?;
        self.forward_packed_projected(query,key,value,query_layout,key_layout,masks,options,groups.heads.clone())
    }
}
