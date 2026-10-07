use ruda_model::tensor::{Tensor,FloatDType,backend::Backend};
use crate::transformer::AdaptedGroupedQueryAttention;
use super::{GroupedQueryAttention,PackedSequenceLayout,PackedAttentionOptions,packed_scaled_dot_product_attention};
use super::{PackedDocumentAttentionMask,packed_scaled_dot_product_attention_masked};

fn heads<B: Backend>(query: Tensor<B,2>,key: Tensor<B,2>,value: Tensor<B,2>,
    query_heads: usize,kv_heads: usize,width: usize) -> (Tensor<B,3>,Tensor<B,3>,Tensor<B,3>) {
    assert!(query_heads > 0 && kv_heads > 0 && width > 0 && query_heads.is_multiple_of(kv_heads),"invalid packed projection head geometry");
    let [queries,query_width] = query.dims();
    let [keys,key_width] = key.dims();
    assert_eq!(query_heads.checked_mul(width),Some(query_width),"packed query projection width differs");
    assert_eq!(kv_heads.checked_mul(width),Some(key_width),"packed key projection width differs");
    assert_eq!(value.dims(),[keys,key_width],"packed key/value projection geometry differs");
    (query.reshape([queries,query_heads,width]),key.reshape([keys,kv_heads,width]),value.reshape([keys,kv_heads,width]))
}

fn context<B: Backend>(query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
    query_heads: usize,kv_heads: usize,width: usize,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
    options: PackedAttentionOptions,dropout: &crate::Dropout) -> Tensor<B,2> {
    let [queries,heads,features] = query.dims();
    assert_eq!((heads,features),(query_heads,width),"packed projected query geometry differs");
    assert_eq!((key.dims()[1],key.dims()[2]),(kv_heads,width),"packed projected key geometry differs");
    assert_eq!((value.dims()[1],value.dims()[2]),(kv_heads,width),"packed projected value geometry differs");
    packed_scaled_dot_product_attention(query,key,value,query_layout,key_layout,options,Some(dropout))
        .reshape([queries,query_heads.checked_mul(width).expect("packed context width overflow")])
}

impl<B: Backend> GroupedQueryAttention<B> {
    /// Actual per-document masks/bias and the original native output projection.
    pub fn forward_packed_masked_projected(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: &[PackedDocumentAttentionMask<B>],
        options: PackedAttentionOptions) -> Tensor<B,2> {
        let [queries,heads,width] = query.dims();
        assert_eq!((heads,width),(self.query_heads,self.head_dimension),"masked packed query heads differ");
        assert_eq!((key.dims()[1],key.dims()[2]),(self.kv_heads,self.head_dimension),"masked packed key heads differ");
        assert_eq!((value.dims()[1],value.dims()[2]),(self.kv_heads,self.head_dimension),"masked packed value heads differ");
        let context = packed_scaled_dot_product_attention_masked(query,key,value,query_layout,key_layout,masks,options,Some(&self.dropout));
        self.output.forward(context.reshape([queries,heads*width]))
    }

    /// Actual flat [tokens,width] projections -> [tokens,heads,head_dimension].
    /// Query and memory token counts may differ; boundaries are passed at attention.
    pub fn project_packed(&self,query: Tensor<B,2>,key: Tensor<B,2>,value: Tensor<B,2>)
        -> (Tensor<B,3>,Tensor<B,3>,Tensor<B,3>) {
        heads(self.query.forward(query),self.key.forward(key),self.value.forward(value),
            self.query_heads,self.kv_heads,self.head_dimension)
    }

    /// Explicit projection arithmetic storage with gradients to original parameter leaves.
    pub fn project_packed_with_compute_dtype(&self,query: Tensor<B,2>,key: Tensor<B,2>,value: Tensor<B,2>,dtype: FloatDType)
        -> (Tensor<B,3>,Tensor<B,3>,Tensor<B,3>) {
        let [queries,query_width] = query.dims();
        let [keys,key_width] = key.dims();
        let [values,value_width] = value.dims();
        let (query,key,value) = self.project_with_compute_dtype(query.reshape([1,queries,query_width]),
            key.reshape([1,keys,key_width]),value.reshape([1,values,value_width]),dtype);
        (query.swap_dims(1,2).reshape([queries,self.query_heads,self.head_dimension]),
            key.swap_dims(1,2).reshape([keys,self.kv_heads,self.head_dimension]),
            value.swap_dims(1,2).reshape([values,self.kv_heads,self.head_dimension]))
    }

    /// Attend corresponding actual documents, then apply the actual output projection.
    /// No global packed N-by-N mask; the existing per-document FP32 attention is reused.
    pub fn forward_packed_projected(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        self.output.forward(context(query,key,value,self.query_heads,self.kv_heads,self.head_dimension,
            query_layout,key_layout,options,&self.dropout))
    }

    /// Packed self/cross attention with explicit actual query/memory document boundaries.
    pub fn forward_packed(&self,query: Tensor<B,2>,key: Tensor<B,2>,value: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        let (query,key,value) = self.project_packed(query,key,value);
        self.forward_packed_projected(query,key,value,query_layout,key_layout,options)
    }
}

impl<B: Backend> AdaptedGroupedQueryAttention<B> {
    /// Masked independent-document attention with actual A/B gradients and score bias.
    pub fn forward_packed_masked_projected(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: &[PackedDocumentAttentionMask<B>],
        options: PackedAttentionOptions) -> Tensor<B,2> {
        let [queries,heads,width] = query.dims();
        assert_eq!((heads,width),(self.query_heads,self.head_dimension),"masked adapted packed query heads differ");
        assert_eq!((key.dims()[1],key.dims()[2]),(self.kv_heads,self.head_dimension),"masked adapted packed key heads differ");
        assert_eq!((value.dims()[1],value.dims()[2]),(self.kv_heads,self.head_dimension),"masked adapted packed value heads differ");
        let context = packed_scaled_dot_product_attention_masked(query,key,value,query_layout,key_layout,masks,options,Some(&self.dropout));
        self.output.forward(context.reshape([queries,heads*width]))
    }

    /// Actual flat dense/LoRA projections -> packed heads, with original A/B gradients.
    pub fn project_packed(&self,query: Tensor<B,2>,key: Tensor<B,2>,value: Tensor<B,2>)
        -> (Tensor<B,3>,Tensor<B,3>,Tensor<B,3>) {
        heads(self.query.forward(query),self.key.forward(key),self.value.forward(value),
            self.query_heads,self.kv_heads,self.head_dimension)
    }

    /// Independent-document grouped attention plus the actual dense/LoRA output layer.
    pub fn forward_packed_projected(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        self.output.forward(context(query,key,value,self.query_heads,self.kv_heads,self.head_dimension,
            query_layout,key_layout,options,&self.dropout))
    }

    /// Adapted packed self/cross attention; no label, separator or padding inference.
    pub fn forward_packed(&self,query: Tensor<B,2>,key: Tensor<B,2>,value: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        let (query,key,value) = self.project_packed(query,key,value);
        self.forward_packed_projected(query,key,value,query_layout,key_layout,options)
    }
}
