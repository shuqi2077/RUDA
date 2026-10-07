use ruda_model::tensor::{Tensor,Bool,backend::Backend};
use super::{PackedSequenceLayout,PackedAttentionOptions,DenseAttentionMask,dense_scaled_dot_product_attention};

/// Actual per-document packed visibility and trainable additive score bias.
#[derive(Clone,Debug)]
pub struct PackedDocumentAttentionMask<B: Backend> {
    /// [queries], True permits a real query row to participate.
    pub query_valid: Option<Tensor<B,1,Bool>>,
    /// [keys], True permits an actual key/value row.
    pub key_valid: Option<Tensor<B,1,Bool>>,
    /// [heads,queries,keys], True allows an edge; singleton broadcasting is explicit.
    pub allowed: Option<Tensor<B,3,Bool>>,
    /// Same broadcast geometry; gradients remain connected to the supplied bias.
    pub bias: Option<Tensor<B,3>>,
}

impl<B: Backend> Default for PackedDocumentAttentionMask<B> {
    fn default() -> Self { Self {query_valid:None,key_valid:None,allowed:None,bias:None} }
}

impl<B: Backend> PackedDocumentAttentionMask<B> {
    fn dense(&self,queries: usize,keys: usize) -> DenseAttentionMask<B> {
        let query_valid = self.query_valid.as_ref().map(|mask| {
            assert_eq!(mask.dims(),[queries],"packed query mask differs from actual document length");
            mask.clone().reshape([1,queries])
        });
        let key_valid = self.key_valid.as_ref().map(|mask| {
            assert_eq!(mask.dims(),[keys],"packed key mask differs from actual document length");
            mask.clone().reshape([1,keys])
        });
        let allowed = self.allowed.as_ref().map(|mask| {
            let [heads,queries,keys] = mask.dims();
            mask.clone().reshape([1,heads,queries,keys])
        });
        let bias = self.bias.as_ref().map(|bias| {
            let [heads,queries,keys] = bias.dims();
            bias.clone().reshape([1,heads,queries,keys])
        });
        DenseAttentionMask {query_valid,key_valid,allowed,bias}
    }
}

/// Corresponding packed documents with explicit masks and differentiable score bias.
///
/// Q/K/V are [tokens,heads,features]. The exact dense masked GQA primitive is
/// applied separately to each actual document; no global packed N-by-N mask,
/// CPU activation execution, synthetic separator, or quantized conversion is used.
/// This is per-document dense attention, not a fused FlashAttention kernel.
pub fn packed_scaled_dot_product_attention_masked<B: Backend>(
    query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
    query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
    masks: &[PackedDocumentAttentionMask<B>],options: PackedAttentionOptions,dropout: Option<&crate::Dropout>,
) -> Tensor<B,3> {
    let [queries,heads,features] = query.dims();
    let [keys,kv_heads,key_features] = key.dims();
    let [values,value_heads,value_features] = value.dims();
    assert_eq!(query_layout.tokens(),queries,"packed query metadata/payload differs");
    assert_eq!(key_layout.tokens(),keys,"packed key metadata/payload differs");
    assert_eq!((values,value_heads,key_features),(keys,kv_heads,features),"packed key/value or QK geometry differs");
    assert_eq!(query_layout.documents(),key_layout.documents(),"paired packed document counts differ");
    assert_eq!(masks.len(),query_layout.documents(),"one explicit mask description per actual document is required");
    if query_layout.documents() == 0 {
        return dense_scaled_dot_product_attention(query.swap_dims(0,1).reshape([1,heads,queries,features]),
            key.swap_dims(0,1).reshape([1,kv_heads,keys,key_features]),
            value.swap_dims(0,1).reshape([1,value_heads,values,value_features]),DenseAttentionMask::default(),options,dropout)
            .reshape([heads,queries,value_features]).swap_dims(0,1);
    }
    let mut outputs = alloc::vec::Vec::with_capacity(query_layout.documents());
    for ((qrange,krange),mask) in query_layout.boundaries().windows(2).zip(key_layout.boundaries().windows(2)).zip(masks) {
        let qlen = qrange[1]-qrange[0];
        let klen = krange[1]-krange[0];
        let q = query.clone().slice_dim(0,qrange[0]..qrange[1]).swap_dims(0,1).reshape([1,heads,qlen,features]);
        let k = key.clone().slice_dim(0,krange[0]..krange[1]).swap_dims(0,1).reshape([1,kv_heads,klen,key_features]);
        let v = value.clone().slice_dim(0,krange[0]..krange[1]).swap_dims(0,1).reshape([1,value_heads,klen,value_features]);
        outputs.push(dense_scaled_dot_product_attention(q,k,v,mask.dense(qlen,klen),options,dropout)
            .reshape([heads,qlen,value_features]).swap_dims(0,1));
    }
    Tensor::cat(outputs,0)
}
