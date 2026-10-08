use ruda_model::tensor::{Bool,Tensor,backend::Backend};
use crate::{Dropout,attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},cache::ProjectedKvCache};
use super::{ProjectedGroupedQueryAttention,TransformerProjection,DenseTransformerNorm};
use super::dense::try_residual_branch;

pub(super) fn attention_branch<B:Backend,P:TransformerProjection<B>,F>(attention:&ProjectedGroupedQueryAttention<B,P>,norm:&DenseTransformerNorm<B>,dropout:&Dropout,
    norm_first:bool,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F) -> Result<Tensor<B,3>,P::Error>
    where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
    try_residual_branch(input,norm,dropout,norm_first,|source| {
        let (query,key,value)=attention.project(source.clone(),source.clone(),source)?;
        let shape=(query.dims(),key.dims());let (query,key)=positions(query,key);
        assert_eq!((query.dims(),key.dims()),shape,"positions changed head geometry");attention.forward_projected(query,key,value,masks,options)
    })
}
pub(super) fn packed_attention_branch<B:Backend,P:TransformerProjection<B>,F>(attention:&ProjectedGroupedQueryAttention<B,P>,norm:&DenseTransformerNorm<B>,dropout:&Dropout,
    norm_first:bool,input:Tensor<B,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F)
    -> Result<Tensor<B,2>,P::Error> where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
    assert_eq!(input.dims()[0],layout.tokens(),"packed document boundaries differ from actual rows");
    try_residual_branch(input,norm,dropout,norm_first,|source| {
        let (query,key,value)=attention.project_packed(source.clone(),source.clone(),source)?;
        let shape=(query.dims(),key.dims());let (query,key)=positions(query,key);
        assert_eq!((query.dims(),key.dims()),shape,"packed positions changed geometry");
        if let Some(masks)=masks {attention.forward_packed_masked_projected(query,key,value,layout,layout,masks,options)}
        else {attention.forward_packed_projected(query,key,value,layout,layout,options)}
    })
}
pub(super) fn cached_attention_branch<B:Backend,P:TransformerProjection<B>,F>(attention:&ProjectedGroupedQueryAttention<B,P>,norm:&DenseTransformerNorm<B>,dropout:&Dropout,
    norm_first:bool,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
    -> Result<Tensor<B,3>,P::Error> where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
    try_residual_branch(input,norm,dropout,norm_first,|source|attention.forward_cached_with_positions(source,visible,cache,masks,options,positions))
}
