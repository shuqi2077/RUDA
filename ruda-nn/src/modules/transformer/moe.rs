use alloc::vec::Vec;
use core::fmt;
use ruda_model::{module::Module,tensor::{Bool,MoeOps,Tensor,backend::Backend}};
use crate::{NativeMoeLayer,NativeMoeLayerError,Dropout,attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::{ProjectedKvCache,TransformerKvCache}};
use super::{ProjectedGroupedQueryAttention,ProjectedFeedForward,ProjectedTransformerBlock,DenseTransformerNorm,TransformerProjectionShape,TransformerProjection};
use super::{dense::try_residual_branch,native_attention::{attention_branch,packed_attention_branch,cached_attention_branch}};

/// Original attention/shared projection or actual native routed expert failure.
#[derive(Debug)]
pub enum NativeMoeTransformerError<P:fmt::Debug,M:fmt::Debug> {
    /// Original selected attention/shared/head projection failure.
    Projection(P),
    /// Original router/native expert branch failure.
    Routed(NativeMoeLayerError<P,M>),
}
impl<P:fmt::Debug,M:fmt::Debug> fmt::Display for NativeMoeTransformerError<P,M> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Projection(error)=>write!(f,"native transformer projection: {error:?}"),Self::Routed(error)=>write!(f,"native routed feed-forward: {error}")}
    }
}
impl<P:fmt::Debug,M:fmt::Debug> core::error::Error for NativeMoeTransformerError<P,M> {}

/// Explicit original routed branch plus optional actual shared FFN, without inferred expert gates/scales.
#[derive(Module,Debug)]
pub struct NativeMoeFeedForward<B:Backend,P:Module<B>> {
    /// Original native routed branch with loaded router/expert values.
    pub routed:NativeMoeLayer<B,P>,
    /// Original optional shared ordinary/gated FFN; None retains original absence.
    pub shared:Option<ProjectedFeedForward<B,P>>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> NativeMoeFeedForward<B,P> {
    /// Connect actual original loaded branches. The shared branch adds to routed output only when supplied.
    pub fn from_parts(routed:NativeMoeLayer<B,P>,shared:Option<ProjectedFeedForward<B,P>>) -> Self {
        routed.validate();if let Some(shared)=&shared {assert_eq!(shared.up.dimensions()[0],routed.width(),"shared FFN input width differs");
            assert_eq!(shared.down.dimensions()[1],routed.width(),"shared FFN output width differs");}Self {routed,shared}
    }
}
impl<B:MoeOps,P:TransformerProjection<B>> NativeMoeFeedForward<B,P> {
    /// Run original routed experts and an explicitly present shared branch on the same actual source rows.
    pub fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        let routed=self.routed.forward(input.clone()).map_err(NativeMoeTransformerError::Routed)?;
        if let Some(shared)=&self.shared {let shared=shared.forward(input).map_err(NativeMoeTransformerError::Projection)?;
            assert_eq!(shared.dims(),routed.dims(),"shared/routed output axes differ");Ok(routed+shared)} else {Ok(routed)}
    }
}
/// Actual native attention then MoE FFN, preserving original normalization and residual dropout order.
#[derive(Module,Debug)]
pub struct NativeMoeTransformerBlock<B:Backend,P:Module<B>> {
    /// Original independent dense/adapter/AWQ/NF4 attention projection choices.
    pub attention:ProjectedGroupedQueryAttention<B,P>,
    /// Original actual routed and optional shared FFN branches.
    pub feed_forward:NativeMoeFeedForward<B,P>,
    /// Original actual attention affine norm.
    pub attention_norm:DenseTransformerNorm<B>,
    /// Original independent FFN affine norm.
    pub feed_forward_norm:DenseTransformerNorm<B>,
    /// Original residual-branch dropout.
    pub residual_dropout:Dropout,
    /// Original pre/post-normalization choice.
    pub norm_first:bool,
}
impl<B:Backend,P:TransformerProjectionShape<B>> NativeMoeTransformerBlock<B,P> {
    /// Validate actual original native branch and residual widths without downloading model values.
    pub fn validate(&self) {
        let width=self.feed_forward.routed.width();self.feed_forward.routed.validate();
        for projection in [&self.attention.query,&self.attention.key,&self.attention.value] {assert_eq!(projection.dimensions()[0],width,"MoE self-attention input width differs");}
        assert_eq!(self.attention.output.dimensions()[1],width,"MoE attention output width differs");
        assert_eq!((self.attention_norm.width(),self.feed_forward_norm.width()),(width,width),"MoE original norm width differs");
    }
}
impl<B:MoeOps,P:TransformerProjection<B>> NativeMoeTransformerBlock<B,P> {
    fn feed_forward<const D:usize>(&self,hidden:Tensor<B,D>) -> Result<Tensor<B,D>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        try_residual_branch(hidden,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,|source|self.feed_forward.forward(source))
    }
    /// Original dense-axis complete native block with explicit actual projected positions.
    pub fn forward_with_positions<F>(&self,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.validate();let hidden=attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,masks,options,positions)
            .map_err(NativeMoeTransformerError::Projection)?;self.feed_forward(hidden)
    }
    /// Original independent packed documents/masks followed by the actual native routed/shared FFN.
    pub fn forward_packed_with_positions<F>(&self,input:Tensor<B,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F)
        -> Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.validate();let hidden=packed_attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,layout,masks,options,positions)
            .map_err(NativeMoeTransformerError::Projection)?;self.feed_forward(hidden)
    }
    /// Original cached new-token attention and MoE FFN only on actual new rows.
    pub fn forward_cached_with_positions<F>(&self,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.validate();let hidden=cached_attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,visible,cache,masks,options,positions)
            .map_err(NativeMoeTransformerError::Projection)?;self.feed_forward(hidden)
    }
}
/// Explicit actual per-layer dense FFN or native routed/shared FFN choice.
#[derive(Module,Debug)]
pub enum NativeMoeTransformerLayer<B:Backend,P:Module<B>> {
    /// Original native dense/adapter/packed ordinary/gated FFN block.
    Dense(ProjectedTransformerBlock<B,P>),
    /// Original native routed/shared FFN block.
    Routed(NativeMoeTransformerBlock<B,P>),
}
impl<B:Backend,P:TransformerProjectionShape<B>> NativeMoeTransformerLayer<B,P> {
    /// Original actual residual width, without quantization or expert decoding.
    pub fn width(&self) -> usize {match self {Self::Dense(block)=>block.attention.query.dimensions()[0],Self::Routed(block)=>block.feed_forward.routed.width()}}
}
impl<B:MoeOps,P:TransformerProjection<B>> NativeMoeTransformerLayer<B,P> {
    /// Apply only this original actual layer choice with explicit dense positions.
    pub fn forward_with_positions<F>(&self,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_with_positions(input,masks,options,positions).map_err(NativeMoeTransformerError::Projection),
            Self::Routed(block)=>block.forward_with_positions(input,masks,options,positions)}
    }
    /// Apply only this original actual layer choice on independent packed documents.
    pub fn forward_packed_with_positions<F>(&self,input:Tensor<B,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F)
        -> Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        match self {Self::Dense(block)=>block.forward_packed_with_positions(input,layout,masks,options,positions).map_err(NativeMoeTransformerError::Projection),
            Self::Routed(block)=>block.forward_packed_with_positions(input,layout,masks,options,positions)}
    }
    /// Apply only this original actual cached new-row layer choice.
    pub fn forward_cached_with_positions<F>(&self,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_cached_with_positions(input,visible,cache,masks,options,positions).map_err(NativeMoeTransformerError::Projection),
            Self::Routed(block)=>block.forward_cached_with_positions(input,visible,cache,masks,options,positions)}
    }
}
/// Exact actual dense/MoE layer order with independently selected projection storage.
#[derive(Module,Debug)]
pub struct NativeMoeTransformerStack<B:Backend,P:Module<B>> {
    /// Every actual original native layer in its loaded order.
    pub layers:Vec<NativeMoeTransformerLayer<B,P>>,
}
impl<B:Backend,P:Module<B>> NativeMoeTransformerStack<B,P> {
    /// Original actual native per-layer cache topology.
    pub fn new_kv_cache(&self,capacity:usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.layers.len(),capacity)}
}
impl<B:MoeOps,P:TransformerProjection<B>> NativeMoeTransformerStack<B,P> {
    /// Whole original native dense-axis stack with explicit per-layer positions.
    pub fn forward_with_positions<F>(&self,mut input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,mut positions:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        for (index,layer) in self.layers.iter().enumerate() {input=layer.forward_with_positions(input,masks.clone(),options,|query,key|positions(index,query,key))?;}Ok(input)
    }
    /// Whole original native independent-document stack, without padded rows or global score matrices.
    pub fn forward_packed_with_positions<F>(&self,mut input:Tensor<B,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,mut positions:F)
        -> Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        for (index,layer) in self.layers.iter().enumerate() {input=layer.forward_packed_with_positions(input,layout,masks,options,|query,key|positions(index,query,key))?;}Ok(input)
    }
    /// Original complete new-token boundary; partial errors retain the actual cache restore contract.
    pub fn forward_cached_with_positions<F>(&self,mut input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,mut positions:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        cache.validate_layers(self.layers.len());let rows=(input.dims()[0],input.dims()[1]);let next=cache.position().checked_add(rows.1).expect("native MoE cached position overflows");
        for (index,layer) in self.layers.iter().enumerate() {input=layer.forward_cached_with_positions(input,visible.clone(),&mut cache.layers_mut()[index],masks.clone(),options,
            |query,key,position|positions(index,query,key,position))?;assert_eq!((input.dims()[0],input.dims()[1]),rows,"native cached layer changed actual rows");}
        cache.finish_chunk(next);Ok(input)
    }
}
