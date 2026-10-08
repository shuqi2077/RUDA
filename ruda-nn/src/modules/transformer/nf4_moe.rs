use alloc::vec::Vec;
use core::fmt;
use ruda_model::{module::Module,tensor::{Tensor,Int,Bool,MoeDispatchOps,backend::Backend}};
use crate::{Nf4MoeLayer,Nf4MoeError,NativeMoeLayerError,FrozenNf4SwiGluExperts,FrozenPackedSwiGluExperts,FrozenExpertGeometry,FrozenSelectedExperts,Dropout,
    attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::{ProjectedKvCache,TransformerKvCache},loss::{CausalCrossEntropyConfig,CausalLoss},pool::SequencePooling};
use super::{TransformerProjectionShape,TransformerProjection,ProjectedGroupedQueryAttention,ProjectedFeedForward,ProjectedTransformerBlock,
    NativeMoeTransformerBlock,NativeMoeTransformerError,DenseTransformerNorm,TransformerEmbeddings,ProjectedTransformerHead,ProjectedTransformerInput,SequenceHeadOutput};
use super::{dense::try_residual_branch,native_attention::{attention_branch,packed_attention_branch,cached_attention_branch},
    projected_paired_model::{embed_projected,embed_packed_projected,check_block}};

/// Original independent projection, floating routed branch or packed routed branch failure.
#[derive(Debug)]
pub enum Nf4MoeTransformerError<P:fmt::Debug,M:fmt::Debug,N:fmt::Debug> {
    /// Actual original attention/shared/head projection failure.
    Projection(P),
    /// Original actual floating-expert layer failure when present in a mixed stack.
    Floating(NativeMoeLayerError<P,M>),
    /// Actual original packed-expert routed branch failure.
    Packed(Nf4MoeError<P,M,N>),
}
impl<P:fmt::Debug,M:fmt::Debug,N:fmt::Debug> fmt::Display for Nf4MoeTransformerError<P,M,N> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {
        Self::Projection(error)=>write!(f,"NF4 transformer projection: {error:?}"),Self::Floating(error)=>write!(f,"floating expert branch: {error}"),
        Self::Packed(error)=>write!(f,"packed expert branch: {error}")}}
}
impl<P:fmt::Debug,M:fmt::Debug,N:fmt::Debug> core::error::Error for Nf4MoeTransformerError<P,M,N> {}
fn floating_error<P:fmt::Debug,M:fmt::Debug,N:fmt::Debug>(error:NativeMoeTransformerError<P,M>) -> Nf4MoeTransformerError<P,M,N> {
    match error {NativeMoeTransformerError::Projection(error)=>Nf4MoeTransformerError::Projection(error),
        NativeMoeTransformerError::Routed(error)=>Nf4MoeTransformerError::Floating(error)}
}
/// Actual native self-attention plus packed routed experts and an explicitly optional shared FFN.
#[derive(Module,Debug)]
pub struct Nf4MoeTransformerBlock<B:Backend,P:Module<B>,E:Module<B> =FrozenNf4SwiGluExperts<B>> {
    /// Actual independent source attention projection choices.
    pub attention:ProjectedGroupedQueryAttention<B,P>,
    /// Actual packed gate/up/down expert branch and original router.
    pub routed:Nf4MoeLayer<B,P,E>,
    /// Explicit optional actual source shared FFN, not an inferred expert gate.
    pub shared:Option<ProjectedFeedForward<B,P>>,
    /// Actual original attention normalization.
    pub attention_norm:DenseTransformerNorm<B>,
    /// Actual independent original FFN normalization.
    pub feed_forward_norm:DenseTransformerNorm<B>,
    /// Original residual-branch dropout.
    pub residual_dropout:Dropout,
    /// Original source pre/post-normalization order.
    pub norm_first:bool,
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:FrozenExpertGeometry<B>> Nf4MoeTransformerBlock<B,P,E> {
    /// Validate actual source residual geometry without decoding expert weights.
    pub fn validate(&self) {
        self.routed.validate();let width=self.routed.width();
        for projection in [&self.attention.query,&self.attention.key,&self.attention.value] {assert_eq!(projection.dimensions()[0],width,"NF4 MoE attention input width differs");}
        assert_eq!(self.attention.output.dimensions()[1],width,"NF4 MoE attention output width differs");
        assert_eq!((self.attention_norm.width(),self.feed_forward_norm.width()),(width,width),"NF4 MoE original norm width differs");
        if let Some(shared)=&self.shared {assert_eq!(shared.up.dimensions()[0],width,"NF4 MoE shared input width differs");
            assert_eq!(shared.down.dimensions()[1],width,"NF4 MoE shared output width differs");}
    }
}
impl<B:MoeDispatchOps,P:TransformerProjection<B>,E:FrozenSelectedExperts<B>> Nf4MoeTransformerBlock<B,P,E> {
    fn feed_forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        try_residual_branch(input,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,|source| {
            let routed=self.routed.forward(source.clone()).map_err(Nf4MoeTransformerError::Packed)?;
            if let Some(shared)=&self.shared {let shared=shared.forward(source).map_err(Nf4MoeTransformerError::Projection)?;
                assert_eq!(shared.dims(),routed.dims(),"NF4 shared/routed actual token axes differ");Ok(routed+shared)}else {Ok(routed)}
        })
    }
    /// Original dense-axis attention/positions followed by actual selected packed/shared branches.
    pub fn forward_with_positions<F>(&self,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.validate();let hidden=attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,masks,options,positions)
            .map_err(Nf4MoeTransformerError::Projection)?;self.feed_forward(hidden)
    }
    /// Exact independent packed-document attention and source-selected packed experts, without padding.
    pub fn forward_packed_with_positions<F>(&self,input:Tensor<B,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F)
        -> Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.validate();let hidden=packed_attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,layout,masks,options,positions)
            .map_err(Nf4MoeTransformerError::Projection)?;self.feed_forward(hidden)
    }
    /// Original KV attention only for actual new rows, then their actual selected packed/shared FFN.
    pub fn forward_cached_with_positions<F>(&self,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.validate();let hidden=cached_attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,visible,cache,masks,options,positions)
            .map_err(Nf4MoeTransformerError::Projection)?;self.feed_forward(hidden)
    }
}
/// Explicit actual ordinary, floating-expert or packed-expert layer, in original loaded order.
#[derive(Module,Debug)]
pub enum Nf4MoeTransformerLayer<B:Backend,P:Module<B>,E:Module<B> =FrozenNf4SwiGluExperts<B>> {
    /// Original actual ordinary/gated source block.
    Dense(ProjectedTransformerBlock<B,P>),
    /// Original actual floating expert source block, without forced quantization.
    Floating(NativeMoeTransformerBlock<B,P>),
    /// Selected native expert implementation; the default retains native NF4 and may explicitly be floating LoRA.
    Packed(Nf4MoeTransformerBlock<B,P,E>),
}

/// Complete original Transformer block with independently selected AWQ/NF4 expert projections.
pub type PackedMoeTransformerBlock<B,P> = Nf4MoeTransformerBlock<B,P,FrozenPackedSwiGluExperts<B>>;
/// Actual loaded ordinary/floating/packed layer choice using the same original model graph.
pub type PackedMoeTransformerLayer<B,P> = Nf4MoeTransformerLayer<B,P,FrozenPackedSwiGluExperts<B>>;
/// Complete mixed packed-expert native token logits, training objectives and KV decoding.
pub type PackedMoeTransformerModel<B,P> = Nf4MoeTransformerModel<B,P,FrozenPackedSwiGluExperts<B>>;
impl<B:Backend,P:TransformerProjectionShape<B>,E:FrozenExpertGeometry<B>> Nf4MoeTransformerLayer<B,P,E> {
    /// Actual residual width, independent of expert storage.
    pub fn width(&self) -> usize {match self {Self::Dense(block)=>block.attention.query.dimensions()[0],
        Self::Floating(block)=>block.feed_forward.routed.width(),Self::Packed(block)=>block.routed.width()}}
}
impl<B:MoeDispatchOps,P:TransformerProjection<B>,E:FrozenSelectedExperts<B>> Nf4MoeTransformerLayer<B,P,E> {
    /// Execute only the original actual dense-axis layer choice.
    pub fn forward_with_positions<F>(&self,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_with_positions(input,masks,options,positions).map_err(Nf4MoeTransformerError::Projection),
            Self::Floating(block)=>block.forward_with_positions(input,masks,options,positions).map_err(floating_error),
            Self::Packed(block)=>block.forward_with_positions(input,masks,options,positions)}
    }
    /// Execute only the actual layer choice with original independent document boundaries.
    pub fn forward_packed_with_positions<F>(&self,input:Tensor<B,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F)
        -> Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        match self {Self::Dense(block)=>block.forward_packed_with_positions(input,layout,masks,options,positions).map_err(Nf4MoeTransformerError::Projection),
            Self::Floating(block)=>block.forward_packed_with_positions(input,layout,masks,options,positions).map_err(floating_error),
            Self::Packed(block)=>block.forward_packed_with_positions(input,layout,masks,options,positions)}
    }
    /// Execute only the actual source new-token layer and retain its original cache semantics.
    pub fn forward_cached_with_positions<F>(&self,input:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_cached_with_positions(input,visible,cache,masks,options,positions).map_err(Nf4MoeTransformerError::Projection),
            Self::Floating(block)=>block.forward_cached_with_positions(input,visible,cache,masks,options,positions).map_err(floating_error),
            Self::Packed(block)=>block.forward_cached_with_positions(input,visible,cache,masks,options,positions)}
    }
}

/// Complete actual token-to-logit native graph with explicitly mixed expert storage.
#[derive(Module,Debug)]
pub struct Nf4MoeTransformerModel<B:Backend,P:Module<B>,E:Module<B> =FrozenNf4SwiGluExperts<B>> {
    /// Original token and optional learned position/type embeddings.
    pub embeddings:TransformerEmbeddings<B>,
    /// Every actual original dense/floating/packed layer in loaded order.
    pub layers:Vec<Nf4MoeTransformerLayer<B,P,E>>,
    /// Original optional independent final affine normalization.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Actual original native vocabulary projection with its norm/dropout.
    pub head:ProjectedTransformerHead<B,P>,
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:FrozenExpertGeometry<B>> Nf4MoeTransformerModel<B,P,E> {
    /// Connect actual source modules and canonical parameters, without synthetic expert cubes.
    pub fn from_parts(embeddings:TransformerEmbeddings<B>,layers:Vec<Nf4MoeTransformerLayer<B,P,E>>,normalization:Option<DenseTransformerNorm<B>>,head:ProjectedTransformerHead<B,P>) -> Self {
        let width=embeddings.token.weight.val().dims()[1];assert_eq!(head.projection.dimensions()[0],width,"NF4 MoE vocabulary head width differs");
        if let Some(norm)=&normalization {assert_eq!(norm.width(),width,"NF4 MoE final norm width differs");}
        for layer in &layers {assert_eq!(layer.width(),width,"NF4 MoE original residual width differs");match layer {
            Nf4MoeTransformerLayer::Dense(block)=>check_block(block,width),Nf4MoeTransformerLayer::Floating(block)=>block.validate(),Nf4MoeTransformerLayer::Packed(block)=>block.validate()}}
        Self {embeddings,layers,normalization,head}
    }
    /// Actual original per-layer KV topology, independent of weight storage.
    pub fn new_kv_cache(&self,capacity:usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.layers.len(),capacity)}
    fn normalize<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {if let Some(norm)=&self.normalization {norm.forward(hidden)}else {hidden}}
}
impl<B:MoeDispatchOps,P:TransformerProjection<B>,E:FrozenSelectedExperts<B>> Nf4MoeTransformerModel<B,P,E> {
    /// Complete original hidden graph with actual caller-owned per-layer attention/position policy.
    pub fn forward_hidden_with<F>(&self,input:ProjectedTransformerInput<B>,mut layer:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,3>)->Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        let rows=input.tokens.dims();let width=self.embeddings.token.weight.val().dims()[1];let mut hidden=embed_projected(&self.embeddings,input);
        for (index,block) in self.layers.iter().enumerate() {hidden=layer(index,block,hidden)?;assert_eq!(hidden.dims(),[rows[0],rows[1],width],"NF4 MoE layer changed original token axes");}
        Ok(self.normalize(hidden))
    }
    /// Original actual vocabulary logits without a dense quantized base shadow.
    pub fn forward_with<F>(&self,input:ProjectedTransformerInput<B>,layer:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,3>)->Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        self.head.forward(self.forward_hidden_with(input,layer)?).map_err(Nf4MoeTransformerError::Projection)
    }
    /// Exact original causal shift/ignore/smoothing with actual chunked vocabulary projection.
    pub fn forward_causal_with<F>(&self,input:ProjectedTransformerInput<B>,labels:Tensor<B,2,Int>,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,layer:F)
        -> Result<CausalLoss<B>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,3>)->Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        criterion.try_forward_hidden_with_smoothing(self.forward_hidden_with(input,layer)?,labels,
            |rows|self.head.forward(rows).map_err(Nf4MoeTransformerError::Projection),label_smoothing)
    }
    /// Original independent-document hidden graph with exact caller-provided boundaries.
    pub fn forward_packed_hidden_with<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,mut layer:F)
        -> Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,2>)->Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        let shape=[layout.tokens(),self.embeddings.token.weight.val().dims()[1]];let mut hidden=embed_packed_projected(&self.embeddings,input,layout);
        for (index,block) in self.layers.iter().enumerate() {hidden=layer(index,block,hidden)?;assert_eq!(hidden.dims(),shape,"NF4 MoE layer changed original document rows");}
        Ok(self.normalize(hidden))
    }
    /// Actual flat-document vocabulary logits with no padded-token expert work.
    pub fn forward_packed_with<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,layer:F)
        -> Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,2>)->Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        self.head.forward(self.forward_packed_hidden_with(input,layout,layer)?).map_err(Nf4MoeTransformerError::Projection)
    }
    /// Exact packed causal targets and original chunked smoothing, excluding cross-document shifts.
    pub fn forward_packed_causal_with<F>(&self,input:ProjectedTransformerInput<B,1>,labels:Tensor<B,1,Int>,layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,layer:F)
        -> Result<CausalLoss<B>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,2>)->Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        criterion.try_forward_packed_hidden_with_smoothing(self.forward_packed_hidden_with(input,layout,layer)?,labels,layout,
            |rows|self.head.forward(rows).map_err(Nf4MoeTransformerError::Projection),label_smoothing)
    }
    /// Complete actual dense token logits with explicit original per-layer positions.
    pub fn forward_with_positions<F>(&self,input:ProjectedTransformerInput<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,mut positions:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_with(input,|index,layer,hidden|layer.forward_with_positions(hidden,masks.clone(),options,|query,key|positions(index,query,key)))
    }
    /// Complete actual packed token logits with independent source document masks/positions.
    pub fn forward_packed_with_positions<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,mut positions:F)
        -> Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_packed_with(input,layout,|index,layer,hidden|layer.forward_packed_with_positions(hidden,layout,masks,options,|query,key|positions(index,query,key)))
    }
    /// Complete chunked native training objective with original actual dense positions/masks.
    pub fn forward_causal_with_positions<F>(&self,input:ProjectedTransformerInput<B>,labels:Tensor<B,2,Int>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,mut positions:F) -> Result<CausalLoss<B>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_causal_with(input,labels,criterion,label_smoothing,
            |index,layer,hidden|layer.forward_with_positions(hidden,masks.clone(),options,|query,key|positions(index,query,key)))
    }
    /// Complete packed native training graph with exact document boundaries and original positions.
    pub fn forward_packed_causal_with_positions<F>(&self,input:ProjectedTransformerInput<B,1>,labels:Tensor<B,1,Int>,layout:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,mut positions:F)
        -> Result<CausalLoss<B>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_packed_causal_with(input,labels,layout,criterion,label_smoothing,
            |index,layer,hidden|layer.forward_packed_with_positions(hidden,layout,masks,options,|query,key|positions(index,query,key)))
    }
    /// Original visible-token sequence pooling with actual native head and source-selected policy.
    pub fn forward_sequence_with<F>(&self,input:ProjectedTransformerInput<B>,visible:Tensor<B,2,Bool>,pooling:SequencePooling,layer:F)
        -> Result<SequenceHeadOutput<B>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,3>)->Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        self.head.forward_sequence(self.forward_hidden_with(input,layer)?,visible,pooling).map_err(Nf4MoeTransformerError::Projection)
    }
    /// Original independent-document sequence pooling, retaining true empty-row/token metadata.
    pub fn forward_packed_sequences_with<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,visible:Option<Tensor<B,1,Bool>>,pooling:SequencePooling,layer:F)
        -> Result<SequenceHeadOutput<B>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,&Nf4MoeTransformerLayer<B,P,E>,Tensor<B,2>)->Result<Tensor<B,2>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>> {
        self.head.forward_packed_sequences(self.forward_packed_hidden_with(input,layout,layer)?,layout,visible,pooling).map_err(Nf4MoeTransformerError::Projection)
    }
    /// Original new-token KV execution and complete stack boundary; partial errors
    /// retain the same caller-owned cache restore contract as the native model.
    pub fn forward_cached_with_positions<F>(&self,input:ProjectedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,mut positions:F)
        -> Result<Tensor<B,3>,Nf4MoeTransformerError<P::Error,B::MoeError,E::Error>>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        cache.validate_layers(self.layers.len());let mut hidden=embed_projected(&self.embeddings,input);let rows=(hidden.dims()[0],hidden.dims()[1]);
        let next=cache.position().checked_add(rows.1).expect("NF4 MoE cached position overflows");
        for (index,layer) in self.layers.iter().enumerate() {hidden=layer.forward_cached_with_positions(hidden,visible.clone(),&mut cache.layers_mut()[index],masks.clone(),options,
            |query,key,position|positions(index,query,key,position))?;assert_eq!((hidden.dims()[0],hidden.dims()[1]),rows,"NF4 MoE cached layer changed actual new rows");}
        cache.finish_chunk(next);self.head.forward(self.normalize(hidden)).map_err(Nf4MoeTransformerError::Projection)
    }
}
