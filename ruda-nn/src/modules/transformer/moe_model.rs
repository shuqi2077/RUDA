use ruda_model::{module::Module,tensor::{Bool,Int,MoeOps,Tensor,backend::Backend}};
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::TransformerKvCache,loss::{CausalCrossEntropyConfig,CausalLoss},pool::SequencePooling};
use super::{TransformerProjectionShape,TransformerProjection,TransformerEmbeddings,DenseTransformerNorm,ProjectedTransformerHead,
    ProjectedTransformerInput,NativeMoeTransformerLayer,NativeMoeTransformerStack,NativeMoeTransformerError,SequenceHeadOutput};
use super::projected_paired_model::{embed_projected,embed_packed_projected,check_block};

/// Complete loaded native token-to-logit graph with the actual dense/routed layer order.
#[derive(Module,Debug)]
pub struct NativeMoeTransformerModel<B:Backend,P:Module<B>> {
    /// Original native token and optional learned-position/type tables.
    pub embeddings:TransformerEmbeddings<B>,
    /// Original explicit dense or routed/shared branch for every layer.
    pub backbone:NativeMoeTransformerStack<B,P>,
    /// Original optional final affine normalization, independent of the head norm.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Actual native output projection with its original dropout and optional norm.
    pub head:ProjectedTransformerHead<B,P>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> NativeMoeTransformerModel<B,P> {
    /// Connect actual caller-loaded parts without initialization, source quantization or guessed architecture.
    pub fn from_parts(embeddings:TransformerEmbeddings<B>,backbone:NativeMoeTransformerStack<B,P>,
        normalization:Option<DenseTransformerNorm<B>>,head:ProjectedTransformerHead<B,P>) -> Self {
        let width=embeddings.token.weight.val().dims()[1];assert_eq!(head.projection.dimensions()[0],width,"native MoE head/input width differs");
        if let Some(norm)=&normalization {assert_eq!(norm.width(),width,"native MoE final norm width differs");}
        for layer in &backbone.layers {assert_eq!(layer.width(),width,"native MoE residual width differs");
            match layer {NativeMoeTransformerLayer::Dense(block)=>check_block(block,width),NativeMoeTransformerLayer::Routed(block)=>block.validate()}}
        Self {embeddings,backbone,normalization,head}
    }
    /// Allocate only the original per-layer KV cache topology, not model replicas.
    pub fn new_kv_cache(&self,capacity:usize) -> TransformerKvCache<B> {self.backbone.new_kv_cache(capacity)}
    fn normalize<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {
        if let Some(norm)=&self.normalization {norm.forward(hidden)} else {hidden}
    }
}
impl<B:MoeOps,P:TransformerProjection<B>> NativeMoeTransformerModel<B,P> {
    /// Complete original hidden graph with caller-owned per-layer attention/position policies.
    pub fn forward_hidden_with<F>(&self,input:ProjectedTransformerInput<B>,mut layer:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        let rows=input.tokens.dims();let width=self.embeddings.token.weight.val().dims()[1];let mut hidden=embed_projected(&self.embeddings,input);
        for (index,block) in self.backbone.layers.iter().enumerate() {hidden=layer(index,block,hidden)?;
            assert_eq!(hidden.dims(),[rows[0],rows[1],width],"native MoE layer changed original token axes");}Ok(self.normalize(hidden))
    }
    /// Complete actual native token logits, preserving original base and adapter storage.
    pub fn forward_with<F>(&self,input:ProjectedTransformerInput<B>,layer:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        self.head.forward(self.forward_hidden_with(input,layer)?).map_err(NativeMoeTransformerError::Projection)
    }
    /// Complete causal training with the original explicit shift, ignore-index and full-vocabulary smoothing.
    /// Only the configured token chunks are projected; the full token/logit matrix is not retained.
    pub fn forward_causal_with<F>(&self,input:ProjectedTransformerInput<B>,labels:Tensor<B,2,Int>,criterion:&CausalCrossEntropyConfig,
        label_smoothing:f64,layer:F) -> Result<CausalLoss<B>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        criterion.try_forward_hidden_with_smoothing(self.forward_hidden_with(input,layer)?,labels,
            |rows|self.head.forward(rows).map_err(NativeMoeTransformerError::Projection),label_smoothing)
    }
    /// Actual flat-document hidden graph, retaining exact document boundaries and caller-owned policies.
    pub fn forward_packed_hidden_with<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,mut layer:F)
        -> Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,2>)->Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        let shape=[layout.tokens(),self.embeddings.token.weight.val().dims()[1]];let mut hidden=embed_packed_projected(&self.embeddings,input,layout);
        for (index,block) in self.backbone.layers.iter().enumerate() {hidden=layer(index,block,hidden)?;
            assert_eq!(hidden.dims(),shape,"native packed MoE layer changed original document rows");}Ok(self.normalize(hidden))
    }
    /// Actual independent-document logits without adding padding or inferring learned positions.
    pub fn forward_packed_with<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,layer:F)
        -> Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,2>)->Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        self.head.forward(self.forward_packed_hidden_with(input,layout,layer)?).map_err(NativeMoeTransformerError::Projection)
    }
    /// Original chunked packed objective, excluding cross-document shifted targets and ignored labels.
    pub fn forward_packed_causal_with<F>(&self,input:ProjectedTransformerInput<B,1>,labels:Tensor<B,1,Int>,layout:&PackedSequenceLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,layer:F) -> Result<CausalLoss<B>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,2>)->Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        criterion.try_forward_packed_hidden_with_smoothing(self.forward_packed_hidden_with(input,layout,layer)?,labels,layout,
            |rows|self.head.forward(rows).map_err(NativeMoeTransformerError::Projection),label_smoothing)
    }
    /// Original explicit visible-token pooling for sequence-level native fine tuning.
    pub fn forward_sequence_with<F>(&self,input:ProjectedTransformerInput<B>,visible:Tensor<B,2,Bool>,pooling:SequencePooling,layer:F)
        -> Result<SequenceHeadOutput<B>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        self.head.forward_sequence(self.forward_hidden_with(input,layer)?,visible,pooling).map_err(NativeMoeTransformerError::Projection)
    }
    /// Original independent-document pooling with actual empty-row/token-count metadata.
    pub fn forward_packed_sequences_with<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,
        visible:Option<Tensor<B,1,Bool>>,pooling:SequencePooling,layer:F) -> Result<SequenceHeadOutput<B>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,&NativeMoeTransformerLayer<B,P>,Tensor<B,2>)->Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>> {
        self.head.forward_packed_sequences(self.forward_packed_hidden_with(input,layout,layer)?,layout,visible,pooling).map_err(NativeMoeTransformerError::Projection)
    }
    /// Complete original dense graph with explicit actual self-attention positions.
    pub fn forward_with_positions<F>(&self,input:ProjectedTransformerInput<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,mut positions:F)
        -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_with(input,|index,layer,hidden|layer.forward_with_positions(hidden,masks.clone(),options,|query,key|positions(index,query,key)))
    }
    /// Complete chunked objective with explicit actual native per-layer positions.
    pub fn forward_causal_with_positions<F>(&self,input:ProjectedTransformerInput<B>,labels:Tensor<B,2,Int>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,mut positions:F) -> Result<CausalLoss<B>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_causal_with(input,labels,criterion,label_smoothing,
            |index,layer,hidden|layer.forward_with_positions(hidden,masks.clone(),options,|query,key|positions(index,query,key)))
    }
    /// Actual packed native logits with exact independent-document masks and positions.
    pub fn forward_packed_with_positions<F>(&self,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,
        options:PackedAttentionOptions,mut positions:F) -> Result<Tensor<B,2>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_packed_with(input,layout,|index,layer,hidden|layer.forward_packed_with_positions(hidden,layout,masks,options,|query,key|positions(index,query,key)))
    }
    /// Actual packed chunked native training graph without cross-document targets.
    pub fn forward_packed_causal_with_positions<F>(&self,input:ProjectedTransformerInput<B,1>,labels:Tensor<B,1,Int>,layout:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,mut positions:F)
        -> Result<CausalLoss<B>,NativeMoeTransformerError<P::Error,B::MoeError>> where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_packed_causal_with(input,labels,layout,criterion,label_smoothing,
            |index,layer,hidden|layer.forward_packed_with_positions(hidden,layout,masks,options,|query,key|positions(index,query,key)))
    }
    /// Original cache processes only actual new-token rows and commits a complete stack boundary.
    pub fn forward_cached_with_positions<F>(&self,input:ProjectedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,cache:&mut TransformerKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F) -> Result<Tensor<B,3>,NativeMoeTransformerError<P::Error,B::MoeError>>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.backbone.forward_cached_with_positions(embed_projected(&self.embeddings,input),visible,cache,masks,options,positions)?;
        self.head.forward(self.normalize(hidden)).map_err(NativeMoeTransformerError::Projection)
    }
}
