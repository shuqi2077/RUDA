use ruda_model::{module::Module,tensor::{Bool,FloatDType,Int,Tensor,backend::Backend}};
use crate::{Dropout,attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::TransformerKvCache};
use super::{AwqTransformerProjection,AwqTransformerStack,DenseTransformerNorm,TransformerEmbeddings,SequenceHeadOutput};
use super::{TransformerProjectionShape,TransformerProjection};
use crate::loss::{CausalCrossEntropyConfig,CausalLoss};
use crate::pool::{pool_sequence,pool_packed_sequences,SequencePooling,SequencePoolOutput};

/// Native original output normalization/dropout plus explicit dense or packed projection.
#[derive(Module,Debug)]
pub struct AwqTransformerHead<B:Backend,P:Module<B>=AwqTransformerProjection<B>> {
    /// Original actual projection, including explicitly selected adapters.
    pub projection:P,
    /// Original optional pre-projection affine norm.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Original head-input dropout, independent of adapter-input dropout.
    pub dropout:Dropout,
}
impl<B:Backend,P:TransformerProjectionShape<B>> AwqTransformerHead<B,P> {
    /// Connect loaded actual projection and explicit head options without initialization.
    pub fn from_projection(projection:P,normalization:Option<DenseTransformerNorm<B>>,dropout:Dropout) -> Self {
        if let Some(norm)=&normalization {assert_eq!(norm.width(),projection.dimensions()[0],"head norm/projection input width differs");}
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid head dropout");
        Self {projection,normalization,dropout}
    }
}
impl<B:Backend,P:TransformerProjection<B>> AwqTransformerHead<B,P> {
    /// Project actual hidden rows to logits, without inferred targets or loss masks.
    pub fn forward<const D:usize>(&self,hidden:Tensor<B,D>) -> Result<Tensor<B,D>,P::Error> {
        let hidden=if let Some(norm)=&self.normalization {norm.forward(hidden)} else {hidden};
        self.projection.forward(self.dropout.forward(hidden))
    }
    /// Project already pooled actual rows, retaining original visibility/count metadata.
    pub fn forward_pooled(&self,pooled:SequencePoolOutput<B>) -> Result<SequenceHeadOutput<B>,P::Error> {
        Ok(SequenceHeadOutput {logits:self.forward(pooled.values)?,valid_rows:pooled.valid_rows,token_counts:pooled.token_counts})
    }
    /// Reuse original explicit native pooling for sequence-level fine tuning.
    pub fn forward_sequence(&self,hidden:Tensor<B,3>,visible:Tensor<B,2,Bool>,pooling:SequencePooling)
        -> Result<SequenceHeadOutput<B>,P::Error> {self.forward_pooled(pool_sequence(hidden,visible,pooling))}
    /// Reuse original independent-document pooling, including empty-row/count metadata.
    pub fn forward_packed_sequences(&self,hidden:Tensor<B,2>,layout:&PackedSequenceLayout,visible:Option<Tensor<B,1,Bool>>,pooling:SequencePooling)
        -> Result<SequenceHeadOutput<B>,P::Error> {self.forward_pooled(pool_packed_sequences(hidden,layout,visible,pooling))}
}

/// Complete native loaded embeddings, mixed packed/dense backbone, final norm and logits.
/// Tokenization, positional transforms and target/loss policy remain explicit caller inputs.
#[derive(Module,Debug)]
pub struct AwqTransformerModel<B:Backend,P:Module<B>=AwqTransformerProjection<B>> {
    /// Actual original input tables, optional learned positions/types and input dropout.
    pub embeddings:TransformerEmbeddings<B>,
    /// Every actual original mixed-projection block in its loaded order.
    pub backbone:AwqTransformerStack<B,P>,
    /// Original optional final affine norm, independent of any head norm.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Actual original output head, including its independent projection choices.
    pub head:AwqTransformerHead<B,P>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> AwqTransformerModel<B,P> {
    /// Assemble actual caller-loaded components, retaining existing IDs and flags.
    pub fn from_parts(embeddings:TransformerEmbeddings<B>,backbone:AwqTransformerStack<B,P>,
        normalization:Option<DenseTransformerNorm<B>>,head:AwqTransformerHead<B,P>) -> Self {
        let width=embeddings.token.weight.val().dims()[1];
        assert_eq!(head.projection.dimensions()[0],width,"model input/head width differs");
        if let Some(norm)=&normalization {assert_eq!(norm.width(),width,"model final norm width differs");}
        for block in &backbone.blocks {
            assert_eq!(block.attention.query.dimensions()[0],width,"self-attention input width differs");
            assert_eq!(block.attention.key.dimensions()[0],width,"self-attention key input width differs");
            assert_eq!(block.attention.value.dimensions()[0],width,"self-attention value input width differs");
            assert_eq!(block.attention.output.dimensions()[1],width,"self-attention residual width differs");
            assert_eq!((block.feed_forward.up.dimensions()[0],block.feed_forward.down.dimensions()[1]),(width,width),"FFN residual width differs");
            assert_eq!((block.attention_norm.width(),block.feed_forward_norm.width()),(width,width),"block normalization width differs");
        }
        Self {embeddings,backbone,normalization,head}
    }
    /// Original native cache topology for the actual loaded backbone.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {self.backbone.new_kv_cache(initial_capacity)}
    fn embed(&self,tokens:Tensor<B,2,Int>,positions:Option<Tensor<B,2,Int>>,token_types:Option<Tensor<B,2,Int>>,
        dtypes:Option<(FloatDType,FloatDType)>) -> Tensor<B,3> {
        match dtypes {Some((compute,output))=>self.embeddings.forward_with_compute_dtype(tokens,positions,token_types,compute,output),
            None=>self.embeddings.forward(tokens,positions,token_types)}
    }
    fn normalize<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {
        if let Some(norm)=&self.normalization {norm.forward(hidden)} else {hidden}
    }
}
impl<B:Backend,P:TransformerProjection<B>> AwqTransformerModel<B,P> {
    /// Actual complete dense-axis graph from explicit native IDs to logits.
    /// Explicit embedding work/output dtypes enable mixed-storage tables without a dense shadow.
    pub fn forward_with_positions<F>(&self,tokens:Tensor<B,2,Int>,positions:Option<Tensor<B,2,Int>>,token_types:Option<Tensor<B,2,Int>>,
        embedding_dtypes:Option<(FloatDType,FloatDType)>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,projected_positions:F)
        -> Result<Tensor<B,3>,P::Error> where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.forward_hidden_with_positions(tokens,positions,token_types,embedding_dtypes,masks,options,projected_positions)?;
        self.head.forward(hidden)
    }
    /// Actual final hidden states for chunked logits or sequence heads, without constructing logits.
    pub fn forward_hidden_with_positions<F>(&self,tokens:Tensor<B,2,Int>,positions:Option<Tensor<B,2,Int>>,token_types:Option<Tensor<B,2,Int>>,
        embedding_dtypes:Option<(FloatDType,FloatDType)>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,projected_positions:F)
        -> Result<Tensor<B,3>,P::Error> where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.embed(tokens,positions,token_types,embedding_dtypes);
        let hidden=self.backbone.forward_with_positions(hidden,masks,options,projected_positions)?;
        Ok(self.normalize(hidden))
    }
    /// Actual flat document rows from explicit native token/table-position/type IDs.
    /// No learned position IDs, padding rows or document-boundary loss rules are invented.
    pub fn forward_packed_with_positions<F>(&self,tokens:Tensor<B,1,Int>,positions:Option<Tensor<B,1,Int>>,token_types:Option<Tensor<B,1,Int>>,
        embedding_dtypes:Option<(FloatDType,FloatDType)>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,
        options:PackedAttentionOptions,projected_positions:F) -> Result<Tensor<B,2>,P::Error>
        where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let hidden=self.forward_packed_hidden_with_positions(tokens,positions,token_types,embedding_dtypes,layout,masks,options,projected_positions)?;
        self.head.forward(hidden)
    }
    /// Actual packed final hidden states for explicit chunked logits or document pooling.
    pub fn forward_packed_hidden_with_positions<F>(&self,tokens:Tensor<B,1,Int>,positions:Option<Tensor<B,1,Int>>,token_types:Option<Tensor<B,1,Int>>,
        embedding_dtypes:Option<(FloatDType,FloatDType)>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<B>]>,
        options:PackedAttentionOptions,projected_positions:F) -> Result<Tensor<B,2>,P::Error>
        where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let count=layout.tokens();assert_eq!(tokens.dims(),[count],"packed IDs/document boundaries differ");
        for ids in positions.iter().chain(token_types.iter()) {assert_eq!(ids.dims(),[count],"packed optional ID shape differs");}
        let width=self.embeddings.token.weight.val().dims()[1];
        let hidden=self.embed(tokens.reshape([1,count]),positions.map(|ids|ids.reshape([1,count])),
            token_types.map(|ids|ids.reshape([1,count])),embedding_dtypes).reshape([count,width]);
        let hidden=self.backbone.forward_packed_with_positions(hidden,layout,masks,options,projected_positions)?;
        Ok(self.normalize(hidden))
    }
    /// Original new-token cached inference to logits, without reprocessing old FFN rows.
    pub fn forward_cached_with_positions<F>(&self,tokens:Tensor<B,2,Int>,positions:Option<Tensor<B,2,Int>>,token_types:Option<Tensor<B,2,Int>>,
        embedding_dtypes:Option<(FloatDType,FloatDType)>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut TransformerKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,projected_positions:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.embed(tokens,positions,token_types,embedding_dtypes);
        let hidden=self.backbone.forward_cached_with_positions(hidden,new_visible,cache,masks,options,projected_positions)?;
        self.head.forward(self.normalize(hidden))
    }
    /// Complete original causal fine-tuning graph, projecting only the caller-configured
    /// token chunks. Label shift, ignore index and full-vocabulary smoothing retain
    /// the existing criterion semantics; packed bases are never expanded as model storage.
    pub fn forward_causal_with_positions<F>(&self,tokens:Tensor<B,2,Int>,positions:Option<Tensor<B,2,Int>>,token_types:Option<Tensor<B,2,Int>>,
        embedding_dtypes:Option<(FloatDType,FloatDType)>,labels:Tensor<B,2,Int>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,projected_positions:F) -> Result<CausalLoss<B>,P::Error>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.forward_hidden_with_positions(tokens,positions,token_types,embedding_dtypes,masks,options,projected_positions)?;
        criterion.try_forward_hidden_with_smoothing(hidden,labels,|rows|self.head.forward(rows),label_smoothing)
    }
    /// Complete flat-document fine tuning without cross-document targets or full token logits.
    pub fn forward_packed_causal_with_positions<F>(&self,tokens:Tensor<B,1,Int>,positions:Option<Tensor<B,1,Int>>,token_types:Option<Tensor<B,1,Int>>,
        embedding_dtypes:Option<(FloatDType,FloatDType)>,labels:Tensor<B,1,Int>,layout:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,criterion:&CausalCrossEntropyConfig,
        label_smoothing:f64,projected_positions:F) -> Result<CausalLoss<B>,P::Error>
        where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let hidden=self.forward_packed_hidden_with_positions(tokens,positions,token_types,embedding_dtypes,layout,masks,options,projected_positions)?;
        criterion.try_forward_packed_hidden_with_smoothing(hidden,labels,layout,|rows|self.head.forward(rows),label_smoothing)
    }
}
