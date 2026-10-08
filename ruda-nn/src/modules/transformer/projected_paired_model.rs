use ruda_model::{module::Module,tensor::{FloatDType,Int,Tensor,backend::Backend}};
use crate::{attention::{PackedSequenceLayout,DenseAttentionMask,DenseAttentionOptions,PackedDocumentAttentionMask,PackedAttentionOptions},
    cache::{EncoderDecoderKvCache,ProjectedKvCache},loss::{CausalCrossEntropyConfig,CausalLoss}};
use super::{TransformerProjectionShape,TransformerProjection,TransformerEmbeddings,DenseTransformerNorm,
    ProjectedTransformerBlock,ProjectedTransformerStack,ProjectedTransformerHead,ProjectedEncoderDecoderLayer,ProjectedEncoderDecoderStack};

/// Actual model input IDs and independent embedding work/output storage; no tokenizer is inferred.
#[derive(Clone,Debug)]
pub struct ProjectedTransformerInput<B:Backend,const D:usize=2> {
    /// Actual dense `[batch,tokens]` or packed `[tokens]` IDs.
    pub tokens:Tensor<B,D,Int>,
    /// Actual optional learned-position table IDs; absence retains original absence.
    pub positions:Option<Tensor<B,D,Int>>,
    /// Actual optional token-type table IDs.
    pub token_types:Option<Tensor<B,D,Int>>,
    /// Explicit `(embedding compute,embedding output)` dtypes; None retains original table policy.
    pub embedding_dtypes:Option<(FloatDType,FloatDType)>,
}
impl<B:Backend,const D:usize> ProjectedTransformerInput<B,D> {
    /// Supply actual token IDs; any required optional table IDs remain explicit caller inputs.
    pub fn new(tokens:Tensor<B,D,Int>) -> Self {Self {tokens,positions:None,token_types:None,embedding_dtypes:None}}
}
pub(super) fn embed_projected<B:Backend>(embeddings:&TransformerEmbeddings<B>,input:ProjectedTransformerInput<B>) -> Tensor<B,3> {
    match input.embedding_dtypes {
        Some((compute,output))=>embeddings.forward_with_compute_dtype(input.tokens,input.positions,input.token_types,compute,output),
        None=>embeddings.forward(input.tokens,input.positions,input.token_types),
    }
}
pub(super) fn embed_packed_projected<B:Backend>(embeddings:&TransformerEmbeddings<B>,input:ProjectedTransformerInput<B,1>,layout:&PackedSequenceLayout) -> Tensor<B,2> {
    let tokens=layout.tokens();assert_eq!(input.tokens.dims(),[tokens],"packed IDs/document boundaries differ");
    for ids in input.positions.iter().chain(input.token_types.iter()) {assert_eq!(ids.dims(),[tokens],"packed optional ID rows differ");}
    let input=ProjectedTransformerInput {tokens:input.tokens.reshape([1,tokens]),positions:input.positions.map(|ids|ids.reshape([1,tokens])),
        token_types:input.token_types.map(|ids|ids.reshape([1,tokens])),embedding_dtypes:input.embedding_dtypes};
    embed_projected(embeddings,input).reshape([tokens,embeddings.token.weight.val().dims()[1]])
}
fn norm<B:Backend,const D:usize>(hidden:Tensor<B,D>,normalization:&Option<DenseTransformerNorm<B>>) -> Tensor<B,D> {
    if let Some(normalization)=normalization {normalization.forward(hidden)} else {hidden}
}
fn check_block<B:Backend,P:TransformerProjectionShape<B>>(block:&ProjectedTransformerBlock<B,P>,width:usize) {
    for projection in [&block.attention.query,&block.attention.key,&block.attention.value] {assert_eq!(projection.dimensions()[0],width,"original self-attention input width differs");}
    assert_eq!(block.attention.output.dimensions()[1],width,"original attention residual width differs");
    assert_eq!((block.feed_forward.up.dimensions()[0],block.feed_forward.down.dimensions()[1]),(width,width),"original FFN residual width differs");
    assert_eq!((block.attention_norm.width(),block.feed_forward_norm.width()),(width,width),"original block norm width differs");
}

/// Complete native paired source encoder/target decoder graph over actual independent packed roles.
#[derive(Module,Debug)]
pub struct ProjectedEncoderDecoderModel<B:Backend,P:Module<B>> {
    /// Original actual independent source token/position/type tables and input norm/dropout.
    pub source_embeddings:TransformerEmbeddings<B>,
    /// Original ordered native source encoder blocks.
    pub encoder:ProjectedTransformerStack<B,P>,
    /// Original optional source encoder final norm.
    pub encoder_normalization:Option<DenseTransformerNorm<B>>,
    /// Original actual independent target input tables.
    pub target_embeddings:TransformerEmbeddings<B>,
    /// Original native self/cross/FFN decoder layer order.
    pub decoder:ProjectedEncoderDecoderStack<B,P>,
    /// Original optional target decoder final norm.
    pub decoder_normalization:Option<DenseTransformerNorm<B>>,
    /// Original actual target vocabulary or sequence-classification head.
    pub head:ProjectedTransformerHead<B,P>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> ProjectedEncoderDecoderModel<B,P> {
    /// Assemble actual loaded native components through explicit independent source/target geometry.
    pub fn from_parts(source_embeddings:TransformerEmbeddings<B>,encoder:ProjectedTransformerStack<B,P>,encoder_normalization:Option<DenseTransformerNorm<B>>,
        target_embeddings:TransformerEmbeddings<B>,decoder:ProjectedEncoderDecoderStack<B,P>,decoder_normalization:Option<DenseTransformerNorm<B>>,
        head:ProjectedTransformerHead<B,P>) -> Self {
        let source=source_embeddings.token.weight.val().dims()[1];let target=target_embeddings.token.weight.val().dims()[1];
        assert_eq!(head.projection.dimensions()[0],target,"original paired head width differs");
        if let Some(norm)=&encoder_normalization {assert_eq!(norm.width(),source,"original encoder final norm width differs");}
        if let Some(norm)=&decoder_normalization {assert_eq!(norm.width(),target,"original decoder final norm width differs");}
        for block in &encoder.blocks {check_block(block,source);}
        for layer in &decoder.layers {
            check_block(&layer.backbone,target);let cross=&layer.cross_attention;
            assert_eq!(cross.query_norm.width(),target,"original cross query norm width differs");
            if let Some(norm)=&cross.memory_norm {assert_eq!(norm.width(),source,"original cross memory norm width differs");}
            assert_eq!(cross.attention.query.dimensions()[0],target,"original cross query width differs");
            assert_eq!(cross.attention.key.dimensions()[0],source,"original cross key width differs");
            assert_eq!(cross.attention.value.dimensions()[0],source,"original cross value width differs");
            assert_eq!(cross.attention.output.dimensions()[1],target,"original cross output width differs");
        }
        Self {source_embeddings,encoder,encoder_normalization,target_embeddings,decoder,decoder_normalization,head}
    }
}
impl<B:Backend,P:TransformerProjection<B>> ProjectedEncoderDecoderModel<B,P> {
    /// Complete actual native paired target logits with independent source/self/cross masks and positions.
    pub fn forward_with_positions<E,F,G>(&self,source:ProjectedTransformerInput<B>,target:ProjectedTransformerInput<B>,source_masks:DenseAttentionMask<B>,
        source_options:DenseAttentionOptions,self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,cross_masks:DenseAttentionMask<B>,
        cross_options:DenseAttentionOptions,mut source_positions:E,mut self_positions:F,mut cross_positions:G) -> Result<Tensor<B,3>,P::Error>
        where E:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),
            G:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_with(source,target,|index,block,hidden|block.forward_with_positions(hidden,source_masks.clone(),source_options,|query,key|source_positions(index,query,key)),
            |index,layer,hidden,memory|layer.forward_with_positions(hidden,memory,self_masks.clone(),self_options,cross_masks.clone(),cross_options,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Complete actual native paired chunked training with independent source/self/cross policies.
    pub fn forward_causal_with_positions<E,F,G>(&self,source:ProjectedTransformerInput<B>,target:ProjectedTransformerInput<B>,labels:Tensor<B,2,Int>,
        source_masks:DenseAttentionMask<B>,source_options:DenseAttentionOptions,self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,
        cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,
        mut source_positions:E,mut self_positions:F,mut cross_positions:G) -> Result<CausalLoss<B>,P::Error>
        where E:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),
            G:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_causal_with(source,target,labels,criterion,label_smoothing,
            |index,block,hidden|block.forward_with_positions(hidden,source_masks.clone(),source_options,|query,key|source_positions(index,query,key)),
            |index,layer,hidden,memory|layer.forward_with_positions(hidden,memory,self_masks.clone(),self_options,cross_masks.clone(),cross_options,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Complete packed native paired target logits with independent source/self/cross document masks.
    pub fn forward_packed_with_positions<E,F,G>(&self,source:ProjectedTransformerInput<B,1>,target:ProjectedTransformerInput<B,1>,source_layout:&PackedSequenceLayout,
        target_layout:&PackedSequenceLayout,source_masks:Option<&[PackedDocumentAttentionMask<B>]>,source_options:PackedAttentionOptions,
        self_masks:Option<&[PackedDocumentAttentionMask<B>]>,self_options:PackedAttentionOptions,cross_masks:Option<&[PackedDocumentAttentionMask<B>]>,
        cross_options:PackedAttentionOptions,mut source_positions:E,mut self_positions:F,mut cross_positions:G) -> Result<Tensor<B,2>,P::Error>
        where E:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),
            G:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_packed_with(source,target,source_layout,target_layout,
            |index,block,hidden|block.forward_packed_with_positions(hidden,source_layout,source_masks,source_options,|query,key|source_positions(index,query,key)),
            |index,layer,hidden,memory|layer.forward_packed_with_positions(hidden,memory,target_layout,source_layout,self_masks,self_options,cross_masks,cross_options,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Complete native packed paired chunked objective, retaining explicit actual target label boundaries.
    pub fn forward_packed_causal_with_positions<E,F,G>(&self,source:ProjectedTransformerInput<B,1>,target:ProjectedTransformerInput<B,1>,labels:Tensor<B,1,Int>,
        source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,source_masks:Option<&[PackedDocumentAttentionMask<B>]>,source_options:PackedAttentionOptions,
        self_masks:Option<&[PackedDocumentAttentionMask<B>]>,self_options:PackedAttentionOptions,cross_masks:Option<&[PackedDocumentAttentionMask<B>]>,cross_options:PackedAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,mut source_positions:E,mut self_positions:F,mut cross_positions:G) -> Result<CausalLoss<B>,P::Error>
        where E:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),
            G:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_packed_causal_with(source,target,labels,source_layout,target_layout,criterion,label_smoothing,
            |index,block,hidden|block.forward_packed_with_positions(hidden,source_layout,source_masks,source_options,|query,key|source_positions(index,query,key)),
            |index,layer,hidden,memory|layer.forward_packed_with_positions(hidden,memory,target_layout,source_layout,self_masks,self_options,cross_masks,cross_options,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Encode actual source rows once, preserving the original memory graph for target backward.
    pub fn encode_with<E>(&self,input:ProjectedTransformerInput<B>,encoder:E) -> Result<Tensor<B,3>,P::Error>
        where E:FnMut(usize,&ProjectedTransformerBlock<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error> {
        let rows=input.tokens.dims();let hidden=self.encoder.forward_with(embed_projected(&self.source_embeddings,input),encoder)?;
        assert_eq!(hidden.dims(),[rows[0],rows[1],self.source_embeddings.token.weight.val().dims()[1]],"encoder changed original source rows");
        Ok(norm(hidden,&self.encoder_normalization))
    }
    /// Decode actual target rows against the supplied native source memory, with no detach or hidden shadow.
    pub fn decode_hidden_with<F>(&self,input:ProjectedTransformerInput<B>,memory:Tensor<B,3>,decoder:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error> {
        let rows=input.tokens.dims();let width=self.target_embeddings.token.weight.val().dims()[1];
        assert_eq!(memory.dims()[0],rows[0],"paired source/target batch rows differ");
        assert_eq!(memory.dims()[2],self.source_embeddings.token.weight.val().dims()[1],"paired memory width differs");
        assert_eq!(memory.device(),input.tokens.device(),"paired source/target devices differ");
        let hidden=self.decoder.forward_with(embed_projected(&self.target_embeddings,input),memory,decoder)?;
        assert_eq!(hidden.dims(),[rows[0],rows[1],width],"decoder changed original target rows");Ok(norm(hidden,&self.decoder_normalization))
    }
    /// Complete actual source/target hidden graph with architecture-owned per-layer policies.
    pub fn forward_hidden_with<E,F>(&self,source:ProjectedTransformerInput<B>,target:ProjectedTransformerInput<B>,encoder:E,decoder:F) -> Result<Tensor<B,3>,P::Error>
        where E:FnMut(usize,&ProjectedTransformerBlock<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error>,
            F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error> {
        self.decode_hidden_with(target,self.encode_with(source,encoder)?,decoder)
    }
    /// Actual complete native target token logits, retaining complete encoder-memory derivatives.
    pub fn forward_with<E,F>(&self,source:ProjectedTransformerInput<B>,target:ProjectedTransformerInput<B>,encoder:E,decoder:F) -> Result<Tensor<B,3>,P::Error>
        where E:FnMut(usize,&ProjectedTransformerBlock<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error>,
            F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error> {
        self.head.forward(self.forward_hidden_with(source,target,encoder,decoder)?)
    }
    /// Full-vocabulary chunked target objective. Already aligned seq2seq labels use the
    /// caller's criterion.shift=false; no extra source/target token or label shift is invented.
    pub fn forward_causal_with<E,F>(&self,source:ProjectedTransformerInput<B>,target:ProjectedTransformerInput<B>,labels:Tensor<B,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,encoder:E,decoder:F) -> Result<CausalLoss<B>,P::Error>
        where E:FnMut(usize,&ProjectedTransformerBlock<B,P>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error>,
            F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error> {
        let hidden=self.forward_hidden_with(source,target,encoder,decoder)?;
        criterion.try_forward_hidden_with_smoothing(hidden,labels,|rows|self.head.forward(rows),label_smoothing)
    }
    /// Complete independent-document native encoder/decoder graph, preserving actual paired boundaries.
    pub fn forward_packed_hidden_with<E,F>(&self,source:ProjectedTransformerInput<B,1>,target:ProjectedTransformerInput<B,1>,source_layout:&PackedSequenceLayout,
        target_layout:&PackedSequenceLayout,encoder:E,decoder:F) -> Result<Tensor<B,2>,P::Error>
        where E:FnMut(usize,&ProjectedTransformerBlock<B,P>,Tensor<B,2>)->Result<Tensor<B,2>,P::Error>,
            F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,P::Error> {
        assert_eq!(source_layout.documents(),target_layout.documents(),"paired packed document counts differ");
        assert_eq!(source.tokens.device(),target.tokens.device(),"paired packed devices differ");
        let memory=self.encoder.forward_with(embed_packed_projected(&self.source_embeddings,source,source_layout),encoder)?;
        assert_eq!(memory.dims(),[source_layout.tokens(),self.source_embeddings.token.weight.val().dims()[1]],"packed encoder changed source rows");
        let memory=norm(memory,&self.encoder_normalization);let hidden=embed_packed_projected(&self.target_embeddings,target,target_layout);
        let hidden=self.decoder.forward_packed_with(hidden,memory,target_layout,source_layout,decoder)?;
        assert_eq!(hidden.dims(),[target_layout.tokens(),self.target_embeddings.token.weight.val().dims()[1]],"packed decoder changed target rows");
        Ok(norm(hidden,&self.decoder_normalization))
    }
    /// Full-vocabulary target objective with actual independent document-local label shifting.
    pub fn forward_packed_causal_with<E,F>(&self,source:ProjectedTransformerInput<B,1>,target:ProjectedTransformerInput<B,1>,labels:Tensor<B,1,Int>,
        source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,encoder:E,decoder:F)
        -> Result<CausalLoss<B>,P::Error>
        where E:FnMut(usize,&ProjectedTransformerBlock<B,P>,Tensor<B,2>)->Result<Tensor<B,2>,P::Error>,
            F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,P::Error> {
        let hidden=self.forward_packed_hidden_with(source,target,source_layout,target_layout,encoder,decoder)?;
        criterion.try_forward_packed_hidden_with_smoothing(hidden,labels,target_layout,|rows|self.head.forward(rows),label_smoothing)
    }
    /// Complete actual packed target token logits, without padding source or target documents.
    pub fn forward_packed_with<E,F>(&self,source:ProjectedTransformerInput<B,1>,target:ProjectedTransformerInput<B,1>,source_layout:&PackedSequenceLayout,
        target_layout:&PackedSequenceLayout,encoder:E,decoder:F) -> Result<Tensor<B,2>,P::Error>
        where E:FnMut(usize,&ProjectedTransformerBlock<B,P>,Tensor<B,2>)->Result<Tensor<B,2>,P::Error>,
            F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,P::Error> {
        self.head.forward(self.forward_packed_hidden_with(source,target,source_layout,target_layout,encoder,decoder)?)
    }
    /// Native only-new-target-row logits over actual previously prepared paired cache state.
    pub fn decode_cached_with<F>(&self,input:ProjectedTransformerInput<B>,cache:&mut EncoderDecoderKvCache<B>,decoder:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,P::Error> {
        let hidden=self.decoder.forward_cached_with(embed_projected(&self.target_embeddings,input),cache,decoder)?;
        self.head.forward(norm(hidden,&self.decoder_normalization))
    }
}
