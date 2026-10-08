use super::*;
use crate::transformer::{TransformerProjection,ProjectedCrossAttentionBlock,ProjectedEncoderDecoderLayer,
    ProjectedEncoderDecoderStack,ProjectedEncoderDecoderModel};
use crate::attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask};
use crate::loss::CausalCrossEntropyConfig;
use ruda_model::tensor::IntegerTensorCollective;
use ruda_model::tensor::Bool;
use crate::cache::{EncoderDecoderKvCache,TransformerKvCache,ProjectedKvCache};
use ruda_autodiff::collective::{CollectiveScope,ScopedTensorCollective};

/// Actual native encoder-memory stage with only local persistent projection/norm leaves.
#[derive(Module,Debug)]
pub struct FullyShardedProjectedCrossAttention<B:Backend,P:Module<B>> {
    /// Original independent query/memory/output projection choices.
    pub attention:FullyShardedProjectedAttention<B,P>,
    /// Original query/residual affine norm slices.
    pub query_norm:FullyShardedTransformerNorm<B>,
    /// Original optional independent memory norm slices.
    pub memory_norm:Option<FullyShardedTransformerNorm<B>>,
    /// Original residual-branch dropout.
    pub residual_dropout:crate::Dropout,
    /// Original pre/post-normalization choice.
    pub norm_first:bool,
}
/// Original self-attention, encoder-memory attention, then FFN with actual local-only storage.
#[derive(Module,Debug)]
pub struct FullyShardedProjectedDecoderLayer<B:Backend,P:Module<B>> {
    /// Original native self-attention and FFN stages.
    pub backbone:FullyShardedProjectedTransformerBlock<B,P>,
    /// Original independently selected encoder-memory stage.
    pub cross_attention:FullyShardedProjectedCrossAttention<B,P>,
}
/// Exact actual native paired-layer order with no complete persistent dense/packed model replica.
#[derive(Module,Debug)]
pub struct FullyShardedProjectedDecoderStack<B:Backend,P:Module<B>> {
    /// Every actual loaded paired layer in its original order.
    pub layers:Vec<FullyShardedProjectedDecoderLayer<B,P>>,
}
/// Complete actual sharded paired source/target graph over original independently selected storage.
#[derive(Module,Debug)]
pub struct FullyShardedProjectedEncoderDecoderModel<B:Backend,P:Module<B>> {
    /// Original source table/norm local leaves and input dropout.
    pub source_embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Original ordered native encoder block local leaves.
    pub encoder:FullyShardedProjectedTransformerStack<B,P>,
    /// Original optional source final affine norm slices.
    pub encoder_normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Original independent target table/norm local leaves.
    pub target_embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Original native three-stage decoder local leaves.
    pub decoder:FullyShardedProjectedDecoderStack<B,P>,
    /// Original optional independent target final affine norm slices.
    pub decoder_normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Original actual output projection/norm/dropout local leaves.
    pub head:FullyShardedProjectedTransformerHead<B,P>,
}
impl<B:Backend> ShardingContext<B> {
    /// Preserve every original query/memory/output role through the same canonical ID context.
    pub fn projected_cross_attention<P:ShardTransformerProjection<B>>(&mut self,cross:ProjectedCrossAttentionBlock<B,P>) -> FullyShardedProjectedCrossAttention<B,P::Sharded> {
        FullyShardedProjectedCrossAttention {attention:self.awq_attention(cross.attention),query_norm:self.normalization(cross.query_norm),
            memory_norm:cross.memory_norm.map(|norm|self.normalization(norm)),residual_dropout:cross.residual_dropout,norm_first:cross.norm_first}
    }
    /// Partition actual loaded three-stage native decoder leaves without changing the original stage order.
    pub fn projected_decoder_layer<P:ShardTransformerProjection<B>>(&mut self,layer:ProjectedEncoderDecoderLayer<B,P>) -> FullyShardedProjectedDecoderLayer<B,P::Sharded> {
        FullyShardedProjectedDecoderLayer {backbone:self.awq_transformer(layer.backbone),cross_attention:self.projected_cross_attention(layer.cross_attention)}
    }
    /// Partition every original decoder layer through this same source/target tie context.
    pub fn projected_decoder_stack<P:ShardTransformerProjection<B>>(&mut self,stack:ProjectedEncoderDecoderStack<B,P>) -> FullyShardedProjectedDecoderStack<B,P::Sharded> {
        FullyShardedProjectedDecoderStack {layers:stack.layers.into_iter().map(|layer|self.projected_decoder_layer(layer)).collect()}
    }
    /// Partition a complete actual native paired model, preserving source/target/head/shared-leaf IDs.
    pub fn projected_encoder_decoder_model<P:ShardTransformerProjection<B>>(&mut self,model:ProjectedEncoderDecoderModel<B,P>) -> FullyShardedProjectedEncoderDecoderModel<B,P::Sharded> {
        FullyShardedProjectedEncoderDecoderModel {source_embeddings:self.transformer_embeddings(model.source_embeddings),encoder:self.awq_transformer_stack(model.encoder),
            encoder_normalization:model.encoder_normalization.map(|norm|self.normalization(norm)),target_embeddings:self.transformer_embeddings(model.target_embeddings),
            decoder:self.projected_decoder_stack(model.decoder),decoder_normalization:model.decoder_normalization.map(|norm|self.normalization(norm)),head:self.awq_transformer_head(model.head)}
    }
}
impl<B:Backend,P:Module<B>> FullyShardedProjectedEncoderDecoderModel<B,P> {
    /// Slice original actual caller-loaded components without creating replacement parameter values.
    pub fn from_full<Q:ShardTransformerProjection<B,Sharded=P>>(model:ProjectedEncoderDecoderModel<B,Q>,rank:usize,world:usize) -> Self {
        ShardingContext::new(rank,world).projected_encoder_decoder_model(model)
    }
}

macro_rules! projected_decoder_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$forward:ident,$packed:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedProjectedCrossAttention<$backend,P> {
            /// Gather only actual original cross-attention projection/norm leaves.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<ProjectedCrossAttentionBlock<$backend,P::Gathered>,C::Error> {
                Ok(ProjectedCrossAttentionBlock::from_parts(self.attention.$gather(communicator.clone())?,self.query_norm.$gather(communicator.clone())?,
                    self.memory_norm.as_ref().map(|norm|norm.$gather(communicator)).transpose()?,self.residual_dropout.clone(),self.norm_first))
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedProjectedDecoderLayer<$backend,P> {
            /// Gather only this original complete three-stage decoder layer, retaining actual local persistent leaves.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<ProjectedEncoderDecoderLayer<$backend,P::Gathered>,C::Error> {
                Ok(ProjectedEncoderDecoderLayer::from_parts(self.backbone.$gather(communicator.clone())?,self.cross_attention.$gather(communicator)?))
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedProjectedDecoderLayer<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Actual native self-attention, cross-attention, then FFN with explicit independent positions.
            pub fn $forward<C,F,G>(&self,input:Tensor<$backend,3>,memory:Tensor<$backend,3>,self_masks:DenseAttentionMask<$backend>,self_options:DenseAttentionOptions,
                cross_masks:DenseAttentionMask<$backend>,cross_options:DenseAttentionOptions,communicator:C,self_positions:F,cross_positions:G)
                -> Result<Tensor<$backend,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>),
                    G:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.$gather(communicator).map_err(FullyShardedProjectedError::Collective)?.forward_with_positions(input,memory,self_masks,self_options,cross_masks,cross_options,self_positions,cross_positions)
                    .map_err(FullyShardedProjectedError::Projection)
            }
            /// Actual independent source/target documents and optional per-document self/cross masks.
            pub fn $packed<C,F,G>(&self,input:Tensor<$backend,2>,memory:Tensor<$backend,2>,query_layout:&PackedSequenceLayout,memory_layout:&PackedSequenceLayout,
                self_masks:Option<&[PackedDocumentAttentionMask<$backend>]>,self_options:PackedAttentionOptions,cross_masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                cross_options:PackedAttentionOptions,communicator:C,self_positions:F,cross_positions:G)
                -> Result<Tensor<$backend,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>),
                    G:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                self.$gather(communicator).map_err(FullyShardedProjectedError::Collective)?.forward_packed_with_positions(input,memory,query_layout,memory_layout,
                    self_masks,self_options,cross_masks,cross_options,self_positions,cross_positions).map_err(FullyShardedProjectedError::Projection)
            }
        }
    };
}
projected_decoder_execution!(B,[B:Backend],gather_inference,forward_inference,forward_packed_inference);
projected_decoder_execution!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward,forward_packed);

impl<B:Backend,P:GatherTransformerProjection<B,B>> FullyShardedProjectedDecoderLayer<B,P>
    where P::Gathered:TransformerProjection<B> {
    /// Actual only-new-row native self/cross/FFN inference against prepared original encoder memory.
    pub fn forward_cached_inference<C,F,G>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,memory:&ProjectedKvCache<B>,
        self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,
        communicator:C,self_positions:F,cross_positions:G) -> Result<Tensor<B,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<B>>::Error>>
        where C:IntegerTensorCollective<B>,F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),G:FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        self.gather_inference(communicator).map_err(FullyShardedProjectedError::Collective)?.forward_cached_with_positions(input,new_visible,cache,memory,
            self_masks,self_options,cross_masks,cross_options,self_positions,cross_positions).map_err(FullyShardedProjectedError::Projection)
    }
}
impl<B:Backend,P:GatherTransformerProjection<B,B>> FullyShardedProjectedEncoderDecoderModel<B,P>
    where P::Gathered:TransformerProjection<B> {
    /// Prepare actual paired per-layer native caches from caller-provided encoded source memory.
    /// The original cache record/reorder APIs remain reusable, without a dense quantized-base shadow.
    pub fn prepare_kv_cache<C,F>(&self,memory:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,start_position:usize,initial_capacity:usize,communicator:C,mut positions:F)
        -> Result<EncoderDecoderKvCache<B>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<B>>::Error>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        assert_eq!(memory.dims()[2],self.source_embeddings.hidden_width(),"prepared source memory width differs");
        let mut memories=Vec::with_capacity(self.decoder.layers.len());
        for (index,layer) in self.decoder.layers.iter().enumerate() {
            let cross=layer.cross_attention.gather_inference(communicator.clone()).map_err(FullyShardedProjectedError::Collective)?;
            memories.push(cross.prepare_cached_memory(memory.clone(),visible.clone(),start_position,|key,position|positions(index,key,position))
                .map_err(FullyShardedProjectedError::Projection)?);
        }
        Ok(EncoderDecoderKvCache::new(TransformerKvCache::new(self.decoder.layers.len(),initial_capacity),memories))
    }
    /// Native actual new-target-row logits, gathering only the current decoder layer.
    /// Partial projection failures preserve the original requirement to restore the actual cache record.
    pub fn decode_cached_inference_with<C,F>(&self,input:FullyShardedTransformerInput<B>,cache:&mut EncoderDecoderKvCache<B>,communicator:C,mut decoder:F)
        -> Result<Tensor<B,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<B>>::Error>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,&FullyShardedProjectedDecoderLayer<B,P>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)
            ->Result<Tensor<B,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<B>>::Error>> {
        let mut hidden=self.target_embeddings.forward_inference(input,communicator.clone()).map_err(FullyShardedProjectedError::Collective)?;
        let (history,memory)=cache.parts_mut();history.validate_layers(self.decoder.layers.len());assert_eq!(memory.len(),self.decoder.layers.len(),"cached paired layer counts differ");
        let rows=(hidden.dims()[0],hidden.dims()[1]);let next=history.position().checked_add(rows.1).expect("cached paired position overflows");
        for (index,layer) in self.decoder.layers.iter().enumerate() {
            hidden=decoder(index,layer,hidden,&mut history.layers_mut()[index],&memory[index])?;
            assert_eq!((hidden.dims()[0],hidden.dims()[1]),rows,"cached decoder changed actual target rows");
        }
        history.finish_chunk(next);
        if let Some(norm)=&self.decoder_normalization {hidden=norm.gather_inference(communicator.clone()).map_err(FullyShardedProjectedError::Collective)?.forward(hidden);}
        self.head.forward_inference(hidden,communicator)
    }
}

macro_rules! projected_paired_hidden {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$embed:ident,$hidden:ident,$packed:ident,$forward:ident,$packed_logits:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedProjectedEncoderDecoderModel<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Complete actual native source-memory-target hidden graph, gathering one original layer at a time.
            pub fn $hidden<C,E,F>(&self,source:FullyShardedTransformerInput<$backend>,target:FullyShardedTransformerInput<$backend>,communicator:C,mut encoder:E,mut decoder:F)
                -> Result<Tensor<$backend,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,E:FnMut(usize,&FullyShardedProjectedTransformerBlock<$backend,P>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>,
                    F:FnMut(usize,&FullyShardedProjectedDecoderLayer<$backend,P>,Tensor<$backend,3>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>> {
                let source_rows=source.tokens.dims();let target_rows=target.tokens.dims();assert_eq!(source_rows[0],target_rows[0],"paired batch rows differ");
                assert_eq!(source.tokens.device(),target.tokens.device(),"paired source/target devices differ");
                let mut memory=self.source_embeddings.$embed(source,communicator.clone()).map_err(FullyShardedProjectedError::Collective)?;
                for (index,block) in self.encoder.blocks.iter().enumerate() {memory=encoder(index,block,memory)?;}
                assert_eq!(memory.dims(),[source_rows[0],source_rows[1],self.source_embeddings.hidden_width()],"encoder changed actual source rows");
                if let Some(norm)=&self.encoder_normalization {memory=norm.$gather(communicator.clone()).map_err(FullyShardedProjectedError::Collective)?.forward(memory);}
                let mut hidden=self.target_embeddings.$embed(target,communicator.clone()).map_err(FullyShardedProjectedError::Collective)?;
                for (index,layer) in self.decoder.layers.iter().enumerate() {hidden=decoder(index,layer,hidden,memory.clone())?;}
                assert_eq!(hidden.dims(),[target_rows[0],target_rows[1],self.target_embeddings.hidden_width()],"decoder changed actual target rows");
                if let Some(norm)=&self.decoder_normalization {hidden=norm.$gather(communicator).map_err(FullyShardedProjectedError::Collective)?.forward(hidden);}
                Ok(hidden)
            }
            /// Complete independent-document native source/target hidden graph, retaining exact actual boundary metadata.
            pub fn $packed<C,E,F>(&self,source:FullyShardedTransformerInput<$backend,1>,target:FullyShardedTransformerInput<$backend,1>,source_layout:&PackedSequenceLayout,
                target_layout:&PackedSequenceLayout,communicator:C,mut encoder:E,mut decoder:F)
                -> Result<Tensor<$backend,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,E:FnMut(usize,&FullyShardedProjectedTransformerBlock<$backend,P>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>,
                    F:FnMut(usize,&FullyShardedProjectedDecoderLayer<$backend,P>,Tensor<$backend,2>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>> {
                assert_eq!(source_layout.documents(),target_layout.documents(),"paired packed document counts differ");assert_eq!(source.tokens.device(),target.tokens.device(),"paired packed devices differ");
                let mut memory=self.source_embeddings.$embed(super::model::packed_input(source,source_layout),communicator.clone()).map_err(FullyShardedProjectedError::Collective)?
                    .reshape([source_layout.tokens(),self.source_embeddings.hidden_width()]);
                for (index,block) in self.encoder.blocks.iter().enumerate() {memory=encoder(index,block,memory)?;}
                assert_eq!(memory.dims(),[source_layout.tokens(),self.source_embeddings.hidden_width()],"packed encoder changed actual source rows");
                if let Some(norm)=&self.encoder_normalization {memory=norm.$gather(communicator.clone()).map_err(FullyShardedProjectedError::Collective)?.forward(memory);}
                let mut hidden=self.target_embeddings.$embed(super::model::packed_input(target,target_layout),communicator.clone()).map_err(FullyShardedProjectedError::Collective)?
                    .reshape([target_layout.tokens(),self.target_embeddings.hidden_width()]);
                for (index,layer) in self.decoder.layers.iter().enumerate() {hidden=decoder(index,layer,hidden,memory.clone())?;}
                assert_eq!(hidden.dims(),[target_layout.tokens(),self.target_embeddings.hidden_width()],"packed decoder changed actual target rows");
                if let Some(norm)=&self.decoder_normalization {hidden=norm.$gather(communicator).map_err(FullyShardedProjectedError::Collective)?.forward(hidden);}
                Ok(hidden)
            }
            /// Original target logits over complete actual native source/decoder graph.
            pub fn $forward<C,E,F>(&self,source:FullyShardedTransformerInput<$backend>,target:FullyShardedTransformerInput<$backend>,communicator:C,encoder:E,decoder:F)
                -> Result<Tensor<$backend,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,E:FnMut(usize,&FullyShardedProjectedTransformerBlock<$backend,P>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>,
                    F:FnMut(usize,&FullyShardedProjectedDecoderLayer<$backend,P>,Tensor<$backend,3>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>> {
                let hidden=self.$hidden(source,target,communicator.clone(),encoder,decoder)?;
                self.head.$gather(communicator).map_err(FullyShardedProjectedError::Collective)?.forward(hidden).map_err(FullyShardedProjectedError::Projection)
            }
            /// Original actual independent-document target logits without padded rows.
            pub fn $packed_logits<C,E,F>(&self,source:FullyShardedTransformerInput<$backend,1>,target:FullyShardedTransformerInput<$backend,1>,source_layout:&PackedSequenceLayout,
                target_layout:&PackedSequenceLayout,communicator:C,encoder:E,decoder:F)
                -> Result<Tensor<$backend,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,E:FnMut(usize,&FullyShardedProjectedTransformerBlock<$backend,P>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>,
                    F:FnMut(usize,&FullyShardedProjectedDecoderLayer<$backend,P>,Tensor<$backend,2>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>> {
                let hidden=self.$packed(source,target,source_layout,target_layout,communicator.clone(),encoder,decoder)?;
                self.head.$gather(communicator).map_err(FullyShardedProjectedError::Collective)?.forward(hidden).map_err(FullyShardedProjectedError::Projection)
            }
        }
    };
}
projected_paired_hidden!(B,[B:Backend],gather_inference,forward_inference,forward_hidden_inference_with,forward_packed_hidden_inference_with,forward_inference_with,forward_packed_inference_with);
projected_paired_hidden!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward,forward_hidden_with,forward_packed_hidden_with,forward_with,forward_packed_with);

impl<B:Backend,S:CheckpointStrategy,P:GatherTransformerProjection<Autodiff<B,S>,B>> FullyShardedProjectedEncoderDecoderModel<Autodiff<B,S>,P>
    where P::Gathered:TransformerProjection<Autodiff<B,S>> {
    /// Complete actual native scoped paired fine tuning with independent source/self/cross policies.
    pub fn forward_causal_with_positions<C,E,F,G>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>>,target:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        source_masks:DenseAttentionMask<Autodiff<B,S>>,source_options:DenseAttentionOptions,self_masks:DenseAttentionMask<Autodiff<B,S>>,self_options:DenseAttentionOptions,
        cross_masks:DenseAttentionMask<Autodiff<B,S>>,cross_options:DenseAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,
        mut source_positions:E,mut self_positions:F,mut cross_positions:G)
        -> Result<FullyShardedLoss<B,S>,FullyShardedProjectedTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
        where C:IntegerTensorCollective<B>,E:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            F:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            G:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_causal_with(source,target,labels,criterion,label_smoothing,communicator,
            |index,block,hidden,transport|block.forward(hidden,source_masks.clone(),source_options,transport,|query,key|source_positions(index,query,key)),
            |index,layer,hidden,memory,transport|layer.forward(hidden,memory,self_masks.clone(),self_options,cross_masks.clone(),cross_options,transport,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Complete native independent-document packed paired objective through the same original loss scope.
    pub fn forward_packed_causal_with_positions<C,E,F,G>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>,1>,target:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,source_masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,source_options:PackedAttentionOptions,
        self_masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,self_options:PackedAttentionOptions,cross_masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,cross_options:PackedAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut source_positions:E,mut self_positions:F,mut cross_positions:G)
        -> Result<FullyShardedLoss<B,S>,FullyShardedProjectedTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
        where C:IntegerTensorCollective<B>,E:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            F:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            G:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        self.forward_packed_causal_with(source,target,labels,source_layout,target_layout,criterion,label_smoothing,communicator,
            |index,block,hidden,transport|block.forward_packed(hidden,source_layout,source_masks,source_options,transport,|query,key|source_positions(index,query,key)),
            |index,layer,hidden,memory,transport|layer.forward_packed(hidden,memory,target_layout,source_layout,self_masks,self_options,cross_masks,cross_options,transport,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Complete actual packed/floating paired model training through one native loss scope.
    /// Encoder-memory gradients remain intact; already aligned labels use criterion.shift=false.
    pub fn forward_causal_with<C,E,F>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>>,target:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut encoder:E,mut decoder:F)
        -> Result<FullyShardedLoss<B,S>,FullyShardedProjectedTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
        where C:IntegerTensorCollective<B>,E:FnMut(usize,&FullyShardedProjectedTransformerBlock<Autodiff<B,S>,P>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>,
            F:FnMut(usize,&FullyShardedProjectedDecoderLayer<Autodiff<B,S>,P>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>> {
        assert_eq!(target.tokens.dims(),labels.dims(),"actual target/label rows differ");
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden_with(source,target,transport.clone(),|index,block,hidden|encoder(index,block,hidden,transport.clone()),
            |index,block,hidden,memory|decoder(index,block,hidden,memory,transport.clone())).map_err(FullyShardedProjectedTrainingError::Model)?;
        let loss=self.head.forward_causal_loss(hidden,labels,criterion,label_smoothing,transport).map_err(FullyShardedProjectedTrainingError::Model)?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedProjectedTrainingError::Loss)
    }
    /// Original full-vocabulary independent-target-document loss with exact global token counts.
    /// All original source/target/adapter reachability participates; no second gradient SUM.
    pub fn forward_packed_causal_with<C,E,F>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>,1>,target:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut encoder:E,mut decoder:F)
        -> Result<FullyShardedLoss<B,S>,FullyShardedProjectedTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
        where C:IntegerTensorCollective<B>,E:FnMut(usize,&FullyShardedProjectedTransformerBlock<Autodiff<B,S>,P>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>,
            F:FnMut(usize,&FullyShardedProjectedDecoderLayer<Autodiff<B,S>,P>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,FullyShardedProjectedError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>> {
        assert_eq!(target.tokens.dims(),labels.dims(),"actual packed target/label rows differ");
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden_with(source,target,source_layout,target_layout,transport.clone(),|index,block,hidden|encoder(index,block,hidden,transport.clone()),
            |index,block,hidden,memory|decoder(index,block,hidden,memory,transport.clone())).map_err(FullyShardedProjectedTrainingError::Model)?;
        let loss=self.head.forward_packed_causal_loss(hidden,labels,target_layout,criterion,label_smoothing,transport).map_err(FullyShardedProjectedTrainingError::Model)?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedProjectedTrainingError::Loss)
    }
}
