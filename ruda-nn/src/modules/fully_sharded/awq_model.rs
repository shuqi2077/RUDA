use super::*;
use ruda_model::tensor::{Bool,IntegerTensorCollective};
use crate::transformer::TransformerProjection;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::TransformerKvCache,transformer::{AwqTransformerHead,AwqTransformerModel}};
use crate::loss::{CausalCrossEntropyConfig,CausalLoss};
use ruda_autodiff::collective::{CollectiveScope,ScopedCollectiveError};

/// Exact native output projection/norm/dropout with integer and floating storage sharded.
#[derive(Module,Debug)]
pub struct FullyShardedAwqTransformerHead<B:Backend,P:Module<B>=FullyShardedAwqProjection<B>> {
    /// Actual original dense/LoRA/AWQ/AWQ-LoRA output projection choice.
    pub projection:P,
    /// Original optional head affine norm.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Original independent head dropout.
    pub dropout:crate::Dropout,
}

/// Complete actual embeddings/backbone/final-norm/head graph with only local persistent leaves.
#[derive(Module,Debug)]
pub struct FullyShardedAwqTransformerModel<B:Backend,P:Module<B>=FullyShardedAwqProjection<B>> {
    /// Original input tables, norms and dropout, retaining all explicit ties.
    pub embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Actual original ordered mixed-projection blocks.
    pub backbone:FullyShardedAwqTransformerStack<B,P>,
    /// Original optional independent model-final norm.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Actual original independently selected output projection/norm/dropout.
    pub head:FullyShardedAwqTransformerHead<B,P>,
}
impl<B:Backend> ShardingContext<B> {
    /// Partition every actual original head leaf through the shared-ID context.
    pub fn awq_transformer_head<P:ShardTransformerProjection<B>>(&mut self,head:AwqTransformerHead<B,P>) -> FullyShardedAwqTransformerHead<B,P::Sharded> {
        FullyShardedAwqTransformerHead {projection:self.awq_projection(head.projection),
            normalization:head.normalization.map(|norm|self.normalization(norm)),dropout:head.dropout}
    }
    /// Partition the complete actual model, retaining one canonical local leaf for each tie.
    pub fn awq_transformer_model<P:ShardTransformerProjection<B>>(&mut self,model:AwqTransformerModel<B,P>) -> FullyShardedAwqTransformerModel<B,P::Sharded> {
        FullyShardedAwqTransformerModel {embeddings:self.transformer_embeddings(model.embeddings),
            backbone:self.awq_transformer_stack(model.backbone),normalization:model.normalization.map(|norm|self.normalization(norm)),
            head:self.awq_transformer_head(model.head)}
    }
}
impl<B:Backend,P:Module<B>> FullyShardedAwqTransformerModel<B,P> {
    /// Convert actual caller-loaded native components without random replacement values.
    pub fn from_full<Q:ShardTransformerProjection<B,Sharded=P>>(model:AwqTransformerModel<B,Q>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).awq_transformer_model(model)}
    /// Prepare the actual native per-layer cache metadata.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {self.backbone.new_kv_cache(initial_capacity)}
}

macro_rules! awq_model_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$embed:ident,$forward:ident,$packed:ident,$hidden:ident,$packed_hidden:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqTransformerHead<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Transient exact original head over actual packed/floating local values.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqTransformerHead<$backend,P::Gathered>,C::Error> {
                Ok(AwqTransformerHead::from_projection(self.projection.gather_projection(communicator.clone())?,
                    self.normalization.as_ref().map(|norm|norm.$gather(communicator)).transpose()?,self.dropout.clone()))
            }
            /// Native logits on actual hidden states after exact original data gathers.
            pub fn $forward<C:IntegerTensorCollective<B>,const D:usize>(&self,hidden:Tensor<$backend,D>,communicator:C)
                -> Result<Tensor<$backend,D>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>> {
                self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?.forward(hidden).map_err(FullyShardedAwqError::Projection)
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqTransformerModel<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Complete actual native ID-to-logits graph with explicit original inputs.
            pub fn $forward<C,F>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,
                options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                let hidden=self.$hidden(input,masks,options,communicator.clone(),positions)?;
                self.head.$forward(hidden,communicator)
            }
            /// Actual final hidden states for chunked logits or pooled sequence heads.
            pub fn $hidden<C,F>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,
                options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                let hidden=self.embeddings.$embed(input,communicator.clone()).map_err(FullyShardedAwqError::Collective)?;
                let hidden=self.backbone.$forward(hidden,masks,options,communicator.clone(),positions)?;
                let hidden=if let Some(norm)=&self.normalization {norm.$gather(communicator.clone()).map_err(FullyShardedAwqError::Collective)?.forward(hidden)} else {hidden};
                Ok(hidden)
            }
            /// Actual flat-document complete model. The native input helper preserves
            /// original token/position/type metadata and explicit mixed lookup-row dtypes.
            pub fn $packed<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,
                masks:Option<&[PackedDocumentAttentionMask<$backend>]>,options:PackedAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                let hidden=self.$packed_hidden(input,layout,masks,options,communicator.clone(),positions)?;
                self.head.$forward(hidden,communicator)
            }
            /// Actual packed final hidden states without constructing a full token-logit tensor.
            pub fn $packed_hidden<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,
                masks:Option<&[PackedDocumentAttentionMask<$backend>]>,options:PackedAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                let input=super::model::packed_input(input,layout);
                let hidden=self.embeddings.$embed(input,communicator.clone()).map_err(FullyShardedAwqError::Collective)?
                    .reshape([layout.tokens(),self.embeddings.hidden_width()]);
                let hidden=self.backbone.$packed(hidden,layout,masks,options,communicator.clone(),positions)?;
                let hidden=if let Some(norm)=&self.normalization {norm.$gather(communicator.clone()).map_err(FullyShardedAwqError::Collective)?.forward(hidden)} else {hidden};
                Ok(hidden)
            }
        }
    };
}
awq_model_execution!(B,[B:Backend],gather_inference,forward_inference,forward_inference,forward_packed_inference,forward_hidden_inference,forward_packed_hidden_inference);
awq_model_execution!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward,forward,forward_packed,forward_hidden,forward_packed_hidden);

impl<B:Backend,P:GatherTransformerProjection<B,B>> FullyShardedAwqTransformerModel<B,P>
    where P::Gathered:TransformerProjection<B> {
    /// Original complete new-token inference with positioned native KV cache and actual logits.
    /// Partial cache failures retain the original cache recovery requirement.
    pub fn forward_cached_inference<C,F>(&self,input:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,mut positions:F)
        -> Result<Tensor<B,3>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<B>>::Error>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let mut hidden=self.embeddings.forward_inference(input,communicator.clone()).map_err(FullyShardedAwqError::Collective)?;
        cache.validate_layers(self.backbone.blocks.len());let rows=(hidden.dims()[0],hidden.dims()[1]);
        let next=cache.position().checked_add(rows.1).expect("cached sharded model position overflows");
        for (index,block) in self.backbone.blocks.iter().enumerate() {
            hidden=block.forward_cached_inference(hidden,visible.clone(),&mut cache.layers_mut()[index],masks.clone(),options,communicator.clone(),
                |query,key,position|positions(index,query,key,position))?;
            assert_eq!((hidden.dims()[0],hidden.dims()[1]),rows,"cached model layer changed actual rows");
        }
        cache.finish_chunk(next);
        let hidden=if let Some(norm)=&self.normalization {norm.gather_inference(communicator.clone()).map_err(FullyShardedAwqError::Collective)?.forward(hidden)} else {hidden};
        self.head.forward_inference(hidden,communicator)
    }
}

/// Original packed model/transport failure or original distributed loss-completion failure.
#[derive(Debug)]
pub enum FullyShardedAwqTrainingError<C:core::fmt::Debug,Q:core::fmt::Debug> {
    /// Original actual native projection or data gather failure.
    Model(FullyShardedAwqError<C,Q>),
    /// Original scoped loss reachability/count/transport completion failure.
    Loss(ScopedCollectiveError<C>),
}
impl<C:core::fmt::Debug,Q:core::fmt::Debug> core::fmt::Display for FullyShardedAwqTrainingError<C,Q> {
    fn fmt(&self,f:&mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {Self::Model(error)=>write!(f,"AWQ training model: {error}"),Self::Loss(error)=>write!(f,"AWQ training loss: {error}")}
    }
}
impl<C:core::fmt::Debug,Q:core::fmt::Debug> core::error::Error for FullyShardedAwqTrainingError<C,Q> {}

macro_rules! awq_head_loss {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$causal:ident,$packed:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedAwqTransformerHead<$backend,P>
            where P::Gathered:TransformerProjection<$backend> {
            /// Gather the actual output head once and reuse it for full-vocabulary token chunks.
            pub fn $causal<C:IntegerTensorCollective<B>>(&self,hidden:Tensor<$backend,3>,labels:Tensor<$backend,2,Int>,
                criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C)
                -> Result<CausalLoss<$backend>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>> {
                let head=self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?;
                criterion.try_forward_hidden_with_smoothing(hidden,labels,|rows|head.forward(rows),label_smoothing).map_err(FullyShardedAwqError::Projection)
            }
            /// Original document-local shifted labels and full-vocabulary smoothing/counts.
            pub fn $packed<C:IntegerTensorCollective<B>>(&self,hidden:Tensor<$backend,2>,labels:Tensor<$backend,1,Int>,layout:&PackedSequenceLayout,
                criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C)
                -> Result<CausalLoss<$backend>,FullyShardedAwqError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error>> {
                let head=self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?;
                criterion.try_forward_packed_hidden_with_smoothing(hidden,labels,layout,|rows|head.forward(rows),label_smoothing).map_err(FullyShardedAwqError::Projection)
            }
        }
    };
}
awq_head_loss!(B,[B:Backend],gather_inference,forward_causal_loss_inference,forward_packed_causal_loss_inference);
awq_head_loss!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward_causal_loss,forward_packed_causal_loss);

impl<B:Backend,S:CheckpointStrategy,P:GatherTransformerProjection<Autodiff<B,S>,B>> FullyShardedAwqTransformerModel<Autodiff<B,S>,P>
    where P::Gathered:TransformerProjection<Autodiff<B,S>> {
    /// Complete actual native fine-tuning graph, globally counted original causal loss
    /// and rank-consistent AD reachability. No second gradient reduction or skip policy.
    pub fn forward_causal_with_positions<C,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        masks:DenseAttentionMask<Autodiff<B,S>>,options:DenseAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,positions:F)
        -> Result<FullyShardedLoss<B,S>,FullyShardedAwqTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden(input,masks,options,transport.clone(),positions).map_err(FullyShardedAwqTrainingError::Model)?;
        let loss=self.head.forward_causal_loss(hidden,labels,criterion,label_smoothing,transport).map_err(FullyShardedAwqTrainingError::Model)?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedAwqTrainingError::Loss)
    }
    /// Complete packed-document fine tuning through real packed/dense/adapted projections.
    /// Original label shifting never crosses document boundaries; exact global integer
    /// normalization/empty-rank backward participation reuse the existing native scope.
    pub fn forward_packed_causal_with_positions<C,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options:PackedAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,positions:F)
        -> Result<FullyShardedLoss<B,S>,FullyShardedAwqTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden(input,layout,masks,options,transport.clone(),positions).map_err(FullyShardedAwqTrainingError::Model)?;
        let loss=self.head.forward_packed_causal_loss(hidden,labels,layout,criterion,label_smoothing,transport).map_err(FullyShardedAwqTrainingError::Model)?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedAwqTrainingError::Loss)
    }
}
