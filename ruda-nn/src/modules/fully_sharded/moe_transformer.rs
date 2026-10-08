use super::*;
use ruda_model::tensor::{MoeOps,Bool,IntegerTensorCollective};
use crate::{NativeMoeFeedForward,NativeMoeTransformerBlock,NativeMoeTransformerLayer,NativeMoeTransformerStack,NativeMoeTransformerModel,
    NativeMoeTransformerError,transformer::TransformerProjection,attention::{DenseAttentionMask,DenseAttentionOptions,
    PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},cache::TransformerKvCache,loss::CausalCrossEntropyConfig};
use ruda_autodiff::collective::{CollectiveScope,ScopedCollectiveError};

/// Actual routed expert cubes/router and optional shared FFN with local-only persistent leaves.
#[derive(Module,Debug)]
pub struct FullyShardedNativeMoeFeedForward<B:Backend,P:Module<B>> {
    /// Original native routed branch, including actual optional correction bias.
    pub routed:FullyShardedNativeMoeLayer<B,P>,
    /// Original explicitly present ordinary/gated shared branch.
    pub shared:Option<FullyShardedProjectedFeedForward<B,P>>,
}
/// Actual native attention/routed/shared block with every loaded parameter data-sharded.
#[derive(Module,Debug)]
pub struct FullyShardedNativeMoeTransformerBlock<B:Backend,P:Module<B>> {
    /// Original native self-attention projections and head geometry.
    pub attention:FullyShardedProjectedAttention<B,P>,
    /// Original actual routed and optional shared branches.
    pub feed_forward:FullyShardedNativeMoeFeedForward<B,P>,
    /// Original attention affine norm and epsilon.
    pub attention_norm:FullyShardedTransformerNorm<B>,
    /// Original independent FFN affine norm and epsilon.
    pub feed_forward_norm:FullyShardedTransformerNorm<B>,
    /// Original residual dropout.
    pub residual_dropout:crate::Dropout,
    /// Original pre/post-norm choice.
    pub norm_first:bool,
}
/// Original actual dense/routed choice, not an independently selected architecture per rank.
#[derive(Module,Debug)]
pub enum FullyShardedNativeMoeTransformerLayer<B:Backend,P:Module<B>> {
    /// Original native ordinary/gated dense branch.
    Dense(FullyShardedProjectedTransformerBlock<B,P>),
    /// Original native routed/shared branch.
    Routed(FullyShardedNativeMoeTransformerBlock<B,P>),
}
/// Original actual mixed layer order over only rank-local persistent parameters.
#[derive(Module,Debug)]
pub struct FullyShardedNativeMoeTransformerStack<B:Backend,P:Module<B>> {
    /// Every actual original loaded layer in order.
    pub layers:Vec<FullyShardedNativeMoeTransformerLayer<B,P>>,
}
/// Complete original native embeddings/mixed-backbone/norm/head graph with data sharding.
#[derive(Module,Debug)]
pub struct FullyShardedNativeMoeTransformerModel<B:Backend,P:Module<B>> {
    /// Actual original token and optional position/type tables.
    pub embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Actual original dense/routed/shared layer order.
    pub backbone:FullyShardedNativeMoeTransformerStack<B,P>,
    /// Original optional independent final norm.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Original independently selected native dense/packed/adapter head.
    pub head:FullyShardedProjectedTransformerHead<B,P>,
}
impl<B:Backend> ShardingContext<B> {
    /// Retain original tied router/expert/shared parameter identities through this full-model context.
    pub fn moe_feed_forward<P:ShardTransformerProjection<B>>(&mut self,feed:NativeMoeFeedForward<B,P>) -> FullyShardedNativeMoeFeedForward<B,P::Sharded> {
        FullyShardedNativeMoeFeedForward {routed:self.moe_layer(feed.routed),shared:feed.shared.map(|shared|self.awq_feed_forward(shared))}
    }
    /// Partition the actual original native block without changing routing/execution choices or norm order.
    pub fn moe_transformer<P:ShardTransformerProjection<B>>(&mut self,block:NativeMoeTransformerBlock<B,P>) -> FullyShardedNativeMoeTransformerBlock<B,P::Sharded> {
        block.validate();FullyShardedNativeMoeTransformerBlock {attention:self.awq_attention(block.attention),feed_forward:self.moe_feed_forward(block.feed_forward),
            attention_norm:self.normalization(block.attention_norm),feed_forward_norm:self.normalization(block.feed_forward_norm),
            residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }
    /// Preserve every actual original dense/routed variant in its original loaded order.
    pub fn moe_transformer_stack<P:ShardTransformerProjection<B>>(&mut self,stack:NativeMoeTransformerStack<B,P>) -> FullyShardedNativeMoeTransformerStack<B,P::Sharded> {
        FullyShardedNativeMoeTransformerStack {layers:stack.layers.into_iter().map(|layer|match layer {
            NativeMoeTransformerLayer::Dense(block)=>FullyShardedNativeMoeTransformerLayer::Dense(self.awq_transformer(block)),
            NativeMoeTransformerLayer::Routed(block)=>FullyShardedNativeMoeTransformerLayer::Routed(self.moe_transformer(block)),
        }).collect()}
    }
    /// Slice the actual complete model, preserving cross-layer and table/head parameter aliases.
    pub fn moe_transformer_model<P:ShardTransformerProjection<B>>(&mut self,model:NativeMoeTransformerModel<B,P>) -> FullyShardedNativeMoeTransformerModel<B,P::Sharded> {
        FullyShardedNativeMoeTransformerModel {embeddings:self.transformer_embeddings(model.embeddings),backbone:self.moe_transformer_stack(model.backbone),
            normalization:model.normalization.map(|norm|self.normalization(norm)),head:self.awq_transformer_head(model.head)}
    }
}
impl<B:Backend,P:Module<B>> FullyShardedNativeMoeTransformerStack<B,P> {
    /// Actual per-layer cache topology, with no full model weight allocation.
    pub fn new_kv_cache(&self,capacity:usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.layers.len(),capacity)}
}
impl<B:Backend,P:Module<B>> FullyShardedNativeMoeTransformerModel<B,P> {
    /// Partition actual native loaded components without independent random expert replicas.
    pub fn from_full<Q:ShardTransformerProjection<B,Sharded=P>>(model:NativeMoeTransformerModel<B,Q>,rank:usize,world:usize) -> Self {
        ShardingContext::new(rank,world).moe_transformer_model(model)
    }
    /// Actual original complete model cache topology.
    pub fn new_kv_cache(&self,capacity:usize) -> TransformerKvCache<B> {self.backbone.new_kv_cache(capacity)}
}
/// Original actual data-gather failure or actual native attention/router/expert execution failure.
#[derive(Debug)]
pub enum FullyShardedNativeMoeTransformerError<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> {
    /// Original integer/floating collective failure.
    Collective(C),
    /// Original actual native graph error, retaining its selected projection or routed source.
    Native(NativeMoeTransformerError<P,M>),
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::fmt::Display for FullyShardedNativeMoeTransformerError<C,P,M> {
    fn fmt(&self,f:&mut core::fmt::Formatter<'_>) -> core::fmt::Result {match self {
        Self::Collective(error)=>write!(f,"native MoE model data transport: {error:?}"),Self::Native(error)=>write!(f,"native MoE model: {error}")}}
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::error::Error for FullyShardedNativeMoeTransformerError<C,P,M> {}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> From<FullyShardedAwqError<C,P>> for FullyShardedNativeMoeTransformerError<C,P,M> {
    fn from(error:FullyShardedAwqError<C,P>) -> Self {match error {
        FullyShardedAwqError::Collective(error)=>Self::Collective(error),FullyShardedAwqError::Projection(error)=>Self::Native(NativeMoeTransformerError::Projection(error))}}
}
macro_rules! moe_transformer_gathers {
    ($backend:ty,[$($generics:tt)*],$gather:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedNativeMoeFeedForward<$backend,P> {
            /// Transient actual routed/shared branch, preserving all original local storage and IDs.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<NativeMoeFeedForward<$backend,P::Gathered>,C::Error> {
                Ok(NativeMoeFeedForward::from_parts(self.routed.$gather(communicator.clone())?,self.shared.as_ref().map(|shared|shared.$gather(communicator)).transpose()?))
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedNativeMoeTransformerBlock<$backend,P> {
            /// Gather only this actual native block, never the whole model as persistent storage.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<NativeMoeTransformerBlock<$backend,P::Gathered>,C::Error> {
                Ok(NativeMoeTransformerBlock {attention:self.attention.$gather(communicator.clone())?,feed_forward:self.feed_forward.$gather(communicator.clone())?,
                    attention_norm:self.attention_norm.$gather(communicator.clone())?,feed_forward_norm:self.feed_forward_norm.$gather(communicator)?,
                    residual_dropout:self.residual_dropout.clone(),norm_first:self.norm_first})
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedNativeMoeTransformerLayer<$backend,P> {
            /// Preserve the original actual layer variant with exact native integer/floating gathers.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<NativeMoeTransformerLayer<$backend,P::Gathered>,C::Error> {
                match self {Self::Dense(block)=>block.$gather(communicator).map(NativeMoeTransformerLayer::Dense),
                    Self::Routed(block)=>block.$gather(communicator).map(NativeMoeTransformerLayer::Routed)}
            }
        }
    };
}
moe_transformer_gathers!(B,[B:Backend],gather_inference);
moe_transformer_gathers!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);

macro_rules! moe_transformer_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$embed:ident,$forward:ident,$packed:ident,$hidden:ident,$packed_hidden:ident) => {
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedNativeMoeTransformerStack<$backend,P> where P::Gathered:TransformerProjection<$backend> {
            /// Original dense-axis graph, gathering the current actual layer before native execution.
            pub fn $forward<C,F>(&self,mut input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedNativeMoeTransformerError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                for (index,layer) in self.layers.iter().enumerate() {input=layer.$gather(communicator.clone()).map_err(FullyShardedNativeMoeTransformerError::Collective)?
                    .forward_with_positions(input,masks.clone(),options,|query,key|positions(index,query,key)).map_err(FullyShardedNativeMoeTransformerError::Native)?;}Ok(input)
            }
            /// Original independent-document graph, retaining full router/input/expert derivatives on AD.
            pub fn $packed<C,F>(&self,mut input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedNativeMoeTransformerError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                for (index,layer) in self.layers.iter().enumerate() {input=layer.$gather(communicator.clone()).map_err(FullyShardedNativeMoeTransformerError::Collective)?
                    .forward_packed_with_positions(input,layout,masks,options,|query,key|positions(index,query,key)).map_err(FullyShardedNativeMoeTransformerError::Native)?;}Ok(input)
            }
        }
        impl<$($generics)*,P:GatherTransformerProjection<$backend,B>> FullyShardedNativeMoeTransformerModel<$backend,P> where P::Gathered:TransformerProjection<$backend> {
            /// Actual complete native token logits with only local persistent table/router/expert/head storage.
            pub fn $forward<C,F>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedNativeMoeTransformerError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                let hidden=self.$hidden(input,masks,options,communicator.clone(),positions)?;self.head.$embed(hidden,communicator).map_err(Into::into)
            }
            /// Actual final hidden rows for original full-vocabulary token chunks or sequence heads.
            pub fn $hidden<C,F>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedNativeMoeTransformerError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                let hidden=self.embeddings.$embed(input,communicator.clone()).map_err(FullyShardedNativeMoeTransformerError::Collective)?;
                let hidden=self.backbone.$forward(hidden,masks,options,communicator.clone(),positions)?;
                Ok(if let Some(norm)=&self.normalization {norm.$gather(communicator).map_err(FullyShardedNativeMoeTransformerError::Collective)?.forward(hidden)} else {hidden})
            }
            /// Actual complete flat-document token logits without adding padding or changing loss boundaries.
            pub fn $packed<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedNativeMoeTransformerError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                let hidden=self.$packed_hidden(input,layout,masks,options,communicator.clone(),positions)?;self.head.$embed(hidden,communicator).map_err(Into::into)
            }
            /// Actual packed final hidden states with original mixed-precision embedding row policy.
            pub fn $packed_hidden<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedNativeMoeTransformerError<C::Error,<P::Gathered as TransformerProjection<$backend>>::Error,<$backend as MoeOps>::MoeError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                let hidden=self.embeddings.$embed(super::model::packed_input(input,layout),communicator.clone()).map_err(FullyShardedNativeMoeTransformerError::Collective)?
                    .reshape([layout.tokens(),self.embeddings.hidden_width()]);let hidden=self.backbone.$packed(hidden,layout,masks,options,communicator.clone(),positions)?;
                Ok(if let Some(norm)=&self.normalization {norm.$gather(communicator).map_err(FullyShardedNativeMoeTransformerError::Collective)?.forward(hidden)} else {hidden})
            }
        }
    };
}
moe_transformer_execution!(B,[B:MoeOps],gather_inference,forward_inference,forward_inference,forward_packed_inference,forward_hidden_inference,forward_packed_hidden_inference);
moe_transformer_execution!(Autodiff<B,S>,[B:MoeOps,S:CheckpointStrategy],gather,forward,forward,forward_packed,forward_hidden,forward_packed_hidden);

/// Original actual native model error or original rank-consistent loss completion failure.
#[derive(Debug)]
pub enum FullyShardedNativeMoeTrainingError<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> {
    /// Original gathered attention/router/expert/head graph failure.
    Model(FullyShardedNativeMoeTransformerError<C,P,M>),
    /// Original exact global token-count/AD-reachability/transport failure.
    Loss(ScopedCollectiveError<C>),
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::fmt::Display for FullyShardedNativeMoeTrainingError<C,P,M> {
    fn fmt(&self,f:&mut core::fmt::Formatter<'_>) -> core::fmt::Result {match self {
        Self::Model(error)=>write!(f,"native MoE training graph: {error}"),Self::Loss(error)=>write!(f,"native MoE training loss: {error}")}}
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::error::Error for FullyShardedNativeMoeTrainingError<C,P,M> {}
impl<B:MoeOps,S:CheckpointStrategy,P:GatherTransformerProjection<Autodiff<B,S>,B>> FullyShardedNativeMoeTransformerModel<Autodiff<B,S>,P>
    where P::Gathered:TransformerProjection<Autodiff<B,S>> {
    /// Complete original causal training with scope-bound native gathers and exact global token normalization.
    pub fn forward_causal_with_positions<C,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        masks:DenseAttentionMask<Autodiff<B,S>>,options:DenseAttentionOptions,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,positions:F)
        -> Result<FullyShardedLoss<B,S>,FullyShardedNativeMoeTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<Autodiff<B,S> as MoeOps>::MoeError>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden(input,masks,options,transport.clone(),positions).map_err(FullyShardedNativeMoeTrainingError::Model)?;
        let loss=self.head.forward_causal_loss(hidden,labels,criterion,label_smoothing,transport)
            .map_err(|error|FullyShardedNativeMoeTrainingError::Model(error.into()))?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedNativeMoeTrainingError::Loss)
    }
    /// Original packed-document causal targets, empty-rank graph participation and full-vocabulary smoothing.
    pub fn forward_packed_causal_with_positions<C,F>(&self,input:FullyShardedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options:PackedAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,positions:F)
        -> Result<FullyShardedLoss<B,S>,FullyShardedNativeMoeTrainingError<C::Error,<P::Gathered as TransformerProjection<Autodiff<B,S>>>::Error,<Autodiff<B,S> as MoeOps>::MoeError>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden(input,layout,masks,options,transport.clone(),positions).map_err(FullyShardedNativeMoeTrainingError::Model)?;
        let loss=self.head.forward_packed_causal_loss(hidden,labels,layout,criterion,label_smoothing,transport)
            .map_err(|error|FullyShardedNativeMoeTrainingError::Model(error.into()))?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(FullyShardedNativeMoeTrainingError::Loss)
    }
}
impl<B:MoeOps,P:GatherTransformerProjection<B,B>> FullyShardedNativeMoeTransformerModel<B,P> where P::Gathered:TransformerProjection<B> {
    /// Original complete cached new-token inference, gathering only each current layer and then the head.
    pub fn forward_cached_inference<C,F>(&self,input:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,cache:&mut TransformerKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,mut positions:F)
        -> Result<Tensor<B,3>,FullyShardedNativeMoeTransformerError<C::Error,<P::Gathered as TransformerProjection<B>>::Error,B::MoeError>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let mut hidden=self.embeddings.forward_inference(input,communicator.clone()).map_err(FullyShardedNativeMoeTransformerError::Collective)?;
        cache.validate_layers(self.backbone.layers.len());let rows=(hidden.dims()[0],hidden.dims()[1]);let next=cache.position().checked_add(rows.1).expect("native MoE cache position overflows");
        for (index,layer) in self.backbone.layers.iter().enumerate() {
            hidden=layer.gather_inference(communicator.clone()).map_err(FullyShardedNativeMoeTransformerError::Collective)?
                .forward_cached_with_positions(hidden,visible.clone(),&mut cache.layers_mut()[index],masks.clone(),options,
                    |query,key,position|positions(index,query,key,position)).map_err(FullyShardedNativeMoeTransformerError::Native)?;
            assert_eq!((hidden.dims()[0],hidden.dims()[1]),rows,"native cached MoE layer changed original rows");}
        cache.finish_chunk(next);let hidden=if let Some(norm)=&self.normalization {norm.gather_inference(communicator.clone()).map_err(FullyShardedNativeMoeTransformerError::Collective)?.forward(hidden)} else {hidden};
        self.head.forward_inference(hidden,communicator).map_err(Into::into)
    }
}
