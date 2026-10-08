use alloc::{vec::Vec,collections::BTreeSet};
use ruda_model::{module::{Module,ModuleVisitor,Param,ParamId},tensor::{Tensor,Int,Bool,MoeDispatchOps,MoeReceivedOps,VariableTensorCollective,backend::Backend}};
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,collective::{CollectiveScope,ScopedTensorCollective,ScopedCollectiveError}};
use crate::{Dropout,expert_parallel::{ExpertParallelMoeLayer,ExpertParallelMoeError,ExpertParallelSwiGluExperts,ExpertParallelGeometry,ExpertParallelReceived},attention::{DenseAttentionMask,DenseAttentionOptions,
    PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},cache::{ProjectedKvCache,TransformerKvCache},
    loss::CausalCrossEntropyConfig,pool::SequencePooling,fully_sharded::{FullyShardedLoss,complete_fully_sharded_loss}};
use super::{ProjectedGroupedQueryAttention,ProjectedFeedForward,DenseTransformerNorm,NativeMoeTransformerLayer,NativeMoeTransformerError,
    TransformerProjectionShape,TransformerProjection,TransformerEmbeddings,ProjectedTransformerHead,ProjectedTransformerInput,SequenceHeadOutput};
use super::{dense::try_residual_branch,native_attention::{attention_branch,packed_attention_branch,cached_attention_branch},
    projected_paired_model::{embed_projected,embed_packed_projected,check_block}};

/// Original self-attention and actual expert-owned routed/shared FFN, retaining residual/norm order.
#[derive(Module,Debug)]
pub struct ExpertParallelTransformerBlock<B:Backend,P:Module<B>,E:Module<B> =ExpertParallelSwiGluExperts<B>> {
    /// Original actual self-attention projections and head geometry.
    pub attention:ProjectedGroupedQueryAttention<B,P>,
    /// Actual expert-world routed branch with only local persistent expert cubes.
    pub routed:ExpertParallelMoeLayer<B,P,E>,
    /// Original optional actual shared ordinary/gated FFN; absence remains absence.
    pub shared:Option<ProjectedFeedForward<B,P>>,
    /// Original attention affine norm/epsilon.
    pub attention_norm:DenseTransformerNorm<B>,
    /// Original independent FFN affine norm/epsilon.
    pub feed_forward_norm:DenseTransformerNorm<B>,
    /// Original residual-branch dropout.
    pub residual_dropout:Dropout,
    /// Original pre/post-normalization choice.
    pub norm_first:bool,
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>> ExpertParallelTransformerBlock<B,P,E> {
    /// Validate the actual loaded residual and original attention/shared/router widths.
    pub fn validate(&self) {
        self.routed.validate();let width=self.routed.width();
        for projection in [&self.attention.query,&self.attention.key,&self.attention.value] {assert_eq!(projection.dimensions()[0],width,"owned-expert attention input width differs");}
        assert_eq!(self.attention.output.dimensions()[1],width,"owned-expert attention output width differs");
        assert_eq!((self.attention_norm.width(),self.feed_forward_norm.width()),(width,width),"owned-expert original norm widths differ");
        if let Some(shared)=&self.shared {assert_eq!((shared.up.dimensions()[0],shared.down.dimensions()[1]),(width,width),"owned-expert shared residual width differs");}
    }
}
/// Original local graph error or actual cross-rank owned-expert execution error.
#[derive(Debug)]
pub enum ExpertParallelTransformerError<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> {
    /// Original local native dense/routed projection error.
    Local(NativeMoeTransformerError<P,M>),
    /// Actual original expert-world native/projection/transport error.
    Expert(ExpertParallelMoeError<C,P,M>),
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::fmt::Display for ExpertParallelTransformerError<C,P,M> {
    fn fmt(&self,f:&mut core::fmt::Formatter<'_>) -> core::fmt::Result {match self {Self::Local(error)=>write!(f,"native model stage: {error}"),Self::Expert(error)=>write!(f,"owned expert stage: {error}")}}
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::error::Error for ExpertParallelTransformerError<C,P,M> {}
fn local_expert_error<B:MoeReceivedOps,E:ExpertParallelReceived<B>,C:core::fmt::Debug,P:core::fmt::Debug>(
    error:NativeMoeTransformerError<P,B::MoeError>) -> ExpertParallelTransformerError<C,P,E::Error> {
    let error=match error {
        NativeMoeTransformerError::Projection(error)=>NativeMoeTransformerError::Projection(error),
        NativeMoeTransformerError::Routed(crate::NativeMoeLayerError::Router(error))=>NativeMoeTransformerError::Routed(crate::NativeMoeLayerError::Router(error)),
        NativeMoeTransformerError::Routed(crate::NativeMoeLayerError::Experts(error))=>NativeMoeTransformerError::Routed(crate::NativeMoeLayerError::Experts(E::routing_error(error))),
    };ExpertParallelTransformerError::Local(error)
}
/// Every actual original local or expert-world layer, preserving its original loaded order.
#[derive(Module,Debug)]
pub enum ExpertParallelTransformerLayer<B:Backend,P:Module<B>,E:Module<B> =ExpertParallelSwiGluExperts<B>> {
    /// Original native dense or local routed/shared block.
    Local(NativeMoeTransformerLayer<B,P>),
    /// Actual expert-owned routed/shared block.
    Parallel(ExpertParallelTransformerBlock<B,P,E>),
}
/// Complete original embeddings, mixed local/expert backbone, final norm and native head.
#[derive(Module,Debug)]
pub struct ExpertParallelTransformerModel<B:Backend,P:Module<B>,E:Module<B> =ExpertParallelSwiGluExperts<B>> {
    /// Actual original token and optional learned-position/type tables.
    pub embeddings:TransformerEmbeddings<B>,
    /// Every actual original local/owned-expert layer in order.
    pub layers:Vec<ExpertParallelTransformerLayer<B,P,E>>,
    /// Original optional independent final affine norm.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Actual original dense/packed/adapter output head.
    pub head:ProjectedTransformerHead<B,P>,
}
struct FloatIds(BTreeSet<ParamId>);
impl<B:Backend> ModuleVisitor<B> for FloatIds {
    fn visit_float<const D:usize>(&mut self,parameter:&Param<Tensor<B,D>>) {self.0.insert(parameter.id);}
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>> ExpertParallelTransformerModel<B,P,E> {
    /// Connect actual original loaded parts without creating unused dense stand-in FFNs or guessed model topology.
    pub fn from_expert_parts(embeddings:TransformerEmbeddings<B>,layers:Vec<ExpertParallelTransformerLayer<B,P,E>>,normalization:Option<DenseTransformerNorm<B>>,head:ProjectedTransformerHead<B,P>) -> Self {
        let width=embeddings.token.weight.val().dims()[1];assert_eq!(head.projection.dimensions()[0],width,"expert model original head/input width differs");
        if let Some(norm)=&normalization {assert_eq!(norm.width(),width,"expert model final norm width differs");}
        for layer in &layers {match layer {
            ExpertParallelTransformerLayer::Local(layer)=>{assert_eq!(layer.width(),width,"local model residual width differs");match layer {
                NativeMoeTransformerLayer::Dense(block)=>check_block(block,width),NativeMoeTransformerLayer::Routed(block)=>block.validate()}},
            ExpertParallelTransformerLayer::Parallel(block)=>{block.validate();assert_eq!(block.routed.width(),width,"owned-expert residual width differs");},
        }}Self {embeddings,layers,normalization,head}
    }
    /// Original actual per-layer cache topology, not replicated expert weights.
    pub fn new_kv_cache(&self,capacity:usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.layers.len(),capacity)}
    /// Actual local expert IDs, for caller-owned optimizer/checkpoint/gradient-distribution policy.
    pub fn owned_expert_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids=BTreeSet::new();for layer in &self.layers {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
            ids.extend(block.routed.experts.parameter_ids());}}ids.into_iter().collect()
    }
    /// Actual non-owned-expert floating IDs. This does not assume that a custom projection is replicated;
    /// callers choose their original replication/data/tensor groups and reductions explicitly.
    pub fn non_expert_parameter_ids(&self) -> Vec<ParamId> {
        let owned=self.owned_expert_parameter_ids().into_iter().collect::<BTreeSet<_>>();let mut ids=FloatIds(BTreeSet::new());self.visit(&mut ids);
        ids.0.difference(&owned).copied().collect()
    }
    fn normalize<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {if let Some(norm)=&self.normalization {norm.forward(hidden)} else {hidden}}
}
impl<B:Backend,P:TransformerProjectionShape<B>> ExpertParallelTransformerModel<B,P> {
    /// Original constructor remains cube-only, including inference when the actual stack contains only local layers.
    pub fn from_parts(embeddings:TransformerEmbeddings<B>,layers:Vec<ExpertParallelTransformerLayer<B,P>>,normalization:Option<DenseTransformerNorm<B>>,head:ProjectedTransformerHead<B,P>) -> Self {
        Self::from_expert_parts(embeddings,layers,normalization,head)
    }
}
macro_rules! expert_model_execution {
    ($backend:ty,[$($generics:tt)*],$routed:ident,$feed:ident,$forward:ident,$packed:ident,$cached:ident,$hidden_with:ident,$packed_hidden_with:ident,$sequence:ident,$packed_sequence:ident) => {
        impl<$($generics)*,P:TransformerProjection<$backend>,E:ExpertParallelReceived<$backend>> ExpertParallelTransformerBlock<$backend,P,E> {
            fn $feed<C:VariableTensorCollective<B>,const D:usize>(&self,hidden:Tensor<$backend,D>,communicator:C)
                -> Result<Tensor<$backend,D>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>> {
                try_residual_branch(hidden,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,|source| {
                    let output=self.routed.$routed(source.clone(),communicator).map_err(ExpertParallelTransformerError::Expert)?;
                    if let Some(shared)=&self.shared {Ok(output+shared.forward(source).map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))?)} else {Ok(output)}
                })
            }
            /// Original self-attention stage followed by actual expert exchange, optional shared FFN and original residual/norm order.
            pub fn $forward<C:VariableTensorCollective<B>,F>(&self,input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.validate();let hidden=attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,masks,options,positions)
                    .map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))?;self.$feed(hidden,communicator)
            }
            /// Original independent packed documents with actual native expert-owned routed/shared FFN rows.
            pub fn $packed<C:VariableTensorCollective<B>,F>(&self,input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,positions:F) -> Result<Tensor<$backend,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                self.validate();let hidden=packed_attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,layout,masks,options,positions)
                    .map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))?;self.$feed(hidden,communicator)
            }
            /// Original cached attention/new-row-only FFN, with all expert-world ranks participating.
            pub fn $cached<C:VariableTensorCollective<B>,F>(&self,input:Tensor<$backend,3>,visible:Option<Tensor<$backend,2,Bool>>,cache:&mut ProjectedKvCache<$backend>,
                masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>,usize)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.validate();let hidden=cached_attention_branch(&self.attention,&self.attention_norm,&self.residual_dropout,self.norm_first,input,visible,cache,masks,options,positions)
                    .map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))?;self.$feed(hidden,communicator)
            }
        }
        impl<$($generics)*,P:TransformerProjection<$backend>,E:ExpertParallelReceived<$backend>> ExpertParallelTransformerLayer<$backend,P,E> {
            /// Execute only actual cached new-token rows, retaining the original local/owned-expert choice.
            pub fn $cached<C:VariableTensorCollective<B>,F>(&self,input:Tensor<$backend,3>,visible:Option<Tensor<$backend,2,Bool>>,cache:&mut ProjectedKvCache<$backend>,
                masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>,usize)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                match self {Self::Local(layer)=>layer.forward_cached_with_positions(input,visible,cache,masks,options,positions).map_err(local_expert_error::<$backend,E,_,_>),
                    Self::Parallel(block)=>block.$cached(input,visible,cache,masks,options,communicator,positions)}
            }
            /// Execute only this actual original local/owned-expert layer choice.
            pub fn $forward<C:VariableTensorCollective<B>,F>(&self,input:Tensor<$backend,3>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnOnce(Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                match self {Self::Local(layer)=>layer.forward_with_positions(input,masks,options,positions).map_err(local_expert_error::<$backend,E,_,_>),
                    Self::Parallel(block)=>block.$forward(input,masks,options,communicator,positions)}
            }
            /// Execute actual original independent-document rows without changing dense/routed layer choices.
            pub fn $packed<C:VariableTensorCollective<B>,F>(&self,input:Tensor<$backend,2>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,positions:F) -> Result<Tensor<$backend,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnOnce(Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                match self {Self::Local(layer)=>layer.forward_packed_with_positions(input,layout,masks,options,positions).map_err(local_expert_error::<$backend,E,_,_>),
                    Self::Parallel(block)=>block.$packed(input,layout,masks,options,communicator,positions)}
            }
        }
        impl<$($generics)*,P:TransformerProjection<$backend>,E:ExpertParallelReceived<$backend>> ExpertParallelTransformerModel<$backend,P,E> {
            /// Complete actual new-token model logits; source caches commit only a whole-stack chunk boundary.
            /// Partial errors retain the original explicit cache restore contract.
            pub fn $cached<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<$backend>,visible:Option<Tensor<$backend,2,Bool>>,cache:&mut TransformerKvCache<$backend>,
                masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>,usize)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                cache.validate_layers(self.layers.len());let rows=input.tokens.dims();let next=cache.position().checked_add(rows[1]).expect("expert model cache position overflows");
                let mut hidden=embed_projected(&self.embeddings,input);for (index,layer) in self.layers.iter().enumerate() {
                    hidden=layer.$cached(hidden,visible.clone(),&mut cache.layers_mut()[index],masks.clone(),options,communicator.clone(),|query,key,position|positions(index,query,key,position))?;
                    assert_eq!((hidden.dims()[0],hidden.dims()[1]),(rows[0],rows[1]),"expert model cached layer changed actual token rows");}
                cache.finish_chunk(next);self.head.forward(self.normalize(hidden)).map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))
            }
            /// Complete original native hidden graph with caller-owned per-layer attention/position/transport policies.
            pub fn $hidden_with<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<$backend>,communicator:C,mut layer:F)
                -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnMut(usize,&ExpertParallelTransformerLayer<$backend,P,E>,Tensor<$backend,3>,C)
                    -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>> {
                let rows=input.tokens.dims();let width=self.embeddings.token.weight.val().dims()[1];let mut hidden=embed_projected(&self.embeddings,input);
                for (index,block) in self.layers.iter().enumerate() {hidden=layer(index,block,hidden,communicator.clone())?;
                    assert_eq!(hidden.dims(),[rows[0],rows[1],width],"expert model layer changed actual source token axes");}Ok(self.normalize(hidden))
            }
            /// Complete original native packed hidden graph with no inferred document positions or attention windows.
            pub fn $packed_hidden_with<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,communicator:C,mut layer:F)
                -> Result<Tensor<$backend,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnMut(usize,&ExpertParallelTransformerLayer<$backend,P,E>,Tensor<$backend,2>,C)
                    -> Result<Tensor<$backend,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>> {
                let shape=[layout.tokens(),self.embeddings.token.weight.val().dims()[1]];let mut hidden=embed_packed_projected(&self.embeddings,input,layout);
                for (index,block) in self.layers.iter().enumerate() {hidden=layer(index,block,hidden,communicator.clone())?;
                    assert_eq!(hidden.dims(),shape,"expert model packed layer changed original document rows");}Ok(self.normalize(hidden))
            }
            /// Complete actual dense-axis token logits over the original local/owned-expert layer order.
            pub fn $forward<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,options:DenseAttentionOptions,communicator:C,mut positions:F)
                -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                let hidden=self.$hidden_with(input,communicator,|index,layer,hidden,transport|layer.$forward(hidden,masks.clone(),options,transport,|query,key|positions(index,query,key)))?;
                self.head.forward(hidden).map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))
            }
            /// Pool only actual visible tokens, retaining the original head, empty-row validity and I64 counts.
            pub fn $sequence<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<$backend>,visible:Tensor<$backend,2,Bool>,
                pooling:SequencePooling,communicator:C,layer:F) -> Result<SequenceHeadOutput<$backend>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnMut(usize,&ExpertParallelTransformerLayer<$backend,P,E>,Tensor<$backend,3>,C)
                    -> Result<Tensor<$backend,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>> {
                self.head.forward_sequence(self.$hidden_with(input,communicator,layer)?,visible,pooling)
                    .map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))
            }
            /// Pool each original packed document independently, without padding or cross-document targets.
            pub fn $packed_sequence<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,
                visible:Option<Tensor<$backend,1,Bool>>,pooling:SequencePooling,communicator:C,layer:F)
                -> Result<SequenceHeadOutput<$backend>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnMut(usize,&ExpertParallelTransformerLayer<$backend,P,E>,Tensor<$backend,2>,C)
                    -> Result<Tensor<$backend,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>> {
                self.head.forward_packed_sequences(self.$packed_hidden_with(input,layout,communicator,layer)?,layout,visible,pooling)
                    .map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))
            }
            /// Complete actual packed token logits with original independent-document masks and positions.
            pub fn $packed<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,masks:Option<&[PackedDocumentAttentionMask<$backend>]>,
                options:PackedAttentionOptions,communicator:C,mut positions:F) -> Result<Tensor<$backend,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>>
                where F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                let hidden=self.$packed_hidden_with(input,layout,communicator,|index,layer,hidden,transport|layer.$packed(hidden,layout,masks,options,transport,|query,key|positions(index,query,key)))?;
                self.head.forward(hidden).map_err(|error|ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error)))
            }
        }
    };
}
expert_model_execution!(B,[B:MoeDispatchOps+MoeReceivedOps],forward_inference,feed_forward_inference,forward_with_positions_inference,forward_packed_with_positions_inference,forward_cached_with_positions_inference,forward_hidden_with_inference,forward_packed_hidden_with_inference,forward_sequence_with_inference,forward_packed_sequences_with_inference);
expert_model_execution!(Autodiff<B,S>,[B:MoeDispatchOps+MoeReceivedOps,S:CheckpointStrategy],forward,feed_forward,forward_with_positions,forward_packed_with_positions,forward_cached_with_positions,forward_hidden_with,forward_packed_hidden_with,forward_sequence_with,forward_packed_sequences_with);

/// Reuse original exact global integer-count loss completion for expert-exchanged derivatives;
/// local expert cubes are NOT data shards, and custom non-expert gradient groups remain explicit.
pub type ExpertParallelLoss<B,S> = FullyShardedLoss<B,S>;
/// Original model failure or exact expert-world loss-completion failure.
#[derive(Debug)]
pub enum ExpertParallelTrainingError<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> {
    /// Actual original model/projection/transport execution failure.
    Model(ExpertParallelTransformerError<C,P,M>),
    /// Original exact global count/AD-reachability/transport failure.
    Loss(ScopedCollectiveError<C>),
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::fmt::Display for ExpertParallelTrainingError<C,P,M> {
    fn fmt(&self,f:&mut core::fmt::Formatter<'_>) -> core::fmt::Result {match self {Self::Model(error)=>write!(f,"expert model: {error}"),Self::Loss(error)=>write!(f,"expert loss: {error}")}}
}
impl<C:core::fmt::Debug,P:core::fmt::Debug,M:core::fmt::Debug> core::error::Error for ExpertParallelTrainingError<C,P,M> {}
impl<B:MoeDispatchOps+MoeReceivedOps,S:CheckpointStrategy,P:TransformerProjection<Autodiff<B,S>>,E:ExpertParallelReceived<Autodiff<B,S>>>
    ExpertParallelTransformerModel<Autodiff<B,S>,P,E> {
    /// Complete original chunked causal objective and exact global effective-token mean basis on the expert world.
    pub fn forward_causal_with<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<Autodiff<B,S>>,labels:Tensor<Autodiff<B,S>,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,layer:F)
        -> Result<ExpertParallelLoss<B,S>,ExpertParallelTrainingError<C::Error,P::Error,E::Error>>
        where F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P,E>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)
            -> Result<Tensor<Autodiff<B,S>,3>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>> {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden_with(input,transport,layer).map_err(ExpertParallelTrainingError::Model)?;
        let loss=criterion.try_forward_hidden_with_smoothing(hidden,labels,|rows|self.head.forward(rows),label_smoothing)
            .map_err(|error|ExpertParallelTrainingError::Model(ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error))))?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(ExpertParallelTrainingError::Loss)
    }
    /// Complete original packed-document chunked objective without cross-document targets or unused-rank backward omissions.
    pub fn forward_packed_causal_with<C:VariableTensorCollective<B>,F>(&self,input:ProjectedTransformerInput<Autodiff<B,S>,1>,labels:Tensor<Autodiff<B,S>,1,Int>,
        layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,layer:F)
        -> Result<ExpertParallelLoss<B,S>,ExpertParallelTrainingError<C::Error,P::Error,E::Error>>
        where F:FnMut(usize,&ExpertParallelTransformerLayer<Autodiff<B,S>,P,E>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)
            -> Result<Tensor<Autodiff<B,S>,2>,ExpertParallelTransformerError<C::Error,P::Error,E::Error>> {
        let scope=CollectiveScope::<B,S>::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden_with(input,layout,transport,layer).map_err(ExpertParallelTrainingError::Model)?;
        let loss=criterion.try_forward_packed_hidden_with_smoothing(hidden,labels,layout,|rows|self.head.forward(rows),label_smoothing)
            .map_err(|error|ExpertParallelTrainingError::Model(ExpertParallelTransformerError::Local(NativeMoeTransformerError::Projection(error))))?;
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator).map_err(ExpertParallelTrainingError::Loss)
    }
}
