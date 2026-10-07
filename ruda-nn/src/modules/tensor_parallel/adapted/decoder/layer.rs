use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Module,Tensor,Bool,
    DenseAttentionMask,DenseAttentionOptions,ProjectedKvCache,geometry,TensorParallelAdaptedStackLayer,TensorParallelAdaptedDecoderCrossAttention};
use crate::transformer::AdaptedEncoderDecoderLayer;

/// Actual independently selected self/cross/FFN adapters in original encoder-decoder order.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedEncoderDecoderLayer<B: Backend> {
    /// Original dense/adapted self-attention and final FFN, with independent actual norms.
    pub backbone: TensorParallelAdaptedStackLayer<B>,
    /// Original dense/adapted source-memory stage between self-attention and the final FFN.
    pub cross_attention: TensorParallelAdaptedDecoderCrossAttention<B>,
}

impl<B: Backend> TensorParallelAdaptedEncoderDecoderLayer<B> {
    /// Connect actual loaded local stages without new target selection or parameter initialization.
    pub fn from_sharded_layer(layer: AdaptedEncoderDecoderLayer<B>) -> Self {
        let backbone = TensorParallelAdaptedStackLayer::from_sharded_layer(layer.backbone);
        let cross_attention = TensorParallelAdaptedDecoderCrossAttention::from_sharded_stage(layer.cross_attention);
        let width = match &backbone {TensorParallelAdaptedStackLayer::Dense(block)=>block.attention.local.query.weight.val().dims()[0],
            TensorParallelAdaptedStackLayer::Adapted(block)=>geometry(&block.attention.local.query)[0]};
        assert_eq!(width,cross_attention.query_width(),"adapted parallel decoder/cross residual widths differ");
        Self {backbone,cross_attention}
    }

    /// Restore original local native containers without merging adapters or copying their state.
    pub fn into_local_layer(self) -> AdaptedEncoderDecoderLayer<B> {
        AdaptedEncoderDecoderLayer {backbone:self.backbone.into_local_layer(),cross_attention:self.cross_attention.into_local_stage()}
    }

    /// Native unmerged actual selected layers with independently supplied self/cross transports.
    pub fn forward_inference<C,D,F,G>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions,self_communicator: C,cross_communicator: D,self_positions: F,cross_positions: G)
        -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,D: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),G: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = self.backbone.forward_attention_inference(input,self_masks,self_options,self_communicator.clone(),self_positions)?;
        let hidden = self.cross_attention.forward_inference(hidden,memory,memory_masks,memory_options,cross_communicator,cross_positions)?;
        self.backbone.forward_feed_forward_inference(hidden,self_communicator)
    }

    /// Native new decoder rows reuse actual adapted prepared source K/V and run final FFN once.
    pub fn forward_cached_inference<C,D,F,G>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,decoder: &mut ProjectedKvCache<B>,memory: &ProjectedKvCache<B>,
        self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions,
        self_communicator: C,cross_communicator: D,self_positions: F,cross_positions: G) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,D: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),G: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let position = decoder.position();
        let hidden = self.backbone.forward_cached_attention_inference(input,visible,decoder,self_masks,self_options,self_communicator.clone(),self_positions)?;
        let hidden = self.cross_attention.forward_cached_inference(hidden,memory,memory_masks,memory_options,cross_communicator,position,cross_positions)?;
        self.backbone.forward_feed_forward_inference(hidden,self_communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>> {
    /// Original native selected self -> source -> FFN training graph and independently declared KV groups.
    pub fn forward<C,K,D,L,F,G>(&self,input: Tensor<Autodiff<B,S>,3>,memory: Tensor<Autodiff<B,S>,3>,self_masks: DenseAttentionMask<Autodiff<B,S>>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<Autodiff<B,S>>,memory_options: DenseAttentionOptions,self_groups: &AttentionParallelGroups<C,K>,memory_groups: &AttentionParallelGroups<D,L>,
        self_positions: F,cross_positions: G) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,D: BroadcastTensorCollective<B,Error=C::Error>,L: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            G: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        let hidden = self.backbone.forward_attention_with_positions(input,self_masks,self_options,self_groups,self_positions)?;
        let hidden = self.cross_attention.forward(hidden,memory,memory_masks,memory_options,memory_groups,cross_positions)?;
        self.backbone.forward_feed_forward(hidden,self_groups.heads.clone())
    }

    /// Inference-only actual selected decoder history and detached immutable source memory.
    pub fn forward_cached<C,K,D,F,G>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        decoder: &mut ProjectedKvCache<Autodiff<B,S>>,memory: &ProjectedKvCache<Autodiff<B,S>>,self_masks: DenseAttentionMask<Autodiff<B,S>>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<Autodiff<B,S>>,memory_options: DenseAttentionOptions,groups: &AttentionParallelGroups<C,K>,cross_communicator: D,
        self_positions: F,cross_positions: G) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,D: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            G: FnOnce(Tensor<Autodiff<B,S>,4>,usize)->Tensor<Autodiff<B,S>,4> {
        let position = decoder.position();
        let hidden = self.backbone.forward_cached_attention(input,visible,decoder,self_masks,self_options,groups,self_positions)?;
        let hidden = self.cross_attention.forward_cached(hidden,memory,memory_masks,memory_options,cross_communicator,position,cross_positions)?;
        self.backbone.forward_feed_forward(hidden,groups.heads.clone())
    }
}
