use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Tensor};
use super::{TensorParallelAdaptedTransformerBlock,TensorParallelAdaptedStackLayer};
use super::super::transformer::residual;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},cache::ProjectedKvCache};
use ruda_model::tensor::Bool;
use crate::attention::{PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask};

impl<B: Backend> TensorParallelAdaptedTransformerBlock<B> {
    /// Native packed adapted self-attention alone, retaining exact independent-document boundaries.
    pub fn forward_packed_attention_inference<C,F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        assert_eq!(input.dims()[0],layout.tokens(),"native adapted packed self boundaries differ from actual rows");
        residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.local.project_packed(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"native adapted packed self positions changed local heads");
            self.attention.forward_packed_projected_inference(query,key,value,layout,layout,masks,options,communicator)
        },|branch|self.residual_dropout.forward(branch))
    }

    /// Native adapted self-attention stage alone, before an actual source-memory stage.
    pub fn forward_attention_inference<C,F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,
        communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.local.project(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"native adapted attention positions changed local heads");
            self.attention.forward_projected_inference(query,key,value,masks,options,communicator)
        },|branch|self.residual_dropout.forward(branch))
    }

    /// Native adapted FFN stage after actual self/cross attention, on dense or flat token rows.
    pub fn forward_feed_forward_inference<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<B,D>,communicator: C)
        -> Result<Tensor<B,D>,C::Error> {
        residual(input,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward_inference(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }

    /// Native new-chunk adapted self-attention without prematurely running the final FFN.
    pub fn forward_cached_attention_inference<C,F>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        residual(input,&self.attention_norm,self.norm_first,|source|
            self.attention.forward_cached_inference(source,visible,cache,masks,options,communicator,positions),|branch|self.residual_dropout.forward(branch))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedTransformerBlock<Autodiff<B,S>> {
    /// Actual packed adapted self-attention stage, before the original paired-source stage.
    pub fn forward_packed_attention<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        assert_eq!(input.dims()[0],layout.tokens(),"adapted parallel packed self boundaries differ from actual rows");
        residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed(source.clone(),source.clone(),source,groups)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted parallel packed self positions changed local geometry");
            self.attention.forward_packed_projected(query,key,value,layout,layout,masks,options,groups.heads.clone())
        },|branch|self.residual_dropout.forward(branch))
    }

    /// Actual selected adapters on the self-attention stage, retaining original norm/residual order.
    pub fn forward_attention_with_positions<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project(source.clone(),source.clone(),source,groups)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted parallel self positions changed local geometry");
            self.attention.forward_projected(query,key,value,masks,options,groups.heads.clone())
        },|branch|self.residual_dropout.forward(branch))
    }

    /// Actual dense/packed FFN stage only; no attention or base-merge replay occurs.
    pub fn forward_feed_forward<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        residual(input,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }

    /// Cached adapted self-attention alone, retaining only actual positioned local K/V.
    pub fn forward_cached_attention<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project(source.clone(),source.clone(),source,groups)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key,cache.position());
            assert_eq!((query.dims(),key.dims()),geometry,"cached adapted self positions changed local heads");
            self.attention.forward_cached_projected(query,key,value,visible,cache,masks,options,groups.heads.clone())
        },|branch|self.residual_dropout.forward(branch))
    }
}

impl<B: Backend> TensorParallelAdaptedStackLayer<B> {
    /// Actual native packed dense/adapted attention stage before source memory.
    pub fn forward_packed_attention_inference<C,F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        match self {Self::Dense(block)=>block.forward_packed_attention_inference(input,layout,masks,options,communicator,positions),
            Self::Adapted(block)=>block.forward_packed_attention_inference(input,layout,masks,options,communicator,positions)}
    }

    /// Native actual dense/adapted self-attention stage before explicit cross-attention.
    pub fn forward_attention_inference<C,F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F)
        -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_attention_inference(input,masks,options,communicator,positions),
            Self::Adapted(block)=>block.forward_attention_inference(input,masks,options,communicator,positions)}
    }

    /// Native actual final FFN stage on flat or dense token rows.
    pub fn forward_feed_forward_inference<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<B,D>,communicator: C) -> Result<Tensor<B,D>,C::Error> {
        match self {Self::Dense(block)=>residual(input,&block.feed_forward_norm,block.norm_first,|source|block.feed_forward.forward_inference(source,communicator),
            |branch|block.residual_dropout.forward(branch)),Self::Adapted(block)=>block.forward_feed_forward_inference(input,communicator)}
    }

    /// Native decoder cache stage on the actual dense/adapted self-attention kind.
    pub fn forward_cached_attention_inference<C,F>(&self,input: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,cache: &mut ProjectedKvCache<B>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_cached_attention_inference(input,visible,cache,masks,options,communicator,positions),
            Self::Adapted(block)=>block.forward_cached_attention_inference(input,visible,cache,masks,options,communicator,positions)}
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedStackLayer<Autodiff<B,S>> {
    /// Actual selected/unselected packed self stage, preserving the original layer choice.
    pub fn forward_packed_attention<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        match self {Self::Dense(block)=>block.forward_packed_attention(input,layout,masks,options,groups,positions),
            Self::Adapted(block)=>block.forward_packed_attention(input,layout,masks,options,groups,positions)}
    }

    /// Original selected/unselected self-attention stage with explicit local positions/groups.
    pub fn forward_attention_with_positions<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        match self {Self::Dense(block)=>block.forward_attention_with_positions(input,masks,options,groups,positions),
            Self::Adapted(block)=>block.forward_attention_with_positions(input,masks,options,groups,positions)}
    }

    /// Actual FFN stage with native full-residual SUM and no premature source-memory omission.
    pub fn forward_feed_forward<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        match self {Self::Dense(block)=>residual(input,&block.feed_forward_norm,block.norm_first,|source|block.feed_forward.forward(source,communicator),
            |branch|block.residual_dropout.forward(branch)),Self::Adapted(block)=>block.forward_feed_forward(input,communicator)}
    }

    /// Actual cached self stage, leaving source attention and final FFN to the original layer order.
    pub fn forward_cached_attention<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        cache: &mut ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>,usize)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        match self {Self::Dense(block)=>block.forward_cached_attention(input,visible,cache,masks,options,groups,positions),
            Self::Adapted(block)=>block.forward_cached_attention(input,visible,cache,masks,options,groups,positions)}
    }
}
