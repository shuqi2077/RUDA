use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Tensor,Dropout,AttentionAdapterTarget,
    TensorParallelAdaptedCrossAttentionBlock,TensorParallelAdaptedDecoderCrossAttention,TensorParallelAdaptedEncoderDecoderLayer};
use crate::attention::{PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask};

impl<B: Backend> TensorParallelAdaptedCrossAttentionBlock<B> {
    /// Native selected cross adapters on exact paired packed source/target documents.
    pub fn forward_packed_inference<C,F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        assert_eq!(input.dims()[0],query_layout.tokens(),"native adapted packed cross query boundaries differ");
        assert_eq!(memory.dims()[0],key_layout.tokens(),"native adapted packed cross source boundaries differ");
        let memory = self.memory(memory);
        let (query,key,value) = self.attention.local.project_packed(self.source(input.clone()),memory.clone(),memory);
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"native adapted packed cross positions changed local geometry");
        Ok(self.finish(input,self.attention.forward_packed_projected_inference(query,key,value,query_layout,key_layout,masks,options,communicator)?))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedCrossAttentionBlock<Autodiff<B,S>> {
    /// Actual native packed cross adapters with source/query widths and head/KV groups unchanged.
    pub fn forward_packed<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,memory: Tensor<Autodiff<B,S>,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        self.forward_packed_with_transforms(input,memory,query_layout,key_layout,masks,options,groups,positions,
            |_,module,input|module.forward(input),|branch|self.residual_dropout.forward(branch))
    }

    /// Actual packed graph with explicit per-target adapter and shared full-residual dropout transforms.
    pub fn forward_packed_with_transforms<C,K,F,A,R>(&self,input: Tensor<Autodiff<B,S>,2>,memory: Tensor<Autodiff<B,S>,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F,mut dropout: A,branch_output: R)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            A: FnMut(AttentionAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,R: FnOnce(Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        assert_eq!(input.dims()[0],query_layout.tokens(),"adapted parallel packed cross query boundaries differ");
        assert_eq!(memory.dims()[0],key_layout.tokens(),"adapted parallel packed cross source boundaries differ");
        let memory = self.memory(memory);
        let (query,key,value) = self.attention.project_packed_with_adapter_dropout(self.source(input.clone()),memory.clone(),memory,groups,&mut dropout)?;
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"adapted parallel packed cross positions changed local heads");
        let branch = self.attention.forward_packed_projected_with_adapter_dropout(query,key,value,query_layout,key_layout,masks,options,groups.heads.clone(),
            |module,input|dropout(AttentionAdapterTarget::Output,module,input))?;
        Ok(self.finish_with(input,branch,branch_output))
    }
}

impl<B: Backend> TensorParallelAdaptedDecoderCrossAttention<B> {
    /// Actual native packed source stage with the original dense/adapted choice.
    pub fn forward_packed_inference<C,F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        match self {Self::Dense(block)=>block.forward_packed_inference(input,memory,query_layout,key_layout,masks,options,communicator,positions),
            Self::Adapted(block)=>block.forward_packed_inference(input,memory,query_layout,key_layout,masks,options,communicator,positions)}
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedDecoderCrossAttention<Autodiff<B,S>> {
    /// Actual selected/unselected source-stage training on corresponding packed documents.
    pub fn forward_packed<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,memory: Tensor<Autodiff<B,S>,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        match self {Self::Dense(block)=>block.forward_packed(input,memory,query_layout,key_layout,masks,options,groups,positions),
            Self::Adapted(block)=>block.forward_packed(input,memory,query_layout,key_layout,masks,options,groups,positions)}
    }
}

impl<B: Backend> TensorParallelAdaptedEncoderDecoderLayer<B> {
    /// Native packed actual self -> source -> FFN sequence with independent local transports.
    pub fn forward_packed_inference<C,D,F,G>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        self_masks: Option<&[PackedDocumentAttentionMask<B>]>,self_options: PackedAttentionOptions,memory_masks: Option<&[PackedDocumentAttentionMask<B>]>,memory_options: PackedAttentionOptions,
        self_communicator: C,cross_communicator: D,self_positions: F,cross_positions: G) -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,D: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),G: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let hidden = self.backbone.forward_packed_attention_inference(input,query_layout,self_masks,self_options,self_communicator.clone(),self_positions)?;
        let hidden = self.cross_attention.forward_packed_inference(hidden,memory,query_layout,key_layout,memory_masks,memory_options,cross_communicator,cross_positions)?;
        self.backbone.forward_feed_forward_inference(hidden,self_communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>> {
    /// Original selected native adapters and independently declared source/query packed policies.
    pub fn forward_packed<C,K,D,L,F,G>(&self,input: Tensor<Autodiff<B,S>,2>,memory: Tensor<Autodiff<B,S>,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        self_masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,self_options: PackedAttentionOptions,
        memory_masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,memory_options: PackedAttentionOptions,
        self_groups: &AttentionParallelGroups<C,K>,memory_groups: &AttentionParallelGroups<D,L>,self_positions: F,cross_positions: G) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,D: BroadcastTensorCollective<B,Error=C::Error>,L: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            G: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        let hidden = self.backbone.forward_packed_attention(input,query_layout,self_masks,self_options,self_groups,self_positions)?;
        let hidden = self.cross_attention.forward_packed(hidden,memory,query_layout,key_layout,memory_masks,memory_options,memory_groups,cross_positions)?;
        self.backbone.forward_feed_forward(hidden,self_groups.heads.clone())
    }
}
