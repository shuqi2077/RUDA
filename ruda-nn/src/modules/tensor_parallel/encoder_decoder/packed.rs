use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Tensor};
use super::{TensorParallelCrossAttentionBlock,TensorParallelEncoderDecoderLayer,TensorParallelEncoderDecoderStack};
use crate::attention::{PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask};

impl<B: Backend> TensorParallelCrossAttentionBlock<B> {
    fn packed_source(&self,input: Tensor<B,2>) -> Tensor<B,2> {if self.norm_first {self.query_norm.forward(input)} else {input}}
    fn packed_memory(&self,input: Tensor<B,2>) -> Tensor<B,2> {if let Some(norm) = &self.memory_norm {norm.forward(input)} else {input}}
    fn packed_finish(&self,input: Tensor<B,2>,branch: Tensor<B,2>) -> Tensor<B,2> {
        let output = input+self.residual_dropout.forward(branch);
        if self.norm_first {output} else {self.query_norm.forward(output)}
    }

    /// Native actual paired query/source documents with independent local positions and widths.
    pub fn forward_packed_inference<C,F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions,communicator: C,positions: F)
        -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        assert_eq!(input.dims()[0],query_layout.tokens(),"native packed cross query boundaries differ");
        assert_eq!(memory.dims()[0],key_layout.tokens(),"native packed cross memory boundaries differ");
        let memory = self.packed_memory(memory);
        let (query,key,value) = self.attention.local.project_packed(self.packed_source(input.clone()),memory.clone(),memory);
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"native packed cross positions changed local query/source geometry");
        Ok(self.packed_finish(input,self.attention.forward_packed_projected_inference(query,key,value,query_layout,key_layout,masks,options,communicator)?))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelCrossAttentionBlock<Autodiff<B,S>> {
    /// Native packed cross training with head-input SUM gradients and exact KV replica groups.
    pub fn forward_packed<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,memory: Tensor<Autodiff<B,S>,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        assert_eq!(input.dims()[0],query_layout.tokens(),"packed parallel cross query boundaries differ");
        assert_eq!(memory.dims()[0],key_layout.tokens(),"packed parallel cross memory boundaries differ");
        let memory = self.packed_memory(memory);
        let (query,key,value) = self.attention.project_packed(self.packed_source(input.clone()),memory.clone(),memory,groups)?;
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"packed parallel cross positions changed actual local heads");
        Ok(self.packed_finish(input,self.attention.forward_packed_projected(query,key,value,query_layout,key_layout,masks,options,groups.heads.clone())?))
    }
}

impl<B: Backend> TensorParallelEncoderDecoderLayer<B> {
    /// Native packed self -> cross -> FFN execution with independent actual layouts/transports.
    pub fn forward_packed_inference<C,D,F,G>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        self_masks: Option<&[PackedDocumentAttentionMask<B>]>,self_options: PackedAttentionOptions,
        memory_masks: Option<&[PackedDocumentAttentionMask<B>]>,memory_options: PackedAttentionOptions,
        self_communicator: C,cross_communicator: D,self_positions: F,cross_positions: G) -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,D: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),G: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let hidden = self.backbone.forward_packed_attention_inference(input,query_layout,self_masks,self_options,self_communicator.clone(),self_positions)?;
        let hidden = self.cross_attention.forward_packed_inference(hidden,memory,query_layout,key_layout,memory_masks,memory_options,cross_communicator,cross_positions)?;
        self.backbone.forward_packed_feed_forward_inference(hidden,self_communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelEncoderDecoderLayer<Autodiff<B,S>> {
    /// Actual paired packed documents, preserving original stage order and all native gradients.
    pub fn forward_packed<C,K,D,L,F,G>(&self,input: Tensor<Autodiff<B,S>,2>,memory: Tensor<Autodiff<B,S>,2>,query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,
        self_masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,self_options: PackedAttentionOptions,
        memory_masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,memory_options: PackedAttentionOptions,
        self_groups: &AttentionParallelGroups<C,K>,memory_groups: &AttentionParallelGroups<D,L>,self_positions: F,cross_positions: G)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            D: BroadcastTensorCollective<B,Error=C::Error>,L: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            G: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        let hidden = self.backbone.forward_packed_attention(input,query_layout,self_masks,self_options,self_groups,self_positions)?;
        let hidden = self.cross_attention.forward_packed(hidden,memory,query_layout,key_layout,memory_masks,memory_options,memory_groups,cross_positions)?;
        self.backbone.forward_packed_feed_forward(hidden,self_groups.heads.clone())
    }
}

impl<B: Backend> TensorParallelEncoderDecoderStack<B> {
    /// Native flat-token layer sequence with original source document rows.
    pub fn forward_packed_inference_with<E,F>(&self,mut input: Tensor<B,2>,memory: Tensor<B,2>,mut layer: F) -> Result<Tensor<B,2>,E>
        where F: FnMut(usize,&TensorParallelEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelEncoderDecoderStack<Autodiff<B,S>> {
    /// Actual flat-token packed training through every original source/target layer.
    pub fn forward_packed_with<E,F>(&self,mut input: Tensor<Autodiff<B,S>,2>,memory: Tensor<Autodiff<B,S>,2>,mut layer: F)
        -> Result<Tensor<Autodiff<B,S>,2>,E>
        where F: FnMut(usize,&TensorParallelEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)
            -> Result<Tensor<Autodiff<B,S>,2>,E> {
        for (index,block) in self.layers.iter().enumerate() {input = layer(index,block,input,memory.clone())?;}
        Ok(input)
    }
}
