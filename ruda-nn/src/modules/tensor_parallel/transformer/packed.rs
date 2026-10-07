use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,TensorParallelTransformerBlock,Tensor,residual};
use crate::attention::{PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask};

impl<B: Backend> TensorParallelTransformerBlock<B> {
    /// Native independent-document inference on actual local heads and flat token rows.
    pub fn forward_packed_inference<C,F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions,communicator: C,positions: F)
        -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        assert_eq!(input.dims()[0],layout.tokens(),"native parallel packed boundaries differ from actual rows");
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.local.project_packed(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"native parallel packed positions changed local heads");
            self.attention.forward_packed_projected_inference(query,key,value,layout,layout,masks,options,communicator.clone())
        },|branch|self.residual_dropout.forward(branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward_inference(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelTransformerBlock<Autodiff<B,S>> {
    /// Actual packed documents with rank-local heads and explicit optional per-document masks.
    /// Neither labels nor synthetic separators determine document boundaries.
    pub fn forward_packed<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        let hidden = self.forward_packed_attention(input,layout,masks,options,groups,positions)?;
        self.forward_packed_feed_forward(hidden,groups.heads.clone())
    }

    /// Packed attention/residual stage alone for explicit encoder-decoder composition.
    pub fn forward_packed_attention<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        assert_eq!(input.dims()[0],layout.tokens(),"parallel packed token metadata differs from actual rows");
        residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed_self(source,groups)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"parallel packed positions changed actual local geometry");
            self.attention.forward_packed_projected(query,key,value,layout,layout,masks,options,groups.heads.clone())
        },|branch|self.residual_dropout.forward(branch))
    }

    /// Flat-token FFN/residual stage with the exact loaded local activation and weights.
    pub fn forward_packed_feed_forward<C: BroadcastTensorCollective<B>>(&self,input: Tensor<Autodiff<B,S>,2>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        residual(input,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }
}
