use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Dropout,Tensor,column,row,row_inference};
use super::{TensorParallelAdaptedGroupedQueryAttention,TensorParallelAdaptedTransformerBlock};
use super::super::transformer::{residual,TensorParallelResidualStage};
use crate::{attention::{PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask,
    packed_scaled_dot_product_attention,packed_scaled_dot_product_attention_masked},
    transformer::{AttentionAdapterTarget,FeedForwardAdapterTarget}};

impl<B: Backend> TensorParallelAdaptedGroupedQueryAttention<B> {
    fn packed_context(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,query_layout: &PackedSequenceLayout,
        key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<B>]>,options: PackedAttentionOptions) -> Tensor<B,2> {
        let [tokens,heads,width] = query.dims();
        assert_eq!((heads,width),(self.local.query_heads,self.local.head_dimension),"adapted packed parallel query geometry differs");
        assert_eq!((key.dims()[1],key.dims()[2]),(self.local.kv_heads,width),"adapted packed parallel key geometry differs");
        assert_eq!((value.dims()[1],value.dims()[2]),(self.local.kv_heads,width),"adapted packed parallel value geometry differs");
        let context = if let Some(masks) = masks {
            packed_scaled_dot_product_attention_masked(query,key,value,query_layout,key_layout,masks,options,Some(&self.local.dropout))
        } else {packed_scaled_dot_product_attention(query,key,value,query_layout,key_layout,options,Some(&self.local.dropout))};
        context.reshape([tokens,heads.checked_mul(width).expect("adapted packed parallel context overflow")])
    }

    /// Actual per-document local-head inference, followed by separate base/adapter row SUMs.
    pub fn forward_packed_projected_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<B>]>,
        options: PackedAttentionOptions,communicator: C) -> Result<Tensor<B,2>,C::Error> {
        row_inference(&self.local.output,self.packed_context(query,key,value,query_layout,key_layout,masks,options),&communicator)
    }

    /// Native packed self/cross inference without merging adapters or gathering global heads.
    pub fn forward_packed_inference<C: BroadcastTensorCollective<B>>(&self,query: Tensor<B,2>,key: Tensor<B,2>,value: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<B>]>,
        options: PackedAttentionOptions,communicator: C) -> Result<Tensor<B,2>,C::Error> {
        let (query,key,value) = self.local.project_packed(query,key,value);
        self.forward_packed_projected_inference(query,key,value,query_layout,key_layout,masks,options,communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedGroupedQueryAttention<Autodiff<B,S>> {
    /// Original flat-token selected adapters and exact A/B storage with explicit KV replicas.
    pub fn project_packed<C,K>(&self,query: Tensor<Autodiff<B,S>,2>,key: Tensor<Autodiff<B,S>,2>,value: Tensor<Autodiff<B,S>,2>,
        groups: &AttentionParallelGroups<C,K>) -> Result<(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.project_packed_with_adapter_dropout(query,key,value,groups,|_,module,input|module.forward(input))
    }

    /// Explicit column-input dropout at each original adapter's actual A dtype.
    /// Replicated A/input draws must match the corresponding declared head ranks.
    pub fn project_packed_with_adapter_dropout<C,K,F>(&self,query: Tensor<Autodiff<B,S>,2>,key: Tensor<Autodiff<B,S>,2>,value: Tensor<Autodiff<B,S>,2>,
        groups: &AttentionParallelGroups<C,K>,mut dropout: F)
        -> Result<(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnMut(AttentionAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        let queries = query.dims()[0];let keys = key.dims()[0];
        assert_eq!(value.dims()[0],keys,"adapted packed parallel K/V token counts differ");
        let query = column(&self.local.query,query,&groups.heads,None::<&K>,|module,input|dropout(AttentionAdapterTarget::Query,module,input))?;
        let key = column(&self.local.key,key,&groups.heads,groups.kv_replicas.as_ref(),|module,input|dropout(AttentionAdapterTarget::Key,module,input))?;
        let value = column(&self.local.value,value,&groups.heads,groups.kv_replicas.as_ref(),|module,input|dropout(AttentionAdapterTarget::Value,module,input))?;
        Ok((query.reshape([queries,self.local.query_heads,self.local.head_dimension]),
            key.reshape([keys,self.local.kv_heads,self.local.head_dimension]),value.reshape([keys,self.local.kv_heads,self.local.head_dimension])))
    }

    /// Local independent documents with actual row-sharded dense/LoRA output derivatives.
    pub fn forward_packed_projected<C: BroadcastTensorCollective<B>>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,
        options: PackedAttentionOptions,communicator: C) -> Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        self.forward_packed_projected_with_adapter_dropout(query,key,value,query_layout,key_layout,masks,options,communicator,|module,input|module.forward(input))
    }

    /// Explicit local-context adapter dropout before the original row A projection.
    pub fn forward_packed_projected_with_adapter_dropout<C,F>(&self,query: Tensor<Autodiff<B,S>,3>,key: Tensor<Autodiff<B,S>,3>,value: Tensor<Autodiff<B,S>,3>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,
        options: PackedAttentionOptions,communicator: C,dropout: F) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        row(&self.local.output,self.packed_context(query,key,value,query_layout,key_layout,masks,options),&communicator,dropout)
    }

    /// Packed generic self/cross training with exact independent query/source boundaries.
    pub fn forward_packed<C,K>(&self,query: Tensor<Autodiff<B,S>,2>,key: Tensor<Autodiff<B,S>,2>,value: Tensor<Autodiff<B,S>,2>,
        query_layout: &PackedSequenceLayout,key_layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,
        options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        let (query,key,value) = self.project_packed(query,key,value,groups)?;
        self.forward_packed_projected(query,key,value,query_layout,key_layout,masks,options,groups.heads.clone())
    }
}

impl<B: Backend> TensorParallelAdaptedTransformerBlock<B> {
    /// Original native flat-token norm/residual/FFN order and caller-owned packed positions.
    pub fn forward_packed_inference<C,F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,masks: Option<&[PackedDocumentAttentionMask<B>]>,
        options: PackedAttentionOptions,communicator: C,positions: F) -> Result<Tensor<B,2>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        assert_eq!(input.dims()[0],layout.tokens(),"adapted native parallel packed boundaries differ from actual rows");
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.local.project_packed(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted native packed positions changed local heads");
            self.attention.forward_packed_projected_inference(query,key,value,layout,layout,masks,options,communicator.clone())
        },|branch|self.residual_dropout.forward(branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,|source|self.feed_forward.forward_inference(source,communicator),
            |branch|self.residual_dropout.forward(branch))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedTransformerBlock<Autodiff<B,S>> {
    /// Actual packed adapter training, retaining per-document masks and original local heads.
    pub fn forward_packed<C,K,F>(&self,input: Tensor<Autodiff<B,S>,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        self.forward_packed_with_dropout(input,layout,masks,options,groups,positions,
            |_,module,input|module.forward(input),|_,module,input|module.forward(input),|_,branch|self.residual_dropout.forward(branch))
    }

    /// Explicit actual per-target adapter and replicated residual dropout transforms.
    pub fn forward_packed_with_dropout<C,K,F,A,G,R>(&self,input: Tensor<Autodiff<B,S>,2>,layout: &PackedSequenceLayout,
        masks: Option<&[PackedDocumentAttentionMask<Autodiff<B,S>>]>,options: PackedAttentionOptions,groups: &AttentionParallelGroups<C,K>,positions: F,
        mut attention_dropout: A,feed_forward_dropout: G,mut branch_output: R) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            A: FnMut(AttentionAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,
            G: FnMut(FeedForwardAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2>,
            R: FnMut(TensorParallelResidualStage,Tensor<Autodiff<B,S>,2>)->Tensor<Autodiff<B,S>,2> {
        assert_eq!(input.dims()[0],layout.tokens(),"adapted parallel packed boundaries differ from actual rows");
        let hidden = residual(input,&self.attention_norm,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed_with_adapter_dropout(source.clone(),source.clone(),source,groups,&mut attention_dropout)?;
            let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted parallel packed positions changed local heads");
            self.attention.forward_packed_projected_with_adapter_dropout(query,key,value,layout,layout,masks,options,groups.heads.clone(),
                |module,input|attention_dropout(AttentionAdapterTarget::Output,module,input))
        },|branch|branch_output(TensorParallelResidualStage::Attention,branch))?;
        residual(hidden,&self.feed_forward_norm,self.norm_first,
            |source|self.feed_forward.forward_with_adapter_dropout(source,groups.heads.clone(),feed_forward_dropout),
            |branch|branch_output(TensorParallelResidualStage::FeedForward,branch))
    }
}
