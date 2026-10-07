use super::*;

impl<B:Backend,S:CheckpointStrategy> FullyShardedEncoderDecoderModel<Autodiff<B,S>> {
    /// Complete paired causal objective with independent source, target-self and cross-memory policies.
    pub fn forward_causal_with_positions<C,E,F,G>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>>,target:FullyShardedTransformerInput<Autodiff<B,S>>,
        labels:Tensor<Autodiff<B,S>,2,Int>,source_masks:DenseAttentionMask<Autodiff<B,S>>,source_options:DenseAttentionOptions,
        self_masks:DenseAttentionMask<Autodiff<B,S>>,self_options:DenseAttentionOptions,cross_masks:DenseAttentionMask<Autodiff<B,S>>,cross_options:DenseAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut source_positions:E,mut self_positions:F,mut cross_positions:G)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            F:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            G:FnMut(usize,Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_causal_with(source,target,labels,criterion,label_smoothing,communicator,
            |index,block,hidden,transport|block.forward(hidden,source_masks.clone(),source_options,transport,|query,key|source_positions(index,query,key)),
            |index,block,hidden,memory,transport|block.forward(hidden,memory,self_masks.clone(),self_options,cross_masks.clone(),cross_options,transport,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Real packed paired training with independent source/self/cross options and original document boundaries.
    pub fn forward_packed_causal_with_positions<C,E,F,G>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>,1>,target:FullyShardedTransformerInput<Autodiff<B,S>,1>,
        labels:Tensor<Autodiff<B,S>,1,Int>,source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,
        source_options:PackedAttentionOptions,self_options:PackedAttentionOptions,cross_options:PackedAttentionOptions,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut source_positions:E,mut self_positions:F,mut cross_positions:G)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            F:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>),
            G:FnMut(usize,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->(Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>) {
        self.forward_packed_causal_with(source,target,labels,source_layout,target_layout,criterion,label_smoothing,communicator,
            |index,block,hidden,transport|block.forward_packed(hidden,source_layout,source_options,transport,|query,key|source_positions(index,query,key)),
            |index,block,hidden,memory,transport|block.forward_packed(hidden,memory,target_layout,source_layout,self_options,cross_options,transport,
                |query,key|self_positions(index,query,key),|query,key|cross_positions(index,query,key)))
    }
    /// Actual original paired model graph and target-aligned full-vocabulary objective.
    /// Encoder memory derivatives remain intact; for already aligned seq2seq labels select criterion.shift=false.
    pub fn forward_causal_with<C,E,F>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>>,target:FullyShardedTransformerInput<Autodiff<B,S>>,
        labels:Tensor<Autodiff<B,S>,2,Int>,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut encoder:E,mut decoder:F)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&FullyShardedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert_eq!(target.tokens.dims(),labels.dims(),"actual sharded paired target/label geometry differs");
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_hidden_with(source,target,transport.clone(),
            |index,block,hidden|encoder(index,block,hidden,transport.clone()),
            |index,block,hidden,memory|decoder(index,block,hidden,memory,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        let loss=criterion.forward_hidden_with_smoothing(hidden,labels,|rows|head.forward(rows),label_smoothing);
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator)
    }
    /// Complete actual independent-document source/target graph, preserving original target shift/boundary masks.
    pub fn forward_packed_causal_with<C,E,F>(&self,source:FullyShardedTransformerInput<Autodiff<B,S>,1>,target:FullyShardedTransformerInput<Autodiff<B,S>,1>,
        labels:Tensor<Autodiff<B,S>,1,Int>,source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,mut encoder:E,mut decoder:F)
        -> Result<FullyShardedLoss<B,S>,ScopedCollectiveError<C::Error>>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&FullyShardedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>,ScopedTensorCollective<C,B,S>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(target.tokens.dims(),labels.dims(),"actual sharded packed paired target/label geometry differs");
        let scope=CollectiveScope::new();let transport=scope.bind(communicator.clone());
        let hidden=self.forward_packed_hidden_with(source,target,source_layout,target_layout,transport.clone(),
            |index,block,hidden|encoder(index,block,hidden,transport.clone()),
            |index,block,hidden,memory|decoder(index,block,hidden,memory,transport.clone())).map_err(ScopedCollectiveError::Collective)?;
        let head=self.head.gather(transport).map_err(ScopedCollectiveError::Collective)?;
        let loss=criterion.forward_packed_hidden_with_smoothing(hidden,labels,target_layout,|rows|head.forward(rows),label_smoothing);
        complete_fully_sharded_loss(&scope,loss.loss_sum,loss.valid_tokens,communicator)
    }
}
