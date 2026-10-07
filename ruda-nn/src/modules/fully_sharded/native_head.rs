use super::*;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout},cache::TransformerKvCache,
    loss::{CausalCrossEntropyConfig,CausalLoss}};
use ruda_model::tensor::Bool;

/// Native complete-class result for this data rank's own actual rows, not replicated data-rank prompts.
pub type FullyShardedGreedySelection<B> = crate::tensor_parallel::VocabParallelGreedySelection<B>;
/// Native complete-class ordered candidates for this data rank's independent prompts.
pub type FullyShardedTopKSelection<B> = crate::tensor_parallel::VocabParallelTopKSelection<B>;

fn last_rows<B:Backend>(hidden:Tensor<B,3>) -> Tensor<B,2> {
    let [batch,tokens,width]=hidden.dims();assert!(tokens>0,"native sharded last-row projection needs an actual token");
    hidden.slice([0..batch,tokens-1..tokens,0..width]).reshape([batch,width])
}

impl<B:Backend> FullyShardedTransformerHead<B> {
    /// Project only actual final incoming rows, avoiding logits for all earlier prompt/chunk tokens.
    /// The caller supplies unpadded last rows or an explicit row-selection policy before this call.
    pub fn forward_last_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,3>,communicator:C)
        -> Result<Tensor<B,2>,C::Error> {self.forward_inference(last_rows(hidden),communicator)}

    /// Complete-class native candidates with the existing FP32/ignore-NaN/lowest-I64-ID tie policy.
    /// Parameter gather uses DP; candidate selection never reduces independent data-rank prompts.
    /// This reuses the exact native O(k*rows*classes) selector, not a new fused top-k kernel.
    pub fn forward_topk_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,2>,communicator:C,k:usize,
        visible:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedTopKSelection<B>,C::Error> {
        assert!(k<=self.classes(),"sharded head top-k exceeds actual output classes");
        let logits=self.forward_inference(hidden,communicator)?;Ok(crate::tensor_parallel::full_logits_topk(logits,k,visible))
    }
    /// Select actual last-row candidates only; no EOS, sampling, beam or automatic next-step policy is added.
    pub fn forward_topk_last_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,3>,communicator:C,k:usize,
        visible:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedTopKSelection<B>,C::Error> {
        self.forward_topk_inference(last_rows(hidden),communicator,k,visible)
    }
    /// Native greedy class ID per actual row; excluded/all-NaN rows retain the original -1/false result.
    pub fn forward_greedy_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,2>,communicator:C,
        visible:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedGreedySelection<B>,C::Error> {
        let rows=hidden.dims()[0];let selected=self.forward_topk_inference(hidden,communicator,1,visible)?;
        Ok(FullyShardedGreedySelection {indices:selected.indices.reshape([rows]),valid:selected.valid.reshape([rows])})
    }
    /// Native greedy projection for each actual last incoming token, without materializing prompt-wide logits.
    pub fn forward_greedy_last_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,3>,communicator:C,
        visible:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedGreedySelection<B>,C::Error> {
        self.forward_greedy_inference(last_rows(hidden),communicator,visible)
    }
    /// Native full-vocabulary evaluation with original causal alignment, ignore sentinel and smoothing.
    /// Gather the actual head once, then reuse its exact native expression for every real token chunk.
    pub fn forward_causal_loss_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,3>,labels:Tensor<B,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C) -> Result<CausalLoss<B>,C::Error> {
        let head=self.gather_inference(communicator)?;
        Ok(criterion.forward_hidden_with_smoothing(hidden,labels,|rows|head.forward(rows),label_smoothing))
    }
    /// Native actual packed-document evaluation; shifted targets never cross original document boundaries.
    pub fn forward_packed_causal_loss_inference<C:BroadcastTensorCollective<B>>(&self,hidden:Tensor<B,2>,labels:Tensor<B,1,Int>,
        layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C) -> Result<CausalLoss<B>,C::Error> {
        let head=self.gather_inference(communicator)?;
        Ok(criterion.forward_packed_hidden_with_smoothing(hidden,labels,layout,|rows|head.forward(rows),label_smoothing))
    }
}

impl<B:Backend> FullyShardedTransformerModel<B> {
    /// Complete native prompt-to-candidates path, projecting only actual last tokens after the full backbone.
    pub fn forward_topk_last_inference_with<C,F>(&self,input:FullyShardedTransformerInput<B>,communicator:C,layer:F,
        k:usize,visible:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedTopKSelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert!(input.tokens.dims()[1]>0,"native sharded prompt candidates need actual input tokens");
        let hidden=self.forward_hidden_inference_with(input,communicator.clone(),layer)?;
        self.head.forward_topk_last_inference(hidden,communicator,k,visible)
    }
    /// Complete cached native new-row candidate path with actual original Q/K positions and cache boundaries.
    pub fn forward_cached_topk_inference<C,P>(&self,input:FullyShardedTransformerInput<B>,token_visibility:Option<Tensor<B,2,Bool>>,
        cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,positions:P,
        k:usize,row_visibility:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedTopKSelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        assert!(input.tokens.dims()[1]>0,"native sharded cached candidates need actual new tokens");
        let hidden=self.forward_cached_hidden_inference(input,token_visibility,cache,masks,options,communicator.clone(),positions)?;
        self.head.forward_topk_last_inference(hidden,communicator,k,row_visibility)
    }
    /// End-to-end native full-vocabulary evaluation without creating whole-prompt [batch,tokens,classes] logits.
    pub fn forward_causal_loss_inference_with<C,F>(&self,input:FullyShardedTransformerInput<B>,labels:Tensor<B,2,Int>,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,layer:F) -> Result<CausalLoss<B>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert_eq!(input.tokens.dims(),labels.dims(),"sharded evaluation token/label geometry differs");
        let hidden=self.forward_hidden_inference_with(input,communicator.clone(),layer)?;
        self.head.forward_causal_loss_inference(hidden,labels,criterion,label_smoothing,communicator)
    }
    /// End-to-end actual packed-document full-vocabulary evaluation with independent shifted boundaries.
    pub fn forward_packed_causal_loss_inference_with<C,F>(&self,input:FullyShardedTransformerInput<B,1>,labels:Tensor<B,1,Int>,
        layout:&PackedSequenceLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,layer:F) -> Result<CausalLoss<B>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        assert_eq!(input.tokens.dims(),labels.dims(),"sharded packed evaluation token/label geometry differs");
        let hidden=self.forward_packed_hidden_inference_with(input,layout,communicator.clone(),layer)?;
        self.head.forward_packed_causal_loss_inference(hidden,labels,layout,criterion,label_smoothing,communicator)
    }
}

impl<B:Backend> FullyShardedEncoderDecoderModel<B> {
    /// Actual paired prompt-to-candidates path; only the final actual target row is projected.
    pub fn forward_topk_last_inference_with<C,E,F>(&self,source:FullyShardedTransformerInput<B>,target:FullyShardedTransformerInput<B>,
        communicator:C,encoder:E,decoder:F,k:usize,visible:Option<Tensor<B,1,Bool>>) -> Result<FullyShardedTopKSelection<B>,C::Error>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&FullyShardedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert!(target.tokens.dims()[1]>0,"native sharded paired candidates need actual target tokens");
        let hidden=self.forward_hidden_inference_with(source,target,communicator.clone(),encoder,decoder)?;
        self.head.forward_topk_last_inference(hidden,communicator,k,visible)
    }
    /// Actual paired model full-vocabulary evaluation with the caller's explicit target alignment.
    /// For already aligned seq2seq labels supply criterion.shift=false, not a guessed second shift.
    pub fn forward_causal_loss_inference_with<C,E,F>(&self,source:FullyShardedTransformerInput<B>,target:FullyShardedTransformerInput<B>,
        labels:Tensor<B,2,Int>,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,communicator:C,encoder:E,decoder:F)
        -> Result<CausalLoss<B>,C::Error>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error>,
            F:FnMut(usize,&FullyShardedEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,C::Error> {
        assert_eq!(target.tokens.dims(),labels.dims(),"sharded paired evaluation target/label geometry differs");
        let hidden=self.forward_hidden_inference_with(source,target,communicator.clone(),encoder,decoder)?;
        self.head.forward_causal_loss_inference(hidden,labels,criterion,label_smoothing,communicator)
    }
    /// Complete paired packed-document target evaluation, retaining independently declared source/target layouts.
    pub fn forward_packed_causal_loss_inference_with<C,E,F>(&self,source:FullyShardedTransformerInput<B,1>,target:FullyShardedTransformerInput<B,1>,
        source_layout:&PackedSequenceLayout,target_layout:&PackedSequenceLayout,labels:Tensor<B,1,Int>,criterion:&CausalCrossEntropyConfig,
        label_smoothing:f64,communicator:C,encoder:E,decoder:F) -> Result<CausalLoss<B>,C::Error>
        where C:BroadcastTensorCollective<B>,E:FnMut(usize,&FullyShardedTransformerBlock<B>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error>,
            F:FnMut(usize,&FullyShardedEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,C::Error> {
        assert_eq!(target.tokens.dims(),labels.dims(),"sharded paired packed evaluation target/label geometry differs");
        let hidden=self.forward_packed_hidden_inference_with(source,target,source_layout,target_layout,communicator.clone(),encoder,decoder)?;
        self.head.forward_packed_causal_loss_inference(hidden,labels,target_layout,criterion,label_smoothing,communicator)
    }
}
