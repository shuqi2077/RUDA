use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy};
use ruda_model::tensor::{Tensor,Int,Bool,DType,backend::Backend};
use crate::{attention::PackedSequenceLayout,loss::{CausalCrossEntropyConfig,CausalLoss}};
use super::{BroadcastTensorCollective,VocabParallelCrossEntropy,VocabParallelLossLayout};

#[cfg(feature="std")]
mod batches;

impl CausalCrossEntropyConfig {
    /// Native FP32 chunked causal objective over actual local vocabulary logits.
    /// Uses this config's original chunk size, target shift and ignore sentinel. Smoothing
    /// covers every real global class through the existing sharded criterion, never a sampled
    /// vocabulary. All ranks supply corresponding hidden/label rows and execute identical chunks.
    /// The callback preserves actual rows and projects only this rank's declared storage classes.
    /// This does not bound total backward graph memory or synchronize data-parallel gradients.
    pub fn forward_sharded_hidden<B,S,C,P>(&self,hidden:Tensor<Autodiff<B,S>,3>,labels:Tensor<Autodiff<B,S>,2,Int>,
        layout:&VocabParallelLossLayout,communicator:C,mut project:P,label_smoothing:f64)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>,
            P:FnMut(Tensor<Autodiff<B,S>,2>,&C)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert!(self.token_chunk_size > 0,"token_chunk_size must be positive");
        assert_eq!(communicator.world_size() as usize,layout.world_size(),"causal vocabulary layout/topology differ");
        let local_width = layout.interval(communicator.rank() as usize).len();
        let criterion = VocabParallelCrossEntropy::new(layout.clone(),label_smoothing,Some(self.ignore_index));
        let [batch,sequence,width] = hidden.dims();
        assert_eq!(labels.dims(),[batch,sequence],"causal hidden/label geometry differs");
        assert_eq!(labels.device(),hidden.device(),"causal hidden/labels must share a device");
        let length = if self.shift {sequence.saturating_sub(1)} else {sequence};
        let count = batch.checked_mul(length).expect("causal target row count overflow");
        if count == 0 {
            let mask = Tensor::<Autodiff<B,S>,3,Bool>::zeros(hidden.dims(),&hidden.device()).bool_not();
            return Ok(CausalLoss {loss_sum:hidden.cast(DType::F32).mask_fill(mask,0).sum(),valid_tokens:Tensor::zeros([1],&labels.device())});
        }
        let (hidden,labels) = if self.shift {
            (hidden.slice([0..batch,0..length,0..width]),labels.slice([0..batch,1..sequence]))
        } else {(hidden,labels)};
        let hidden = hidden.reshape([count,width]);let labels = labels.reshape([count]);
        let valid_tokens = labels.clone().equal_elem(self.ignore_index).bool_not().int().sum();
        let mut loss_sum = Tensor::<Autodiff<B,S>,1>::zeros([1],(&hidden.device(),DType::F32));
        for start in (0..count).step_by(self.token_chunk_size) {
            let end = start.saturating_add(self.token_chunk_size).min(count);
            let logits = project(hidden.clone().slice([start..end,0..width]),&communicator)?;
            assert_eq!(logits.dims(),[end-start,local_width],"causal projection changed actual token rows/local vocabulary storage");
            let terms = criterion.forward_terms(logits.cast(DType::F32),labels.clone().slice([start..end]),communicator.clone(),None,None)?;
            loss_sum = loss_sum+terms.loss_sum();
        }
        Ok(CausalLoss {loss_sum,valid_tokens})
    }

    /// Packed native causal supervision; document starts are excluded only when shifting targets.
    /// A predecessor in one document never supervises the next document's first token.
    pub fn forward_sharded_packed_hidden<B,S,C,P>(&self,hidden:Tensor<Autodiff<B,S>,2>,labels:Tensor<Autodiff<B,S>,1,Int>,
        packed:&PackedSequenceLayout,layout:&VocabParallelLossLayout,communicator:C,project:P,label_smoothing:f64)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>,
            P:FnMut(Tensor<Autodiff<B,S>,2>,&C)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        let [tokens,width] = hidden.dims();
        assert_eq!(tokens,packed.tokens(),"packed causal hidden/document geometry differs");
        assert_eq!(labels.dims(),[tokens],"packed causal hidden/label lengths differ");
        assert_eq!(labels.device(),hidden.device(),"packed causal hidden/labels must share a device");
        let labels = if self.shift {
            labels.clone().mask_fill(packed.document_starts::<Autodiff<B,S>>(&labels.device()),self.ignore_index)
        } else {labels};
        self.forward_sharded_hidden(hidden.reshape([1,tokens,width]),labels.reshape([1,tokens]),layout,communicator,project,label_smoothing)
    }

    /// Apply the same native causal shift/chunks to already-produced rank-local logits.
    /// This is an explicit preprojected entry point, not a bounded-logit hidden projection.
    pub fn forward_sharded_logits<B,S,C>(&self,logits:Tensor<Autodiff<B,S>,3>,labels:Tensor<Autodiff<B,S>,2,Int>,
        layout:&VocabParallelLossLayout,communicator:C,label_smoothing:f64) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        self.forward_sharded_hidden(logits,labels,layout,communicator,|rows,_|Ok(rows),label_smoothing)
    }
}
