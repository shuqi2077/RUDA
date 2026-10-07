use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::tensor::{Tensor,Int,Bool,DType,backend::Backend};
use crate::loss::{KLDivLoss,LossTerms};
use super::{BroadcastTensorCollective,VocabParallelLossLayout,loss::{floating,work_dtype}};

impl KLDivLoss {
    /// Native per-row KL over actual rank-local slices of a complete logical class distribution.
    /// Predictions are already global log probabilities; target space is this module's log_target.
    /// Reuses native zero-target, selection and selected-row normalization without epsilon clipping,
    /// target detachment, mass renormalization or summing replicated target counts by world size.
    pub fn forward_sharded_terms<B,S,C>(&self,predictions:Tensor<Autodiff<B,S>,2>,targets:Tensor<Autodiff<B,S>,2>,
        layout:&VocabParallelLossLayout,communicator:C,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>)
        -> Result<LossTerms<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        assert_eq!(predictions.dims(),targets.dims(),"sharded KL prediction/target class geometry differs");
        assert_eq!(predictions.device(),targets.device(),"sharded KL prediction/targets must share a device");
        floating(predictions.dtype());floating(targets.dtype());
        assert_eq!(communicator.world_size() as usize,layout.world_size(),"sharded KL topology/layout differ");
        let interval = layout.interval(communicator.rank() as usize);let [rows,width] = predictions.dims();
        assert_eq!(width,interval.len(),"sharded KL columns differ from actual rank vocabulary storage");
        if let Some(visible) = &visible {
            assert_eq!(visible.dims(),[rows],"sharded KL visibility differs from actual rows");
            assert_eq!(visible.device(),predictions.device(),"sharded KL visibility must share the device");
        }
        let dtype = work_dtype::<B,C>(predictions.dtype() == DType::F64 || targets.dtype() == DType::F64,&predictions.device(),&communicator)?;
        let predictions = predictions.cast(dtype);let targets = targets.cast(dtype);
        if rows == 0 {return Ok(self.forward_terms(predictions,targets,visible));}
        let padding = Tensor::<Autodiff<B,S>,1,Int>::arange(interval.start as i64..interval.end as i64,(&predictions.device(),DType::I64))
            .greater_equal_elem(layout.vocabulary_size() as i64).reshape([1,width]).expand([rows,width]);
        let predictions = predictions.mask_fill(padding.clone(),0);
        let targets = targets.mask_fill(padding,if self.log_target {f64::NEG_INFINITY} else {0.});
        let local = self.forward_terms(predictions,targets,visible);
        let values = region::reduce_from_region(local.values,communicator)?;
        Ok(LossTerms {values,normalizers:local.normalizers,valid:local.valid})
    }

    /// Explicit local-logit student normalization followed by the same native sharded KL objective.
    /// Shared normalizer gradients combine the complete target mass, not only one shard's mass.
    /// Teacher targets remain in the caller's declared probability/log-probability space.
    pub fn forward_sharded_logits_terms<B,S,C>(&self,logits:Tensor<Autodiff<B,S>,2>,targets:Tensor<Autodiff<B,S>,2>,
        layout:&VocabParallelLossLayout,communicator:C,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>)
        -> Result<LossTerms<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        assert_eq!(logits.dims(),targets.dims(),"sharded KL logit/target class geometry differs");
        assert_eq!(logits.device(),targets.device(),"sharded KL logit/targets must share a device");floating(targets.dtype());
        let predictions = layout.log_softmax_for_dtype(logits,communicator.clone(),visible.clone(),Some(targets.dtype()))?;
        self.forward_sharded_terms(predictions,targets,layout,communicator,visible)
    }

    /// Actual per-token KL of complete class distributions supplied as local vocabulary slices.
    pub fn forward_sharded_token_terms<B,S,C>(&self,predictions:Tensor<Autodiff<B,S>,3>,targets:Tensor<Autodiff<B,S>,3>,
        layout:&VocabParallelLossLayout,communicator:C,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>)
        -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        let [batch,tokens,width] = predictions.dims();assert_eq!(targets.dims(),[batch,tokens,width],"sharded token KL targets differ");
        let rows = batch.checked_mul(tokens).expect("sharded token KL row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"sharded token KL visibility differs");mask.reshape([rows])});
        self.forward_sharded_terms(predictions.reshape([rows,width]),targets.reshape([rows,width]),layout,communicator,visible)
            .map(|terms|terms.reshape([batch,tokens]))
    }

    /// Actual per-token student logits and teacher targets, with explicit complete-class normalization.
    pub fn forward_sharded_token_logits_terms<B,S,C>(&self,logits:Tensor<Autodiff<B,S>,3>,targets:Tensor<Autodiff<B,S>,3>,
        layout:&VocabParallelLossLayout,communicator:C,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>)
        -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();assert_eq!(targets.dims(),[batch,tokens,width],"sharded token KL logit targets differ");
        let rows = batch.checked_mul(tokens).expect("sharded token KL logit row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"sharded token KL logit visibility differs");mask.reshape([rows])});
        self.forward_sharded_logits_terms(logits.reshape([rows,width]),targets.reshape([rows,width]),layout,communicator,visible)
            .map(|terms|terms.reshape([batch,tokens]))
    }

    /// Native NCHW class-sharded per-pixel KL, retaining original [batch,height,width] output axes.
    pub fn forward_sharded_pixel_terms<B,S,C>(&self,predictions:Tensor<Autodiff<B,S>,4>,targets:Tensor<Autodiff<B,S>,4>,
        layout:&VocabParallelLossLayout,communicator:C,visible:Option<Tensor<Autodiff<B,S>,3,Bool>>)
        -> Result<LossTerms<Autodiff<B,S>,3>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        let [batch,classes,height,width] = predictions.dims();
        assert_eq!(targets.dims(),[batch,classes,height,width],"sharded pixel KL target geometry differs");
        let rows = batch.checked_mul(height).and_then(|rows|rows.checked_mul(width)).expect("sharded pixel KL row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,height,width],"sharded pixel KL visibility differs");mask.reshape([rows])});
        self.forward_sharded_terms(predictions.permute([0,2,3,1]).reshape([rows,classes]),targets.permute([0,2,3,1]).reshape([rows,classes]),
            layout,communicator,visible).map(|terms|terms.reshape([batch,height,width]))
    }
}
