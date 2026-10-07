use alloc::vec::Vec;
use ruda_model::tensor::{Tensor,TensorPrimitive,TensorData,Int,Bool,DType,backend::Backend};
use super::{VocabParallelLossLayout,BroadcastTensorCollective,loss::floating,selection::{gather_index_matrix,actual_maximum}};

/// Actual globally ordered native top-k candidates from sharded vocabulary logits.
/// Selection is distinct from threshold-tie-retaining categorical sampling.
#[derive(Clone,Debug)]
pub struct VocabParallelTopKSelection<B:Backend,const D:usize=2> {
    /// Selected actual FP32 scores, descending; invalid candidate slots contain negative infinity.
    pub scores:Tensor<B,D>,
    /// Exact I64 global class IDs, with lowest ID first on equal scores and -1 for invalid slots.
    pub indices:Tensor<B,D,Int>,
    /// True only for selected actual real non-NaN classes; storage padding is never selected.
    pub valid:Tensor<B,D,Bool>,
}

fn integer<B:Backend>(value:i64,shape:[usize;2],device:&B::Device) -> Tensor<B,2,Int> {
    Tensor::<B,2,Int>::from_data(TensorData::new(alloc::vec![value],[1,1]),(device,DType::I64)).expand(shape)
}

fn select<B:Backend>(scores:Tensor<B,2>,ids:Tensor<B,2,Int>,mut valid:Tensor<B,2,Bool>,k:usize) -> (Tensor<B,2>,Tensor<B,2,Int>) {
    let [rows,width] = scores.dims();let sentinel = integer(i64::MAX,[rows,width],&scores.device());
    let offsets = Tensor::<B,1,Int>::arange(0..i64::try_from(width).expect("top-k candidate positions exceed signed index range"),(&scores.device(),DType::I64))
        .reshape([1,width]).expand([rows,width]);
    let mut score_columns = Vec::with_capacity(k);let mut index_columns = Vec::with_capacity(k);
    for _ in 0..k {
        let maximum = actual_maximum(&scores,Some(&valid));
        let candidate = scores.clone().equal(maximum.clone()).bool_and(valid.clone());
        let selected = ids.clone().mask_where(candidate.bool_not(),sentinel.clone()).min_dim(1);
        let missing = selected.clone().equal(integer(i64::MAX,[rows,1],&scores.device()));
        let chosen = ids.clone().equal(selected.clone().expand([rows,width])).bool_and(valid.clone());
        let offset = offsets.clone().mask_where(chosen.clone().bool_not(),sentinel.clone()).min_dim(1).mask_fill(missing.clone(),0);
        score_columns.push(scores.clone().gather(1,offset).mask_fill(missing,f32::NEG_INFINITY));index_columns.push(selected);
        valid = valid.bool_and(chosen.bool_not());
    }
    (Tensor::cat(score_columns,1),Tensor::cat(index_columns,1))
}

/// The same native candidate policy for a complete-class head on an independent data rank.
/// No tensor/data transport is used: data ranks may own different prompts, scores and visibility.
pub(crate) fn full_logits_topk<B:Backend>(logits:Tensor<B,2>,k:usize,visible:Option<Tensor<B,1,Bool>>)
    -> VocabParallelTopKSelection<B> {
    floating(logits.dtype());let [rows,width]=logits.dims();
    assert!(width>0 && k<=width,"complete-class native top-k geometry differs");
    let limit=i64::try_from(width).expect("complete-class native class IDs exceed signed index range");
    if let Some(mask)=&visible {
        assert_eq!(mask.dims(),[rows],"complete-class candidate row visibility differs");
        assert_eq!(mask.device(),logits.device(),"complete-class visibility must share the score device");
    }
    let device=logits.device();
    if rows==0 || k==0 {
        return VocabParallelTopKSelection {scores:Tensor::zeros([rows,k],(&device,DType::F32)),
            indices:Tensor::zeros([rows,k],(&device,DType::I64)),valid:Tensor::zeros([rows,k],&device)};
    }
    let logits=logits.cast(DType::F32);
    let ids=Tensor::<B,1,Int>::arange(0..limit,(&device,DType::I64)).reshape([1,width]).expand([rows,width]);
    let mut valid=logits.clone().is_nan().bool_not();
    if let Some(mask)=visible {valid=valid.bool_and(mask.reshape([rows,1]).expand([rows,width]));}
    let (scores,indices)=select(logits,ids,valid,k);
    let valid=indices.clone().not_equal(integer(i64::MAX,[rows,k],&device));
    VocabParallelTopKSelection {scores,indices:indices.mask_fill(valid.clone().bool_not(),-1),valid}
}

impl VocabParallelLossLayout {
    /// Exact native FP32/ignore-NaN top-k over actual real global classes, without full-logit gathering.
    /// All scores, masks and candidate comparisons stay on the backend; sorting never invokes a host fallback.
    /// Communicates k scores and four exact FP32 index words per row/rank. Native reduction work is
    /// O(k*rows*local_classes), plus O(rows*world_size*k*k) merging; this is not a fused top-k kernel.
    /// k follows native top-k geometry (0 through real vocabulary size). Rows with fewer eligible
    /// classes return invalid trailing slots; actual negative-infinity classes remain valid candidates.
    pub fn topk_indices_inference<B,C>(&self,logits:Tensor<B,2>,communicator:C,k:usize,visible:Option<Tensor<B,1,Bool>>)
        -> Result<VocabParallelTopKSelection<B>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        floating(logits.dtype());assert!(k <= self.vocabulary_size(),"native global top-k exceeds real vocabulary size");
        assert_eq!(communicator.world_size() as usize,self.world_size(),"top-k vocabulary topology/layout differ");
        let interval = self.interval(communicator.rank() as usize);let [rows,width] = logits.dims();
        assert_eq!(width,interval.len(),"top-k logits differ from actual rank vocabulary storage");
        if let Some(visible) = &visible {
            assert_eq!(visible.dims(),[rows],"top-k row visibility differs");assert_eq!(visible.device(),logits.device(),"top-k visibility must share the device");
        }
        let device = logits.device();
        if k == 0 || rows == 0 {
            return Ok(VocabParallelTopKSelection {scores:Tensor::zeros([rows,k],(&device,DType::F32)),
                indices:Tensor::zeros([rows,k],(&device,DType::I64)),valid:Tensor::zeros([rows,k],&device)});
        }
        let local_k = k.min(interval.end.min(self.vocabulary_size()).saturating_sub(interval.start));
        let (mut scores,mut indices) = if local_k == 0 {
            (Tensor::<B,2>::full([rows,k],f32::NEG_INFINITY,(&device,DType::F32)),integer(i64::MAX,[rows,k],&device))
        } else {
            let logits = logits.cast(DType::F32);
            let ids = Tensor::<B,1,Int>::arange(interval.start as i64..interval.end as i64,(&device,DType::I64)).reshape([1,width]).expand([rows,width]);
            let mut valid = logits.clone().is_nan().bool_not().bool_and(ids.clone().lower(integer(self.vocabulary_size() as i64,[rows,width],&device)));
            if let Some(visible) = visible {valid = valid.bool_and(visible.reshape([rows,1]).expand([rows,width]));}
            select(logits,ids,valid,local_k)
        };
        if local_k > 0 && local_k != k {
            let padding = k-local_k;scores = Tensor::cat(alloc::vec![scores,Tensor::full([rows,padding],f32::NEG_INFINITY,(&device,DType::F32))],1);
            indices = Tensor::cat(alloc::vec![indices,integer(i64::MAX,[rows,padding],&device)],1);
        }
        let (scores,indices) = if self.world_size() == 1 {(scores,indices)} else {
            let candidate_width = self.world_size().checked_mul(k).expect("native top-k candidate width overflow");
            let gathered_rows = self.world_size().checked_mul(rows).expect("native top-k candidate row count overflow");
            let gathered = communicator.all_gather_float(scores.into_primitive().tensor())?;
            let gathered = Tensor::<B,2>::from_primitive(TensorPrimitive::Float(gathered));
            assert_eq!(gathered.dims(),[gathered_rows,k],"top-k score transport returned incompatible geometry");
            let gathered = gathered.reshape([self.world_size(),rows,k]).swap_dims(0,1).reshape([rows,candidate_width]);
            let indices = gather_index_matrix(indices,&communicator)?.swap_dims(0,1).reshape([rows,candidate_width]);
            let valid = indices.clone().not_equal(integer(i64::MAX,[rows,candidate_width],&device));
            select(gathered,indices,valid,k)
        };
        let valid = indices.clone().not_equal(integer(i64::MAX,[rows,k],&device));
        Ok(VocabParallelTopKSelection {scores,indices:indices.mask_fill(valid.clone().bool_not(),-1),valid})
    }

    /// Actual per-token global top-k candidates, preserving batch/token axes and exact I64 IDs.
    pub fn topk_token_indices_inference<B,C>(&self,logits:Tensor<B,3>,communicator:C,k:usize,visible:Option<Tensor<B,2,Bool>>)
        -> Result<VocabParallelTopKSelection<B,3>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();let rows = batch.checked_mul(tokens).expect("top-k token row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"top-k token visibility differs");mask.reshape([rows])});
        self.topk_indices_inference(logits.reshape([rows,width]),communicator,k,visible).map(|result|VocabParallelTopKSelection {
            scores:result.scores.reshape([batch,tokens,k]),indices:result.indices.reshape([batch,tokens,k]),valid:result.valid.reshape([batch,tokens,k])})
    }
}
