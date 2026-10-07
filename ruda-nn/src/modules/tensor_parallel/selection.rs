use alloc::vec::Vec;
use ruda_model::tensor::{Tensor,TensorPrimitive,TensorData,Int,Bool,DType,FloatDType,IntDType,ElementConversion,backend::Backend};
use super::{BroadcastTensorCollective,VocabParallelLossLayout,loss::floating};

/// One replicated native greedy result per actual row, without full-vocabulary logits gathering.
#[derive(Clone,Debug)]
pub struct VocabParallelGreedySelection<B:Backend,const D:usize = 1> {
    /// Exact I64 real global class IDs; -1 means no selected non-NaN real class.
    pub indices:Tensor<B,D,Int>,
    /// True exactly when a real class was selected; storage padding is never a candidate.
    pub valid:Tensor<B,D,Bool>,
}

pub(super) fn actual_maximum<B:Backend>(values:&Tensor<B,2>,valid:Option<&Tensor<B,2,Bool>>) -> Tensor<B,2> {
    let work = if let Some(valid) = valid {values.clone().mask_fill(valid.clone().bool_not(),f32::NEG_INFINITY)} else {values.clone()};
    let maximum = work.max_dim(1);
    let present = values.clone().equal(maximum.clone());
    let present = if let Some(valid) = valid {present.bool_and(valid.clone())} else {present};
    maximum.mask_fill(present.any_dim(1).bool_not(),f32::NEG_INFINITY)
}

pub(super) fn gather_index_matrix<B:Backend,C:BroadcastTensorCollective<B>>(indices:Tensor<B,2,Int>,communicator:&C) -> Result<Tensor<B,3,Int>,C::Error> {
    let [rows,width] = indices.dims();assert!(width > 0,"index gather needs actual candidate columns");
    if communicator.world_size() == 1 {return Ok(indices.reshape([1,rows,width]));}
    let count = rows.checked_mul(communicator.world_size() as usize).expect("candidate index gather geometry overflow");
    let encoded_width = width.checked_mul(4).expect("candidate index encoding geometry overflow");
    let mask = Tensor::<B,2,Int>::from_data(TensorData::new(alloc::vec![65535_i64],[1,1]),(&indices.device(),DType::I64)).expand([rows,width]);
    let mut words = Vec::with_capacity(4);
    for word in 0usize..4 {
        let part = indices.clone().bitwise_right_shift_scalar(((word*16) as i32).elem()).bitwise_and(mask.clone());
        // Each exact 16-bit word is representable in FP32 independently of device-default dtype.
        words.push(Tensor::<B,2>::from_primitive(TensorPrimitive::Float(B::int_into_float(part.into_primitive(),FloatDType::F32))));
    }
    let encoded = Tensor::cat(words,1);
    let gathered = communicator.all_gather_float(encoded.into_primitive().tensor())?;
    let gathered = Tensor::<B,2>::from_primitive(TensorPrimitive::Float(gathered));
    assert_eq!(gathered.dims(),[count,encoded_width],"candidate index transport returned incompatible geometry");
    let mut result:Option<Tensor<B,2,Int>> = None;
    for word in 0usize..4 {
        let part = gathered.clone().slice_dim(1,word*width..(word+1)*width).into_primitive().tensor();
        let part = Tensor::<B,2,Int>::from_primitive(B::float_into_int(part,IntDType::I64)).bitwise_left_shift_scalar(((word*16) as i32).elem());
        result = Some(match result {Some(value)=>value.bitwise_or(part),None=>part});
    }
    Ok(result.expect("four actual candidate words").reshape([communicator.world_size() as usize,rows,width]))
}

fn gather_indices<B:Backend,C:BroadcastTensorCollective<B>>(indices:Tensor<B,2,Int>,communicator:&C) -> Result<Tensor<B,2,Int>,C::Error> {
    let rows = indices.dims()[0];assert_eq!(indices.dims()[1],1,"greedy index gather needs one actual candidate per row");
    gather_index_matrix(indices,communicator).map(|gathered|gathered.reshape([communicator.world_size() as usize,rows]))
}

impl VocabParallelLossLayout {
    /// Native generation's FP32/ignore-NaN greedy policy over actual sharded vocabulary rows.
    /// Equal maximum scores select the lowest real global class ID; all-NaN/excluded rows return -1.
    /// Communicates one score and four exact FP32 words per row/rank, not complete logits. Index
    /// words preserve I64 IDs above FP32's integer precision without FP64 or host numerical fallback.
    /// All ranks provide corresponding rows/visibility and the same explicit class layout.
    pub fn greedy_indices_inference<B,C>(&self,logits:Tensor<B,2>,communicator:C,visible:Option<Tensor<B,1,Bool>>)
        -> Result<VocabParallelGreedySelection<B>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        floating(logits.dtype());
        assert_eq!(communicator.world_size() as usize,self.world_size(),"greedy vocabulary topology/layout differ");
        let interval = self.interval(communicator.rank() as usize);let [rows,width] = logits.dims();
        assert_eq!(width,interval.len(),"greedy vocabulary logits differ from actual rank storage");
        if let Some(visible) = &visible {
            assert_eq!(visible.dims(),[rows],"greedy row visibility differs");
            assert_eq!(visible.device(),logits.device(),"greedy row visibility must share the device");
        }
        if rows == 0 {
            return Ok(VocabParallelGreedySelection {indices:Tensor::zeros([0],(&logits.device(),DType::I64)),valid:Tensor::zeros([0],&logits.device())});
        }
        let logits = logits.cast(DType::F32);
        let ids = Tensor::<B,1,Int>::arange(interval.start as i64..interval.end as i64,(&logits.device(),DType::I64)).reshape([1,width]).expand([rows,width]);
        let limit = Tensor::<B,2,Int>::from_data(TensorData::new(alloc::vec![self.vocabulary_size() as i64],[1,1]),(&logits.device(),DType::I64)).expand([rows,width]);
        let mut valid = logits.clone().is_nan().bool_not().bool_and(ids.clone().lower(limit));
        if let Some(visible) = visible {valid = valid.bool_and(visible.reshape([rows,1]).expand([rows,width]));}
        let local_maximum = actual_maximum(&logits,Some(&valid));
        let maximum = if communicator.world_size() == 1 {local_maximum} else {
            let gathered = communicator.all_gather_float(local_maximum.into_primitive().tensor())?;
            let gathered = Tensor::<B,2>::from_primitive(TensorPrimitive::Float(gathered)).reshape([self.world_size(),rows]).transpose();
            actual_maximum(&gathered,None)
        };
        let candidates = logits.equal(maximum).bool_and(valid);
        let sentinel = Tensor::<B,2,Int>::from_data(TensorData::new(alloc::vec![i64::MAX],[1,1]),(&ids.device(),DType::I64)).expand([rows,width]);
        let local = ids.mask_where(candidates.bool_not(),sentinel).min_dim(1);
        let gathered = gather_indices(local,&communicator)?;
        let indices = gathered.min_dim(0).reshape([rows]);
        let sentinel = Tensor::<B,1,Int>::from_data(TensorData::new(alloc::vec![i64::MAX],[1]),(&indices.device(),DType::I64)).expand([rows]);
        let valid = indices.clone().not_equal(sentinel);
        Ok(VocabParallelGreedySelection {indices:indices.mask_fill(valid.clone().bool_not(),-1),valid})
    }

    /// Native per-token greedy indices, retaining actual batch/token axes without shifting them.
    pub fn greedy_token_indices_inference<B,C>(&self,logits:Tensor<B,3>,communicator:C,visible:Option<Tensor<B,2,Bool>>)
        -> Result<VocabParallelGreedySelection<B,2>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();let rows = batch.checked_mul(tokens).expect("greedy token row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"greedy token visibility differs");mask.reshape([rows])});
        self.greedy_indices_inference(logits.reshape([rows,width]),communicator,visible).map(|selection|VocabParallelGreedySelection {
            indices:selection.indices.reshape([batch,tokens]),valid:selection.valid.reshape([batch,tokens])})
    }
}
