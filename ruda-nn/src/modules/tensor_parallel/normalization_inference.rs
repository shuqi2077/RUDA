use ruda_model::tensor::{Tensor,TensorPrimitive,TensorData,Int,Bool,DType,backend::Backend};
use super::{BroadcastTensorCollective,VocabParallelLossLayout,loss::{floating,work_dtype}};

struct Normalized<B:Backend> {shifted:Tensor<B,2>,denominator:Tensor<B,2>,selected:Tensor<B,2,Bool>,valid:Tensor<B,1,Bool>}

fn prepare<B:Backend,C:BroadcastTensorCollective<B>>(layout:&VocabParallelLossLayout,logits:Tensor<B,2>,communicator:&C,
    visible:Option<Tensor<B,1,Bool>>) -> Result<Normalized<B>,C::Error> {
    floating(logits.dtype());assert_eq!(communicator.world_size() as usize,layout.world_size(),"native normalization topology/layout differ");
    let interval = layout.interval(communicator.rank() as usize);let [rows,width] = logits.dims();
    assert_eq!(width,interval.len(),"native normalization logits differ from actual rank class storage");
    let valid = if let Some(visible) = visible {
        assert_eq!(visible.dims(),[rows],"native normalization visibility differs from actual rows");
        assert_eq!(visible.device(),logits.device(),"native normalization visibility must share the device");visible
    } else {Tensor::<B,1,Bool>::zeros([rows],&logits.device()).bool_not()};
    let dtype = work_dtype::<B,C>(logits.dtype() == DType::F64,&logits.device(),communicator)?;
    let logits = logits.cast(dtype);
    let ids = Tensor::<B,1,Int>::arange(interval.start as i64..interval.end as i64,(&logits.device(),DType::I64)).reshape([1,width]).expand([rows,width]);
    let bound = Tensor::<B,2,Int>::from_data(TensorData::new(alloc::vec![layout.vocabulary_size() as i64],[1,1]),(&logits.device(),DType::I64)).expand([rows,width]);
    let real = ids.lower(bound);let selected = real.clone().bool_and(valid.clone().reshape([rows,1]).expand([rows,width]));
    if rows == 0 {return Ok(Normalized {shifted:logits,denominator:Tensor::zeros([0,1],(&valid.device(),dtype)),selected,valid});}
    let logits = logits.mask_fill(valid.clone().bool_not().reshape([rows,1]).expand([rows,width]),0).mask_fill(real.bool_not(),f64::NEG_INFINITY);
    let local = logits.clone().max_dim(1);
    let maximum = if communicator.world_size() == 1 {local} else {
        let gathered = communicator.all_gather_float(local.into_primitive().tensor())?;
        Tensor::<B,2>::from_primitive(TensorPrimitive::Float(gathered)).reshape([layout.world_size(),rows]).max_dim(0).reshape([rows,1])
    };
    let shifted = logits-maximum;let denominator = shifted.clone().exp().sum_dim(1);
    let denominator = if communicator.world_size() == 1 {denominator} else {
        Tensor::from_primitive(TensorPrimitive::Float(communicator.all_reduce_sum(denominator.into_primitive().tensor())?))
    };
    Ok(Normalized {shifted,denominator,selected,valid})
}

impl VocabParallelLossLayout {
    /// Native-backend complete-vocabulary log probabilities from actual local logit shards.
    /// Uses FP32 for half storage or globally selected FP64, excluding declared storage padding.
    /// Does not gather logits, renormalize rank-local classes alone or add differentiable regions.
    /// Selected nonfinite logits retain native log-softmax behavior; excluded rows are exactly zero.
    pub fn log_softmax_inference<B,C>(&self,logits:Tensor<B,2>,communicator:C,visible:Option<Tensor<B,1,Bool>>) -> Result<Tensor<B,2>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        let [rows,width] = logits.dims();let Normalized {shifted,denominator,selected,valid} = prepare(self,logits,&communicator,visible)?;
        if rows == 0 {return Ok(shifted);}
        Ok((shifted-denominator.log()).mask_fill(selected.bool_not(),f64::NEG_INFINITY)
            .mask_fill(valid.bool_not().reshape([rows,1]).expand([rows,width]),0))
    }

    /// Native complete-vocabulary probabilities with stable global max/sum statistics.
    /// Padding and explicitly excluded rows are zero, without log/exp roundtrip normalization.
    pub fn softmax_inference<B,C>(&self,logits:Tensor<B,2>,communicator:C,visible:Option<Tensor<B,1,Bool>>) -> Result<Tensor<B,2>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        let Normalized {shifted,denominator,selected,..} = prepare(self,logits,&communicator,visible)?;
        if shifted.dims()[0] == 0 {return Ok(shifted);}
        Ok((shifted.exp()/denominator).mask_fill(selected.bool_not(),0))
    }

    /// Native per-token global-class log probabilities, preserving actual batch/token axes.
    pub fn log_softmax_tokens_inference<B,C>(&self,logits:Tensor<B,3>,communicator:C,visible:Option<Tensor<B,2,Bool>>) -> Result<Tensor<B,3>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();let rows = batch.checked_mul(tokens).expect("native token normalization row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"native token normalization visibility differs");mask.reshape([rows])});
        self.log_softmax_inference(logits.reshape([rows,width]),communicator,visible).map(|values|values.reshape([batch,tokens,width]))
    }

    /// Native per-token global-class probabilities, preserving actual batch/token axes.
    pub fn softmax_tokens_inference<B,C>(&self,logits:Tensor<B,3>,communicator:C,visible:Option<Tensor<B,2,Bool>>) -> Result<Tensor<B,3>,C::Error>
        where B:Backend,C:BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();let rows = batch.checked_mul(tokens).expect("native token probability row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"native token probability visibility differs");mask.reshape([rows])});
        self.softmax_inference(logits.reshape([rows,width]),communicator,visible).map(|values|values.reshape([batch,tokens,width]))
    }
}
