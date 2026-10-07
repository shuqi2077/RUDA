use super::*;

/// Actual native global fractional weight and matching pending local gradients/counters/placement.
#[derive(Clone)]
pub struct FullyShardedWeightedGradientsRecord<B:Backend> {
    window:FullyShardedGradientsRecord,
    global_weight:Tensor<B,1>,
}
impl<B:Backend> Record<B> for FullyShardedWeightedGradientsRecord<B> {
    type Item<S:PrecisionSettings>=(<FullyShardedGradientsRecord as Record<B>>::Item<S>,<Tensor<B,1> as Record<B>>::Item<S>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        (<FullyShardedGradientsRecord as Record<B>>::into_item::<S>(self.window),<Tensor<B,1> as Record<B>>::into_item::<S>(self.global_weight))
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        let window=<FullyShardedGradientsRecord as Record<B>>::from_item::<S>(item.0,device);
        let global_weight=<Tensor<B,1> as Record<B>>::from_item::<S>(item.1,device).cast(window.state.dtype);
        Self {window,global_weight}
    }
}

/// Actual globally weighted accumulation result, preserving selected-element counts separately.
pub struct FullyShardedWeightedAccumulatedGradients<B:Backend> {
    /// Actual local gradients and exact issued-window/count/precision metadata.
    pub window:FullyShardedAccumulatedGradients,
    /// Actual native whole-window fractional denominator, not the integer selected-element count.
    pub global_weight:Tensor<B,1>,
}

/// Native weighted accumulation for scope-completed classification/binary/regression/KL objectives.
/// Weights are caller-coordinated actual nonnegative finite effective denominators, already globally summed.
/// Every backward participates, including a locally empty rank; no gradient all-reduce or skip policy is added.
pub struct FullyShardedWeightedGradientsAccumulator<M,B:AutodiffBackend> {
    window:FullyShardedGradientsAccumulator<M>,
    global_weight:Tensor<B::InnerBackend,1>,
}
impl<M:AutodiffModule<B>,B:AutodiffBackend> FullyShardedWeightedGradientsAccumulator<M,B> {
    /// Bind actual local shards and explicit work precision/scale; normalizer arithmetic stays native.
    pub fn new<C:BroadcastTensorCollective<B::InnerBackend>>(module:&M,parameters:&[FullyShardedOptimizerParameter<C>],
        dtype:FloatDType,loss_scale:f64,device:&B::Device) -> Result<Self,FullyShardedAccumulationError> {
        let window=FullyShardedGradientsAccumulator::new::<B,C>(module,parameters,dtype,loss_scale)?;
        Ok(Self {window,global_weight:Tensor::zeros([1],(device,DType::from(dtype)))})
    }
    /// Original exact integer selected-element and issued-microbatch metadata.
    pub fn state(&self) -> &FullyShardedAccumulationState {self.window.state()}
    /// Actual native accumulated GLOBAL weight, without host numerical reduction.
    pub fn global_weight(&self) -> Tensor<B::InnerBackend,1> {self.global_weight.clone()}
    /// Actual pending local derivatives only; unused leaves remain absent.
    pub fn pending(&self) -> &GradientsParams {self.window.pending()}
    fn next_weight(&self,weight:Tensor<B::InnerBackend,1>) -> Result<Tensor<B::InnerBackend,1>,FullyShardedAccumulationError> {
        if weight.dims()!=[1] || weight.device()!=self.global_weight.device() || !matches!(weight.dtype(),DType::F32|DType::F64) {
            return Err(FullyShardedAccumulationError::Placement("actual global loss weight scalar/device/work storage differs"));
        }
        Ok(self.global_weight.clone()+weight.cast(self.window.state.dtype))
    }
    /// Add original scale * global SUM local-shard derivatives and the actual GLOBAL denominator/count.
    pub fn accumulate_sum(&mut self,module:&M,gradients:&GradientsParams,global_weight:Tensor<B::InnerBackend,1>,global_count:u64)
        -> Result<(),FullyShardedAccumulationError> {
        let weight=self.next_weight(global_weight)?;
        self.window.accumulate_sum::<B>(module,gradients,global_count)?;self.global_weight=weight;Ok(())
    }
    /// Backward the real unnormalized scope-completed objective on every rank, applying only the fixed scale.
    /// The weight is the native globally reduced normalizer from the same forward, not a local weight.
    pub fn backward_sum(&mut self,module:&M,loss_sum:Tensor<B,1>,global_weight:Tensor<B::InnerBackend,1>,global_count:u64)
        -> Result<(),FullyShardedAccumulationError> {
        let weight=self.next_weight(global_weight)?;
        self.window.backward_sum::<B>(module,loss_sum,global_count)?;self.global_weight=weight;Ok(())
    }
    /// Drain original scaled SUM derivatives and their true fractional denominator without normalization.
    pub fn finish_sums(&mut self) -> FullyShardedWeightedAccumulatedGradients<B::InnerBackend> {
        let global_weight=self.global_weight.clone();
        let result=FullyShardedWeightedAccumulatedGradients {window:self.window.finish_sums(),global_weight};
        self.global_weight=Tensor::zeros([1],(&self.global_weight.device(),self.global_weight.dtype()));result
    }
    /// Normalize once by scale and native whole-window GLOBAL weight; fractional weights stay fractional.
    /// Exactly zero weight gives original zero derivatives, without deciding whether an optimizer should step.
    pub fn finish_mean(&mut self,module:&M) -> Result<FullyShardedWeightedAccumulatedGradients<B::InnerBackend>,FullyShardedAccumulationError> {
        if self.window.state.microbatches==0 {return Err(FullyShardedAccumulationError::EmptyWindow);}
        inspect::<B,M>(module,&self.window.placement,true)?;let dtype=work_dtype(&self.window.state)?;
        let mut gradients=self.window.pending().unscaled_for::<B,M>(module,self.window.state.loss_scale,dtype)?;
        for id in gradients.container.ids().into_iter().copied().collect::<Vec<_>>() {
            let gradient=gradients.remove::<B::InnerBackend,1>(id).ok_or(FullyShardedAccumulationError::State)?;
            let weight=self.global_weight.clone().to_device(&gradient.device());let empty=weight.clone().equal_elem(0);
            let shape=gradient.dims();
            let gradient=gradient.mask_fill(empty.clone().expand(shape),0)/weight.mask_fill(empty,1);
            gradients.register(id,gradient);
        }
        let mut result=self.finish_sums();result.window.gradients=gradients;Ok(result)
    }
    /// Snapshot pending gradients/counters and the actual native fractional normalizer together.
    /// Full-precision recorder settings retain the caller-selected F32/F64 normalizer arithmetic.
    pub fn try_to_record(&self) -> Result<FullyShardedWeightedGradientsRecord<B::InnerBackend>,RecorderError> {
        Ok(FullyShardedWeightedGradientsRecord {window:self.window.try_to_record::<B>()?,global_weight:self.global_weight.clone()})
    }
    /// Async readback of local pending derivatives, with the same actual native weight state.
    pub async fn to_record_async(&self) -> Result<FullyShardedWeightedGradientsRecord<B::InnerBackend>,RecorderError> {
        Ok(FullyShardedWeightedGradientsRecord {window:self.window.to_record_async::<B>().await?,global_weight:self.global_weight.clone()})
    }
    /// Restore original pending gradients and the same window denominator; rejected metadata leaves state unchanged.
    pub fn load_record(&mut self,module:&M,record:FullyShardedWeightedGradientsRecord<B::InnerBackend>,device:&B::Device) -> Result<(),RecorderError> {
        if record.global_weight.dims()!=[1] || record.global_weight.dtype()!=self.window.state.dtype {
            return Err(RecorderError::Unknown(FullyShardedAccumulationError::State.to_string()));
        }
        let weight=record.global_weight.to_device(&self.global_weight.device());
        self.window.load_record::<B>(module,record.window,device)?;self.global_weight=weight;Ok(())
    }
}
