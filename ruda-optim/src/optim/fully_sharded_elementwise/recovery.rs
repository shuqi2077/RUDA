use super::*;

/// Original rank-local model and native input derivatives after a returned FSDP
/// preparation error. The configured optimizer's original histories were not
/// committed. This owns the original model, not a cloned/materialized substitute.
#[derive(Debug)]
pub struct FullyShardedElementwiseStepFailure<M,E:fmt::Debug> {
    pub module:M,
    pub gradients:GradientsParams,
    pub error:FullyShardedElementwiseError<E>,
}
impl<M,E:fmt::Debug> fmt::Display for FullyShardedElementwiseStepFailure<M,E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {fmt::Display::fmt(&self.error,f)}
}
impl<M:fmt::Debug,E:fmt::Debug> core::error::Error for FullyShardedElementwiseStepFailure<M,E> {}

impl<O,M,B,C> FullyShardedElementwiseOptimizer<O,M,B,C>
where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
    O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    /// Same original numerical update/collective sequence as `try_step`, with
    /// original rank-local inputs returned on a preparation error. No model
    /// clone, second gradient SUM/normalization, implicit retry or state reset.
    /// Native gradient primitives are copied before consumption; retained work
    /// dtype/device/IDs and original model leaves/record mappers are unchanged.
    /// Transport/peer progress and asynchronous device failures are not rolled
    /// back: replay requires the caller's matching common boundary/transports.
    pub fn try_step_recoverable(&mut self,lr:LearningRate,module:M,gradients:GradientsParams)
        -> Result<M,FullyShardedElementwiseStepFailure<M,C::Error>> {
        if !lr.is_finite() || lr<0.0 {
            return Err(FullyShardedElementwiseStepFailure {module,gradients,
                error:FullyShardedElementwiseError::Configuration("learning rate must be finite and nonnegative")});
        }
        let optimizer=self.optimizer.clone();let clipping=self.clipping.clone();
        self.step_recoverable_configured(module,gradients,move |_|(optimizer.clone(),clipping.clone(),lr))
    }
    pub(super) fn step_recoverable_configured<F>(&mut self,module:M,gradients:GradientsParams,configuration:F)
        -> Result<M,FullyShardedElementwiseStepFailure<M,C::Error>>
    where F:FnMut(ParamId)->(O,Option<GradientClipping>,LearningRate) {
        let input=gradients.clone_native::<B::InnerBackend>();
        let (states,mut mapper)=match self.prepare_configured(&module,gradients,configuration) {
            Ok(proposed)=>proposed,
            Err(error)=>return Err(FullyShardedElementwiseStepFailure {module,gradients:input,error}),
        };
        let module=module.map(&mut mapper);self.states=states;Ok(module)
    }
}
