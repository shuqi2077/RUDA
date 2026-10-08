use super::*;
use core::fmt;

/// Original native model and pending input derivatives on a returned Muon/AdamW
/// preparation error. Native local histories/master values were not committed.
#[derive(Debug)]
pub struct FullyShardedMuonStepFailure<M,E:fmt::Debug> {
    pub module:M,
    pub gradients:GradientsParams,
    pub error:MuonShardedError<E>,
}
impl<M,E:fmt::Debug> fmt::Display for FullyShardedMuonStepFailure<M,E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {fmt::Display::fmt(&self.error,f)}
}
impl<M:fmt::Debug,E:fmt::Debug> core::error::Error for FullyShardedMuonStepFailure<M,E> {}

impl<M,B,C> FullyShardedMuonAdamW<M,B,C>
where B:AutodiffBackend,M:AutodiffModule<B>,C:BroadcastTensorCollective<B::InnerBackend> {
    /// Original full-logical Muon and auxiliary AdamW preparation, retaining the
    /// rank-local inputs on a returned error. Reuses the same role/configuration,
    /// complete-matrix expression, clipping and native/FP32-master branches.
    /// No model clone, repeated loss/backward, gradient SUM or automatic retry.
    /// Native input gradient primitives retain their original work dtype/devices.
    /// Transport/peer progress and asynchronous device failures are not rolled
    /// back; an explicit replay needs matching common state and transports.
    pub fn try_step_recoverable_with_lrs(&mut self,muon_lr:LearningRate,adamw_lr:LearningRate,module:M,gradients:GradientsParams)
        -> Result<M,FullyShardedMuonStepFailure<M,C::Error>> {
        let input=gradients.clone_native::<B::InnerBackend>();
        let (states,mut mapper)=match self.prepare_step_with_lrs(muon_lr,adamw_lr,&module,gradients) {
            Ok(proposed)=>proposed,
            Err(error)=>return Err(FullyShardedMuonStepFailure {module,gradients:input,error}),
        };
        let module=module.map(&mut mapper);self.states=states;Ok(module)
    }
}
