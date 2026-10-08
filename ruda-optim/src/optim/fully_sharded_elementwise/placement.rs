use super::*;

impl<O,M,B,C> FullyShardedElementwiseOptimizer<O,M,B,C>
where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
    O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    /// Restore original local histories/master buffers and place each immediately
    /// on its actual prepared native leaf's device. Original logical ownership,
    /// dtype/trainable flags and optimizer-history identity still use the existing
    /// exact restore checks; missing/frozen histories are not initialized/reset.
    pub fn try_load_record_for_model(self,module:&M,record:FullyShardedElementwiseRecord<B,O>)
        -> Result<Self,FullyShardedElementwiseError<C::Error>> {
        inspect::<B,M>(module,&self.placement,true).map_err(FullyShardedElementwiseError::Arguments)?;
        let mut restored=self.try_load_record(record)?;restored.place_histories_on_model(module)?;Ok(restored)
    }
    pub(super) fn place_histories_on_model(&mut self,module:&M) -> Result<(),FullyShardedElementwiseError<C::Error>> {
        let devices=crate::adaptor::placement::parameter_devices::<B,M,_>(module,self.states.keys().map(|id|(*id,1)))
            .map_err(|_|FullyShardedElementwiseError::State("restored native history parameter rank/device differs"))?;
        self.states=core::mem::take(&mut self.states).into_iter().map(|(id,state)| {
            let device=devices.get(&id).expect("validated original native flat history placement");(id,O::to_device(state,device))
        }).collect();
        Ok(())
    }
}
