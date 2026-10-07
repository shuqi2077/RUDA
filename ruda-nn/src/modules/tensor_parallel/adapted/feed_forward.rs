use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,Dropout,Module,Tensor,column,row,row_inference,geometry};
use crate::transformer::{AdaptedFeedForward,FeedForwardAdapterTarget};

/// Actual selected dense/LoRA up/gate columns and down rows of a native FFN.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedFeedForward<B: Backend> {
    /// Original base IDs, A/B storage/scales, activation and intermediate dropout.
    pub local: AdaptedFeedForward<B>,
}

impl<B: Backend> TensorParallelAdaptedFeedForward<B> {
    /// Connect actual locally partitioned adapters without reinitialization or base merge.
    pub fn from_shard(local: AdaptedFeedForward<B>) -> Self {
        let [width,inner] = geometry(&local.up);
        assert!(width > 0 && inner > 0,"adapted parallel FFN widths must be positive");
        assert_eq!(geometry(&local.down),[inner,width],"adapted parallel FFN output rows differ");
        if let Some(gate) = &local.gate {assert_eq!(geometry(gate),[width,inner],"adapted parallel gate columns differ");}
        Self {local}
    }

    /// Native non-autodiff FFN inference retains configured adapters and actual A/B dtypes.
    /// No dense base merge, full intermediate gather or training dropout override occurs.
    pub fn forward_inference<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<B,D>,communicator: C)
        -> Result<Tensor<B,D>,C::Error> {
        assert!(D > 0,"adapted native parallel FFN requires a feature axis");
        let up = self.local.up.forward(input.clone());
        let value = if let Some(gate) = &self.local.gate {
            let activated = self.local.activation.forward(gate.forward(input));
            assert_eq!(activated.dims(),up.dims(),"adapted native gate changed actual local geometry");
            activated*up
        } else {self.local.activation.forward(up)};
        row_inference(&self.local.down,self.local.dropout.forward(value),&communicator)
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedFeedForward<Autodiff<B,S>> {
    /// Original adapter input dropout; replicated column inputs must share its actual values.
    pub fn forward<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_adapter_dropout(input,communicator,|_,module,input|module.forward(input))
    }

    /// Explicit per-projection dropout transform at the original adapter A dtype.
    /// Column A gradients SUM across ranks; row A stays local, and row B consumes
    /// reduced rank activations once. All original stored parameter IDs remain intact.
    pub fn forward_with_adapter_dropout<C,F,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,communicator: C,mut dropout: F)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnMut(FeedForwardAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        assert!(D > 0,"adapted parallel FFN requires an actual feature axis");
        let up = column(&self.local.up,input.clone(),&communicator,None::<&C>,|module,input|dropout(FeedForwardAdapterTarget::Up,module,input))?;
        let value = if let Some(gate) = &self.local.gate {
            let gate = column(gate,input,&communicator,None::<&C>,|module,input|dropout(FeedForwardAdapterTarget::Gate,module,input))?;
            let activated = self.local.activation.forward(gate);
            assert_eq!(activated.dims(),up.dims(),"adapted parallel gate activation changed actual local geometry");
            activated*up
        } else {self.local.activation.forward(up)};
        row(&self.local.down,self.local.dropout.forward(value),&communicator,|module,input|dropout(FeedForwardAdapterTarget::Down,module,input))
    }
}
