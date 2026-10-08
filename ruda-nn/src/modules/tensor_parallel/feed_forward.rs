use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::{module::Module,tensor::{Tensor,backend::Backend,module::linear}};
use crate::transformer::DenseFeedForward;
use region::BroadcastTensorCollective;
use ruda_model::tensor::NativeSwiGluOps;
use crate::transformer::NativeFeedForwardError;

/// Actual up/gate columns and down rows of a caller-sharded native FFN.
/// Stateful activation parameters are the caller's actual local partition;
/// this does not automatically split global channel-mixing activation weights.
#[derive(Module,Debug)]
pub struct TensorParallelFeedForward<B: Backend> {
    /// Original IDs, plain/gated activation, local intermediates and native dropout.
    pub local: DenseFeedForward<B>,
}

impl<B: Backend> TensorParallelFeedForward<B> {
    /// Connect supplied intermediate-channel shards without initializing full global weights.
    pub fn from_shard(local: DenseFeedForward<B>) -> Self {
        let [width,inner] = local.up.weight.val().dims();
        assert!(width > 0 && inner > 0,"parallel feed-forward widths must be positive");
        assert_eq!(local.down.weight.val().dims(),[inner,width],"parallel feed-forward output rows differ");
        if let Some(gate) = &local.gate {assert_eq!(gate.weight.val().dims(),[width,inner],"parallel gate/value columns differ");}
        Self {local}
    }

    fn partial<const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {
        let up = self.local.up.forward(input.clone());
        let value = if let Some(gate) = &self.local.gate {
            let activated = self.local.activation.forward(gate.forward(input));
            assert_eq!(activated.dims(),up.dims(),"parallel gate activation changed local intermediate geometry");
            activated*up
        } else {self.local.activation.forward(up)};
        linear(self.local.dropout.forward(value),self.local.down.weight.val(),None)
    }

    fn bias<const D: usize>(&self,output: Tensor<B,D>) -> Tensor<B,D> {
        if let Some(bias) = &self.local.down.bias {
            let mut shape = [1;D];shape[D-1] = bias.val().dims()[0];
            output+bias.val().reshape(shape)
        } else {output}
    }

    /// Native inference keeps intermediates local, reduces down contributions and adds bias once.
    pub fn forward_inference<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<B,D>,communicator: C)
        -> Result<Tensor<B,D>,C::Error> {
        assert!(D > 0,"parallel FFN requires a feature axis");
        let output = communicator.all_reduce_sum(self.partial(input).into_primitive().tensor())?;
        Ok(self.bias(Tensor::from_primitive(ruda_model::tensor::TensorPrimitive::Float(output))))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelFeedForward<Autodiff<B,S>> {
    /// One shared input-copy node SUMs combined gate/value derivatives. Local FFN
    /// parameters keep shard-local gradients; the down SUM has identity backward.
    /// Replicated full residual bias is added after reduction and is not rank-multiplied.
    pub fn forward<C: BroadcastTensorCollective<B>,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,communicator: C)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_activation(input,communicator,|module,input|Ok(module.forward(input)))
    }

    /// Explicit actual activation transform for shard-local or replicated activation state.
    /// Input gradients still use one shared region and output bias remains a single post-SUM addition.
    pub fn forward_with_activation<C,F,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,communicator: C,activation: F)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C: BroadcastTensorCollective<B>,
            F: FnOnce(&crate::activation::Activation<Autodiff<B,S>>,Tensor<Autodiff<B,S>,D>)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        assert!(D > 0,"parallel FFN requires a feature axis");
        let input = region::copy_to_region(input,communicator.clone())?;
        let up = self.local.up.forward(input.clone());
        let value = if let Some(gate) = &self.local.gate {
            let activated = activation(&self.local.activation,gate.forward(input))?;
            assert_eq!(activated.dims(),up.dims(),"parallel activation changed actual gate geometry");
            activated*up
        } else {activation(&self.local.activation,up)?};
        let partial = linear(self.local.dropout.forward(value),self.local.down.weight.val(),None);
        Ok(self.bias(region::reduce_from_region(partial,communicator)?))
    }

    /// SUM derivatives for explicitly replicated activation parameters, such as one shared PReLU slope.
    /// Channel-mixing activation placement and replica membership are explicitly caller-owned.
    pub fn forward_with_replicated_activation<C,K,const D: usize>(&self,input: Tensor<Autodiff<B,S>,D>,communicator: C,activation_group: K)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error> {
        self.forward_with_activation(input,communicator,|module,input|
            super::copy_replicated_module_to_region::<B,S,K,_>(module.clone(),activation_group).map(|module|module.forward(input)))
    }
}

impl<B: NativeSwiGluOps> TensorParallelFeedForward<B> {
    fn partial_native<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>, B::SwiGluError> {
        let up = self.local.up.forward(input.clone());
        let value = if let Some(gate) = &self.local.gate {
            self.local.activation.try_forward_gated_native(gate.forward(input), up)?
        } else { self.local.activation.try_forward_native(up)? };
        Ok(linear(self.local.dropout.forward(value), self.local.down.weight.val(), None))
    }

    /// Local native gate/up activation, then the original down SUM and one replicated bias addition.
    pub fn try_forward_native_inference<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<B, D>, communicator: C,
    ) -> Result<Tensor<B, D>, NativeFeedForwardError<C::Error, B::SwiGluError>> {
        assert!(D > 0, "parallel FFN requires a feature axis");
        let partial = self.partial_native(input).map_err(NativeFeedForwardError::Activation)?;
        let output = communicator.all_reduce_sum(partial.into_primitive().tensor())
            .map_err(NativeFeedForwardError::Execution)?;
        Ok(self.bias(Tensor::from_primitive(ruda_model::tensor::TensorPrimitive::Float(output))))
    }
}

impl<B: NativeSwiGluOps, S: CheckpointStrategy> TensorParallelFeedForward<Autodiff<B, S>> {
    /// One shared input-copy node for combined native gate/up VJPs; down SUM has identity backward.
    /// Local weights remain shard-local, and bias remains outside the down reduction.
    pub fn try_forward_native<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C,
    ) -> Result<Tensor<Autodiff<B, S>, D>, NativeFeedForwardError<C::Error,
        <Autodiff<B, S> as NativeSwiGluOps>::SwiGluError>> {
        assert!(D > 0, "parallel FFN requires a feature axis");
        let input = region::copy_to_region(input, communicator.clone()).map_err(NativeFeedForwardError::Execution)?;
        let partial = self.partial_native(input).map_err(NativeFeedForwardError::Activation)?;
        let output = region::reduce_from_region(partial, communicator).map_err(NativeFeedForwardError::Execution)?;
        Ok(self.bias(output))
    }

    /// Explicit native activation with SUM derivatives for the caller's replicated activation parameters.
    pub fn try_forward_native_with_replicated_activation<C, K, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C, activation_group: K,
    ) -> Result<Tensor<Autodiff<B, S>, D>, NativeFeedForwardError<C::Error,
        <Autodiff<B, S> as NativeSwiGluOps>::SwiGluError>>
        where C: BroadcastTensorCollective<B>, K: BroadcastTensorCollective<B, Error = C::Error> {
        assert!(D > 0, "parallel FFN requires a feature axis");
        let input = region::copy_to_region(input, communicator.clone()).map_err(NativeFeedForwardError::Execution)?;
        let activation = super::copy_replicated_module_to_region::<B, S, K, _>(self.local.activation.clone(), activation_group)
            .map_err(NativeFeedForwardError::Execution)?;
        let up = self.local.up.forward(input.clone());
        let value = if let Some(gate) = &self.local.gate {
            activation.try_forward_gated_native(gate.forward(input), up)
        } else { activation.try_forward_native(up) }.map_err(NativeFeedForwardError::Activation)?;
        let partial = linear(self.local.dropout.forward(value), self.local.down.weight.val(), None);
        let output = region::reduce_from_region(partial, communicator).map_err(NativeFeedForwardError::Execution)?;
        Ok(self.bias(output))
    }
}
