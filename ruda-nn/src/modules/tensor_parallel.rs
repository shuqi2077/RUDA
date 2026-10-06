//! Local column/row shards using the model-parallel derivatives of ruda-autodiff.
use crate::Linear;
use ruda_autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor_parallel as region};
use region::BroadcastTensorCollective;
use ruda_model::{
    module::Module,
    tensor::{Tensor, backend::Backend, module::linear},
};

/// Output-feature shard of a global projection, with an optional local bias shard.
/// Construct local weights directly or load an explicitly partitioned checkpoint.
#[derive(Module, Debug)]
pub struct ColumnParallelLinear<B: Backend> {
    /// Local weight `[global_input, local_output]` and local-output bias.
    pub local: Linear<B>,
}

/// Input-feature shard of a global projection, with a replicated output bias.
#[derive(Module, Debug)]
pub struct RowParallelLinear<B: Backend> {
    /// Local weight `[local_input, global_output]` and optional full-output bias.
    pub local: Linear<B>,
}

impl<B: Backend> ColumnParallelLinear<B> {
    /// Use caller-provided rank-local weights. Does not construct a full global weight.
    pub fn from_shard(local: Linear<B>) -> Self { Self { local } }
}
impl<B: Backend> RowParallelLinear<B> {
    /// Use caller-provided rank-local weights and, if present, identical replicated biases.
    pub fn from_shard(local: Linear<B>) -> Self { Self { local } }
}

impl<B: Backend, S: CheckpointStrategy> ColumnParallelLinear<Autodiff<B, S>> {
    /// Project a replicated input; sum its shard gradients and optionally gather output features.
    /// `gather_output=false` connects directly to a row-parallel layer's sharded input.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C, gather_output: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        assert!(D > 0, "linear projection requires a feature axis");
        let input = region::copy_to_region(input, communicator.clone())?;
        let output = self.local.forward(input);
        if gather_output { region::gather_from_region(output, communicator, D - 1) }
        else { Ok(output) }
    }
}

impl<B: Backend, S: CheckpointStrategy> RowParallelLinear<Autodiff<B, S>> {
    /// Project a rank-local input shard or explicitly scatter a replicated input.
    /// Sum partial outputs before adding bias, so bias is not multiplied by world size.
    /// Replicated losses must be identical on the ranks in this tensor-parallel group.
    pub fn forward<C: BroadcastTensorCollective<B>, const D: usize>(
        &self, input: Tensor<Autodiff<B, S>, D>, communicator: C, input_is_parallel: bool,
    ) -> Result<Tensor<Autodiff<B, S>, D>, C::Error> {
        assert!(D > 0, "linear projection requires a feature axis");
        let input = if input_is_parallel { input }
            else { region::scatter_to_region(input, communicator.clone(), D - 1)? };
        let partial = linear(input, self.local.weight.val(), None);
        let output = region::reduce_from_region(partial, communicator)?;
        Ok(match &self.local.bias {
            Some(bias) => {
                let mut shape = [1; D];
                shape[D - 1] = bias.val().dims()[0];
                output + bias.val().reshape(shape)
            }
            None => output,
        })
    }
}
