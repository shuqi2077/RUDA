use super::GradientsParams;
use alloc::vec::Vec;
use ruda_model::tensor::{TensorPrimitive, backend::Backend};
use ruccl::{
    ReduceOperation,
    rank::communicator::RankCommunicator,
    tensor_device::{TensorDevice, TensorDeviceError},
};

impl GradientsParams {
    /// Synchronize gradients using an explicitly owned ruCCL rank communicator.
    ///
    /// Peers must have matching parameter IDs, gradient shapes and dtypes, and
    /// invoke this operation in the same order. Missing gradients are not
    /// replaced with zeros. Use the inner backend when training with autodiff.
    pub fn all_reduce_with<B: Backend>(
        mut self,
        communicator: &RankCommunicator<TensorDevice<B>>,
        operation: ReduceOperation,
    ) -> Result<Self, TensorDeviceError> {
        let mut ids = self.container.ids().into_iter().copied().collect::<Vec<_>>();
        ids.sort();
        for id in ids {
            let gradient = self.container.remove::<B>(&id).ok_or(
                TensorDeviceError::InvalidBuffer("gradient parameter is missing"),
            )?;
            let TensorPrimitive::Float(gradient) = gradient else {
                return Err(TensorDeviceError::InvalidOperation(
                    "quantized gradients cannot be all-reduced",
                ));
            };
            let gradient = communicator.all_reduce_float(gradient, operation)?;
            self.container.register::<B>(id, TensorPrimitive::Float(gradient));
        }
        Ok(self)
    }
}
