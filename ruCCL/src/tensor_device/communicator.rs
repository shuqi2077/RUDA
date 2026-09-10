use super::{TensorDevice, TensorDeviceError, TensorElement};
use crate::in_process::{Communicator, collective::InProcessCollective};
use crate::rank::{communicator::RankCommunicator, device_collective::DeviceCollective};
use ruda_tensor::Backend;

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// Select the wire element type from the typed tensor buffer contract.
    pub fn tensor_collective<T: TensorElement>(&self) -> DeviceCollective<'_, T, TensorDevice<B>> {
        self.device_collective(T::ELEMENT_TYPE)
    }
}

impl<B: Backend> Communicator<TensorDevice<B>> {
    pub fn tensor_collective<T: TensorElement>(
        &self,
    ) -> InProcessCollective<'_, T, TensorDevice<B>, TensorDeviceError> {
        self.collective()
    }
}
