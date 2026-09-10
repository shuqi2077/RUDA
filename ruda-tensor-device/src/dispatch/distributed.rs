use ruda_tensor::{
    DeviceId, TensorMetadata,
    distributed::{CollectiveTensor, DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};

use crate::{
    BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement,
    dispatch::numeric::{self, zeros_client},
};

impl<R, F, I, BT> DistributedBackend for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        let device = &tensor.device.clone();
        let out_tensor = if tensor.handle.can_mut() && tensor.is_contiguous() {
            tensor
        } else {
            let zeros_tensor = zeros_client::<R>(
                tensor.client.clone(),
                device.clone(),
                tensor.shape(),
                tensor.dtype(),
            );
            numeric::add(zeros_tensor, tensor)
        };

        let op = match op {
            ReduceOperation::Sum => ruda_kernel::dsl::server::ReduceOperation::Sum,
            ReduceOperation::Mean => ruda_kernel::dsl::server::ReduceOperation::Mean,
        };

        let mut client = R::client(device);
        client.all_reduce(
            out_tensor.handle.clone(),
            out_tensor.handle.clone(),
            out_tensor.dtype.into(),
            device_ids.clone(),
            op,
        );
        CollectiveTensor::new(out_tensor)
    }

    fn sync_collective(device: &Device<Self>) {
        let client = R::client(device);
        client.sync_collective();
    }
}
