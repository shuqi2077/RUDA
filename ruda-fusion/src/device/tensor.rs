use super::RudaFusionHandle;
use ruda_core::tensor::{Metadata, Shape};
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use ruda_tensor::DeviceOps;

pub fn into_tensor<R: Runtime>(handle: RudaFusionHandle<R>, shape: Shape) -> RudaTensor<R>
where
    R::Device: DeviceOps,
{
    RudaTensor {
        client: handle.client.clone(),
        handle: handle.handle.clone(),
        device: handle.device.clone(),
        meta: Box::new(Metadata::new(shape, handle.strides.clone())),
        dtype: handle.dtype,
        qparams: handle.qparams.clone(),
    }
}

impl<R: Runtime> From<RudaTensor<R>> for RudaFusionHandle<R>
where
    R::Device: DeviceOps,
{
    fn from(value: RudaTensor<R>) -> Self {
        Self {
            client: value.client.clone(),
            handle: value.handle.clone(),
            device: value.device.clone(),
            strides: value.meta.strides.clone(),
            dtype: value.dtype,
            qparams: value.qparams.clone(),
        }
    }
}
