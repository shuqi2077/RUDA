use crate::{DeviceBackend, DeviceRuntime, FloatElement, IntElement, element::BoolElement};
use ruda_tensor::{DType, ops::{ActivationOps, FloatTensorOps}, tensor::FloatTensor};

impl<R, F, I, BT> ActivationOps<Self> for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn silu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        if matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16) {
            ruprim::elementwise::unary::silu::launch(tensor)
        } else {
            Self::float_mul(tensor.clone(), Self::sigmoid(tensor))
        }
    }
}
