use crate::{DeviceBackend, DeviceRuntime, FloatElement, IntElement, element::BoolElement};
use ruda_tensor::{frozen_awq::FrozenAwqOps, tensor::{FloatTensor, IntTensor}};
use rublas::tensor_int4::{AwqGemm, Int4Error};

impl<R, F, I, BT> FrozenAwqOps for DeviceBackend<R, F, I, BT>
where R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement {
    type AwqError = Int4Error;

    fn frozen_awq_forward(
        input: FloatTensor<Self>, qweight: IntTensor<Self>, qzeros: IntTensor<Self>,
        scales: FloatTensor<Self>, bias: Option<FloatTensor<Self>>, group_size: usize,
    ) -> Result<FloatTensor<Self>, Self::AwqError> {
        AwqGemm::new(qweight, qzeros, scales, bias, group_size)?.forward_with_input_dtype(input)
    }

    fn frozen_awq_input_backward(
        gradient: FloatTensor<Self>, qweight: IntTensor<Self>, qzeros: IntTensor<Self>,
        scales: FloatTensor<Self>, group_size: usize,
    ) -> Result<FloatTensor<Self>, Self::AwqError> {
        AwqGemm::new(qweight, qzeros, scales, None, group_size)?.input_backward(gradient)
    }
}
