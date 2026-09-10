use super::*;
use rublas::tensor_int4::AwqGemm;
use ruda_tensor::TensorPrimitive;

#[derive(Debug)]
pub(super) enum AwqProjection<R: DeviceRuntime>
where
    R::Device: DeviceOps,
{
    Packed(AwqGemm<R>),
    Dense(Linear<AwqBackend<R>>),
}

impl<R: DeviceRuntime> AwqProjection<R>
where
    R::Device: DeviceOps,
{
    pub(super) fn forward(&self, input: Tensor<AwqBackend<R>, 3>) -> Tensor<AwqBackend<R>, 3> {
        match self {
            Self::Dense(linear) => linear.forward(input),
            Self::Packed(linear) => {
                let output = linear
                    .forward(input.into_primitive().tensor())
                    .expect("AWQ projection input does not match its validated layout");
                Tensor::from_primitive(TensorPrimitive::Float(output))
            }
        }
    }
}
