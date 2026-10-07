use crate::{Fusion,FusionBackend,get_client,ops::NoOp,stream::OperationStreams};
use ruda_tensor::{TensorMetadata,frozen_awq::FrozenAwqOps,tensor::{FloatTensor,IntTensor},
    graph::{InitOperationIr,OperationIr,OperationOutput}};

// This packed GEMM is a native execution boundary, not an elementwise fusion.
// Resolve existing native handles without downloading model values to the host.
// Returning the native Result here retains synchronous launch/validation errors.
pub(super) fn register_output<B:FusionBackend>(output:FloatTensor<B>) -> FloatTensor<Fusion<B>> {
    let client=get_client::<B>(&B::float_device(&output));
    let shape=output.shape();let dtype=output.dtype();
    let handle=B::float_tensor_handle(output);
    let desc=InitOperationIr::create(shape,dtype,||client.register_tensor_handle(handle));
    client.register(OperationStreams::default(),OperationIr::Init(desc),NoOp::<B>::new()).output()
}

impl<B:FusionBackend+FrozenAwqOps> FrozenAwqOps for Fusion<B> {
    type AwqError=B::AwqError;

    fn frozen_awq_forward(input:FloatTensor<Self>,qweight:IntTensor<Self>,qzeros:IntTensor<Self>,
        scales:FloatTensor<Self>,bias:Option<FloatTensor<Self>>,group_size:usize) -> Result<FloatTensor<Self>,Self::AwqError> {
        let input=input.client.clone().resolve_tensor_float::<B>(input);
        let qweight=qweight.client.clone().resolve_tensor_int::<B>(qweight);
        let qzeros=qzeros.client.clone().resolve_tensor_int::<B>(qzeros);
        let scales=scales.client.clone().resolve_tensor_float::<B>(scales);
        let bias=bias.map(|value|value.client.clone().resolve_tensor_float::<B>(value));
        B::frozen_awq_forward(input,qweight,qzeros,scales,bias,group_size).map(register_output::<B>)
    }

    fn frozen_awq_input_backward(gradient:FloatTensor<Self>,qweight:IntTensor<Self>,qzeros:IntTensor<Self>,
        scales:FloatTensor<Self>,group_size:usize) -> Result<FloatTensor<Self>,Self::AwqError> {
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);
        let qweight=qweight.client.clone().resolve_tensor_int::<B>(qweight);
        let qzeros=qzeros.client.clone().resolve_tensor_int::<B>(qzeros);
        let scales=scales.client.clone().resolve_tensor_float::<B>(scales);
        B::frozen_awq_input_backward(gradient,qweight,qzeros,scales,group_size).map(register_output::<B>)
    }
}
