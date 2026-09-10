use crate::{BackendRouter, RunnerChannel, RunnerClient};
use ruda_tensor::{
    graph::{FloatOperationIr, OperationIr, OperationOutput, UnaryOpIr},
    ops::ActivationOps,
    tensor::FloatTensor,
};

impl<R: RunnerChannel> ActivationOps<Self> for BackendRouter<R> {
    fn silu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Silu(desc),
            ))
            .output()
    }
}
