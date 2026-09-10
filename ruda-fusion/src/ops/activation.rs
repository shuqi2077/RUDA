use crate::{
    Fusion, FusionBackend,
    client::GlobalFusionClient,
    stream::{OperationStreams, execution::Operation},
    unary_float_ops,
};
use ruda_tensor::{
    graph::{FloatOperationIr, HandleContainer, OperationIr, OperationOutput, UnaryOpIr},
    ops::ActivationOps,
    tensor::FloatTensor,
};
use std::marker::PhantomData;

impl<B: FusionBackend> ActivationOps<Self> for Fusion<B> {
    fn silu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(SiluOps, B::silu);

        let streams = OperationStreams::with_inputs([&tensor]);
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Silu(desc.clone())),
                SiluOps::<B>::new(desc),
            )
            .output()
    }
}
