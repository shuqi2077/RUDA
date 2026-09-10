use core::marker::PhantomData;

/// Simply transfers tensors between backends via the underlying [tensor data](ruda_tensor::TensorData).
pub struct ByteBridge<Backends> {
    backends: PhantomData<Backends>,
}
