use ruda_core::tensor::{Shape, FloatDType, TensorMetadata, host::HostTensor};

pub fn embedding(weights: HostTensor, indices: HostTensor) -> HostTensor {
    let [batch_size, seq_length] = indices.shape().dims();
    let [_, d_model] = weights.shape().dims();

    let indices = HostTensor::reshape(&indices, Shape::from(alloc::vec![batch_size * seq_length]));
    let output = ruprim_host::gather_scatter::dispatch_float::float_select(weights, 0, indices);
    HostTensor::reshape(
        &output,
        Shape::from(alloc::vec![batch_size, seq_length, d_model]),
    )
}

pub fn embedding_backward(
    weights: HostTensor,
    output_grad: HostTensor,
    indices: HostTensor,
) -> HostTensor {
    let [batch_size, seq_length] = indices.shape().dims();
    let [n_embeddings, d_model] = weights.shape().dims();
    let dtype = output_grad.dtype();

    let indices = HostTensor::reshape(&indices, Shape::from(alloc::vec![batch_size * seq_length]));
    let output_grad = HostTensor::reshape(
        &output_grad,
        Shape::from(alloc::vec![batch_size * seq_length, d_model]),
    );
    let grad = HostTensor::zeros(
        Shape::from(alloc::vec![n_embeddings, d_model]),
        FloatDType::from(dtype).into(),
    );
    ruprim_host::gather_scatter::dispatch_float::float_select_add(grad, 0, indices, output_grad)
}

