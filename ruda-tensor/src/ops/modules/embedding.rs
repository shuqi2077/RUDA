use crate::{
    Backend, TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use ruda_core::tensor::Shape;

pub fn embedding<B: Backend>(weights: FloatTensor<B>, indices: IntTensor<B>) -> FloatTensor<B> {
    let [batch_size, seq_length] = indices.shape().dims();
    let [_, d_model] = weights.shape().dims();
    let indices = B::int_reshape(indices, Shape::new([batch_size * seq_length]));
    let output = B::float_select(weights, 0, indices);
    B::float_reshape(output, Shape::new([batch_size, seq_length, d_model]))
}

pub fn embedding_backward<B: Backend>(
    weights: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    indices: IntTensor<B>,
) -> FloatTensor<B> {
    let [batch_size, seq_length] = indices.shape().dims();
    let [n_embeddings, d_model] = weights.shape().dims();
    let device = B::float_device(&weights);
    let dtype = output_grad.dtype();
    let indices = B::int_reshape(indices, Shape::new([batch_size * seq_length]));
    let output_grad = B::float_reshape(output_grad, Shape::new([batch_size * seq_length, d_model]));
    let grad = B::float_zeros(Shape::new([n_embeddings, d_model]), &device, dtype.into());
    B::float_select_add(grad, 0, indices, output_grad)
}
