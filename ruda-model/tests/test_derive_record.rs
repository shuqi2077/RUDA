use ruda_model::record::Record;

use ruda_tensor::api::Tensor;
use ruda_tensor::api::backend::Backend;

// It compiles
#[derive(Record)]
pub struct TestWithBackendRecord<B: Backend> {
    tensor: Tensor<B, 2>,
}

// It compiles
#[derive(Record)]
pub struct TestWithoutBackendRecord {
    _tensor: usize,
}
