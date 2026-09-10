use crate::TensorSnapshot;
use ruda_model::module::ParamId;
use ruda_tensor::api::{DType, TensorData};

/// Helper to create a test TensorSnapshot
#[allow(dead_code)]
pub fn create_test_snapshot(
    name: String,
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
) -> TensorSnapshot {
    TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data, shape, dtype),
        vec![name],
        vec![],
        ParamId::new(),
    )
}
