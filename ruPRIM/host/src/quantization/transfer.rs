use super::*;

pub fn q_from_data(data: TensorData) -> HostQTensor {
    let scheme = match data.dtype {
        DType::QFloat(scheme) => scheme,
        _ => panic!("Expected quantized dtype, got {:?}", data.dtype),
    };

    let shape = data.shape.clone();
    let num_elements = data.num_elements();

    let q_bytes = QuantizedBytes {
        bytes: data.into_bytes(),
        scheme,
        num_elements,
    };

    let (values, qparams) = q_bytes.into_vec_i8_with_shape(&shape);
    let tensor_data = TensorData::new(values, shape);
    let tensor = HostTensor::from_data(tensor_data);

    // Use native storage since we've unpacked to i8
    let scheme = scheme.with_store(QuantStore::Native);

    HostQTensor::new(tensor, scheme, qparams.scales)
}

pub async fn q_into_data(tensor: HostQTensor) -> Result<TensorData, ExecutionError> {
    let shape = tensor.tensor.shape();
    let scheme = tensor.scheme;
    let qt = tensor.tensor.to_contiguous();
    let values: Vec<i8> = qt.storage::<i8>().to_vec();

    Ok(TensorData::quantized(
        values,
        shape.to_vec(),
        scheme,
        &tensor.scales,
    ))
}

