use ruccl::{
    in_process::Communicator,
    rank::ReductionOperation,
    tensor_device::{TensorDevice, TensorDeviceError},
};
use ruda_tensor::{Backend, TensorData, read_sync};

#[cfg(feature = "cuda")]
type Compute = ruda_tensor_device::cuda::Cuda<f32>;
#[cfg(not(feature = "cuda"))]
type Compute = ruda_tensor_host::Host;

fn main() -> Result<(), TensorDeviceError> {
    let contexts: Vec<_> = (0..3)
        .map(|_| TensorDevice::<Compute>::new(Default::default()))
        .collect();
    let (communicator, sum) = Communicator::new(contexts.clone(), |context| {
        context.reduction_kernel::<f32>(ReductionOperation::Sum)
    })?;
    let collective = communicator.tensor_collective::<f32>();
    let tensors = contexts.iter().enumerate().map(|(rank, context)| {
        let values: Vec<f32> = (0..257).map(|index| index as f32 + rank as f32).collect();
        let input = Compute::float_from_data(TensorData::new(values, [257]), context.device());
        let twice = Compute::float_add(input.clone(), input);
        context.import_float::<f32>(twice)
    }).collect::<Result<Vec<_>, _>>()?;
    let original = tensors[0].float_tensor()?;
    let buffer = collective.import_buffers(tensors.clone())?;
    collective.all_reduce_sum(&buffer, &sum)?;
    for tensor in &tensors {
        // Export directly into the next tensor computation, not via host values.
        let value = tensor.float_tensor()?;
        let squared = Compute::float_mul(value.clone(), value);
        let actual = read_sync(Compute::float_into_data(squared))?
            .into_vec::<f32>().map_err(|error| TensorDeviceError::Data(format!("{error:?}")))?;
        let expected: Vec<_> = (0..257).map(|index| (6.0 * index as f32 + 6.0).powi(2)).collect();
        assert_eq!(actual, expected);
    }
    let unchanged = read_sync(Compute::float_into_data(original))?
        .into_vec::<f32>().map_err(|error| TensorDeviceError::Data(format!("{error:?}")))?;
    assert_eq!(unchanged, (0..257).map(|index| 2.0 * index as f32).collect::<Vec<_>>());
    contexts[0].synchronize()?;
    println!("tensor → in-process ring → tensor: 3 logical ranks, 257 F32 elements, {}",
        Compute::name(contexts[0].device()));
    Ok(())
}
