use ruccl::{
    AllReduceStrategy, CollectiveConfig, ReduceOperation, all_reduce, finish_collective, register,
};
use ruda_tensor::api::{Tensor, TensorData, TensorPrimitive};
use ruda_tensor_device::cuda::{Cuda, CudaDevice};

type Backend = Cuda<f32>;
const RANKS: usize = 4;
const ELEMENTS: usize = 257;

fn main() {
    // Four logical ranks share GPU 0; this example does not require four GPUs.
    let config = CollectiveConfig::default()
        .with_num_devices(RANKS)
        .with_local_all_reduce_strategy(AllReduceStrategy::Ring);
    let workers: Vec<_> = (0..RANKS)
        .map(|rank| {
            let config = config.clone();
            std::thread::spawn(move || {
                let device = CudaDevice::default();
                let peer = rank.into();
                register::<Backend>(peer, device.clone(), config).unwrap();
                let values: Vec<f32> = (0..ELEMENTS)
                    .map(|index| index as f32 * 0.25 + rank as f32)
                    .collect();
                let input = Tensor::<Backend, 1>::from_data(
                    TensorData::new(values.clone(), [ELEMENTS]),
                    &device,
                );
                for (operation, divisor) in [
                    (ReduceOperation::Sum, 1.0_f32),
                    (ReduceOperation::Mean, 4.0),
                ] {
                    let transformed = input.clone().mul_scalar(2.0).add_scalar(1.0);
                    let output = all_reduce::<Backend>(
                        peer,
                        transformed.into_primitive().tensor(),
                        operation,
                    )
                    .unwrap();
                    let output =
                        Tensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(output));
                    let actual = output.to_data().to_vec::<f32>().unwrap();
                    let expected: Vec<f32> = (0..ELEMENTS)
                        .map(|index| (2.0 * index as f32 + 16.0) / divisor)
                        .collect();
                    assert_eq!(actual, expected);
                    assert_eq!(input.to_data().to_vec::<f32>().unwrap(), values);
                }
                finish_collective::<Backend>(peer).unwrap();
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    println!(
        "Ring Sum/Mean verified: {RANKS} logical ranks, {ELEMENTS} FP32 elements, one CUDA GPU"
    );
}
