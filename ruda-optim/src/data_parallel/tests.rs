use super::*;
use crate::{Optimizer, SgdConfig};
use ruccl::rank::{TcpRendezvousServer, UniqueId};
use ruda_autodiff::Autodiff;
use ruda_model::{
    module::Module,
    tensor::{TensorData, Tolerance},
};
use ruda_nn::Linear;
use ruda_tensor_host::Host;
use std::{sync::Arc, thread, time::Duration};
type B = Autodiff<Host>;

mod parallel_training;

fn world<R: Send + 'static>(
    run: impl Fn(u32, RankCommunicator<TensorDevice<Host>>) -> R + Send + Sync + 'static,
) -> Vec<R> {
    let id = UniqueId::new();
    let server = TcpRendezvousServer::bind("127.0.0.1:0", id, 2).unwrap();
    let address = server.local_addr().unwrap();
    let coordinator = thread::spawn(move || server.run());
    let run = Arc::new(run);
    let workers = (0..2)
        .map(|rank| {
            let run = run.clone();
            thread::spawn(move || {
                let communicator = RankCommunicator::connect(
                    || Ok::<_, TensorDeviceError>(TensorDevice::<Host>::new(Default::default())),
                    address,
                    id,
                    rank,
                    2,
                    Duration::from_secs(10),
                    "data-parallel-test",
                )
                .unwrap();
                run(rank, communicator)
            })
        })
        .collect::<Vec<_>>();
    let results = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    coordinator.join().unwrap().unwrap();
    results
}

#[test]
fn integer_collectives_preserve_wide_values_shapes_and_backend_reductions() {
    use ruccl::rank::ReductionOperation;
    use ruda_model::tensor::{IntDType, TensorCreationOptions};
    for dtype in [IntDType::I32, IntDType::I64] {
        world(move |rank, communicator| {
            let device: <Host as ruda_model::tensor::backend::BackendTypes>::Device =
                Default::default();
            let large = if dtype == IntDType::I64 {
                9_007_199_254_740_993_i64
            } else {
                16_777_217_i64
            };
            let values = [large, -9, 7, large + 3];
            let tensor = |values: [i64; 4]| {
                Tensor::<Host, 2, Int>::from_data(
                    TensorData::new(values.to_vec(), [2, 2]),
                    TensorCreationOptions::<Host>::new(device.clone()).with_dtype(dtype.into()),
                )
            };
            let input = tensor([
                large + rank as i64,
                7 + rank as i64,
                -9 + rank as i64,
                large + 3 + rank as i64,
            ])
            .swap_dims(0, 1);
            let broadcast = communicator
                .broadcast_int(input.clone().into_primitive(), 1)
                .unwrap();
            let broadcast = Tensor::<Host, 2, Int>::from_primitive(broadcast);
            assert_eq!(broadcast.dims(), [2, 2]);
            assert_eq!(broadcast.dtype(), dtype.into());
            assert_eq!(
                broadcast
                    .cast(IntDType::I64)
                    .into_data()
                    .to_vec::<i64>()
                    .unwrap(),
                values.map(|value| value + 1).to_vec()
            );
            for operation in [
                ReductionOperation::Sum,
                ReductionOperation::Minimum,
                ReductionOperation::Maximum,
                ReductionOperation::BitAnd,
                ReductionOperation::BitOr,
                ReductionOperation::BitXor,
            ] {
                let reduced = communicator
                    .all_reduce_int(input.clone().into_primitive(), operation)
                    .unwrap();
                let expected = values.map(|a| {
                    let b = a + 1;
                    match operation {
                        ReductionOperation::Sum => a + b,
                        ReductionOperation::Minimum => a.min(b),
                        ReductionOperation::Maximum => a.max(b),
                        ReductionOperation::BitAnd => a & b,
                        ReductionOperation::BitOr => a | b,
                        ReductionOperation::BitXor => a ^ b,
                        _ => unreachable!(),
                    }
                });
                let reduced = Tensor::<Host, 2, Int>::from_primitive(reduced);
                assert_eq!(reduced.dtype(), dtype.into());
                assert_eq!(reduced.dims(), [2, 2]);
                assert_eq!(
                    reduced
                        .cast(IntDType::I64)
                        .into_data()
                        .to_vec::<i64>()
                        .unwrap(),
                    expected.to_vec()
                );
            }
            assert_eq!(
                input
                    .cast(IntDType::I64)
                    .into_data()
                    .to_vec::<i64>()
                    .unwrap(),
                values.map(|value| value + rank as i64).to_vec()
            );
            let input = tensor(if rank == 0 {
                [-9, 7, 3, 4]
            } else {
                [2, -1, 5, -6]
            });
            let reduced = communicator
                .all_reduce_int(input.into_primitive(), ReductionOperation::Product)
                .unwrap();
            assert_eq!(
                Tensor::<Host, 2, Int>::from_primitive(reduced)
                    .cast(IntDType::I64)
                    .into_data()
                    .to_vec::<i64>()
                    .unwrap(),
                vec![-18, -7, 15, -24]
            );
            let empty = Tensor::<Host, 2, Int>::empty([2, 0], &device).cast(dtype);
            let reduced = communicator
                .all_reduce_int(empty.into_primitive(), ReductionOperation::Sum)
                .unwrap();
            assert_eq!(
                Tensor::<Host, 2, Int>::from_primitive(reduced).dims(),
                [2, 0]
            );
        });
    }
}

#[test]
fn sharded_collectives_backward_exchanges_rank_gradients_and_preserves_leaf_views() {
    use ruda_autodiff::collective::{all_gather, reduce_scatter_mean, reduce_scatter_sum};
    use ruda_model::tensor::FloatDType;
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
        world(move |rank, communicator| {
            let device = Default::default();
            let input = Tensor::<B, 2>::from_data([[1., 2.], [3., 4.]], &device)
                .cast(dtype)
                .add_scalar(rank as f32)
                .swap_dims(0, 1)
                .detach()
                .require_grad();
            let output = all_gather(input.clone(), communicator.clone()).unwrap();
            assert_eq!(output.dims(), [4, 2]);
            assert_eq!(output.dtype(), dtype.into());
            assert_eq!(
                output
                    .clone()
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![1., 3., 2., 4., 2., 4., 3., 5.]
            );
            let weights = Tensor::<B, 2>::from_data(
                TensorData::new(
                    (1..=8).map(|n| n as f32 * (rank + 1) as f32).collect(),
                    [4, 2],
                ),
                &device,
            )
            .cast(dtype);
            let gradients = (output * weights).sum().backward();
            let gradient = input.grad(&gradients).unwrap();
            assert_eq!(gradient.dtype(), dtype.into());
            assert_eq!(gradient.dims(), [2, 2]);
            assert_eq!(
                gradient
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                (1..=4)
                    .map(|n| (4 * rank + n) as f32 * 3.)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                input
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                [1., 3., 2., 4.].map(|n| n + rank as f32).to_vec()
            );

            for mean in [false, true] {
                let input =
                    Tensor::<B, 2>::from_data([[1., 2., 3., 4.], [5., 6., 7., 8.]], &device)
                        .cast(dtype)
                        .mul_scalar(rank + 1)
                        .swap_dims(0, 1)
                        .detach()
                        .require_grad();
                let output = if mean {
                    reduce_scatter_mean(input.clone(), communicator.clone()).unwrap()
                } else {
                    reduce_scatter_sum(input.clone(), communicator.clone()).unwrap()
                };
                let scale = if mean { 0.5 } else { 1. };
                assert_eq!(output.dims(), [2, 2]);
                assert_eq!(output.dtype(), dtype.into());
                let values = if rank == 0 {
                    [3., 15., 6., 18.]
                } else {
                    [9., 21., 12., 24.]
                };
                assert_eq!(
                    output
                        .clone()
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    values.map(|value| value * scale).to_vec()
                );
                let weights = Tensor::<B, 2>::from_data([[1., 2.], [3., 4.]], &device)
                    .cast(dtype)
                    .mul_scalar(rank + 1);
                let gradients = (output * weights).sum().backward();
                let gradient = input.grad(&gradients).unwrap();
                assert_eq!(gradient.dtype(), dtype.into());
                assert_eq!(gradient.dims(), [4, 2]);
                assert_eq!(
                    gradient
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [1., 2., 3., 4., 2., 4., 6., 8.]
                        .map(|value| value * scale)
                        .to_vec()
                );
            }
            let empty = Tensor::<B, 2>::empty([2, 0], &device)
                .cast(dtype)
                .require_grad();
            let gathered = all_gather(empty.clone(), communicator.clone()).unwrap();
            assert_eq!(gathered.dims(), [4, 0]);
            let gradients = gathered.sum().backward();
            assert_eq!(empty.grad(&gradients).unwrap().dims(), [2, 0]);
        });
    }
}

#[test]
fn broadcast_backward_accumulates_only_into_requested_root_and_preserves_placeholders() {
    use ruda_autodiff::collective::broadcast;
    use ruda_model::tensor::FloatDType;
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
        world(move |rank, communicator| {
            let device = Default::default();
            for root in [0u32, 1] {
                let input = Tensor::<B, 2>::from_data([[1., 2.], [3., 4.]], &device)
                    .cast(dtype)
                    .mul_scalar(rank + 1)
                    .swap_dims(0, 1)
                    .detach()
                    .require_grad();
                let output = broadcast(input.clone(), communicator.clone(), root).unwrap();
                assert_eq!(output.dtype(), dtype.into());
                assert_eq!(output.dims(), [2, 2]);
                assert_eq!(
                    output
                        .clone()
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [1., 3., 2., 4.]
                        .map(|value| value * (root + 1) as f32)
                        .to_vec()
                );
                let weights = Tensor::<B, 2>::from_data([[1., 2.], [3., 4.]], &device)
                    .cast(dtype)
                    .mul_scalar(rank + 1);
                let gradients = (output * weights).sum().backward();
                let gradient = input.grad(&gradients).unwrap();
                assert_eq!(gradient.dtype(), dtype.into());
                assert_eq!(gradient.dims(), [2, 2]);
                let scale = if rank == root { 3. } else { 0. };
                assert_eq!(
                    gradient
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [1., 2., 3., 4.].map(|value| value * scale).to_vec()
                );
                assert_eq!(
                    input
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [1., 3., 2., 4.]
                        .map(|value| value * (rank + 1) as f32)
                        .to_vec()
                );
            }
            let empty = Tensor::<B, 2>::empty([0, 2], &device)
                .cast(dtype)
                .require_grad();
            let output = broadcast(empty.clone(), communicator, 1).unwrap();
            let gradients = output.sum().backward();
            assert_eq!(empty.grad(&gradients).unwrap().dims(), [0, 2]);
        });
    }
}

#[test]
fn replicated_collectives_backward_sums_rank_local_losses_without_mutating_leaves() {
    use ruda_autodiff::collective::{all_reduce_mean, all_reduce_sum};
    use ruda_model::tensor::FloatDType;
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
        world(move |rank, communicator| {
            let device = Default::default();
            for mean in [false, true] {
                let input = Tensor::<B, 2>::from_data([[1., 2.], [3., 4.]], &device)
                    .cast(dtype)
                    .mul_scalar(rank + 1)
                    .swap_dims(0, 1)
                    .detach()
                    .require_grad();
                let output = if mean {
                    all_reduce_mean(input.clone(), communicator.clone()).unwrap()
                } else {
                    all_reduce_sum(input.clone(), communicator.clone()).unwrap()
                };
                let scale = if mean { 0.5 } else { 1. };
                assert_eq!(output.dtype(), dtype.into());
                assert_eq!(output.dims(), [2, 2]);
                assert_eq!(
                    output
                        .clone()
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [1., 3., 2., 4.].map(|value| value * 3. * scale).to_vec()
                );
                let weights = Tensor::<B, 2>::from_data([[1., 2.], [3., 4.]], &device)
                    .cast(dtype)
                    .mul_scalar(rank + 1);
                let gradients = (output * weights).sum().backward();
                let gradient = input.grad(&gradients).unwrap();
                assert_eq!(gradient.dtype(), dtype.into());
                assert_eq!(gradient.dims(), [2, 2]);
                assert_eq!(
                    gradient
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [1., 2., 3., 4.].map(|value| value * 3. * scale).to_vec()
                );
                assert_eq!(
                    input
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [1., 3., 2., 4.]
                        .map(|value| value * (rank + 1) as f32)
                        .to_vec()
                );
            }
            let empty = Tensor::<B, 2>::empty([0, 2], &device)
                .cast(dtype)
                .require_grad();
            let output = all_reduce_sum(empty.clone(), communicator).unwrap();
            assert_eq!(output.dims(), [0, 2]);
            let gradients = output.sum().backward();
            assert_eq!(empty.grad(&gradients).unwrap().dims(), [0, 2]);
        });
    }
}

#[test]
fn sharded_collectives_support_hidden_axes_and_restore_strided_leaf_gradients() {
    use ruda_autodiff::collective::{
        all_gather_dim, reduce_scatter_mean_dim, reduce_scatter_sum_dim,
    };
    use ruda_model::tensor::FloatDType;
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
        world(move |rank, communicator| {
            let device = Default::default();
            let input = Tensor::<B, 3>::from_data(
                TensorData::new(
                    (1..=8).map(|n| n as f32 + 8. * rank as f32).collect(),
                    [2, 2, 2],
                ),
                &device,
            )
            .cast(dtype)
            .swap_dims(0, 2)
            .detach()
            .require_grad();
            let output = all_gather_dim(input.clone(), communicator.clone(), -1i64).unwrap();
            assert_eq!(output.dims(), [2, 2, 4]);
            assert_eq!(output.dtype(), dtype.into());
            assert_eq!(
                output
                    .clone()
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![
                    1., 5., 9., 13., 3., 7., 11., 15., 2., 6., 10., 14., 4., 8., 12., 16.
                ]
            );
            let weights = Tensor::<B, 3>::from_data(
                TensorData::new(
                    (1..=16).map(|n| n as f32 * (rank + 1) as f32).collect(),
                    [2, 2, 4],
                ),
                &device,
            )
            .cast(dtype);
            let gradients = (output * weights).sum().backward();
            let gradient = input.grad(&gradients).unwrap();
            assert_eq!(gradient.dtype(), dtype.into());
            assert_eq!(gradient.dims(), [2, 2, 2]);
            let expected = if rank == 0 {
                [1., 2., 5., 6., 9., 10., 13., 14.]
            } else {
                [3., 4., 7., 8., 11., 12., 15., 16.]
            };
            assert_eq!(
                gradient
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                expected.map(|value| value * 3.).to_vec()
            );
            for mean in [false, true] {
                let input = Tensor::<B, 3>::from_data(
                    TensorData::new(
                        (1..=16).map(|n| n as f32 * (rank + 1) as f32).collect(),
                        [4, 2, 2],
                    ),
                    &device,
                )
                .cast(dtype)
                .swap_dims(0, 2)
                .detach()
                .require_grad();
                let output = if mean {
                    reduce_scatter_mean_dim(input.clone(), communicator.clone(), 2).unwrap()
                } else {
                    reduce_scatter_sum_dim(input.clone(), communicator.clone(), 2).unwrap()
                };
                let scale = if mean { 0.5 } else { 1. };
                assert_eq!(output.dtype(), dtype.into());
                assert_eq!(output.dims(), [2, 2, 2]);
                let expected = if rank == 0 {
                    [1., 5., 3., 7., 2., 6., 4., 8.]
                } else {
                    [9., 13., 11., 15., 10., 14., 12., 16.]
                };
                assert_eq!(
                    output
                        .clone()
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    expected.map(|value| value * 3. * scale).to_vec()
                );
                let weights = Tensor::<B, 3>::from_data(
                    TensorData::new(
                        (1..=8).map(|n| n as f32 * (rank + 1) as f32).collect(),
                        [2, 2, 2],
                    ),
                    &device,
                )
                .cast(dtype);
                let gradients = (output * weights).sum().backward();
                let gradient = input.grad(&gradients).unwrap();
                assert_eq!(gradient.dims(), [2, 2, 4]);
                assert_eq!(gradient.dtype(), dtype.into());
                assert_eq!(
                    gradient
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    [
                        1., 2., 2., 4., 3., 4., 6., 8., 5., 6., 10., 12., 7., 8., 14., 16.
                    ]
                    .map(|value| value * scale)
                    .to_vec()
                );
            }
            let empty = Tensor::<B, 2>::empty([0, 2], &device)
                .cast(dtype)
                .require_grad();
            let output = all_gather_dim(empty.clone(), communicator, 1).unwrap();
            assert_eq!(output.dims(), [0, 4]);
            let gradients = output.sum().backward();
            assert_eq!(empty.grad(&gradients).unwrap().dims(), [0, 2]);
        });
    }
}

#[test]
fn sharded_tensor_collectives_retain_rank_order_dtypes_views_and_input_values() {
    use ruccl::rank::ReductionOperation;
    use ruda_model::tensor::{FloatDType, IntDType, Shape, TensorCreationOptions};
    for dtype in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
        world(move |rank, communicator| {
            let device: <Host as ruda_model::tensor::backend::BackendTypes>::Device =
                Default::default();
            let input = Tensor::<Host, 2>::from_data([[1., 2.], [3., 4.]], &device)
                .cast(dtype)
                .add_scalar(rank as f32)
                .swap_dims(0, 1);
            let gathered = Tensor::<Host, 2>::from_primitive(TensorPrimitive::Float(
                communicator
                    .all_gather_float(input.clone().into_primitive().tensor())
                    .unwrap(),
            ));
            assert_eq!(gathered.dims(), [4, 2]);
            assert_eq!(gathered.dtype(), dtype.into());
            assert_eq!(
                gathered
                    .clone()
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![1., 3., 2., 4., 2., 4., 3., 5.]
            );
            for operation in [ReduceOperation::Sum, ReduceOperation::Mean] {
                let output = Tensor::<Host, 2>::from_primitive(TensorPrimitive::Float(
                    communicator
                        .reduce_scatter_float(gathered.clone().into_primitive().tensor(), operation)
                        .unwrap(),
                ));
                assert_eq!(output.dims(), [2, 2]);
                assert_eq!(output.dtype(), dtype.into());
                let scale = if operation == ReduceOperation::Sum {
                    2.
                } else {
                    1.
                };
                let expected = [1., 3., 2., 4.].map(|value| (value + rank as f32) * scale);
                assert_eq!(
                    output
                        .cast(FloatDType::F32)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap(),
                    expected.to_vec()
                );
            }
            assert_eq!(
                input
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                [1., 3., 2., 4.].map(|value| value + rank as f32).to_vec()
            );
            assert!(
                communicator
                    .reduce_scatter_float(
                        Tensor::<Host, 2>::zeros([3, 2], &device)
                            .cast(dtype)
                            .into_primitive()
                            .tensor(),
                        ReduceOperation::Sum
                    )
                    .is_err()
            );
            let empty = Tensor::<Host, 2>::empty([2, 0], &device).cast(dtype);
            let gathered = communicator
                .all_gather_float(empty.into_primitive().tensor())
                .unwrap();
            assert_eq!(gathered.shape(), Shape::new([4, 0]));
            assert_eq!(
                communicator
                    .reduce_scatter_float(gathered, ReduceOperation::Sum)
                    .unwrap()
                    .shape(),
                Shape::new([2, 0])
            );
        });
    }
    for dtype in [IntDType::I32, IntDType::I64] {
        world(move |rank, communicator| {
            let device: <Host as ruda_model::tensor::backend::BackendTypes>::Device =
                Default::default();
            let large = if dtype == IntDType::I64 {
                9_007_199_254_740_993_i64
            } else {
                16_777_217_i64
            };
            let input = Tensor::<Host, 2, Int>::from_data(
                TensorData::new(vec![large + rank as i64, -3, 7, 2], [2, 2]),
                TensorCreationOptions::<Host>::new(device.clone()).with_dtype(dtype.into()),
            )
            .swap_dims(0, 1);
            let gathered = Tensor::<Host, 2, Int>::from_primitive(
                communicator
                    .all_gather_int(input.clone().into_primitive())
                    .unwrap(),
            );
            assert_eq!(gathered.dims(), [4, 2]);
            assert_eq!(gathered.dtype(), dtype.into());
            assert_eq!(
                gathered
                    .clone()
                    .cast(IntDType::I64)
                    .into_data()
                    .to_vec::<i64>()
                    .unwrap(),
                vec![large, 7, -3, 2, large + 1, 7, -3, 2]
            );
            for operation in [
                ReductionOperation::Sum,
                ReductionOperation::Minimum,
                ReductionOperation::Maximum,
                ReductionOperation::BitAnd,
                ReductionOperation::BitOr,
                ReductionOperation::BitXor,
            ] {
                let output = Tensor::<Host, 2, Int>::from_primitive(
                    communicator
                        .reduce_scatter_int(gathered.clone().into_primitive(), operation)
                        .unwrap(),
                );
                let expected = [large + rank as i64, 7, -3, 2].map(|value| match operation {
                    ReductionOperation::Sum => value * 2,
                    ReductionOperation::BitXor => 0,
                    _ => value,
                });
                assert_eq!(output.dims(), [2, 2]);
                assert_eq!(output.dtype(), dtype.into());
                assert_eq!(
                    output
                        .cast(IntDType::I64)
                        .into_data()
                        .to_vec::<i64>()
                        .unwrap(),
                    expected.to_vec()
                );
            }
            assert_eq!(
                input
                    .cast(IntDType::I64)
                    .into_data()
                    .to_vec::<i64>()
                    .unwrap(),
                vec![large + rank as i64, 7, -3, 2]
            );
            let empty = Tensor::<Host, 2, Int>::empty([2, 0], &device).cast(dtype);
            let gathered = communicator.all_gather_int(empty.into_primitive()).unwrap();
            assert_eq!(gathered.shape(), Shape::new([4, 0]));
            assert_eq!(
                communicator
                    .reduce_scatter_int(gathered, ReductionOperation::Sum)
                    .unwrap()
                    .shape(),
                Shape::new([2, 0])
            );
        });
    }
}

fn model(rank: u32) -> Linear<B> {
    let device = Default::default();
    Linear {
        weight: Param::from_tensor(Tensor::from_floats([[rank as f32 + 0.5], [-0.25]], &device)),
        bias: Some(Param::from_tensor(Tensor::from_floats(
            [rank as f32 + 0.25],
            &device,
        ))),
    }
}

fn loss(model: &Linear<B>, rank: u32) -> Tensor<B, 1> {
    let device = Default::default();
    let (x, y) = if rank == 0 {
        (vec![1.0, 2.0, -1.0, 1.0], vec![1.0, 0.0])
    } else {
        (vec![2.0, -1.0], vec![0.5])
    };
    let count = y.len();
    let residual = model.forward::<2>(Tensor::from_floats(TensorData::new(x, [count, 2]), &device))
        - Tensor::from_floats(TensorData::new(y, [count, 1]), &device);
    residual.clone().mul(residual).sum()
}

#[test]
fn independent_ids_broadcast_and_token_weighted_sgd_match_single_replica() {
    let ids = world(|rank, communicator| {
        let original = model(rank);
        let local_id = original.weight.id;
        let (ddp, mut replica) = DataParallel::<B>::initialize(communicator, original, 0).unwrap();
        assert_eq!(local_id, replica.weight.id);
        assert_eq!(ddp.rank(), rank);
        assert_eq!(ddp.world_size(), 2);
        replica
            .weight
            .val()
            .to_data()
            .assert_eq(&TensorData::from([[0.5], [-0.25]]), false);
        let mut reference = model(0);
        let mut optimizer = SgdConfig::new().init();
        let mut reference_optimizer = SgdConfig::new().init();
        for _ in 0..3 {
            let grads = GradientsParams::from_grads(loss(&replica, rank).backward(), &replica);
            let reduced = ddp
                .reduce(
                    &replica,
                    grads,
                    if rank == 0 { 2 } else { 1 },
                    MissingGradientPolicy::Error,
                )
                .unwrap();
            assert_eq!(reduced.global_weight, 3);
            replica = optimizer.step(0.125, replica, reduced.gradients);
            let mean = (loss(&reference, 0) + loss(&reference, 1)).div_scalar(3);
            let expected = GradientsParams::from_grads(mean.backward(), &reference);
            reference = reference_optimizer.step(0.125, reference, expected);
            replica.weight.val().to_data().assert_approx_eq::<f32>(
                &reference.weight.val().to_data(),
                Tolerance::absolute(1e-6),
            );
            replica
                .bias
                .as_ref()
                .unwrap()
                .val()
                .to_data()
                .assert_approx_eq::<f32>(
                    &reference.bias.as_ref().unwrap().val().to_data(),
                    Tolerance::absolute(1e-6),
                );
        }
        local_id
    });
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn zero_weight_rank_and_explicit_unused_gradient_policy() {
    world(|rank, communicator| {
        let (ddp, replica) = DataParallel::<B>::initialize(communicator, model(rank), 1).unwrap();
        replica
            .weight
            .val()
            .to_data()
            .assert_eq(&TensorData::from([[1.5], [-0.25]]), false);
        let mut gradients = GradientsParams::new();
        if rank == 0 {
            gradients.register(
                replica.weight.id,
                Tensor::<Host, 2>::from_floats([[6.0], [9.0]], &Default::default()),
            );
        }
        let reduced = ddp
            .reduce(
                &replica,
                gradients,
                if rank == 0 { 3 } else { 0 },
                MissingGradientPolicy::Zero,
            )
            .unwrap();
        assert_eq!(reduced.global_weight, 3);
        assert_eq!(reduced.gradients.len(), 1);
        reduced
            .gradients
            .get::<Host, 2>(replica.weight.id)
            .unwrap()
            .to_data()
            .assert_eq(&TensorData::from([[2.0], [3.0]]), false);
        let result = ddp.reduce(
            &replica,
            GradientsParams::new(),
            1,
            MissingGradientPolicy::Error,
        );
        assert!(matches!(result, Err(DataParallelError::Contract(_))));
        // A rejected window leaves the communicator usable for the next window.
        let mut gradients = GradientsParams::new();
        gradients.register(
            replica.weight.id,
            Tensor::<Host, 2>::ones([2, 1], &Default::default()),
        );
        gradients.register(
            replica.bias.as_ref().unwrap().id,
            Tensor::<Host, 1>::ones([1], &Default::default()),
        );
        assert!(
            ddp.reduce(&replica, gradients, 1, MissingGradientPolicy::Error)
                .is_ok()
        );
        assert!(
            ddp.reduce(
                &replica,
                GradientsParams::new(),
                0,
                MissingGradientPolicy::Zero
            )
            .is_err()
        );
    });
}

#[test]
fn inconsistent_replicas_reject_before_parameter_broadcast() {
    world(|rank, communicator| {
        let mut replica = model(rank);
        if rank == 1 {
            replica.bias = None;
        }
        assert!(matches!(
            DataParallel::<B>::initialize(communicator, replica, 0),
            Err(DataParallelError::Contract(_))
        ));
    });
}

#[derive(Module, Debug)]
struct Tied<B: ruda_model::tensor::backend::Backend> {
    first: Linear<B>,
    second: Linear<B>,
}

#[derive(Module, Debug)]
struct BufferedReplica<B: ruda_model::tensor::backend::Backend> {
    weight: Param<Tensor<B, 1>>,
    counter: Param<Tensor<B, 1, Int>>,
    counter_alias: Param<Tensor<B, 1, Int>>,
    flags: Param<Tensor<B, 1, Bool>>,
    flags_alias: Param<Tensor<B, 1, Bool>>,
}

fn buffered_replica(rank: u32) -> BufferedReplica<B> {
    let device: <B as ruda_model::tensor::backend::BackendTypes>::Device = Default::default();
    let counter = Param::initialized(
        ParamId::new(),
        Tensor::<B, 1, Int>::from_data(
            TensorData::from([9_007_199_254_740_993_i64 + rank as i64]),
            ruda_model::tensor::TensorCreationOptions::<B>::new(device.clone())
                .with_dtype(DType::I64),
        ),
    );
    let flags = Param::initialized(
        ParamId::new(),
        Tensor::<B, 1, Bool>::from_data([rank != 0, rank == 0], &device),
    );
    BufferedReplica {
        weight: Param::from_tensor(Tensor::<B, 1>::full([2], rank + 1, &device)),
        counter_alias: counter.clone(),
        counter,
        flags_alias: flags.clone(),
        flags,
    }
}

#[test]
fn explicit_buffer_initialization_preserves_aliases_and_excludes_buffers_from_updates() {
    world(|rank, communicator| {
        let module = buffered_replica(rank);
        let counter_id = module.counter.id;
        let flags_id = module.flags.id;
        let flags_dtype = module.flags.val().dtype();
        let (ddp, replica) =
            DataParallel::<B>::initialize_with_buffers(communicator, module, 1).unwrap();
        assert_eq!(replica.counter.id, counter_id);
        assert_eq!(replica.counter_alias.id, counter_id);
        assert_eq!(replica.flags.id, flags_id);
        assert_eq!(replica.flags_alias.id, flags_id);
        let gradients =
            GradientsParams::from_grads(replica.weight.val().sum().backward(), &replica);
        let reduced = ddp
            .reduce_fp32(&replica, gradients, 1, MissingGradientPolicy::Error)
            .unwrap();
        assert_eq!(reduced.gradients.len(), 1);
        let updated = SgdConfig::new()
            .init()
            .step(0.5, replica, reduced.gradients);
        updated
            .weight
            .val()
            .to_data()
            .assert_eq(&TensorData::from([1.5, 1.5]), false);
        for param in [updated.counter, updated.counter_alias] {
            assert_eq!(param.id, counter_id);
            assert_eq!(param.val().dtype(), DType::I64);
            assert_eq!(
                param.val().into_data().to_vec::<i64>().unwrap(),
                vec![9_007_199_254_740_994]
            );
        }
        for param in [updated.flags, updated.flags_alias] {
            assert_eq!(param.id, flags_id);
            assert_eq!(param.val().dtype(), flags_dtype);
            assert_eq!(
                param.val().into_data().to_vec::<bool>().unwrap(),
                vec![true, false]
            );
        }
    });
}

#[test]
fn implicit_buffer_initialization_and_rank_mode_mismatch_are_rejected_collectively() {
    world(|rank, communicator| {
        assert!(DataParallel::<B>::initialize(communicator, buffered_replica(rank), 0).is_err());
    });
    world(|rank, communicator| {
        let replica = model(rank);
        let result = if rank == 0 {
            DataParallel::<B>::initialize(communicator, replica, 0)
        } else {
            DataParallel::<B>::initialize_with_buffers(communicator, replica, 0)
        };
        assert!(result.is_err());
    });
}

#[derive(Module, Debug)]
struct FrozenAliasReplica<B: ruda_model::tensor::backend::Backend> {
    frozen_alias: Param<Tensor<B, 1>>,
    first: Param<Tensor<B, 1>>,
    alias: Param<Tensor<B, 1>>,
}

#[test]
fn diverged_frozen_alias_broadcast_preserves_values_and_shared_master_updates() {
    for dtype in [
        ruda_model::tensor::FloatDType::F16,
        ruda_model::tensor::FloatDType::BF16,
    ] {
        world(move |rank, communicator| {
            let device = Default::default();
            let first = Param::from_tensor(Tensor::<B, 1>::full([2], rank + 1, &device));
            let frozen_alias = first.clone().no_grad();
            let first = first.map(|tensor| (tensor + 1.).detach().require_grad());
            let replica = FrozenAliasReplica {
                frozen_alias,
                first: first.clone(),
                alias: first,
            }
            .to_dtype(dtype);
            let id = replica.first.id;
            let (ddp, replica) = DataParallel::<B>::initialize(communicator, replica, 0).unwrap();
            assert_eq!(replica.first.id, id);
            assert_eq!(replica.frozen_alias.id, id);
            assert_eq!(replica.alias.id, id);
            assert!(!replica.frozen_alias.val().is_require_grad());
            assert!(replica.first.val().is_require_grad());
            replica
                .first
                .val()
                .cast(DType::F32)
                .to_data()
                .assert_eq(&TensorData::from([2., 2.]), false);
            replica
                .frozen_alias
                .val()
                .cast(DType::F32)
                .to_data()
                .assert_eq(&TensorData::from([1., 1.]), false);
            let result = replica.first.val() + replica.alias.val() + replica.frozen_alias.val();
            let gradients = result.sum().backward();
            assert!(replica.frozen_alias.val().grad(&gradients).is_none());
            let gradients = GradientsParams::from_grads(gradients, &replica);
            let reduced = ddp
                .reduce_fp32(&replica, gradients, 1, MissingGradientPolicy::Error)
                .unwrap();
            assert_eq!(reduced.global_weight, 2);
            assert_eq!(reduced.gradients.len(), 1);
            reduced
                .gradients
                .get::<Host, 1>(id)
                .unwrap()
                .to_data()
                .assert_eq(&TensorData::from([2., 2.]), false);
            let mut optimizer = crate::Fp32MasterOptimizer::new(SgdConfig::new().build::<Host>())
                .init::<B, FrozenAliasReplica<B>>();
            let updated = optimizer.step(0.125, replica, reduced.gradients);
            for param in [updated.first, updated.alias] {
                assert_eq!(param.val().dtype(), dtype.into());
                param
                    .val()
                    .cast(DType::F32)
                    .to_data()
                    .assert_eq(&TensorData::from([1.75, 1.75]), false);
            }
            updated
                .frozen_alias
                .val()
                .cast(DType::F32)
                .to_data()
                .assert_eq(&TensorData::from([1., 1.]), false);
        });
    }
}

#[test]
fn tied_parameters_broadcast_and_reduce_once() {
    world(|rank, communicator| {
        let first = model(rank);
        let tied = Tied {
            second: first.clone(),
            first,
        };
        let (ddp, tied) = DataParallel::<B>::initialize(communicator, tied, 0).unwrap();
        let x = Tensor::<B, 2>::ones([1, 2], &Default::default());
        let result = tied.first.forward(x.clone()) + tied.second.forward(x);
        let gradients = GradientsParams::from_grads(result.sum().backward(), &tied);
        let reduced = ddp
            .reduce(&tied, gradients, 1, MissingGradientPolicy::Error)
            .unwrap();
        assert_eq!(reduced.gradients.len(), 2);
        reduced
            .gradients
            .get::<Host, 2>(tied.first.weight.id)
            .unwrap()
            .to_data()
            .assert_eq(&TensorData::from([[2.0], [2.0]]), false);
        reduced
            .gradients
            .get::<Host, 1>(tied.first.bias.as_ref().unwrap().id)
            .unwrap()
            .to_data()
            .assert_eq(&TensorData::from([2.0]), false);
    });
}

#[test]
fn gradient_shape_and_policy_mismatch_reject_on_every_rank() {
    world(|rank, communicator| {
        let (ddp, replica) = DataParallel::<B>::initialize(communicator, model(rank), 0).unwrap();
        let mut gradients = GradientsParams::new();
        gradients.register(
            replica.weight.id,
            Tensor::<Host, 2>::ones([if rank == 0 { 2 } else { 3 }, 1], &Default::default()),
        );
        assert!(
            ddp.reduce(&replica, gradients, 1, MissingGradientPolicy::Zero)
                .is_err()
        );
        let mut gradients = GradientsParams::new();
        gradients.register(
            replica.weight.id,
            Tensor::<Host, 1>::ones([2], &Default::default()),
        );
        assert!(
            ddp.reduce(&replica, gradients, 1, MissingGradientPolicy::Zero)
                .is_err()
        );
        let gradients = GradientsParams::from_grads(loss(&replica, rank).backward(), &replica);
        assert!(
            ddp.reduce(
                &replica,
                gradients,
                1,
                if rank == 0 {
                    MissingGradientPolicy::Error
                } else {
                    MissingGradientPolicy::Zero
                }
            )
            .is_err()
        );
    });
}

#[test]
fn fp32_reduction_keeps_half_parameter_gradient_precision_and_checks_rank_mode() {
    for dtype in [DType::F16, DType::BF16] {
        world(move |rank, communicator| {
            let device = Default::default();
            let original = Linear::<B> {
                weight: Param::from_tensor(Tensor::<B, 2>::ones([1, 1], &device).cast(dtype)),
                bias: None,
            };
            let (ddp, replica) = DataParallel::<B>::initialize(communicator, original, 0).unwrap();
            let gradient = || {
                let mut gradients = GradientsParams::new();
                let value = if rank == 0 { 1024.125 } else { -1024. };
                gradients.register(
                    replica.weight.id,
                    Tensor::<Host, 2>::full([1, 1], value, &device),
                );
                gradients
            };
            assert!(
                ddp.reduce(&replica, gradient(), 1, MissingGradientPolicy::Error)
                    .is_err()
            );
            let result = ddp
                .reduce_fp32(&replica, gradient(), 1, MissingGradientPolicy::Error)
                .unwrap();
            assert_eq!(result.global_weight, 2);
            let gradient = result.gradients.get::<Host, 2>(replica.weight.id).unwrap();
            assert_eq!(gradient.dtype(), DType::F32);
            assert_eq!(gradient.into_scalar(), 0.0625);
            assert_eq!(replica.weight.val().dtype(), dtype);
            let gradients = GradientsParams::new();
            let result = if rank == 0 {
                ddp.reduce(&replica, gradients, 0, MissingGradientPolicy::Zero)
            } else {
                ddp.reduce_fp32(&replica, gradients, 1, MissingGradientPolicy::Zero)
            };
            assert!(result.is_err());
        });
    }
}
