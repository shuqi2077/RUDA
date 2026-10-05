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
                weight: Param::from_tensor(Tensor::ones([1, 1], &device).cast(dtype)),
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
