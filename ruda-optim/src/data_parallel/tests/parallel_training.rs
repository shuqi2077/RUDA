use super::*;
use crate::{AdamW, AdamWConfig};
use crate::data_parallel::zero::{Zero1, Zero1Record};
use ruda_model::record::{FullPrecisionSettings, Record};
use ruda_nn::modules::tensor_parallel::{ColumnParallelLinear, RowParallelLinear};

#[test]
fn zero1_adamw_state_shards_match_dense_updates_and_resume() {
    world(|rank, communicator| {
        let (session, mut replica) = DataParallel::<B>::initialize(communicator, model(rank), 0).unwrap();
        let mut zero = Zero1::new(session, &replica, AdamWConfig::new().build(), vec![0, 1]).unwrap();
        let mut reference = model(0);
        let mut reference_optimizer = AdamWConfig::new().init();
        for _ in 0..3 {
            let gradients = GradientsParams::from_grads(loss(&replica, rank).backward(), &replica);
            let result = zero.step(0.01, replica, gradients, if rank == 0 { 2 } else { 1 }, MissingGradientPolicy::Error).unwrap();
            assert_eq!(result.global_weight, 3);
            replica = result.model;
            let mean = (loss(&reference, 0) + loss(&reference, 1)) / 3.;
            let gradients = GradientsParams::from_grads(mean.backward(), &reference);
            reference = reference_optimizer.step(0.01, reference, gradients);
            replica.weight.val().to_data().assert_approx_eq::<f32>(&reference.weight.val().to_data(), Tolerance::absolute(1e-6));
            replica.bias.as_ref().unwrap().val().to_data().assert_approx_eq::<f32>(&reference.bias.as_ref().unwrap().val().to_data(), Tolerance::absolute(1e-6));
        }
        assert_eq!(zero.owned_parameter_count(), 1);
        assert_eq!(zero.state_parameter_count(), 1);
        let saved_model = replica.clone();
        let saved = zero.to_record().into_item::<FullPrecisionSettings>();
        let gradients = GradientsParams::from_grads(loss(&replica, rank).backward(), &replica);
        let expected = zero.step(0.01, replica, gradients, if rank == 0 { 2 } else { 1 }, MissingGradientPolicy::Error).unwrap().model;
        let saved = Zero1Record::<B, Linear<B>, AdamW>::from_item::<FullPrecisionSettings>(saved, &Default::default());
        zero.load_record(saved).unwrap();
        let gradients = GradientsParams::from_grads(loss(&saved_model, rank).backward(), &saved_model);
        let resumed = zero.step(0.01, saved_model, gradients, if rank == 0 { 2 } else { 1 }, MissingGradientPolicy::Error).unwrap().model;
        resumed.weight.val().to_data().assert_eq(&expected.weight.val().to_data(), false);
        resumed.bias.unwrap().val().to_data().assert_eq(&expected.bias.unwrap().val().to_data(), false);
    });
}

#[test]
fn zero1_ownership_mismatch_rejects_collectively() {
    world(|rank, communicator| {
        let (session, replica) = DataParallel::<B>::initialize(communicator, model(rank), 0).unwrap();
        let owners = if rank == 0 { vec![0, 1] } else { vec![1, 0] };
        assert!(Zero1::new(session, &replica, AdamWConfig::new().build(), owners).is_err());
    });
}

#[test]
fn tensor_parallel_column_row_chain_matches_dense_forward_and_every_gradient() {
    world(|rank, communicator| {
        let device = Default::default();
        let first = Linear::<B> {
            weight: Param::from_tensor(Tensor::from_floats([[1., 2., 3., 4.], [-1., 0.5, 2., -2.]], &device)),
            bias: Some(Param::from_tensor(Tensor::from_floats([0.1, 0.2, 0.3, 0.4], &device))),
        };
        let second = Linear::<B> {
            weight: Param::from_tensor(Tensor::from_floats([[0.5], [1.], [-1.], [0.25]], &device)),
            bias: Some(Param::from_tensor(Tensor::from_floats([0.25], &device))),
        };
        let range = rank as usize * 2..(rank as usize + 1) * 2;
        let column = ColumnParallelLinear::from_shard(Linear::<B> {
            weight: Param::from_tensor(first.weight.val().slice_dim(1, range.clone()).detach()),
            bias: Some(Param::from_tensor(first.bias.as_ref().unwrap().val().slice_dim(0, range.clone()).detach())),
        });
        let row = RowParallelLinear::from_shard(Linear::<B> {
            weight: Param::from_tensor(second.weight.val().slice_dim(0, range.clone()).detach()),
            bias: Some(Param::from_tensor(second.bias.as_ref().unwrap().val().detach())),
        });
        let input = Tensor::<B, 2>::from_floats([[1., 2.], [-1., 0.5]], &device).require_grad();
        let reference_input = input.clone().detach().require_grad();
        let output = column.forward(input.clone(), communicator.clone(), false).unwrap().square();
        let output = row.forward(output, communicator, true).unwrap();
        let reference = second.forward(first.forward(reference_input.clone()).square());
        output.to_data().assert_approx_eq::<f32>(&reference.to_data(), Tolerance::absolute(1e-5));
        let gradients = output.square().sum().backward();
        let expected = reference.square().sum().backward();
        let compare = |actual: Tensor<Host, 2>, expected: Tensor<Host, 2>| {
            actual.to_data().assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(1e-3));
        };
        compare(input.grad(&gradients).unwrap(), reference_input.grad(&expected).unwrap());
        compare(column.local.weight.val().grad(&gradients).unwrap(), first.weight.val().grad(&expected).unwrap().slice_dim(1, range.clone()));
        compare(row.local.weight.val().grad(&gradients).unwrap(), second.weight.val().grad(&expected).unwrap().slice_dim(0, range.clone()));
        column.local.bias.as_ref().unwrap().val().grad(&gradients).unwrap().to_data().assert_approx_eq::<f32>(
            &first.bias.as_ref().unwrap().val().grad(&expected).unwrap().slice_dim(0, range).to_data(), Tolerance::absolute(1e-3));
        row.local.bias.as_ref().unwrap().val().grad(&gradients).unwrap().to_data().assert_approx_eq::<f32>(
            &second.bias.as_ref().unwrap().val().grad(&expected).unwrap().to_data(), Tolerance::absolute(1e-3));
    });
}

#[test]
fn tensor_parallel_gather_scatter_preserves_replicated_loss_derivative() {
    world(|_, communicator| {
        use ruda_autodiff::tensor_parallel::{gather_from_region, scatter_to_region};
        let input = Tensor::<B, 2>::from_floats([[1., 2., 3., 4.]], &Default::default()).require_grad();
        let shard = scatter_to_region(input.clone(), communicator.clone(), 1).unwrap();
        let output = gather_from_region(shard, communicator, 1).unwrap();
        output.to_data().assert_eq(&input.to_data(), false);
        let gradients = output.square().sum().backward();
        input.grad(&gradients).unwrap().to_data().assert_eq(&TensorData::from([[2., 4., 6., 8.]]), false);
    });
}
