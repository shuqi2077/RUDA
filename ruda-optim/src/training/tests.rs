use super::TrainingRecord;
use crate::{Adam, AdamConfig, GradientsAccumulator, GradientsParams, Optimizer, adaptor::OptimizerAdaptor,
    SgdConfig, TestAutodiffBackend as B, lr_scheduler::{LrScheduler, step::{StepLrScheduler, StepLrSchedulerConfig}}};
use ruda_model::{module::Param, record::{BinBytesRecorder, FullPrecisionSettings}, tensor::{Tensor, TensorData, Tolerance}};
use ruda_nn::Linear;

type Model = Linear<B>;
type AdamOptimizer = OptimizerAdaptor<Adam, Model, B>;
type Snapshot = TrainingRecord<B, Model, AdamOptimizer, StepLrScheduler, (usize, usize)>;

fn model() -> Model {
    let device = Default::default();
    Linear {
        weight: Param::from_tensor(Tensor::from_floats([[0.25], [-0.5]], &device)),
        bias: Some(Param::from_tensor(Tensor::from_floats([0.1], &device))),
    }
}

fn loss(model: &Model, batch: usize) -> Tensor<B, 1> {
    let device = Default::default();
    let (x, y) = if batch % 2 == 0 {
        ([[1.0, 0.0], [0.0, 1.0]], [[2.5], [-2.5]])
    } else {
        ([[1.0, 1.0], [-1.0, 2.0]], [[-0.5], [-7.5]])
    };
    let difference = model.forward::<2>(Tensor::from_floats(x, &device)) - Tensor::from_floats(y, &device);
    difference.clone().mul(difference).mean()
}

fn accumulate(model: &Model, accumulator: &mut GradientsAccumulator<Model>, batch: usize) {
    accumulator.accumulate(model, GradientsParams::from_grads(loss(model, batch).backward(), model));
}

#[test]
fn linear_mse_sgd_matches_analytic_update() {
    let device = Default::default();
    let model = Linear::<B> { weight: Param::from_tensor(Tensor::zeros([2, 1], &device)), bias: None };
    let x = Tensor::<B, 2>::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    let y = Tensor::from_floats([[2.0], [-3.0]], &device);
    let residual = model.forward(x) - y;
    let loss = residual.clone().mul(residual).mean();
    assert!((loss.clone().into_scalar() - 6.5).abs() < 1e-6);
    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    let mut optimizer = SgdConfig::new().init();
    let model = optimizer.step(0.1, model, gradients);
    model.weight.val().to_data().assert_approx_eq::<f32>(&TensorData::from([[0.2], [-0.3]]), Tolerance::absolute(1e-6));
}

#[test]
fn adam_resume_preserves_pending_gradients_and_next_updates() {
    let device = Default::default();
    let mut model = model();
    let initial_loss = loss(&model, 0).into_scalar() + loss(&model, 1).into_scalar();
    let config = AdamConfig::new().with_amsgrad(true);
    let schedule = StepLrSchedulerConfig::new(0.05, 2).with_gamma(0.8);
    let mut optimizer: AdamOptimizer = config.init();
    let mut scheduler = schedule.init().unwrap();
    let mut accumulator = GradientsAccumulator::new();
    for step in 0..3 {
        accumulate(&model, &mut accumulator, step * 2);
        accumulate(&model, &mut accumulator, step * 2 + 1);
        model = optimizer.step(scheduler.step(), model, accumulator.grads());
    }
    accumulate(&model, &mut accumulator, 6);
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = Snapshot::capture(&model, &optimizer, &scheduler, &accumulator, (3, 1))
        .unwrap().save(&recorder, ()).unwrap();
    let snapshot = Snapshot::load(&recorder, bytes, &device).unwrap();
    let mut restored = snapshot.restore(self::model(), config.init(), schedule.init().unwrap(), &device).unwrap();
    assert_eq!(restored.state, (3, 1));
    assert_eq!(model.weight.id, restored.model.weight.id);
    for step in 3..8 {
        if step != 3 {
            accumulate(&model, &mut accumulator, step * 2);
            accumulate(&restored.model, &mut restored.accumulator, step * 2);
        }
        accumulate(&model, &mut accumulator, step * 2 + 1);
        accumulate(&restored.model, &mut restored.accumulator, step * 2 + 1);
        let lr = scheduler.step();
        assert_eq!(lr, restored.scheduler.step());
        model = optimizer.step(lr, model, accumulator.grads());
        restored.model = restored.optimizer.step(lr, restored.model, restored.accumulator.grads());
        model.weight.val().to_data().assert_eq(&restored.model.weight.val().to_data(), false);
        model.bias.as_ref().unwrap().val().to_data()
            .assert_eq(&restored.model.bias.as_ref().unwrap().val().to_data(), false);
        loss(&model, 0).to_data().assert_eq(&loss(&restored.model, 0).to_data(), false);
    }
    let final_loss = loss(&model, 0).into_scalar() + loss(&model, 1).into_scalar();
    assert!(final_loss < initial_loss, "loss did not decrease: {initial_loss} -> {final_loss}");
}
