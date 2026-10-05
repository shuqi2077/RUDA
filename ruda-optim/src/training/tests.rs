use super::TrainingRecord;
use crate::{Adam, AdamConfig, GradientsAccumulator, GradientsParams, Optimizer, adaptor::OptimizerAdaptor,
    SgdConfig, TestAutodiffBackend as B, lr_scheduler::{LrScheduler, step::{StepLrScheduler, StepLrSchedulerConfig}}};
use ruda_model::{module::Param, record::{BinBytesRecorder, FullPrecisionSettings}, tensor::{Tensor, TensorData, Tolerance}};
use ruda_nn::Linear;

type Model = Linear<B>;
type AdamOptimizer = OptimizerAdaptor<Adam, Model, B>;
type Snapshot = TrainingRecord<B, Model, AdamOptimizer, StepLrScheduler, (usize, usize)>;

type MasterOptimizer = OptimizerAdaptor<crate::Fp32MasterOptimizer<crate::AdamW>, Model, B>;
type MixedSnapshot = TrainingRecord<B, Model, MasterOptimizer, StepLrScheduler, (usize, usize)>;
type StoredMixedSnapshot = TrainingRecord<B, Model, MasterOptimizer, StepLrScheduler,
    (ruda_model::module::ModuleDTypeRecord, (usize, usize))>;

fn mixed_model(dtype: ruda_model::tensor::FloatDType) -> Model {
    let device = Default::default();
    Linear {
        weight: Param::from_tensor(Tensor::<B, 2>::from_floats([[0.25], [-0.5]], &device).cast(dtype)),
        bias: Some(Param::from_tensor(Tensor::<B, 1>::from_floats([0.1], &device))),
    }
}

fn accumulate_mixed(model: &Model, accumulator: &mut GradientsAccumulator<Model>, batch: usize) {
    use ruda_model::tensor::{DType, FloatDType};
    let device = Default::default();
    let input = if batch % 2 == 0 { [[1., 0.], [0., 1.]] } else { [[1., 1.], [-1., 2.]] };
    let input = Tensor::<B, 2>::from_floats(input, &device).cast(model.weight.val().dtype());
    let prediction = input.matmul(model.weight.val()).cast(DType::F32)
        + model.bias.as_ref().unwrap().val().unsqueeze::<2>();
    let loss = prediction.square().mean() * 8.;
    accumulator.accumulate_with_dtype(model,
        GradientsParams::from_grads(loss.backward(), model), FloatDType::F32);
}

fn assert_fp32_gradients<const D: usize>(expected: &GradientsParams, actual: &GradientsParams,
    id: ruda_model::module::ParamId) {
    let expected = expected.get::<crate::TestBackend, D>(id).unwrap();
    let actual = actual.get::<crate::TestBackend, D>(id).unwrap();
    assert_eq!(expected.dtype(), ruda_model::tensor::DType::F32);
    assert_eq!(actual.dtype(), ruda_model::tensor::DType::F32);
    expected.to_data().assert_eq(&actual.to_data(), false);
}

#[test]
fn mixed_storage_resume_preserves_fp32_pending_gradients_and_master_updates() {
    use crate::{AdamWConfig, Fp32MasterOptimizer};
    use ruda_model::tensor::{DType, FloatDType};
    let device = Default::default();
    for dtype in [FloatDType::F16, FloatDType::BF16] {
        let config = AdamWConfig::new().with_amsgrad(true);
        let schedule = StepLrSchedulerConfig::new(0.001, 2).with_gamma(0.8);
        let mut model = mixed_model(dtype);
        let mut optimizer: MasterOptimizer = Fp32MasterOptimizer::new(config.build())
            .with_gradient_scale(16.).init();
        let mut scheduler = schedule.init().unwrap();
        let mut accumulator = GradientsAccumulator::new();
        for step in 0..3 {
            accumulate_mixed(&model, &mut accumulator, step * 2);
            accumulate_mixed(&model, &mut accumulator, step * 2 + 1);
            model = optimizer.step(scheduler.step(), model, accumulator.grads());
        }
        accumulate_mixed(&model, &mut accumulator, 6);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = MixedSnapshot::capture_with_dtypes(
            &model, &optimizer, &scheduler, &accumulator, (3, 1))
            .unwrap().save(&recorder, ()).unwrap();
        let snapshot = StoredMixedSnapshot::load(&recorder, bytes, &device).unwrap();
        let mut restored = snapshot.restore_with_dtypes(self::model(),
            Fp32MasterOptimizer::new(config.build()).with_gradient_scale(16.).init(),
            schedule.init().unwrap(), &device).unwrap();
        assert_eq!(restored.state, (3, 1));
        assert_eq!(restored.model.weight.val().dtype(), dtype.into());
        assert_eq!(restored.model.bias.as_ref().unwrap().val().dtype(), DType::F32);
        assert_eq!(restored.model.weight.id, model.weight.id);
        assert!(restored.model.weight.val().is_require_grad());
        for step in 3..6 {
            if step != 3 {
                accumulate_mixed(&model, &mut accumulator, step * 2);
                accumulate_mixed(&restored.model, &mut restored.accumulator, step * 2);
            }
            accumulate_mixed(&model, &mut accumulator, step * 2 + 1);
            accumulate_mixed(&restored.model, &mut restored.accumulator, step * 2 + 1);
            let lr = scheduler.step();
            assert_eq!(lr, restored.scheduler.step());
            let gradients = accumulator.grads();
            let restored_gradients = restored.accumulator.grads();
            assert_fp32_gradients::<2>(&gradients, &restored_gradients, model.weight.id);
            assert_fp32_gradients::<1>(&gradients, &restored_gradients,
                model.bias.as_ref().unwrap().id);
            model = optimizer.step(lr, model, gradients);
            restored.model = restored.optimizer.step(lr, restored.model, restored_gradients);
            assert_eq!(restored.model.weight.val().dtype(), dtype.into());
            model.weight.val().to_data().assert_eq(&restored.model.weight.val().to_data(), false);
            model.bias.as_ref().unwrap().val().to_data().assert_eq(
                &restored.model.bias.as_ref().unwrap().val().to_data(), false);
        }
    }
}

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
