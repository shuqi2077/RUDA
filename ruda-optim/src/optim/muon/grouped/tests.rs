// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::{MuonMatrixLayout, MuonMomentumMode, TestAutodiffBackend, TestBackend};
use ruda_model::{module::Module, record::FullPrecisionSettings, tensor::backend::Backend};

#[derive(Module, Debug)]
struct SmallModel<B: Backend> {
    hidden: Param<Tensor<B, 2>>,
    head: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}
fn model() -> SmallModel<TestAutodiffBackend> {
    let device = Default::default();
    SmallModel {
        hidden: Param::initialized(101u64.into(), Tensor::from_data([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], &device).require_grad()),
        head: Param::initialized(102u64.into(), Tensor::from_data([[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]], &device).require_grad()),
        bias: Param::initialized(103u64.into(), Tensor::from_data([0.1, -0.1], &device).require_grad()),
    }
}
fn config() -> MuonAdamWConfig {
    MuonAdamWConfig::new().with_muon(MuonConfig::new()
        .with_momentum_mode(MuonMomentumMode::Ema).with_stable_normalization(true)
        .with_matrix_layout(MuonMatrixLayout::InputOutput))
}
fn gradients(step: usize) -> GradientsParams {
    let device = Default::default();
    let mut g = GradientsParams::new();
    g.register::<TestBackend, 2>(101u64.into(), Tensor::from_data([[0.1, 0.3, -0.2], [0.7, 0.1, -0.1]], &device).mul_scalar(1.0+step as f32*0.1));
    g.register::<TestBackend, 2>(102u64.into(), Tensor::from_data([[0.25, -0.125], [0.5, 0.125], [0.75, -0.5]], &device));
    g.register::<TestBackend, 1>(103u64.into(), Tensor::from_data([0.3, -0.4], &device));
    g
}
fn same_model(a: &SmallModel<TestAutodiffBackend>, b: &SmallModel<TestAutodiffBackend>) {
    assert_eq!(a.hidden.val().to_data().to_vec::<f32>().unwrap(), b.hidden.val().to_data().to_vec::<f32>().unwrap());
    assert_eq!(a.head.val().to_data().to_vec::<f32>().unwrap(), b.head.val().to_data().to_vec::<f32>().unwrap());
    assert_eq!(a.bias.val().to_data().to_vec::<f32>().unwrap(), b.bias.val().to_data().to_vec::<f32>().unwrap());
}

#[test]
fn muon_groups_route_hidden_only_and_preserve_parameter_ids() {
    let mut mixed_model = model();
    let mut expected = mixed_model.clone();
    let config = config();
    let mut mixed = config.init(&mixed_model, &[mixed_model.hidden.id]).unwrap();
    let mut only_muon = config.muon.init();
    let mut only_adam = config.adamw.init();
    for step in 0..4 {
        let mut grads = gradients(step);
        let mut selected = GradientsParams::new();
        selected.register::<TestBackend, 2>(expected.hidden.id, grads.remove(expected.hidden.id).unwrap());
        expected = only_muon.step(0.02, expected, selected);
        expected = only_adam.step(0.0003, expected, grads);
        mixed_model = mixed.try_step_with_lrs(0.02, 0.0003, mixed_model, gradients(step)).unwrap();
        same_model(&mixed_model, &expected);
        assert_eq!(mixed_model.hidden.id.val(), 101);
        assert!(mixed_model.hidden.is_require_grad());
    }
    let record = mixed.to_record();
    assert_eq!(record.muon_state_count(), 1);
    assert_eq!(record.adamw_state_count(), 2); // Includes the 2D output head.
}

#[test]
fn muon_groups_missing_gradient_skips_decay_and_state() {
    let original = model();
    let mut optim = config().init(&original, &[original.hidden.id]).unwrap();
    let mut grads = gradients(0);
    grads.remove::<TestBackend, 2>(original.head.id);
    let next = optim.try_step_with_lrs(0.02, 0.0003, original.clone(), grads).unwrap();
    assert_eq!(next.head.val().to_data().to_vec::<f32>().unwrap(), original.head.val().to_data().to_vec::<f32>().unwrap());
    assert_eq!(optim.to_record().adamw_state_count(), 1);
}

#[test]
fn muon_groups_explicit_skip_preserves_model_and_all_state() {
    let original = model();
    let mut optim = config().init(&original, &[original.hidden.id]).unwrap();
    let next = optim.try_step_or_skip(0.02, 0.0003, original.clone(), gradients(0), true).unwrap();
    same_model(&original, &next);
    assert_eq!(optim.to_record().muon_state_count(), 0);
    assert_eq!(optim.to_record().adamw_state_count(), 0);
}

#[test]
fn muon_groups_reject_bias_unknown_duplicate_and_empty_selection() {
    let m = model();
    assert!(matches!(config().init(&m, &[m.bias.id]), Err(MuonError::ExpectedMatrix { rank: 1 })));
    assert!(matches!(config().init(&m, &[999u64.into()]), Err(MuonError::UnknownParameter(999))));
    assert!(matches!(config().init(&m, &[m.hidden.id, m.hidden.id]), Err(MuonError::DuplicateParameter(101))));
    assert!(matches!(config().init(&m, &[]), Err(MuonError::EmptyMuonGroup)));
}

#[test]
fn muon_groups_reject_frozen_hidden_matrix() {
    let mut m = model();
    m.hidden = m.hidden.set_require_grad(false);
    assert!(matches!(config().init(&m, &[m.hidden.id]), Err(MuonError::FrozenParameter(101))));
}

#[test]
fn muon_groups_validate_both_groups_before_updating() {
    let m = model();
    let mut optim = config().init(&m, &[m.hidden.id]).unwrap();
    let mut g = gradients(0);
    g.register::<TestBackend, 1>(m.bias.id, Tensor::ones([1], &Default::default()));
    assert!(matches!(optim.try_step_with_lrs(0.02, 0.0003, m, g), Err(MuonError::ShapeMismatch("gradient"))));
    assert_eq!(optim.to_record().muon_state_count(), 0);
    assert_eq!(optim.to_record().adamw_state_count(), 0);
}

#[test]
fn muon_groups_reject_unknown_gradient_and_changed_model() {
    let mut m = model();
    let mut optim = config().init(&m, &[m.hidden.id]).unwrap();
    let mut g = gradients(0);
    g.register::<TestBackend, 1>(999u64.into(), Tensor::ones([1], &Default::default()));
    assert!(matches!(optim.try_step_with_lrs(0.02, 0.0003, m.clone(), g), Err(MuonError::UnusedGradients)));
    m.head = Param::initialized(m.head.id, Tensor::ones([4, 2], &Default::default()).require_grad());
    assert!(matches!(optim.try_step_with_lrs(0.02, 0.0003, m, GradientsParams::new()), Err(MuonError::ModelChanged)));
}

#[test]
fn muon_groups_record_roundtrip_continues_exactly() {
    let mut m = model();
    let mut optim = config().init(&m, &[m.hidden.id]).unwrap();
    m = optim.step(0.02, m, gradients(0));
    let item = optim.to_record().into_item::<FullPrecisionSettings>();
    let restored = MuonAdamWRecord::<TestAutodiffBackend>::from_item::<FullPrecisionSettings>(item, &Default::default());
    let mut other = config().init(&m, &[m.hidden.id]).unwrap().try_load_record(restored).unwrap();
    let a = optim.step(0.02, m.clone(), gradients(1));
    let b = other.step(0.02, m, gradients(1));
    same_model(&a, &b);
}

#[test]
fn muon_groups_reject_record_role_and_config_changes() {
    let m = model();
    let mut optim = config().init(&m, &[m.hidden.id]).unwrap();
    let _ = optim.step(0.02, m.clone(), gradients(0));
    assert!(matches!(config().init(&m, &[m.head.id]).unwrap().try_load_record(optim.to_record()), Err(MuonError::IncompatibleRecord)));
    assert!(matches!(config().with_adamw_lr_ratio(0.1).init(&m, &[m.hidden.id]).unwrap().try_load_record(optim.to_record()), Err(MuonError::IncompatibleRecord)));
    let altered = config().with_muon(MuonConfig::new());
    assert!(matches!(altered.init(&m, &[m.hidden.id]).unwrap().try_load_record(optim.to_record()), Err(MuonError::IncompatibleRecord)));
}

#[test]
fn muon_groups_invalid_auxiliary_config_and_rates() {
    let m = model();
    assert!(config().with_adamw(AdamWConfig::new().with_beta_1(f32::NAN)).init(&m, &[m.hidden.id]).is_err());
    assert!(config().with_adamw_lr_ratio(-1.0).init(&m, &[m.hidden.id]).is_err());
    let mut optim = config().init(&m, &[m.hidden.id]).unwrap();
    assert!(optim.try_step_with_lrs(0.02, f64::NAN, m, gradients(0)).is_err());
    assert_eq!(optim.to_record().muon_state_count(), 0);
}

#[derive(Module, Debug)]
struct TiedModel<B: Backend> { left: Param<Tensor<B, 2>>, right: Param<Tensor<B, 2>> }
#[test]
fn muon_groups_tied_parameter_updated_once() {
    let m = model();
    let tied = TiedModel { left: m.hidden.clone(), right: m.hidden.clone() };
    let mut optim = config().init(&tied, &[tied.left.id]).unwrap();
    let mut g = GradientsParams::new();
    let grad = Tensor::<TestBackend, 2>::ones([2, 3], &Default::default());
    g.register(tied.left.id, grad.clone());
    let (expected, _) = config().muon.build::<TestBackend>().try_step(0.02, tied.left.val().inner(), grad, None).unwrap();
    let next = optim.step(0.02, tied, g);
    assert_eq!(next.left.val().to_data().to_vec::<f32>().unwrap(), expected.to_data().to_vec::<f32>().unwrap());
    assert_eq!(next.left.val().to_data().to_vec::<f32>().unwrap(), next.right.val().to_data().to_vec::<f32>().unwrap());
    assert_eq!(optim.to_record().muon_state_count(), 1);
}

#[test]
#[should_panic(expected = "complete synchronized matrices")]
fn muon_groups_refuse_implicit_multi_device_shards() {
    let m = model();
    let mut optim = config().init(&m, &[m.hidden.id]).unwrap();
    let _ = optim.step_multi(0.02, m, MultiGradientsParams::default());
}
