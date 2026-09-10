use super::*;
use crate::{TestBackend, TestAutodiffBackend};
use crate::{GradientsParams, Optimizer};
use ruda_model::module::{Module, Param};
use ruda_model::tensor::{Distribution, Tensor, TensorData};
use ruda_nn::{Linear, LinearConfig, LinearRecord};


const TOLERANCE: f64 = 1e-8;

fn given_linear_layer_no_bias(weight: TensorData) -> Linear<TestAutodiffBackend> {
    let device = Default::default();
    let record = LinearRecord {
        weight: Param::from_data(weight, &device),
        bias: None, //No bias for Muon optimizer
    };

    LinearConfig::new(4, 4)
        .with_bias(false)
        .init(&device)
        .load_record(record)
}

#[test]
fn test_adjust_lr_fn_original() {
    let method = AdjustLrFn::Original;

    // Square matrix [512, 512] -> sqrt(1) = 1.0
    let ratio = method.adjustment_ratio(&[512, 512]);
    assert!((ratio - 1.0).abs() < TOLERANCE);

    // Tall matrix [1024, 512] -> sqrt(2) ≈ 1.414
    let ratio = method.adjustment_ratio(&[1024, 512]);
    let expected = (2.0f64).sqrt();
    assert!((ratio - expected).abs() < TOLERANCE);

    // Wide matrix [512, 1024] -> max(1, 0.5) = 1.0
    let ratio = method.adjustment_ratio(&[512, 1024]);
    assert!((ratio - 1.0).abs() < TOLERANCE);
}

#[test]
fn test_adjust_lr_fn_match_rms_adamw() {
    let method = AdjustLrFn::MatchRmsAdamW;

    // [1024, 512] -> 0.2 * sqrt(1024) = 6.4
    let ratio = method.adjustment_ratio(&[1024, 512]);
    let expected = 0.2 * 1024.0f64.sqrt();
    assert!((ratio - expected).abs() < TOLERANCE);

    // [512, 512] -> 0.2 * sqrt(512) ≈ 4.525
    let ratio = method.adjustment_ratio(&[512, 512]);
    let expected = 0.2 * 512.0f64.sqrt();
    assert!((ratio - expected).abs() < TOLERANCE);
}

#[test]
#[should_panic(expected = "Newton-Schulz iteration requires 2D tensors, got 1D")]
fn test_1d_tensor_panics() {
    let device = Default::default();
    let config = MuonConfig::new();
    let optim: Muon<TestBackend> = config.build();

    let tensor_1d = Tensor::<TestBackend, 1>::zeros([512], &device);
    let grad_1d = Tensor::<TestBackend, 1>::ones([512], &device);

    let _ = optim.step(0.01, tensor_1d, grad_1d, None);
}

#[test]
fn test_muon_optimizer_save_load_state() {
    let device = Default::default();
    // Use Linear layer WITHOUT bias for Muon optimizer
    let linear = LinearConfig::new(6, 6)
        .with_bias(false) // No bias - only 2D weight matrix
        .init::<TestAutodiffBackend>(&device);

    let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);

    let mut optimizer =
        MuonConfig::new().init::<TestAutodiffBackend, Linear<TestAutodiffBackend>>();
    let grads = linear.forward(x).backward();
    let grads = GradientsParams::from_grads(grads, &linear);
    let _linear = optimizer.step(0.01, linear, grads);

    let state_before = optimizer.to_record();
    let state_before_copy = optimizer.to_record();

    let optimizer_new =
        MuonConfig::new().init::<TestAutodiffBackend, Linear<TestAutodiffBackend>>();
    let optimizer_loaded = optimizer_new.load_record(state_before_copy);
    let state_after = optimizer_loaded.to_record();

    assert_eq!(state_before.len(), state_after.len());
}

#[test]
fn test_muon_with_weight_decay() {
    let device = Default::default();
    // Create Linear layer WITHOUT bias for Muon
    let linear = given_linear_layer_no_bias(TensorData::from([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]));

    let x = Tensor::<TestAutodiffBackend, 2>::from_floats(
        [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        &device,
    )
    .require_grad();

    let mut optimizer = MuonConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(0.01)))
        .init::<TestAutodiffBackend, Linear<TestAutodiffBackend>>();

    let grads = linear.forward(x).backward();
    let grads = GradientsParams::from_grads(grads, &linear);
    let linear = optimizer.step(0.01, linear, grads);

    let state = linear.into_record();
    let weight = state.weight.to_data();

    for val in weight.as_slice::<f32>().unwrap() {
        assert!(
            *val < 1.0,
            "Weight should be reduced by weight decay, got {}",
            val
        );
    }
}

#[test]
fn test_newton_schulz_matches_polynomial_not_exact_identity() {
    let device = Default::default();
    let matrix = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.5], [0.5, 1.0]], &device);
    let muon: Muon<TestBackend> = MuonConfig::new().build();
    let out = muon.zeropower_via_newtonschulz(matrix);
    let expected = scalar_reference::orthogonalize(&[1.0, 0.5, 0.5, 1.0], 2, 2, 5, 1e-7, false);
    assert_close(&values(out), &expected, 3e-5);
    // Five quintic iterations do not promise an exact orthogonal factor.
}

#[test]
fn test_tall_matrix_transpose() {
    // Test that tall matrices (A > B) are transposed during Newton-Schulz iteration
    // and then transposed back
    let device = Default::default();

    // Create a tall matrix: [8, 4] (more rows than columns)
    let tall_matrix = Tensor::<TestBackend, 2>::from_floats(
        [
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.5],
            [0.2, 0.1, 0.5, 1.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.2, 0.4, 0.1, 0.3],
            [0.3, 0.1, 0.4, 0.2],
        ],
        &device,
    );

    let config = MuonConfig::new();
    let muon: Muon<TestBackend> = config.build();

    // Perform Newton-Schulz orthogonalization
    let orthogonalized = muon.zeropower_via_newtonschulz(tall_matrix.clone());

    // Verify shape is preserved (should be transposed internally but returned in original shape)
    let original_shape = tall_matrix.shape();
    let result_shape = orthogonalized.shape();
    assert_eq!(
        original_shape.dims::<2>(),
        result_shape.dims::<2>(),
        "Shape should be preserved: [8, 4]"
    );

    // Verify output is different from input (orthogonalization happened)
    let original_data = tall_matrix.into_data();
    let result_data = orthogonalized.into_data();
    assert_ne!(
        original_data.as_slice::<f32>().unwrap(),
        result_data.as_slice::<f32>().unwrap(),
        "Orthogonalized matrix should differ from input"
    );

    // For comparison, test a wide matrix [4, 8] should NOT be transposed
    let wide_matrix = Tensor::<TestBackend, 2>::from_floats(
        [
            [1.0, 0.5, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3],
            [0.5, 1.0, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.5, 0.3, 0.2, 0.1, 0.4],
            [0.2, 0.1, 0.5, 1.0, 0.4, 0.1, 0.3, 0.2],
        ],
        &device,
    );

    let orthogonalized_wide = muon.zeropower_via_newtonschulz(wide_matrix.clone());

    // Verify wide matrix shape is also preserved
    let wide_original_shape = wide_matrix.shape();
    let wide_result_shape = orthogonalized_wide.shape();
    assert_eq!(
        wide_original_shape.dims::<2>(),
        wide_result_shape.dims::<2>(),
        "Wide matrix shape should be preserved: [4, 8]"
    );
}

#[test]
fn test_zero_gradient() {
    // Test that Muon handles zero gradients gracefully
    let device = Default::default();

    let tensor = Tensor::<TestBackend, 2>::from_floats(
        [
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.5],
            [0.2, 0.1, 0.5, 1.0],
        ],
        &device,
    );

    // Zero gradient - all zeros
    let zero_grad = Tensor::<TestBackend, 2>::zeros([4, 4], &device);

    let config = MuonConfig::new();
    let muon: Muon<TestBackend> = config.build();

    // Should not panic or produce NaN
    let (updated_tensor, state) = muon.step(0.01, tensor.clone(), zero_grad, None);

    // Verify state was created
    assert!(state.is_some());

    // With zero gradient and no weight decay, tensor should remain unchanged
    let original_data = tensor.into_data();
    let updated_data = updated_tensor.clone().into_data();

    let original_vals = original_data.as_slice::<f32>().unwrap();
    let updated_vals = updated_data.as_slice::<f32>().unwrap();

    for (orig, upd) in original_vals.iter().zip(updated_vals.iter()) {
        assert!(
            (orig - upd).abs() < 1e-6,
            "With zero gradient, tensor should remain unchanged (or very close)"
        );
    }

    // Verify no NaN values
    for val in updated_vals {
        assert!(
            !val.is_nan(),
            "Result should not contain NaN values with zero gradient"
        );
    }

    // Test with weight decay - should still work
    let muon_with_decay: Muon<TestBackend> = MuonConfig::new().with_weight_decay(Some(WeightDecayConfig::new(0.01))).build();

    let tensor2 = Tensor::<TestBackend, 2>::from_floats(
        [
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.5],
            [0.2, 0.1, 0.5, 1.0],
        ],
        &device,
    );
    let zero_grad2 = Tensor::<TestBackend, 2>::zeros([4, 4], &device);

    let (updated_tensor_decay, _) =
        muon_with_decay.step(0.01, tensor2.clone(), zero_grad2, None);

    // With zero gradient but with weight decay, tensor should be slightly reduced
    let updated_decay_data = updated_tensor_decay.into_data();
    let updated_decay_vals = updated_decay_data.as_slice::<f32>().unwrap();

    for val in updated_decay_vals {
        assert!(
            !val.is_nan(),
            "Result should not contain NaN with zero gradient and weight decay"
        );
    }

    // With weight decay, values should be slightly smaller than original
    let original_vals2 = tensor2.into_data().as_slice::<f32>().unwrap().to_vec();
    for (orig, upd) in original_vals2.iter().zip(updated_decay_vals.iter()) {
        if orig.abs() > 1e-6 {
            // Non-zero values should be reduced by weight decay
            assert!(
                upd.abs() < orig.abs(),
                "Weight decay should reduce magnitude: original={}, updated={}",
                orig,
                upd
            );
        }
    }
}

#[path = "test_reference.rs"]
mod scalar_reference;

fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().as_slice::<f32>().unwrap().to_vec()
}
fn assert_close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && e.is_finite());
        assert!((*a as f64 - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: {a} vs {e}");
    }
}
fn data_tensor(values: &[f64], rows: usize, cols: usize) -> Tensor<TestBackend, 2> {
    Tensor::from_data(TensorData::new(values.iter().map(|v| *v as f32).collect(), [rows, cols]), &Default::default())
}

#[test]
fn muon_numerics_multistep_against_independent_f64() {
    for (rows, cols) in [(3, 5), (5, 3), (3, 3)] {
        for ema in [false, true] {
            for nesterov in [false, true] {
                for stable in [false, true] {
                    for rms in [false, true] {
                        let config = MuonConfig::new()
                            .with_momentum(MomentumConfig::new().with_momentum(0.95).with_dampening(0.0).with_nesterov(nesterov))
                            .with_momentum_mode(if ema { MuonMomentumMode::Ema } else { MuonMomentumMode::Sgd })
                            .with_stable_normalization(stable)
                            .with_adjust_lr_fn(if rms { AdjustLrFn::MatchRmsAdamW } else { AdjustLrFn::Original })
                            .with_weight_decay(Some(WeightDecayConfig::new(0.01)));
                        let optim: Muon<TestBackend> = config.build();
                        let reference_config = scalar_reference::Settings { ema, nesterov, stable, rms, ..Default::default() };
                        let mut weights: Vec<f64> = (0..rows*cols).map(|i| (i as f64 - 6.0) * 0.03).collect();
                        let mut tensor = data_tensor(&weights, rows, cols);
                        let mut state = None;
                        let mut reference_state = None;
                        for step in 0..4 {
                            let grad: Vec<f64> = (0..rows*cols).map(|i| (((i*7 + step*3)%17) as f64 - 8.0)*0.125).collect();
                            let expected = scalar_reference::step(&weights, &grad, reference_state.as_deref(), rows, cols, &reference_config);
                            let (next, next_state) = optim.try_step(0.02, tensor, data_tensor(&grad, rows, cols), state).unwrap();
                            assert_close(&values(next.clone()), &expected.0, 2e-4);
                            assert_close(&values(next_state.as_ref().unwrap().momentum.velocity().clone()), &expected.1, 2e-5);
                            tensor = next; state = next_state;
                            weights = expected.0; reference_state = Some(expected.1);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn muon_invalid_config_is_rejected_before_build() {
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(MuonConfig::new().with_epsilon(bad).validate().is_err());
    }
    for bad in [0, 100, usize::MAX] {
        assert!(MuonConfig::new().with_ns_steps(bad).validate().is_err());
    }
    for bad in [-1.0, 1.0, f64::NAN, f64::INFINITY] {
        assert!(MuonConfig::new().with_momentum(MomentumConfig::new().with_momentum(bad)).validate().is_err());
    }
    assert!(MuonConfig::new().with_ns_coefficients((f32::NAN, 1.0, 1.0)).validate().is_err());
    assert!(MuonConfig::new().with_weight_decay(Some(WeightDecayConfig::new(-0.1))).validate().is_err());
}

#[test]
fn muon_ema_rejects_dampening_and_invalid_nesterov() {
    assert!(MuonConfig::new().with_momentum_mode(MuonMomentumMode::Ema)
        .with_momentum(MomentumConfig::new().with_dampening(0.1)).validate().is_err());
    assert!(MuonConfig::new().with_momentum(MomentumConfig::new()
        .with_momentum(0.0).with_dampening(0.0).with_nesterov(true)).validate().is_err());
    assert!(MuonConfig::new().with_momentum(MomentumConfig::new()
        .with_momentum(0.0).with_dampening(0.0).with_nesterov(false)).validate().is_ok());
}

#[test]
fn muon_shape_broadcast_is_rejected() {
    let device = Default::default();
    let optim: Muon<TestBackend> = MuonConfig::new().build();
    assert!(matches!(optim.try_step::<2>(0.02, Tensor::ones([2, 3], &device), Tensor::ones([1, 3], &device), None),
        Err(MuonError::ShapeMismatch("gradient"))));
}

#[test]
fn muon_bad_state_shape_is_rejected() {
    let device = Default::default();
    let optim: Muon<TestBackend> = MuonConfig::new().build();
    let bad = MuonState::new(MomentumState::new(Tensor::zeros([2, 4], &device)));
    assert!(matches!(optim.try_step::<2>(0.02, Tensor::ones([2, 3], &device), Tensor::ones([2, 3], &device), Some(bad)),
        Err(MuonError::ShapeMismatch("momentum"))));
}

#[test]
fn muon_nonmatrix_returns_error_without_indexing() {
    let device = Default::default();
    let optim: Muon<TestBackend> = MuonConfig::new().build();
    assert!(matches!(optim.try_step::<1>(0.02, Tensor::ones([3], &device), Tensor::ones([3], &device), None),
        Err(MuonError::ExpectedMatrix { rank: 1 })));
}

#[test]
fn muon_negative_or_nonfinite_lr_is_rejected() {
    let device = Default::default();
    let optim: Muon<TestBackend> = MuonConfig::new().build();
    for lr in [-0.01, f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(optim.try_step::<2>(lr, Tensor::ones([2, 3], &device), Tensor::ones([2, 3], &device), None).is_err());
    }
}

#[test]
fn muon_zero_lr_does_not_change_weights_but_updates_momentum() {
    let w = data_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let optim: Muon<TestBackend> = MuonConfig::new().with_momentum_mode(MuonMomentumMode::Ema).build();
    let (out, state) = optim.try_step(0.0, w, data_tensor(&[1.0; 4], 2, 2), None).unwrap();
    assert_close(&values(out), &[1.0, 2.0, 3.0, 4.0], 0.0);
    assert_close(&values(state.unwrap().momentum.velocity().clone()), &[0.05; 4], 1e-6);
}

#[test]
fn muon_stable_normalization_extreme_finite_and_zero() {
    let optim: Muon<TestBackend> = MuonConfig::new().with_stable_normalization(true).build();
    for scale in [0.0, 1e-30, 1e30] {
        let values_in = [scale, -scale*0.5, scale*0.25, scale*0.75];
        let out = optim.zeropower_via_newtonschulz(data_tensor(&values_in, 2, 2));
        let reference = scalar_reference::orthogonalize(&values_in, 2, 2, 5, 1e-7, true);
        assert_close(&values(out), &reference, 5e-5);
    }
}

#[test]
fn muon_orientation_changes_original_scale_not_decay() {
    let normal: Muon<TestBackend> = MuonConfig::new().build();
    let io: Muon<TestBackend> = MuonConfig::new().with_matrix_layout(MuonMatrixLayout::InputOutput).build();
    assert!((normal.adjust_lr(0.02, &[2, 8]) - 0.02).abs() < 1e-12);
    assert!((io.adjust_lr(0.02, &[2, 8]) - 0.04).abs() < 1e-12);
    let optim: Muon<TestBackend> = MuonConfig::new().with_matrix_layout(MuonMatrixLayout::InputOutput)
        .with_weight_decay(Some(WeightDecayConfig::new(0.1))).build();
    let device = Default::default();
    let (out, _) = optim.try_step::<2>(0.02, Tensor::ones([2, 8], &device), Tensor::zeros([2, 8], &device), None).unwrap();
    assert_close(&values(out), &[0.998; 16], 1e-6);
}

#[test]
fn muon_legacy_default_is_preserved() {
    let config = MuonConfig::new();
    assert_eq!(config.momentum_mode, MuonMomentumMode::Sgd);
    assert!(!config.stable_normalization);
    assert_eq!(config.matrix_layout, MuonMatrixLayout::AsStored);
    let optim: Muon<TestBackend> = config.build();
    let (out, state) = optim.try_step(0.02, data_tensor(&[1.0; 4], 2, 2), data_tensor(&[0.25; 4], 2, 2), None).unwrap();
    assert!(values(out).iter().all(|v| v.is_finite()));
    assert_close(&values(state.unwrap().momentum.velocity().clone()), &[0.25; 4], 0.0);
}

#[test]
fn muon_state_roundtrip_continuation_matches() {
    use ruda_model::record::FullPrecisionSettings;
    let optim: Muon<TestBackend> = MuonConfig::new().with_momentum_mode(MuonMomentumMode::Ema).build();
    let (w, state) = optim.try_step(0.02, data_tensor(&[1.0; 6], 2, 3), data_tensor(&[0.25; 6], 2, 3), None).unwrap();
    let state = state.unwrap();
    let item = state.clone().into_item::<FullPrecisionSettings>();
    let loaded = MuonState::<TestBackend, 2>::from_item::<FullPrecisionSettings>(item, &Default::default());
    let grad = data_tensor(&[-0.125, 0.25, 0.5, 0.25, 0.75, -0.5], 2, 3);
    let (a, _) = optim.try_step(0.02, w.clone(), grad.clone(), Some(state)).unwrap();
    let (b, _) = optim.try_step(0.02, w, grad, Some(loaded)).unwrap();
    assert_eq!(values(a), values(b));
}
