use super::*;
use crate::TestAutodiffBackend;
use crate::{GradientsParams, Optimizer};
use ruda_model::module::{Module, Param};
use ruda_model::tensor::{Distribution, Tensor, TensorData};
use ruda_nn::{Linear, LinearConfig, LinearRecord};

type TestBackend = ruda_tensor_host::Host;

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
    let optim: Muon<TestBackend> = Muon {
        momentum: Momentum::new(&config.momentum),
        ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
        weight_decay_penalty: None,
        epsilon: config.epsilon,
        adjust_lr_fn: config.adjust_lr_fn,
    };

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
fn test_newton_schulz_orthogonalization() {
    let device = Default::default();
    let matrix = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.5], [0.5, 1.0]], &device);

    let config = MuonConfig::new();
    let muon: Muon<TestBackend> = Muon {
        momentum: Momentum::new(&config.momentum),
        ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
        weight_decay_penalty: None,
        epsilon: config.epsilon,
        adjust_lr_fn: config.adjust_lr_fn,
    };

    let orthogonalized = muon.zeropower_via_newtonschulz(matrix);
    let o_t = orthogonalized.clone().transpose();
    let product = orthogonalized.matmul(o_t);

    let data = product.into_data();
    let values = data.as_slice::<f32>().unwrap();

    assert!(
        (values[0] - 1.0).abs() < 0.1,
        "Product[0,0] should be ~1.0, got {}",
        values[0]
    );
    assert!(
        (values[3] - 1.0).abs() < 0.1,
        "Product[1,1] should be ~1.0, got {}",
        values[3]
    );
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
    let muon: Muon<TestBackend> = Muon {
        momentum: Momentum::new(&config.momentum),
        ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
        weight_decay_penalty: None,
        epsilon: config.epsilon,
        adjust_lr_fn: config.adjust_lr_fn,
    };

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
    let muon: Muon<TestBackend> = Muon {
        momentum: Momentum::new(&config.momentum),
        ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
        weight_decay_penalty: None,
        epsilon: config.epsilon,
        adjust_lr_fn: config.adjust_lr_fn,
    };

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
    let muon_with_decay: Muon<TestBackend> = Muon {
        momentum: Momentum::new(&config.momentum),
        ns_params: NewtonSchulzParams::new(config.ns_coefficients, config.ns_steps),
        weight_decay_penalty: Some(0.01),
        epsilon: config.epsilon,
        adjust_lr_fn: config.adjust_lr_fn,
    };

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
