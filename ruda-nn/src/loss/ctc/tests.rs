use super::*;
use ruda_tensor_host::{Host, HostDevice};

type TestBackend = Host;

fn assert_approx_equal(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: actual {} vs expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "Mismatch at index {}: expected {:.6}, got {:.6} (diff: {:.6})",
            i,
            e,
            a,
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------

#[test]
#[should_panic(expected = "blank index")]
fn test_ctc_loss_panics_invalid_blank_index() {
    let device = HostDevice;
    // blank=5 is out of bounds for num_classes=3
    let ctc = CTCLossConfig::new().with_blank(5).init();

    let log_probs = Tensor::<TestBackend, 3>::zeros([2, 1, 3], &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1], &device);

    ctc.forward(log_probs, targets, input_lengths, target_lengths);
}

#[test]
#[should_panic(expected = "must equal batch_size")]
fn test_ctc_loss_panics_mismatched_batch_size() {
    let device = HostDevice;
    let ctc = CTCLossConfig::new().init();

    // Logits batch size = 2
    let log_probs = Tensor::<TestBackend, 3>::zeros([2, 2, 3], &device);
    // Targets batch size = 1 (Mismatch)
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2, 2], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1, 1], &device);

    ctc.forward(log_probs, targets, input_lengths, target_lengths);
}

#[test]
#[should_panic(expected = "input_lengths length")]
fn test_ctc_loss_panics_input_lengths_mismatch() {
    let device = HostDevice;
    let ctc = CTCLossConfig::new().init();

    // Logits batch size = 2
    let log_probs = Tensor::<TestBackend, 3>::zeros([2, 2, 3], &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1], [2]], &device);

    // Input lengths size = 1 (Mismatch)
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1, 1], &device);

    ctc.forward(log_probs, targets, input_lengths, target_lengths);
}

#[test]
#[should_panic(expected = "target_lengths length")]
fn test_ctc_loss_panics_target_lengths_mismatch() {
    let device = HostDevice;
    let ctc = CTCLossConfig::new().init();

    // Logits batch size = 2
    let log_probs = Tensor::<TestBackend, 3>::zeros([2, 2, 3], &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1], [2]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2, 2], &device);

    // Target lengths size = 1 (Mismatch)
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1], &device);

    ctc.forward(log_probs, targets, input_lengths, target_lengths);
}

// ---------------------------------------------------------------
// Edge Case & Config Tests
// ---------------------------------------------------------------

#[test]
fn test_ctc_loss_repeated_labels_minimum_input_length() {
    // T=3, N=1, C=2, blank=0, target=[1, 1], uniform P = 1/2.
    //
    // The minimum T for target [1, 1] is 3: the only valid path is (1, 0, 1).
    // prob = (1/2)^3 = 1/8
    // Loss = -ln(1/8) = 3 * ln(2)
    let device = HostDevice;
    let ctc = CTCLossConfig::new().init();

    let log_probs = Tensor::<TestBackend, 3>::full([3, 1, 2], 0.5_f32.ln(), &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i32, 1]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([3_i32], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i32], &device);

    let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
    let loss_data = loss.into_data().to_vec::<f32>().unwrap();
    let expected = 3.0 * 2.0_f32.ln();
    assert_approx_equal(&loss_data, &[expected], 1e-3);
}

#[test]
fn test_ctc_loss_custom_blank_uniform() {
    // T=3, N=1, C=3, blank=2, target=[0, 1], uniform P = 1/3.
    //
    // Two distinct labels, 3 classes, 3 time steps, just with
    // blank=2 instead of 0.
    // 5 valid paths → total = 5/27
    // Loss = -ln(5/27)
    let device = HostDevice;
    let ctc = CTCLossConfig::new().with_blank(2).init();

    let log_probs = Tensor::<TestBackend, 3>::full([3, 1, 3], (1.0_f32 / 3.0).ln(), &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[0_i32, 1]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([3_i32], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i32], &device);

    let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
    let loss_data = loss.into_data().to_vec::<f32>().unwrap();
    let expected = -(5.0_f32 / 27.0).ln();
    assert_approx_equal(&loss_data, &[expected], 1e-3);
}

// ---------------------------------------------------------------
// zero_infinity tests
// ---------------------------------------------------------------

#[test]
fn test_ctc_loss_zero_infinity_produces_inf_when_disabled() {
    // T=2, N=1, C=3, blank=0, target=[1, 1], input_length=2
    // Target [1, 1] requires at least 3 time steps → no valid paths → loss = +inf
    let device = HostDevice;
    let ctc = CTCLossConfig::new().with_zero_infinity(false).init();

    let log_probs = Tensor::<TestBackend, 3>::full([2, 1, 3], (1.0_f32 / 3.0).ln(), &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i32, 1]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i32], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i32], &device);

    let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
    let loss_data = loss.into_data().to_vec::<f32>().unwrap();
    assert!(
        loss_data[0].is_infinite() && loss_data[0] > 0.0,
        "Expected +inf, got {}",
        loss_data[0]
    );
}

#[test]
fn test_ctc_loss_zero_infinity_masks_inf_when_enabled() {
    // Same inputs as above, but zero_infinity=true → loss should be 0.0
    let device = HostDevice;
    let ctc = CTCLossConfig::new().with_zero_infinity(true).init();

    let log_probs = Tensor::<TestBackend, 3>::full([2, 1, 3], (1.0_f32 / 3.0).ln(), &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i32, 1]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i32], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i32], &device);

    let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
    let loss_data = loss.into_data().to_vec::<f32>().unwrap();
    assert_approx_equal(&loss_data, &[0.0], 1e-6);
}

#[test]
fn test_ctc_loss_zero_infinity_does_not_affect_finite_loss() {
    // Verify that zero_infinity=true does not change a finite loss value.
    let device = HostDevice;
    let ctc = CTCLossConfig::new().with_zero_infinity(true).init();

    let log_probs = Tensor::<TestBackend, 3>::full([2, 1, 2], 0.5_f32.ln(), &device);
    let targets = Tensor::<TestBackend, 2, Int>::from_data([[1_i32]], &device);
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([2_i32], &device);
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([1_i32], &device);

    let loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
    let loss_data = loss.into_data().to_vec::<f32>().unwrap();
    let expected = -(0.75_f32).ln();
    assert_approx_equal(&loss_data, &[expected], 1e-3);
}
