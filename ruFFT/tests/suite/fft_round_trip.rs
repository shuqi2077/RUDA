use ruda_kernel::dsl as kernel_dsl;
#[cfg(feature = "heavy")]
use ruda_kernel::dsl::Runtime;
use ruda_test_runtime::TestRuntime;
use ruda_kernel::dsl::prelude::RudaPrimitive;
#[cfg(feature = "heavy")]
use rufft::{irfft, rfft};
//use rudafx_engine::{SignalSpec, phase_shift_effect};
#[cfg(feature = "heavy")]
use ruda_test_utils::{HostData, TestInput, assert_equals_approx};

#[test]
#[cfg(feature = "heavy")]
fn large_fft_roundtrip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    let shape = [431, 2, 2048];

    let (original_signal, signal_data) = TestInput::builder(client.clone(), shape)
        .dtype(dtype)
        .uniform(42, -1., 1.)
        .generate_with_f32_host_data();

    let (spectrum_re, spectrum_im) = rfft(original_signal, shape.len() - 1, dtype);
    let signal_back = irfft(spectrum_re, spectrum_im, shape.len() - 1, dtype);

    assert_equals_approx(
        &HostData::from_tensor_handle(&client, signal_back, ruda_test_utils::HostDataType::F32),
        &signal_data,
        0.03,
    )
    .as_test_outcome()
    .enforce();
}
