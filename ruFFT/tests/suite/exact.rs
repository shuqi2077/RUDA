//! Real GPU tests when TestRuntime=cuda/hip/wgpu. The release gate pins cuda;
//! these are NOT replaced by the Python host mathematical reference tests.
use realfft::RealFftPlanner;
use ruda_kernel::dsl::{prelude::*, RudaElement};
use ruda_kernel::library::tensor::TensorHandle;
use ruda_test_runtime::TestRuntime;
use rufft::{FftMode, RealFftPlan};

type Client = ComputeClient<TestRuntime>;
type Tensor = TensorHandle<TestRuntime>;
fn client() -> Client {
    let runtime = core::any::type_name::<TestRuntime>();
    println!("EXACT_FFT_RUNTIME,{runtime}");
    if std::env::var("RUDA_FFT_REQUIRE_CUDA").as_deref() == Ok("1") {
        assert!(runtime.contains("CudaRuntime"), "release validation must not run on a CPU/fallback runtime");
    }
    TestRuntime::client(&Default::default())
}

fn tensor(client: &Client, shape: Vec<usize>, data: Option<&[f32]>) -> Tensor {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let elements: usize = shape.iter().product();
    let memory = match data {
        Some(data) => { assert_eq!(data.len(), elements); client.create_from_slice(f32::as_bytes(data)) },
        None => client.empty(elements * 4),
    };
    Tensor::new_contiguous(shape, memory, dtype)
}
fn read(client: &Client, value: &Tensor) -> Vec<f32> {
    f32::from_bytes(&client.read_one(value.handle.clone()).unwrap()).to_vec()
}
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![1; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() { result[dim] = result[dim + 1] * shape[dim + 1]; }
    result
}
fn offset(mut row: usize, shape: &[usize], dim: usize, steps: &[usize]) -> usize {
    let mut base = 0;
    for axis in 0..shape.len() {
        if axis != dim { base += row % shape[axis] * steps[axis]; row /= shape[axis]; }
    }
    base
}
fn close(actual: f32, expected: f32, context: &str) {
    assert!(actual.is_finite() && (actual - expected).abs() <= 0.002 + 0.00005 * expected.abs(),
        "{context}: actual={actual} expected={expected}");
}
fn case(shape: &[usize], dim: usize, n: usize, used: usize) {
    let client = client();
    let data: Vec<f32> = (0..shape.iter().product()).map(|i| ((i * 17 + 3) as f32 * 0.013).sin()).collect();
    let input = tensor(&client, shape.to_vec(), Some(&data));
    let mut spec_shape = shape.to_vec(); spec_shape[dim] = n / 2 + 1;
    let mut out_shape = shape.to_vec(); out_shape[dim] = n;
    let re = tensor(&client, spec_shape.clone(), None);
    let im = tensor(&client, spec_shape.clone(), None);
    let restored = tensor(&client, out_shape.clone(), None);
    let mut forward = RealFftPlan::new(client.clone(), n, FftMode::Forward).unwrap();
    let mut inverse = RealFftPlan::new(client.clone(), n, FftMode::Inverse).unwrap();
    forward.forward(input.clone().binding(), re.clone().binding(), im.clone().binding(), dim, used).unwrap();
    inverse.inverse(re.clone().binding(), im.clone().binding(), restored.clone().binding(), dim, n / 2 + 1).unwrap();
    let retained = forward.retained_bytes();
    // Reuse exactly the same chirps and batched scratch; no intervening host
    // wait. Repeat output must agree after normal same-stream ordering.
    forward.forward(input.clone().binding(), re.clone().binding(), im.clone().binding(), dim, used).unwrap();
    assert_eq!(forward.retained_bytes(), retained);
    let actual_re = read(&client, &re); let actual_im = read(&client, &im);
    let actual_x = read(&client, &restored);
    let rows: usize = shape.iter().enumerate().filter(|(axis, _)| *axis != dim).map(|(_, &v)| v).product();
    let a_stride = strides(shape); let s_stride = strides(&spec_shape); let o_stride = strides(&out_shape);
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    for row in 0..rows {
        let ib = offset(row, shape, dim, &a_stride);
        let sb = offset(row, &spec_shape, dim, &s_stride);
        let ob = offset(row, &out_shape, dim, &o_stride);
        let mut signal = vec![0.0; n];
        for k in 0..used { signal[k] = data[ib + k * a_stride[dim]]; }
        for k in 0..n { close(actual_x[ob + k * o_stride[dim]], signal[k], "roundtrip"); }
        let mut expected = fft.make_output_vec(); fft.process(&mut signal, &mut expected).unwrap();
        for k in 0..expected.len() {
            close(actual_re[sb + k * s_stride[dim]], expected[k].re, "real");
            close(actual_im[sb + k * s_stride[dim]], expected[k].im, "imag");
        }
    }
    forward.clear_workspace();
    if let Some(m) = forward.convolution_len() { assert_eq!(forward.retained_bytes(), 8 * (n + m)); }
}

#[test] fn exact_fft_six_points() { case(&[6], 0, 6, 6); }
#[test] fn exact_fft_odd_prime() { case(&[3, 7], 1, 7, 7); }
#[test] fn exact_fft_middle_dimension() { case(&[2, 6, 3], 1, 6, 6); }
#[test] fn exact_fft_truncation() { case(&[2, 20], 1, 11, 11); }
#[test] fn exact_fft_virtual_padding() { case(&[2, 5], 1, 13, 5); }
#[test] fn exact_fft_empty_source() { case(&[2, 0], 1, 7, 0); }
#[test] fn exact_fft_single_sample() { case(&[3, 1], 1, 1, 1); }
#[test] fn exact_fft_empty_single_sample() { case(&[2, 0], 1, 1, 0); }
#[test] fn exact_fft_radix2_padding() { case(&[2, 5], 1, 8, 5); }
#[test] fn exact_fft_radix2_empty() { case(&[2, 0], 1, 8, 0); }
#[test] fn exact_fft_radix2_packed_empty() { case(&[1, 0], 1, 8192, 0); }
#[test] fn exact_fft_radix2_four_step_empty() { case(&[1, 0], 1, 16384, 0); }
#[test] fn exact_fft_large_prime() { case(&[2, 1009], 1, 1009, 1009); }
#[test] fn exact_fft_four_step_convolution() { case(&[1, 4097], 1, 4097, 4097); }

#[test]
fn exact_fft_irfft_real_only_bins() {
    let client = client();
    for n in [1, 6, 7, 8, 8192] {
        let bins = n / 2 + 1;
        let re = tensor(&client, vec![1, bins], Some(&vec![1.0; bins]));
        let mut imag = vec![0.0; bins]; imag[0] = f32::NAN;
        if n % 2 == 0 { imag[n / 2] = f32::NAN; }
        let im = tensor(&client, vec![1, bins], Some(&imag));
        let out = tensor(&client, vec![1, n], None);
        let mut plan = RealFftPlan::new(client.clone(), n, FftMode::Inverse).unwrap();
        plan.inverse(re.binding(), im.binding(), out.clone().binding(), 1, bins).unwrap();
        let values = read(&client, &out);
        for (i, value) in values.into_iter().enumerate() { close(value, if i == 0 {1.0} else {0.0}, "real-only bins"); }
    }
}

#[cfg(feature = "tensor")]
#[test]
fn exact_fft_tensor_api_preserves_legacy_shape() {
    use ruda_core::tensor::{data::TensorData, TensorMetadata};
    use ruda_kernel::tensor::{transfer::from_data, readback::into_data_sync};
    let device = Default::default();
    let x = from_data::<TestRuntime>(TensorData::new(vec![1.0f32, 2., 3., 4., 5., 6.], [6]), &device);
    let (re, im) = rufft::tensor::rfft_exact(x.clone(), 0, None);
    assert_eq!(re.shape()[0], 4);
    let x_back = rufft::tensor::irfft_exact(re, im, 0, Some(6));
    let values = into_data_sync(x_back).to_vec::<f32>().unwrap();
    for (i, v) in values.into_iter().enumerate() { close(v, i as f32 + 1., "tensor API"); }
    let (legacy_re, _) = rufft::tensor::rfft(x, 0, None);
    assert_eq!(legacy_re.shape()[0], 5);
}

#[test]
#[ignore = "real GPU benchmark; compares exact N with exact N, not padded FFT"]
fn exact_fft_plan_reuse_benchmark() {
    use ruda_core::future::block_on;
    use std::time::Instant;
    let client = client();
    println!("EXACT_FFT_RUNTIME,{}", core::any::type_name::<TestRuntime>());
    for n in [6, 1009, 4097] {
        let data = vec![0.25f32; 16 * n];
        let input = tensor(&client, vec![16, n], Some(&data));
        let re = tensor(&client, vec![16, n / 2 + 1], None);
        let im = tensor(&client, vec![16, n / 2 + 1], None);
        let mut plan = RealFftPlan::new(client.clone(), n, FftMode::Forward).unwrap();
        plan.forward(input.clone().binding(), re.clone().binding(), im.clone().binding(), 1, n).unwrap();
        block_on(client.sync()).unwrap();
        for reuse in [false, true] {
            let start = Instant::now();
            for _ in 0..50 {
                if reuse {
                    plan.forward(input.clone().binding(), re.clone().binding(), im.clone().binding(), 1, n).unwrap();
                } else {
                    let mut temporary = RealFftPlan::new(client.clone(), n, FftMode::Forward).unwrap();
                    temporary.forward(input.clone().binding(), re.clone().binding(), im.clone().binding(), 1, n).unwrap();
                }
            }
            block_on(client.sync()).unwrap();
            println!("EXACT_FFT_TIMING,n={n},batch=16,reuse={reuse},host_and_gpu_us={},retained_bytes={}",
                start.elapsed().as_secs_f64() * 1e6 / 50., plan.retained_bytes());
        }
    }
}
