use realfft::RealFftPlanner;
use ruda_core::future::block_on;
use ruda_kernel::dsl::{RudaElement, Runtime, prelude::*};
use ruda_kernel::library::tensor::TensorHandle;
use ruda_test_runtime::TestRuntime;
use rufft::{irfft_launch_padded, rfft_launch_padded};
use std::time::Instant;

type Tensor = TensorHandle<TestRuntime>;
type Client = ComputeClient<TestRuntime>;

fn tensor(client: &Client, shape: Vec<usize>, data: Option<&[f32]>) -> Tensor {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let handle = match data {
        Some(data) => client.create_from_slice(f32::as_bytes(data)),
        None => client.empty(shape.iter().product::<usize>() * 4),
    };
    Tensor::new_contiguous(shape, handle, dtype)
}

fn read(client: &Client, tensor: &Tensor) -> Vec<f32> {
    let bytes = client.read_one(tensor.handle.clone()).unwrap();
    f32::from_bytes(&bytes).to_vec()
}

fn launch(
    client: &Client,
    signal: &Tensor,
    re: &Tensor,
    im: &Tensor,
    dim: usize,
    used: usize,
    inverse: bool,
) {
    let dtype = f32::as_type_native_unchecked().storage_type();
    if inverse {
        irfft_launch_padded(
            client,
            re.clone().binding(),
            im.clone().binding(),
            signal.clone().binding(),
            dim,
            used,
            dtype,
        )
        .unwrap();
    } else {
        rfft_launch_padded(
            client,
            signal.clone().binding(),
            re.clone().binding(),
            im.clone().binding(),
            dim,
            used,
            dtype,
        )
        .unwrap();
    }
}

fn measure(client: &Client, run: impl Fn() + Send + Sync) -> (f64, f64) {
    for _ in 0..3 {
        run();
    }
    block_on(client.sync()).unwrap();
    let mut interval = Vec::new();
    let mut host = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        let (_, duration) = client.profile(&run, "fft").unwrap();
        interval.push(block_on(duration.resolve()).duration().as_secs_f64() * 1e6);
        host.push(start.elapsed().as_secs_f64() * 1e6);
    }
    interval.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    (interval[5], host[5])
}

#[test]
#[ignore = "bounded GPU FFT benchmark"]
fn gpu_fft_timings() {
    let client = TestRuntime::client(&Default::default());
    println!("FFT_RUNTIME,{}", core::any::type_name::<TestRuntime>());
    println!("FFT_TIMING_METHOD,{}", client.properties().timing_method);
    let selected = std::env::var("RUDA_FFT_CASE").ok();
    let cases = [
        (64, 1),
        (64, 1024),
        (1024, 1),
        (1024, 256),
        (4096, 1),
        (4096, 64),
        (8192, 1),
        (8192, 64),
        (16384, 1),
        (16384, 64),
        (65536, 1),
        (65536, 16),
    ];
    let cases: Vec<_> = cases
        .into_iter()
        .filter(|(n, batch)| {
            selected
                .as_ref()
                .is_none_or(|value| value == &format!("{n},{batch}"))
        })
        .collect();
    assert!(
        !cases.is_empty(),
        "RUDA_FFT_CASE does not match a benchmark case"
    );
    for (index, &(n, batch)) in cases.iter().enumerate() {
        let data: Vec<f32> = (0..n * batch).map(|i| (i as f32 * 0.013).sin()).collect();
        let signal = tensor(&client, vec![batch, n], Some(&data));
        let re = tensor(&client, vec![batch, n / 2 + 1], None);
        let im = tensor(&client, vec![batch, n / 2 + 1], None);
        let output = tensor(&client, vec![batch, n], None);
        let forward = measure(&client, || launch(&client, &signal, &re, &im, 1, n, false));
        let inverse = measure(&client, || {
            launch(&client, &output, &re, &im, 1, n / 2 + 1, true)
        });
        println!(
            "GPU_FFT,{n},{batch},forward,{:.3},{:.3}",
            forward.0, forward.1
        );
        println!(
            "GPU_FFT,{n},{batch},inverse,{:.3},{:.3}",
            inverse.0, inverse.1
        );
        println!("GPU_FFT_PROGRESS,{}/{}", index + 1, cases.len());
    }
}

fn assert_close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a as f64 - e).abs() <= tolerance * (1.0 + e.abs()),
            "index={i}: {a} != {e}"
        );
    }
}

fn accuracy_case(client: &Client, n: usize, inner: usize, used: usize, bins: usize) {
    accuracy_layout(client, vec![2, n, inner], 1, used, bins);
}

fn accuracy_layout(client: &Client, shape: Vec<usize>, dim: usize, used: usize, bins: usize) {
    let n = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let mut data: Vec<f32> = (0..outer * n * inner)
        .map(|i| ((i * 17 % 257) as f32 - 128.0) / 128.0)
        .collect();
    for o in 0..outer {
        for k in used..n {
            for lane in 0..inner {
                data[(o * n + k) * inner + lane] =
                    if k % 2 == 0 { f32::NAN } else { f32::INFINITY };
            }
        }
    }
    let signal = tensor(client, shape.clone(), Some(&data));
    let mut spectrum_shape = shape.clone();
    spectrum_shape[dim] = n / 2 + 1;
    let re = tensor(client, spectrum_shape.clone(), None);
    let im = tensor(client, spectrum_shape, None);
    launch(client, &signal, &re, &im, dim, used, false);
    let actual_re = read(client, &re);
    let actual_im = read(client, &im);
    let mut expected_re = vec![0.0f64; actual_re.len()];
    let mut expected_im = vec![0.0f64; actual_im.len()];
    let mut planner = RealFftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(n);
    for o in 0..outer {
        for lane in 0..inner {
            let mut input: Vec<f64> = (0..n)
                .map(|k| {
                    if k < used {
                        data[(o * n + k) * inner + lane] as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut spectrum = forward.make_output_vec();
            forward.process(&mut input, &mut spectrum).unwrap();
            for (k, v) in spectrum.iter().enumerate() {
                let index = (o * (n / 2 + 1) + k) * inner + lane;
                expected_re[index] = v.re;
                expected_im[index] = v.im;
            }
        }
    }
    assert_close(&actual_re, &expected_re, 2e-5 * (n as f64).sqrt());
    assert_close(&actual_im, &expected_im, 2e-5 * (n as f64).sqrt());

    let mut spec_shape = shape.clone();
    spec_shape[dim] = bins;
    let spec_re: Vec<f32> = (0..outer * bins * inner)
        .map(|i| ((i * 13 % 97) as f32 - 48.0) / 48.0)
        .collect();
    let spec_im: Vec<f32> = (0..spec_re.len())
        .map(|i| ((i * 19 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let re = tensor(client, spec_shape.clone(), Some(&spec_re));
    let im = tensor(client, spec_shape, Some(&spec_im));
    let result = tensor(client, shape, None);
    launch(client, &result, &re, &im, dim, bins, true);
    let actual = read(client, &result);
    let mut expected = vec![0.0; actual.len()];
    let inverse = planner.plan_fft_inverse(n);
    for o in 0..outer {
        for lane in 0..inner {
            let mut spectrum = inverse.make_input_vec();
            for (k, z) in spectrum.iter_mut().enumerate().take(bins) {
                z.re = spec_re[(o * bins + k) * inner + lane] as f64;
                if k != 0 && k != n / 2 {
                    z.im = spec_im[(o * bins + k) * inner + lane] as f64;
                }
            }
            let mut output = inverse.make_output_vec();
            inverse.process(&mut spectrum, &mut output).unwrap();
            for k in 0..n {
                expected[(o * n + k) * inner + lane] = output[k] / n as f64;
            }
        }
    }
    assert_close(&actual, &expected, 2e-6);
    println!("GPU_FFT_ACCURACY,{n},{inner},{used},{bins}");
}

#[test]
fn gpu_fft_reference_matrix() {
    let client = TestRuntime::client(&Default::default());
    for n in [2, 4, 8, 16, 64, 256, 1024, 4096, 8192, 16384, 65536] {
        accuracy_case(&client, n, 1, n, n / 2 + 1);
    }
    for n in [8, 64, 4096, 8192, 16384] {
        accuracy_case(&client, n, 3, n - 3, n / 4);
    }
    accuracy_case(&client, 8192, 3, 0, 1);
}

#[test]
fn gpu_fft_fused_axes() {
    let client = TestRuntime::client(&Default::default());
    accuracy_layout(&client, vec![8192, 3], 0, 8191, 4097);
    accuracy_layout(&client, vec![2, 3, 8192], 2, 8192, 4097);
    accuracy_layout(&client, vec![2, 8192, 3, 2], 1, 4001, 2049);
}
