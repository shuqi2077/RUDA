use realfft::RealFftPlanner;
use ruda_core::tensor::{DType, data::TensorData, host::HostTensor};
use rufft_host::{irfft_f32, irfft_f64, rfft_f32, rfft_f64};

fn tensor(data: Vec<f64>, shape: &[usize]) -> HostTensor {
    HostTensor::from_data(TensorData::new(data, shape))
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "index={i}: {a} != {e}"
        );
    }
}

#[test]
fn f64_forward_matches_reference() {
    let mut planner = RealFftPlanner::<f64>::new();
    for n in [1, 2, 4, 8, 16, 64, 256, 1024, 16384, 131072] {
        let mut data: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.137).sin() + 0.1234567890123)
            .collect();
        let (re, im) = rfft_f64(tensor(data.clone(), &[n]), 0, None);
        let fft = planner.plan_fft_forward(n);
        let mut expected = fft.make_output_vec();
        fft.process(&mut data, &mut expected).unwrap();
        close(
            re.storage(),
            &expected.iter().map(|z| z.re).collect::<Vec<_>>(),
            n as f64 * 2e-14,
        );
        close(
            im.storage(),
            &expected.iter().map(|z| z.im).collect::<Vec<_>>(),
            n as f64 * 2e-14,
        );
    }
}

#[test]
fn f64_inverse_matches_independent_spectrum() {
    let mut planner = RealFftPlanner::<f64>::new();
    for n in [2, 4, 8, 16, 64, 1024, 16384] {
        let fft = planner.plan_fft_inverse(n);
        let mut spectrum = fft.make_input_vec();
        for (k, z) in spectrum.iter_mut().enumerate() {
            z.re = (k as f64 * 0.19).cos() + 1e-10;
            z.im = (k as f64 * 0.37).sin();
        }
        spectrum[0].im = 0.0;
        spectrum[n / 2].im = 0.0;
        let re = tensor(spectrum.iter().map(|z| z.re).collect(), &[n / 2 + 1]);
        let im = tensor(spectrum.iter().map(|z| z.im).collect(), &[n / 2 + 1]);
        let actual = irfft_f64(re, im, 0, Some(n));
        assert_eq!(actual.dtype(), DType::F64);
        let mut expected = fft.make_output_vec();
        fft.process(&mut spectrum, &mut expected).unwrap();
        for v in &mut expected {
            *v /= n as f64;
        }
        close(actual.storage(), &expected, 2e-13);
    }
}

#[test]
fn f64_preserves_range_and_small_differences() {
    for scale in [1e-100, 1.0, 1e100] {
        let data: Vec<f64> = (0..64).map(|i| scale * (1.0 + i as f64 * 1e-10)).collect();
        let (re, im) = rfft_f64(tensor(data.clone(), &[64]), 0, None);
        let result = irfft_f64(re, im, 0, None);
        let values: &[f64] = result.storage();
        close(
            &values.iter().map(|v| v / scale).collect::<Vec<_>>(),
            &data.iter().map(|v| v / scale).collect::<Vec<_>>(),
            2e-14,
        );
    }
}

#[test]
fn batched_and_strided_axes_match_single_fibers() {
    for dims in [[8, 64, 4], [2, 4, 64], [64, 4, 2]] {
        for dim in 0..3 {
            let data: Vec<f64> = (0..dims.iter().product())
                .map(|i| (i as f64 * 0.031).sin())
                .collect();
            let input = tensor(data, &dims);
            for input in [input.clone(), input.permute(&[2, 1, 0])] {
                let shape = input.layout().shape().as_slice().to_vec();
                let contiguous = input.clone().to_contiguous();
                let original: &[f64] = contiguous.storage();
                let n = shape[dim];
                let inner: usize = shape[dim + 1..].iter().product();
                let outer: usize = shape[..dim].iter().product();
                let bins = n / 2 + 1;
                let (re, im) = rfft_f64(input.clone(), dim, None);
                for o in 0..outer {
                    for lane in 0..inner {
                        let fiber = (0..n)
                            .map(|k| original[(o * n + k) * inner + lane])
                            .collect();
                        let (er, ei) = rfft_f64(tensor(fiber, &[n]), 0, None);
                        for (actual, expected) in [(&re, er), (&im, ei)] {
                            let actual: &[f64] = actual.storage();
                            let expected: &[f64] = expected.storage();
                            close(
                                &(0..bins)
                                    .map(|k| actual[(o * bins + k) * inner + lane])
                                    .collect::<Vec<_>>(),
                                expected,
                                1e-13,
                            );
                        }
                    }
                }
                close(
                    irfft_f64(re, im, dim, Some(n)).to_contiguous().storage(),
                    original,
                    2e-13,
                );
                let f32_input = HostTensor::from_data(TensorData::new(
                    original.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                    shape,
                ));
                let (re, im) = rfft_f32(f32_input, dim, None);
                let result = irfft_f32(re, im, dim, Some(n)).to_contiguous();
                let values: &[f32] = result.storage();
                close(
                    &values.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                    original,
                    2e-6,
                );
            }
        }
    }
}

#[test]
fn f64_padding_truncation_and_single_bin() {
    for requested in [1usize, 2, 6, 8, 16, 64] {
        let data: Vec<f64> = (0..9).map(|i| i as f64 + 1e-9).collect();
        let padded_n = requested.next_power_of_two();
        let mut expected = vec![0.0; padded_n];
        let used = requested.min(data.len());
        expected[..used].copy_from_slice(&data[..used]);
        let (re, im) = rfft_f64(tensor(data, &[9]), 0, Some(requested));
        let result = irfft_f64(re, im, 0, Some(requested)).to_contiguous();
        close(result.storage(), &expected[..requested], 2e-13);
    }
    for bins in [1usize, 3, 9] {
        let re: Vec<f64> = (0..bins).map(|k| k as f64 + 0.123456789123).collect();
        let im: Vec<f64> = (0..bins)
            .map(|k| if k == 0 || k == 4 { 0.0 } else { 0.25 })
            .collect();
        let actual = irfft_f64(
            tensor(re.clone(), &[bins]),
            tensor(im.clone(), &[bins]),
            0,
            Some(6),
        );
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_inverse(8);
        let mut spectrum = fft.make_input_vec();
        for k in 0..bins.min(5) {
            spectrum[k].re = re[k];
            spectrum[k].im = im[k];
        }
        let mut expected = fft.make_output_vec();
        fft.process(&mut spectrum, &mut expected).unwrap();
        close(
            actual.to_contiguous().storage(),
            &expected[..6].iter().map(|v| v / 8.0).collect::<Vec<_>>(),
            2e-13,
        );
    }
}
