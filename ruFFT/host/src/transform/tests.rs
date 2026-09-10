    use super::*;
    use ruda_core::tensor::{DType, data::{TensorData, Tolerance}};

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> HostTensor {
        HostTensor::from_data(TensorData::new(data, shape))
    }

    fn make_f64(data: Vec<f64>, shape: Vec<usize>) -> HostTensor {
        HostTensor::from_data(TensorData::new(data, shape))
    }

    fn assert_approx(tensor: HostTensor, expected: &[f32], tol: f32) {
        let shape = tensor.layout().shape().as_slice().to_vec();
        tensor.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected.to_vec(), shape),
            Tolerance::absolute(tol),
        );
    }

    fn assert_approx_f64(tensor: HostTensor, expected: &[f64], tol: f64) {
        tensor
            .into_data()
            .assert_approx_eq::<f64>(&TensorData::from(expected), Tolerance::absolute(tol));
    }

    // ---- N=1 ----

    #[test]
    fn rfft_n1() {
        let signal = make_f32(vec![5.0], vec![1]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[5.0], 1e-6);
        assert_approx(im, &[0.0], 1e-6);
    }

    // ---- N=2 ----

    #[test]
    fn rfft_n2() {
        let signal = make_f32(vec![1.0, -1.0], vec![2]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[0.0, 2.0], 1e-6);
        assert_approx(im, &[0.0, 0.0], 1e-6);
    }

    // ---- N=4: known DFT of [1,0,0,0] = [1,1,1] (all real) ----

    #[test]
    fn rfft_n4_impulse() {
        let signal = make_f32(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[1.0, 1.0, 1.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=4: constant signal [1,1,1,1] -> DC only ----

    #[test]
    fn rfft_n4_constant() {
        let signal = make_f32(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[4.0, 0.0, 0.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=4: zeros ----

    #[test]
    fn rfft_n4_zeros() {
        let signal = make_f32(vec![0.0; 4], vec![4]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[0.0, 0.0, 0.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=8 ----

    #[test]
    fn rfft_n8_impulse() {
        let mut signal = vec![0.0f32; 8];
        signal[0] = 1.0;
        let (re, im) = rfft_f32(make_f32(signal, vec![8]), 0, None);
        // DFT of impulse is all 1s
        assert_approx(re, &[1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn rfft_n8_cosine() {
        // cos(2*pi*k/8) for k=0..7 -> energy at bin 1
        let signal: Vec<f32> = (0..8)
            .map(|k| (2.0 * std::f32::consts::PI * k as f32 / 8.0).cos())
            .collect();
        let (re, im) = rfft_f32(make_f32(signal, vec![8]), 0, None);
        // Bin 1 should have amplitude 4 (real), rest ~0
        assert_approx(re, &[0.0, 4.0, 0.0, 0.0, 0.0], 1e-4);
        assert_approx(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-4);
    }

    // ---- Larger size: N=256 ----

    #[test]
    fn rfft_n256_impulse() {
        let mut signal = vec![0.0f32; 256];
        signal[0] = 1.0;
        let (re, im) = rfft_f32(make_f32(signal, vec![256]), 0, None);
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 129);
        for &v in re_vals {
            assert!((v - 1.0).abs() < 1e-5, "re bin should be 1.0, got {v}");
        }
        for &v in im_vals {
            assert!(v.abs() < 1e-5, "im bin should be 0.0, got {v}");
        }
    }

    // ---- Multi-dimensional: FFT along dim 1 ----

    #[test]
    fn rfft_2d_dim1() {
        // 2 rows, each of length 4: impulse and constant
        let data = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: impulse
            1.0, 1.0, 1.0, 1.0, // row 1: constant
        ];
        let signal = make_f32(data, vec![2, 4]);
        let (re, im) = rfft_f32(signal, 1, None);
        // Shape should be [2, 3]
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 6); // 2 * 3
        // Row 0 (impulse): [1, 1, 1]
        assert!((re_vals[0] - 1.0).abs() < 1e-5);
        assert!((re_vals[1] - 1.0).abs() < 1e-5);
        assert!((re_vals[2] - 1.0).abs() < 1e-5);
        // Row 1 (constant): [4, 0, 0]
        assert!((re_vals[3] - 4.0).abs() < 1e-5);
        assert!((re_vals[4]).abs() < 1e-5);
        assert!((re_vals[5]).abs() < 1e-5);
        // All imaginary should be ~0
        for &v in im_vals {
            assert!(v.abs() < 1e-5);
        }
    }

    // ---- FFT along dim 0 ----

    #[test]
    fn rfft_2d_dim0() {
        // 4 rows, 2 cols: impulse in each column
        let data = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let signal = make_f32(data, vec![4, 2]);
        let (re, _im) = rfft_f32(signal, 0, None);
        // Shape should be [3, 2]
        let re_data = re.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 6);
        // Each column is an impulse -> all bins = 1
        for &v in re_vals {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    // ---- f64 dtype ----

    #[test]
    fn rfft_f64_n4_impulse() {
        let signal = make_f64(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f64(signal, 0, None);
        assert_approx_f64(re, &[1.0, 1.0, 1.0], 1e-10);
        assert_approx_f64(im, &[0.0, 0.0, 0.0], 1e-10);
    }

    #[test]
    fn rfft_f64_n8_cosine() {
        let signal: Vec<f64> = (0..8)
            .map(|k| (2.0 * std::f64::consts::PI * k as f64 / 8.0).cos())
            .collect();
        let (re, im) = rfft_f64(make_f64(signal, vec![8]), 0, None);
        assert_approx_f64(re, &[0.0, 4.0, 0.0, 0.0, 0.0], 1e-6);
        assert_approx_f64(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    // ---- f16 dtype ----

    #[test]
    fn rfft_f16_n4_impulse() {
        use half::f16;
        let f16_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
        ];
        let signal = HostTensor::new(
            Bytes::from_elems(f16_data),
            Layout::contiguous(Shape::from(vec![4])),
            DType::F16,
        );
        let (re, _im) = rfft_f16(signal, 0, None);
        // Verify via round-trip to f32
        let re_f32 = ruda_core::tensor::host::cast::cast_to_f32(re, f16::to_f32);
        let re_data = re_f32.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 3);
        for &v in re_vals {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    // ---- Const twiddle accuracy ----

    #[test]
    fn const_sin_cos_accuracy() {
        let test_angles = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, -1.0, -3.0, 6.0];
        for &angle in &test_angles {
            let cs = const_sin(angle);
            let cc = const_cos(angle);
            let rs = angle.sin();
            let rc = angle.cos();
            assert!(
                (cs - rs).abs() < 1e-12,
                "const_sin({angle}) = {cs}, expected {rs}"
            );
            assert!(
                (cc - rc).abs() < 1e-12,
                "const_cos({angle}) = {cc}, expected {rc}"
            );
        }
    }

    // ---- N=1024 round-trip with known property: Parseval's theorem ----
    // Sum of |x|^2 = (1/N) * Sum of |X|^2

    #[test]
    fn rfft_n1024_parseval() {
        let n = 1024;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let time_energy: f64 = signal.iter().map(|&x| (x as f64) * (x as f64)).sum();

        let (re, im) = rfft_f32(make_f32(signal, vec![n]), 0, None);
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();

        // Frequency energy: DC and Nyquist count once, others count double
        let out_len = n / 2 + 1;
        let mut freq_energy = 0.0f64;
        for k in 0..out_len {
            let mag2 = (re_vals[k] as f64).powi(2) + (im_vals[k] as f64).powi(2);
            if k == 0 || k == n / 2 {
                freq_energy += mag2;
            } else {
                freq_energy += 2.0 * mag2;
            }
        }
        freq_energy /= n as f64;

        let rel_err = (freq_energy - time_energy).abs() / time_energy;
        assert!(
            rel_err < 1e-4,
            "Parseval's theorem violated: time={time_energy}, freq={freq_energy}, rel_err={rel_err}"
        );
    }

    // ---- irfft tests ----

    #[test]
    fn irfft_roundtrip_n4() {
        let signal = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let (re, im) = rfft_f32(signal.clone(), 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        assert_approx(reconstructed, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn irfft_roundtrip_n8() {
        let data: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
        let signal = make_f32(data.clone(), vec![8]);
        let (re, im) = rfft_f32(signal, 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        assert_approx(reconstructed, &data, 1e-5);
    }

    #[test]
    fn rfft_vs_realfft() {
        // Verify our rfft matches realfft (rustfft-backed) for non-trivial input
        // at sizes that exercise radix-4 (n>=16) and the complex packing trick.
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();

        for &n in &[4, 8, 16, 32, 64, 256, 1024, 4096] {
            let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 0.5).collect();

            // Our rfft
            let signal = make_f32(data.clone(), vec![n]);
            let (re_out, im_out) = rfft_f32(signal, 0, None);
            let re_data = re_out.into_data();
            let im_data = im_out.into_data();
            let our_re = re_data.as_slice::<f32>().unwrap();
            let our_im = im_data.as_slice::<f32>().unwrap();

            // Reference: realfft
            let r2c = planner.plan_fft_forward(n);
            let mut input = data.clone();
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();

            let out_len = n / 2 + 1;
            assert_eq!(our_re.len(), out_len);
            assert_eq!(spectrum.len(), out_len);

            let max_re_err = our_re
                .iter()
                .zip(spectrum.iter())
                .map(|(&a, b)| (a - b.re).abs())
                .fold(0.0f32, f32::max);
            let max_im_err = our_im
                .iter()
                .zip(spectrum.iter())
                .map(|(&a, b)| (a - b.im).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_re_err < 1e-3 && max_im_err < 1e-3,
                "rfft vs realfft mismatch at n={n}: max_re_err={max_re_err}, max_im_err={max_im_err}"
            );
        }
    }

    #[test]
    fn irfft_vs_realfft() {
        // Verify our irfft matches realfft's inverse for non-trivial spectra.
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();

        for &n in &[4, 8, 16, 32, 64, 256, 1024, 4096] {
            // Generate a spectrum via realfft forward
            let r2c = planner.plan_fft_forward(n);
            let c2r = planner.plan_fft_inverse(n);
            let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 0.5).collect();
            let mut input = data.clone();
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();

            // Our irfft
            let out_len = n / 2 + 1;
            let spec_re: Vec<f32> = spectrum.iter().map(|c| c.re).collect();
            let spec_im: Vec<f32> = spectrum.iter().map(|c| c.im).collect();
            let re_tensor = make_f32(spec_re, vec![out_len]);
            let im_tensor = make_f32(spec_im, vec![out_len]);
            let our_result = irfft_f32(re_tensor, im_tensor, 0, None);
            let our_data = our_result.into_data();
            let our_vals = our_data.as_slice::<f32>().unwrap();

            // Reference: realfft inverse (note: realfft doesn't normalize, so scale)
            let mut spec_copy = spectrum.clone();
            let mut ref_output = c2r.make_output_vec();
            c2r.process(&mut spec_copy, &mut ref_output).unwrap();
            let scale = 1.0 / n as f32;
            let ref_scaled: Vec<f32> = ref_output.iter().map(|&v| v * scale).collect();

            let max_err = our_vals
                .iter()
                .zip(ref_scaled.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-3,
                "irfft vs realfft mismatch at n={n}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn forward_complex_fft_impulse() {
        // DFT of impulse [1,0,0,...,0] should be all 1s
        for &n in &[4, 8, 16, 32, 64] {
            let tw = get_twiddles(n);
            let mut re = vec![0.0f32; n];
            let mut im = vec![0.0f32; n];
            re[0] = 1.0;
            complex_fft(&mut re, &mut im, n, &tw);
            let max_re_err = re.iter().map(|&v| (v - 1.0).abs()).fold(0.0f32, f32::max);
            let max_im_err = im.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_re_err < 1e-5 && max_im_err < 1e-5,
                "forward FFT impulse n={n}: max_re_err={max_re_err}, max_im_err={max_im_err}"
            );
        }
    }

    #[test]
    fn inverse_complex_fft_roundtrip() {
        for &n in &[4, 8, 16, 32, 64, 256] {
            let tw = get_twiddles(n);
            let mut re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
            let mut im = vec![0.0f32; n];
            let orig_re = re.clone();

            complex_fft(&mut re, &mut im, n, &tw);
            inverse_complex_fft(&mut re, &mut im, n, &tw);

            let max_err = re
                .iter()
                .zip(orig_re.iter())
                .map(|(&got, &expected)| (got - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-4,
                "inverse_complex_fft roundtrip n={n}: max error {max_err}"
            );
        }
    }

    #[test]
    fn irfft_roundtrip_n256() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).cos()).collect();
        let signal = make_f32(data.clone(), vec![256]);
        let (re, im) = rfft_f32(signal, 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        let result = reconstructed.into_data();
        let vals = result.as_slice::<f32>().unwrap();
        let max_err = vals
            .iter()
            .zip(data.iter())
            .map(|(&got, &expected)| (got - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 5e-3,
            "irfft_roundtrip_n256: max error {max_err} exceeds tolerance"
        );
    }

    #[test]
    fn irfft_roundtrip_2d_dim1() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
        ];
        let signal = make_f32(data.clone(), vec![2, 4]);
        let (re, im) = rfft_f32(signal, 1, None);
        let reconstructed = irfft_f32(re, im, 1, None);
        assert_approx(reconstructed, &data, 1e-5);
    }

    #[test]
    fn irfft_roundtrip_2d_dim0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let signal = make_f32(data.clone(), vec![4, 2]);
        let (re, im) = rfft_f32(signal, 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        assert_approx(reconstructed, &data, 1e-5);
    }

    #[test]
    fn irfft_known_spectrum() {
        // DC=4, all others zero -> constant signal [1,1,1,1]
        let re = make_f32(vec![4.0, 0.0, 0.0], vec![3]);
        let im = make_f32(vec![0.0, 0.0, 0.0], vec![3]);
        let signal = irfft_f32(re, im, 0, None);
        assert_approx(signal, &[1.0, 1.0, 1.0, 1.0], 1e-5);
    }

    #[test]
    fn irfft_f64_roundtrip() {
        // irfft_f64 truncates to f32 internally, so tolerance is f32-level
        let data: Vec<f64> = (0..8).map(|i| (i as f64 * 0.3).sin()).collect();
        let signal = make_f64(data.clone(), vec![8]);
        let (re, im) = rfft_f64(signal, 0, None);
        let reconstructed = irfft_f64(re, im, 0, None);
        assert_approx_f64(reconstructed, &data, 1e-5);
    }

    // Coverage for the n=Some(pow2) path on flex.

    #[test]
    fn rfft_n_larger_than_signal_zero_pads() {
        // n=8 zero-pads the signal; output has 8/2+1 = 5 bins.
        let signal = make_f32(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0, Some(8));
        assert_eq!(re.layout().shape()[0], 5);
        assert_eq!(im.layout().shape()[0], 5);
        let re_vals = re.into_data().as_slice::<f32>().unwrap().to_vec();
        for (k, v) in re_vals.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "impulse DFT re[{k}] should be 1.0, got {v}"
            );
        }
    }

    #[test]
    fn rfft_n_smaller_than_signal_truncates_first() {
        // Signal length 8, n=4 -> truncate to 4, compute 4-point DFT of [1,0,0,0].
        let signal = make_f32(vec![1.0, 0.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0], vec![8]);
        let (re, _im) = rfft_f32(signal, 0, Some(4));
        assert_eq!(re.layout().shape()[0], 3);
        let re_vals = re.into_data().as_slice::<f32>().unwrap().to_vec();
        for v in &re_vals {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn rfft_irfft_roundtrip_with_pow2_n() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let signal = make_f32(data.clone(), vec![8]);
        let (re, im) = rfft_f32(signal, 0, Some(8));
        let reconstructed = irfft_f32(re, im, 0, Some(8));
        assert_approx(reconstructed, &data, 1e-4);
    }

    #[test]
    fn rfft_f64_with_pow2_n_and_truncation() {
        // Signal length 8, n=4. Output has 4/2+1 = 3 bins.
        let data: Vec<f64> = (0..8).map(|i| (i as f64 * 0.3).sin()).collect();
        let signal = make_f64(data, vec![8]);
        let (re, im) = rfft_f64(signal, 0, Some(4));
        assert_eq!(re.layout().shape()[0], 3);
        assert_eq!(im.layout().shape()[0], 3);
    }

    #[test]
    fn rfft_vs_realfft_with_pow2_n_and_padding() {
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();
        // Signal shorter than n (pow2); backend zero-pads before the FFT.
        for &(sig_len, n) in &[(3usize, 4usize), (5, 8), (6, 8), (9, 16)] {
            let data: Vec<f32> = (0..sig_len)
                .map(|i| (i as f32 * 0.41).cos() - 0.2)
                .collect();
            let mut padded = data.clone();
            padded.resize(n, 0.0);

            let r2c = planner.plan_fft_forward(n);
            let mut input = padded;
            let mut ref_spec = r2c.make_output_vec();
            r2c.process(&mut input, &mut ref_spec).unwrap();

            let signal = make_f32(data, vec![sig_len]);
            let (re, im) = rfft_f32(signal, 0, Some(n));
            let re_v = re.into_data().as_slice::<f32>().unwrap().to_vec();
            let im_v = im.into_data().as_slice::<f32>().unwrap().to_vec();

            assert_eq!(re_v.len(), n / 2 + 1);
            for (k, refc) in ref_spec.iter().enumerate() {
                let err_re = (re_v[k] - refc.re).abs();
                let err_im = (im_v[k] - refc.im).abs();
                assert!(
                    err_re < 1e-3 && err_im < 1e-3,
                    "sig_len={sig_len} n={n} bin={k}: got ({}, {}), ref ({}, {})",
                    re_v[k],
                    im_v[k],
                    refc.re,
                    refc.im
                );
            }
        }
    }
