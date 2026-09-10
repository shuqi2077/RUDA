    use super::*;
    use ruda_core::tensor::data::TensorData;
    use half::bf16;

    #[test]
    fn test_conv1d_direct_path() {
        // c_in=64 >= 32, kernel=3, stride=2, out_w=49 <= 800 -> hits direct path
        let c_in = 64;
        let c_out = 32;
        let in_w = 100;
        let kw = 3;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1; // 49

        // Deterministic input and weights
        let x_data: Vec<f32> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f32> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
            .collect();

        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        assert_eq!(out.len(), c_out * out_w);

        // Verify against naive reference implementation
        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = 0.0f32;
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_direct_path_kw2() {
        // Test with kw=2 (L5/L6 shapes use kw=2)
        let c_in = 64;
        let c_out = 32;
        let in_w = 50;
        let kw = 2;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1; // 24

        let x_data: Vec<f32> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f32> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
            .collect();

        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = 0.0f32;
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_direct_path_f64() {
        let c_in = 64;
        let c_out = 16;
        let in_w = 50;
        let kw = 3;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1;

        let x_data: Vec<f64> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f64 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f64> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f64 / 50.0) - 0.5)
            .collect();

        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();

        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = 0.0f64;
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "f64 mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_direct_path_with_bias() {
        let c_in = 64;
        let c_out = 32;
        let in_w = 50;
        let kw = 2;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1;

        let x_data: Vec<f32> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f32> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
            .collect();
        let bias_data: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.1).collect();

        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let bias = HostTensor::from_data(TensorData::new(bias_data.clone(), vec![c_out]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f32(x, weight, Some(bias), &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = bias_data[co];
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "bias mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv2d_f64() {
        let x_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data = vec![1.0f64; 4];
        let weight = HostTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(
            out,
            vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0]
        );
    }

    #[test]
    fn test_conv2d_f16() {
        let x_data: Vec<f16> = (1..=16).map(|x| f16::from_f32(x as f32)).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<f16> = vec![f16::from_f32(1.0); 4];
        let weight = HostTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f16(x, weight, None, &options);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();
        let expected = vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5);
        }
    }

    #[test]
    fn test_conv2d_bf16() {
        let x_data: Vec<bf16> = (1..=16).map(|x| bf16::from_f32(x as f32)).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<bf16> = vec![bf16::from_f32(1.0); 4];
        let weight = HostTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_bf16(x, weight, None, &options);
        let out: Vec<bf16> = result.into_data().to_vec().unwrap();
        let expected = vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5);
        }
    }

    // ========================================================================
    // Depthwise conv fast-path tests
    // ========================================================================
    //
    // These tests exercise `conv3d_depthwise_impl` and cross-check against a
    // naive reference to catch any indexing, bounds, or accumulation mistakes.

    /// Naive NCHW depthwise conv2d reference implementation.
    /// Returns output with shape `[batch, channels, out_h, out_w]`.
    #[allow(clippy::too_many_arguments)]
    fn naive_depthwise_conv2d_f32(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> (Vec<f32>, usize, usize) {
        let out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        let mut out = vec![0.0f32; batch * channels * out_h * out_w];
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0f32;
                        for kh in 0..kernel_h {
                            let ih = oh as isize * stride_h as isize
                                + kh as isize * dilation_h as isize
                                - pad_h as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            for kw in 0..kernel_w {
                                let iw = ow as isize * stride_w as isize
                                    + kw as isize * dilation_w as isize
                                    - pad_w as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                let x_idx =
                                    ((b * channels + c) * in_h + ih as usize) * in_w + iw as usize;
                                let w_idx = (c * kernel_h + kh) * kernel_w + kw;
                                acc += x[x_idx] * w[w_idx];
                            }
                        }
                        if let Some(bias) = bias {
                            acc += bias[c];
                        }
                        let o_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
        (out, out_h, out_w)
    }

    /// Build deterministic pseudo-random input for a depthwise test.
    fn seeded_vec_f32(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) & 0xffff;
                (v as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    /// Shared helper: run conv2d via the public dispatch (which must pick the
    /// depthwise path for these shapes) and compare to the naive reference.
    #[allow(clippy::too_many_arguments)]
    fn check_depthwise_conv2d_f32(
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        with_bias: bool,
    ) {
        let x_vec = seeded_vec_f32(batch * channels * in_h * in_w, 1);
        let w_vec = seeded_vec_f32(channels * kernel_h * kernel_w, 2);
        let bias_vec = if with_bias {
            Some(seeded_vec_f32(channels, 3))
        } else {
            None
        };

        let (expected, out_h, out_w) = naive_depthwise_conv2d_f32(
            &x_vec,
            &w_vec,
            bias_vec.as_deref(),
            batch,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
        );

        let x = HostTensor::from_data(TensorData::new(x_vec, vec![batch, channels, in_h, in_w]));
        let weight = HostTensor::from_data(TensorData::new(
            w_vec,
            vec![channels, 1, kernel_h, kernel_w],
        ));
        let bias = bias_vec.map(|v| HostTensor::from_data(TensorData::new(v, vec![channels])));
        let options = ConvOptions::new(stride, padding, dilation, channels);
        let result = conv2d_f32(x, weight, bias, &options);

        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels, out_h, out_w],
            "output shape mismatch"
        );

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out.len(), expected.len());
        for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-4,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn test_conv2d_depthwise_3x3_no_pad() {
        // Canonical depthwise: groups == channels_in == channels_out = 8, 3x3.
        check_depthwise_conv2d_f32(2, 8, 16, 16, 3, 3, [1, 1], [0, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_3x3_pad1() {
        // Same input and output size via padding=1.
        check_depthwise_conv2d_f32(2, 8, 16, 16, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_3x3_stride2_pad1() {
        // Halve spatial via stride 2, with padding.
        check_depthwise_conv2d_f32(1, 16, 32, 32, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_5x5_pad2() {
        check_depthwise_conv2d_f32(2, 4, 10, 10, 5, 5, [1, 1], [2, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_7x7_pad3() {
        // Large kernel like ConvNeXt's 7x7 depthwise.
        check_depthwise_conv2d_f32(2, 24, 14, 14, 7, 7, [1, 1], [3, 3], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_dilated() {
        // Dilation 2 means the effective receptive field is 5x5 but with gaps.
        check_depthwise_conv2d_f32(1, 8, 12, 12, 3, 3, [1, 1], [2, 2], [2, 2], false);
    }

    #[test]
    fn test_conv2d_depthwise_with_bias() {
        check_depthwise_conv2d_f32(2, 8, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], true);
    }

    #[test]
    fn test_conv2d_depthwise_single_channel() {
        // groups = channels_in = 1 is a degenerate depthwise which should
        // still go through this path (it also looks identical to an
        // ungrouped 1-channel conv, but validates the range math).
        check_depthwise_conv2d_f32(1, 1, 5, 5, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_asymmetric_kernel() {
        // Non-square kernel.
        check_depthwise_conv2d_f32(1, 4, 8, 12, 3, 5, [1, 1], [1, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_f64() {
        // Smoke test f64 dispatch through depthwise path.
        let x_data: Vec<f64> = (0..2 * 4 * 5 * 5).map(|i| (i as f64) * 0.1).collect();
        let w_data: Vec<f64> = (0..4 * 3 * 3).map(|i| (i as f64) * 0.01).collect();
        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![2, 4, 5, 5]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![4, 1, 3, 3]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 4);
        let result = conv2d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();

        // Verify against a naive f64 reference for one element (center of channel 2).
        let b = 0usize;
        let c = 2usize;
        let oh = 2usize;
        let ow = 2usize;
        let mut expected = 0.0f64;
        for kh in 0..3 {
            for kw in 0..3 {
                let ih = oh as isize + kh as isize - 1;
                let iw = ow as isize + kw as isize - 1;
                if ih >= 0 && ih < 5 && iw >= 0 && iw < 5 {
                    let x_idx = ((b * 4 + c) * 5 + ih as usize) * 5 + iw as usize;
                    let w_idx = (c * 3 + kh) * 3 + kw;
                    expected += x_data[x_idx] * w_data[w_idx];
                }
            }
        }
        let out_idx = ((b * 4 + c) * 5 + oh) * 5 + ow;
        assert!((out[out_idx] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_depthwise_f16() {
        // Validate f16 depthwise dispatch against a naive f32 reference.
        // Shape check alone would miss indexing/accumulation bugs that still
        // happen to produce the right shape.
        use half::f16;
        let x_data_f32: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let w_data_f32: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let x_data: Vec<f16> = x_data_f32.iter().copied().map(f16::from_f32).collect();
        let w_data: Vec<f16> = w_data_f32.iter().copied().map(f16::from_f32).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 4, 2, 2]));
        let weight = HostTensor::from_data(TensorData::new(w_data, vec![4, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 4);
        let result = conv2d_f16(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 4, 1, 1]);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();

        // Depthwise: out[c] = sum over (kh, kw) of x[c, kh, kw] * w[c, 0, kh, kw].
        // The input per-channel is 4 elements (2x2) and the kernel is 2x2, so
        // there's exactly one output pixel per channel.
        for c in 0..4 {
            let mut expected = 0.0f32;
            for k in 0..4 {
                // c * 4 indexes into channel c's 2x2 plane; both x and w share
                // the same [c, ...] offset because the depthwise weight layout
                // is [c_out, 1, kh, kw].
                expected += x_data_f32[c * 4 + k] * w_data_f32[c * 4 + k];
            }
            let actual = out[c].to_f32();
            assert!(
                (actual - expected).abs() < 1e-2,
                "f16 depthwise mismatch at c={c}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_conv1d_depthwise() {
        // conv1d -> conv3d expansion (kd=kh=1), depthwise on the last axis.
        let channels = 4;
        let in_w = 16;
        let kw = 3;
        let x_data = seeded_vec_f32(channels * in_w, 10);
        let w_data = seeded_vec_f32(channels * kw, 20);
        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![1, channels, in_w]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![channels, 1, kw]));
        let options = ConvOptions::new([1], [1], [1], channels);
        let result = conv1d_f32(x, weight, None, &options);
        let out_w = in_w;
        assert_eq!(result.layout().shape().to_vec(), vec![1, channels, out_w]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        // Naive reference.
        for c in 0..channels {
            for o in 0..out_w {
                let mut expected = 0.0f32;
                for k in 0..kw {
                    let i = o as isize + k as isize - 1;
                    if i >= 0 && i < in_w as isize {
                        expected += x_data[c * in_w + i as usize] * w_data[c * kw + k];
                    }
                }
                let actual = out[c * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "conv1d depthwise mismatch at c={c}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_depthwise_stride_batch_bias() {
        // Depthwise conv1d with batch > 1, stride > 1, padding, and bias.
        // conv1d flows through the same conv3d_depthwise_impl as conv2d
        // because conv1d expands to a 3D shape with kd=kh=1. This test
        // validates the full dispatch path with a non-trivial stride.
        let batch = 3;
        let channels = 6;
        let in_w = 32;
        let kw = 5;
        let stride = 2;
        let pad = 2;
        let out_w = (in_w + 2 * pad - kw) / stride + 1;

        let x_data = seeded_vec_f32(batch * channels * in_w, 30);
        let w_data = seeded_vec_f32(channels * kw, 40);
        let bias_data = seeded_vec_f32(channels, 50);

        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![batch, channels, in_w]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![channels, 1, kw]));
        let bias = HostTensor::from_data(TensorData::new(bias_data.clone(), vec![channels]));
        let options = ConvOptions::new([stride], [pad], [1], channels);
        let result = conv1d_f32(x, weight, Some(bias), &options);
        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels, out_w]
        );
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        // Naive reference.
        for b in 0..batch {
            for c in 0..channels {
                for o in 0..out_w {
                    let mut expected = bias_data[c];
                    for k in 0..kw {
                        let i = (o as isize * stride as isize) + k as isize - pad as isize;
                        if i >= 0 && i < in_w as isize {
                            let x_idx = (b * channels + c) * in_w + i as usize;
                            let w_idx = c * kw + k;
                            expected += x_data[x_idx] * w_data[w_idx];
                        }
                    }
                    let actual = out[(b * channels + c) * out_w + o];
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "conv1d depthwise mismatch at b={b}, c={c}, o={o}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Small-channel conv fast-path tests
    // ========================================================================
    //
    // These tests exercise `conv3d_small_channel_impl` for groups=1 convs
    // with `channels_in <= SMALL_CHANNEL_IN_THRESHOLD`. They cross-check the
    // output against a naive NCHW reference that sums over both input
    // channels and kernel positions.

    #[allow(clippy::too_many_arguments)]
    fn naive_conv2d_f32(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> (Vec<f32>, usize, usize) {
        let out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        let mut out = vec![0.0f32; batch * channels_out * out_h * out_w];
        for b in 0..batch {
            for co in 0..channels_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0f32;
                        for ci in 0..channels_in {
                            for kh in 0..kernel_h {
                                let ih = oh as isize * stride_h as isize
                                    + kh as isize * dilation_h as isize
                                    - pad_h as isize;
                                if ih < 0 || ih >= in_h as isize {
                                    continue;
                                }
                                for kw in 0..kernel_w {
                                    let iw = ow as isize * stride_w as isize
                                        + kw as isize * dilation_w as isize
                                        - pad_w as isize;
                                    if iw < 0 || iw >= in_w as isize {
                                        continue;
                                    }
                                    let x_idx = ((b * channels_in + ci) * in_h + ih as usize)
                                        * in_w
                                        + iw as usize;
                                    let w_idx =
                                        ((co * channels_in + ci) * kernel_h + kh) * kernel_w + kw;
                                    acc += x[x_idx] * w[w_idx];
                                }
                            }
                        }
                        if let Some(bias) = bias {
                            acc += bias[co];
                        }
                        let o_idx = ((b * channels_out + co) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
        (out, out_h, out_w)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_small_channel_conv2d_f32(
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        with_bias: bool,
    ) {
        let x_vec = seeded_vec_f32(batch * channels_in * in_h * in_w, 100);
        let w_vec = seeded_vec_f32(channels_out * channels_in * kernel_h * kernel_w, 200);
        let bias_vec = if with_bias {
            Some(seeded_vec_f32(channels_out, 300))
        } else {
            None
        };

        let (expected, out_h, out_w) = naive_conv2d_f32(
            &x_vec,
            &w_vec,
            bias_vec.as_deref(),
            batch,
            channels_in,
            channels_out,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
        );

        let x = HostTensor::from_data(TensorData::new(x_vec, vec![batch, channels_in, in_h, in_w]));
        let weight = HostTensor::from_data(TensorData::new(
            w_vec,
            vec![channels_out, channels_in, kernel_h, kernel_w],
        ));
        let bias = bias_vec.map(|v| HostTensor::from_data(TensorData::new(v, vec![channels_out])));
        let options = ConvOptions::new(stride, padding, dilation, 1);
        let result = conv2d_f32(x, weight, bias, &options);

        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels_out, out_h, out_w],
            "output shape mismatch"
        );

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out.len(), expected.len());
        for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-3,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn test_conv2d_small_channel_3in_8out_k3x3_pad1() {
        // Sobel-like: 3 input channels, several output channels, 3x3 kernel.
        check_small_channel_conv2d_f32(2, 3, 8, 16, 16, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_3in_3out_k3x3_no_pad() {
        // Channels_in == channels_out == 3, same count but groups=1 so each
        // output channel combines all input channels (not depthwise).
        check_small_channel_conv2d_f32(1, 3, 3, 10, 10, 3, 3, [1, 1], [0, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_3in_16out_k5x5_pad2() {
        check_small_channel_conv2d_f32(2, 3, 16, 12, 12, 5, 5, [1, 1], [2, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_4in_8out_k3x3_stride2() {
        // Threshold channels_in == 4.
        check_small_channel_conv2d_f32(1, 4, 8, 16, 16, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_2in_4out_dilated() {
        check_small_channel_conv2d_f32(1, 2, 4, 12, 12, 3, 3, [1, 1], [2, 2], [2, 2], false);
    }

    #[test]
    fn test_conv2d_small_channel_with_bias() {
        check_small_channel_conv2d_f32(2, 3, 8, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], true);
    }

    #[test]
    fn test_conv2d_small_channel_asymmetric_kernel() {
        // 1x3 and 3x1 kernels are common for separable edge filters.
        check_small_channel_conv2d_f32(1, 3, 6, 16, 16, 1, 3, [1, 1], [0, 1], [1, 1], false);
        check_small_channel_conv2d_f32(1, 3, 6, 16, 16, 3, 1, [1, 1], [1, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_f64() {
        // Smoke test f64 dispatch.
        let x_data: Vec<f64> = (0..2 * 3 * 5 * 5).map(|i| (i as f64) * 0.1).collect();
        let w_data: Vec<f64> = (0..4 * 3 * 3 * 3).map(|i| (i as f64) * 0.01).collect();
        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![2, 3, 5, 5]));
        let weight = HostTensor::from_data(TensorData::new(w_data.clone(), vec![4, 3, 3, 3]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result = conv2d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();

        // Verify against a naive reference for one element (center of channel 2).
        let b = 0usize;
        let co = 2usize;
        let oh = 2usize;
        let ow = 2usize;
        let mut expected = 0.0f64;
        for ci in 0..3 {
            for kh in 0..3 {
                for kw in 0..3 {
                    let ih = oh as isize + kh as isize - 1;
                    let iw = ow as isize + kw as isize - 1;
                    if ih >= 0 && ih < 5 && iw >= 0 && iw < 5 {
                        let x_idx = ((b * 3 + ci) * 5 + ih as usize) * 5 + iw as usize;
                        let w_idx = ((co * 3 + ci) * 3 + kh) * 3 + kw;
                        expected += x_data[x_idx] * w_data[w_idx];
                    }
                }
            }
        }
        let out_idx = ((b * 4 + co) * 5 + oh) * 5 + ow;
        assert!(
            (out[out_idx] - expected).abs() < 1e-10,
            "got {}, expected {expected}",
            out[out_idx]
        );
    }

    #[test]
    fn test_conv2d_small_channel_f16() {
        // Validate f16 dispatch through the small-channel path by running
        // the same shape through the f32 path and comparing element-wise.
        // This catches indexing/accumulation bugs in addition to regressions
        // in the `num_traits::Float` monomorphization for f16.
        use half::f16;
        let x_data_f32: Vec<f32> = (0..3 * 4 * 4).map(|i| i as f32 * 0.1).collect();
        let w_data_f32: Vec<f32> = (0..4 * 3 * 3 * 3).map(|i| i as f32 * 0.01).collect();

        let x_data_f16: Vec<f16> = x_data_f32.iter().copied().map(f16::from_f32).collect();
        let w_data_f16: Vec<f16> = w_data_f32.iter().copied().map(f16::from_f32).collect();

        let x_f16 = HostTensor::from_data(TensorData::new(x_data_f16, vec![1, 3, 4, 4]));
        let weight_f16 = HostTensor::from_data(TensorData::new(w_data_f16, vec![4, 3, 3, 3]));
        let x_f32 = HostTensor::from_data(TensorData::new(x_data_f32, vec![1, 3, 4, 4]));
        let weight_f32 = HostTensor::from_data(TensorData::new(w_data_f32, vec![4, 3, 3, 3]));

        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result_f16 = conv2d_f16(x_f16, weight_f16, None, &options);
        let result_f32 = conv2d_f32(x_f32, weight_f32, None, &options);
        assert_eq!(result_f16.layout().shape().to_vec(), vec![1, 4, 4, 4]);
        assert_eq!(result_f32.layout().shape().to_vec(), vec![1, 4, 4, 4]);

        let out_f16: Vec<f16> = result_f16.into_data().to_vec().unwrap();
        let out_f32: Vec<f32> = result_f32.into_data().to_vec().unwrap();
        assert_eq!(out_f16.len(), out_f32.len());

        // f16 has ~11 bits of mantissa (~0.1% relative precision). With
        // accumulation across `c_in * k_spatial = 27` FMAs the relative
        // error can grow to a few times that. Use a relative tolerance
        // scaled by the expected magnitude with a small absolute floor for
        // values near zero.
        let rel_tol = 3e-3f32;
        let abs_tol = 1e-2f32;
        for (i, (actual, expected)) in out_f16.iter().zip(out_f32.iter()).enumerate() {
            let actual_f32 = actual.to_f32();
            let bound = (expected.abs() * rel_tol).max(abs_tol);
            assert!(
                !actual_f32.is_nan() && (actual_f32 - expected).abs() <= bound,
                "f16 small-channel mismatch at {i}: got {actual_f32}, expected {expected}, bound {bound}"
            );
        }
    }

    #[test]
    fn test_conv2d_small_channel_bias_length_mismatch_panics() {
        // The small-channel impl must panic loudly when bias length does not
        // match channels_out. A silent truncation here would make models with
        // misconfigured bias silently produce wrong output.
        let x = HostTensor::from_data(TensorData::new(vec![0.0f32; 48], vec![1, 3, 4, 4]));
        let weight = HostTensor::from_data(TensorData::new(vec![0.0f32; 108], vec![4, 3, 3, 3]));
        // Bias has only 2 elements but there are 4 output channels.
        let bias = HostTensor::from_data(TensorData::new(vec![0.0f32, 0.0], vec![2]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            conv2d_f32(x, weight, Some(bias), &options)
        }));
        assert!(result.is_err(), "expected panic on bias length mismatch");
    }

    #[test]
    fn test_conv2d_depthwise_bias_length_mismatch_panics() {
        // Mirror of the small-channel bias check for the depthwise path.
        // Depthwise dispatch requires groups == channels_in == channels_out, so
        // we construct a 3->3 depthwise conv and deliberately underspecify the
        // bias length to exercise the assert_eq in conv3d_depthwise_impl.
        let x = HostTensor::from_data(TensorData::new(vec![0.0f32; 48], vec![1, 3, 4, 4]));
        let weight = HostTensor::from_data(TensorData::new(vec![0.0f32; 27], vec![3, 1, 3, 3]));
        // Bias has only 2 elements but there are 3 depthwise channels.
        let bias = HostTensor::from_data(TensorData::new(vec![0.0f32, 0.0], vec![2]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 3); // groups=3 => depthwise
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            conv2d_f32(x, weight, Some(bias), &options)
        }));
        assert!(result.is_err(), "expected panic on bias length mismatch");
    }

    #[test]
    fn test_conv_plane_accumulate_accumulates_into_prefilled() {
        // Pin the documented precondition that `conv_plane_accumulate`
        // accumulates into `out_plane` rather than overwriting it. Both
        // in-tree callers pre-zero before the first call, then call it
        // repeatedly across input channels with the result acting as the
        // running accumulator. A regression that overwrites instead of
        // accumulating would silently drop every input-channel contribution
        // except the last.
        //
        // Test shape: single 2x2 input -> single 2x2 output, 1x1 kernel with
        // weight 1.0. Pre-fill out_plane with [10, 20, 30, 40] so the
        // expected post-call value is x + prefill.
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32]; // 1x1 kernel, weight = 1
        let mut out_plane = vec![10.0f32, 20.0, 30.0, 40.0];

        // 1x1 kernel, no padding: the only kernel position contributes to
        // every output row and column (indices [0, out_h) and [0, out_w)).
        let oh_ranges = [(0usize, 2usize)]; // per kh: (out_start, out_end)
        let ow_ranges = [(0usize, 2usize)]; // per kw: (out_start, out_end)

        super::conv_plane_accumulate::<f32>(
            &mut out_plane,
            &x,
            &w,
            /* kernel_h */ 1,
            /* kernel_w */ 1,
            /* in_w */ 2,
            /* out_w */ 2,
            /* stride_h */ 1,
            /* stride_w */ 1,
            /* pad_h */ 0,
            /* pad_w */ 0,
            /* dilation_h */ 1,
            /* dilation_w */ 1,
            &oh_ranges,
            &ow_ranges,
        );

        // Expected: prefill + (x * 1.0) element-wise.
        assert_eq!(out_plane, vec![11.0f32, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_conv1d_small_channel_3in() {
        // conv1d -> conv3d expansion with groups=1 and 3 input channels.
        let batch = 2;
        let channels_in = 3;
        let channels_out = 5;
        let in_w = 24;
        let kw = 5;
        let stride = 1;
        let pad = 2;
        let out_w = in_w; // same padding

        let x_data = seeded_vec_f32(batch * channels_in * in_w, 400);
        let w_data = seeded_vec_f32(channels_out * channels_in * kw, 500);
        let x = HostTensor::from_data(TensorData::new(
            x_data.clone(),
            vec![batch, channels_in, in_w],
        ));
        let weight = HostTensor::from_data(TensorData::new(
            w_data.clone(),
            vec![channels_out, channels_in, kw],
        ));
        let options = ConvOptions::new([stride], [pad], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels_out, out_w]
        );
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for b in 0..batch {
            for co in 0..channels_out {
                for o in 0..out_w {
                    let mut expected = 0.0f32;
                    for ci in 0..channels_in {
                        for k in 0..kw {
                            let i = o as isize + k as isize - pad as isize;
                            if i >= 0 && i < in_w as isize {
                                let x_idx = (b * channels_in + ci) * in_w + i as usize;
                                let w_idx = (co * channels_in + ci) * kw + k;
                                expected += x_data[x_idx] * w_data[w_idx];
                            }
                        }
                    }
                    let actual = out[(b * channels_out + co) * out_w + o];
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "mismatch at b={b}, co={co}, o={o}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_conv2d_small_channel_single_input() {
        // 1 input channel is allowed (<= threshold) but this also matches
        // many existing conv tests that passed through conv3d_impl before
        // this path existed. Cross-check against the naive reference.
        check_small_channel_conv2d_f32(1, 1, 4, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_threshold_exact() {
        // Verify the thresholds are honored:
        // - channels_in in [1, 4] AND channels_out in [1, 16] triggers the
        //   small-channel path
        // - channels_in == 5 does not
        // - channels_out == 17 does not
        // - groups != 1 does not
        //
        // conv2d options are expanded to 3D as `[0, pad_h, pad_w]` and
        // `[1, stride_h, stride_w]` etc. - the d-axis is always trivial.

        // c_in == 4, c_out == 16: exactly at the thresholds, should trigger.
        assert!(should_use_small_channel_conv(
            &[1, 4, 1, 8, 8],
            &[16, 4, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // c_in == 5: excluded.
        assert!(!should_use_small_channel_conv(
            &[1, 5, 1, 8, 8],
            &[2, 5, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // c_in == 3, c_out == 17: excluded (large output channel count).
        assert!(!should_use_small_channel_conv(
            &[1, 3, 1, 8, 8],
            &[17, 3, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // c_in == 3, c_out == 64 (ImageNet first layer): excluded.
        assert!(!should_use_small_channel_conv(
            &[1, 3, 1, 224, 224],
            &[64, 3, 1, 7, 7],
            &ConvOptions::new([1, 2, 2], [0, 3, 3], [1, 1, 1], 1),
        ));
        // Non-groups=1 is excluded.
        assert!(!should_use_small_channel_conv(
            &[1, 4, 1, 8, 8],
            &[4, 1, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 4),
        ));

        // Cross-check c_in == 5 against naive reference (this goes through
        // the generic conv3d_impl path, not the small-channel path).
        check_small_channel_conv2d_f32(1, 5, 4, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    // The tests below exercise `conv_plane_accumulate_oh_outer`, the variant
    // selected when the output plane exceeds `CONV_PLANE_OH_OUTER_THRESHOLD`
    // (8192 elements). Every test above this point uses shapes where the plane
    // is <= 256 elements and stays on the kh-outer variant; without these,
    // the oh-outer code path ships uncovered.
    //
    // 96x96 = 9216 > 8192, so one element past the threshold. 97x97 = 9409
    // leaves a little more slack in case someone nudges the threshold.

    #[test]
    fn test_conv2d_depthwise_oh_outer_k3x3() {
        // Depthwise path, oh-outer dispatch, 3x3 kernel, stride 1.
        // Plane = 96 * 96 = 9216 elements.
        check_depthwise_conv2d_f32(1, 2, 96, 96, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_oh_outer_k3x3_stride2() {
        // Depthwise, oh-outer, stride 2: exercises the `stride_w != 1`
        // inner branch inside the oh-outer variant. Plane = 96*96 = 9216.
        check_depthwise_conv2d_f32(1, 2, 192, 192, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_oh_outer_k5x1() {
        // Depthwise, oh-outer, 5x1 asymmetric kernel: the exact shape
        // pattern that motivated the loop reorder (Sobel-style separable
        // filter on a large plane). Plane = 97*97 = 9409.
        check_depthwise_conv2d_f32(1, 2, 97, 97, 5, 1, [1, 1], [2, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_oh_outer_k1x5() {
        // Depthwise, oh-outer, 1x5 asymmetric kernel. Plane = 97*97 = 9409.
        check_depthwise_conv2d_f32(1, 2, 97, 97, 1, 5, [1, 1], [0, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k3x3() {
        // Small-channel path, oh-outer dispatch, stride 1.
        // Plane = 96 * 96 = 9216.
        check_small_channel_conv2d_f32(1, 3, 4, 96, 96, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k3x3_stride2() {
        // Small-channel, oh-outer, stride 2: exercises the stride != 1
        // inner branch through small-channel dispatch. Plane = 96*96.
        check_small_channel_conv2d_f32(1, 3, 4, 192, 192, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k5x1_sobel() {
        // Small-channel, oh-outer, 5x1 asymmetric kernel: the shape from
        // the user-reported Sobel regression on RGB, on a plane large
        // enough to cross the threshold. Plane = 97*97 = 9409.
        check_small_channel_conv2d_f32(1, 3, 3, 97, 97, 5, 1, [1, 1], [2, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k1x5_sobel() {
        // Small-channel, oh-outer, 1x5 asymmetric kernel. Plane = 97*97.
        check_small_channel_conv2d_f32(1, 3, 3, 97, 97, 1, 5, [1, 1], [0, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_with_bias_and_dilation() {
        // Small-channel, oh-outer, with bias and dilation > 1.
        // Plane = 96*96 = 9216.
        check_small_channel_conv2d_f32(1, 3, 8, 100, 100, 3, 3, [1, 1], [2, 2], [2, 2], true);
    }

    #[test]
    fn test_conv2d_depthwise_predicate_triggers() {
        // Directly assert `should_use_depthwise_conv` returns true for
        // representative canonical depthwise shapes, and false for shapes
        // that look depthwise-ish but are not. Without this, a future change
        // that tightens the predicate could silently revert depthwise shapes
        // to the generic path while the correctness tests still pass.
        //
        // conv2d options expand to 3D as `[1, stride_h, stride_w]` etc., with
        // the d-axis always trivial. conv1d expands with `kd == kh == 1` and
        // `in_d == in_h == 1`.

        // Canonical depthwise 3x3, groups == c_in == c_out == 32.
        assert!(should_use_depthwise_conv(
            &[4, 32, 1, 56, 56],
            &[32, 1, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 32),
        ));
        // Canonical depthwise 7x7, the ConvNeXt shape Thomas reported.
        assert!(should_use_depthwise_conv(
            &[4, 48, 1, 56, 56],
            &[48, 1, 1, 7, 7],
            &ConvOptions::new([1, 1, 1], [0, 3, 3], [1, 1, 1], 48),
        ));
        // Conv1d depthwise (kh == 1 via the conv1d -> conv3d expansion).
        assert!(should_use_depthwise_conv(
            &[8, 64, 1, 1, 1024],
            &[64, 1, 1, 1, 3],
            &ConvOptions::new([1, 1, 1], [0, 0, 1], [1, 1, 1], 64),
        ));
        // Strided + dilated depthwise.
        assert!(should_use_depthwise_conv(
            &[1, 16, 1, 32, 32],
            &[16, 1, 1, 3, 3],
            &ConvOptions::new([1, 2, 2], [0, 1, 1], [1, 2, 2], 16),
        ));

        // Not depthwise: groups == 1.
        assert!(!should_use_depthwise_conv(
            &[1, 8, 1, 16, 16],
            &[16, 8, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // Not depthwise: channels_per_group > 1 (grouped but not depthwise).
        assert!(!should_use_depthwise_conv(
            &[1, 8, 1, 16, 16],
            &[8, 4, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 2),
        ));
        // Not depthwise: groups == c_in but c_out != c_in (depth multiplier
        // > 1; the canonical depthwise path is restricted to multiplier 1).
        assert!(!should_use_depthwise_conv(
            &[1, 8, 1, 16, 16],
            &[16, 1, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 8),
        ));
        // Not depthwise: pure 3D with kd > 1 (the `d`-axis restriction).
        assert!(!should_use_depthwise_conv(
            &[1, 8, 4, 16, 16],
            &[8, 1, 3, 3, 3],
            &ConvOptions::new([1, 1, 1], [1, 1, 1], [1, 1, 1], 8),
        ));
    }

    #[test]
    fn test_valid_out_range_basics() {
        // Sanity checks for the analytic range helper.
        // 3x3, no pad, stride 1: every k position produces the full range.
        let (s, e) = valid_out_range(0, 1, 0, 1, 5, 3);
        assert_eq!((s, e), (0, 3));
        let (s, e) = valid_out_range(2, 1, 0, 1, 5, 3);
        assert_eq!((s, e), (0, 3));
        // 3x3, pad 1, stride 1: kernel position 0 needs o*1 + 0 >= 1 -> o >= 1.
        let (s, e) = valid_out_range(0, 1, 1, 1, 5, 5);
        assert_eq!((s, e), (1, 5));
        // kernel position 2 needs o*1 + 2 - 1 < 5 -> o < 4.
        let (s, e) = valid_out_range(2, 1, 1, 1, 5, 5);
        assert_eq!((s, e), (0, 4));
        // stride 2: o*2 + 0 in [0, in). With in=5, out=3: o in [0, 3).
        let (s, e) = valid_out_range(0, 1, 0, 2, 5, 3);
        assert_eq!((s, e), (0, 3));
        // dilation 2, pad 2, stride 1: kernel position 0 iw = o*1 + 0 - 2 -> o >= 2.
        let (s, e) = valid_out_range(0, 2, 2, 1, 5, 5);
        assert_eq!((s, e), (2, 5));
    }
