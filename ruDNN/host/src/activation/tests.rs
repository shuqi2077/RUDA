    use alloc::vec;
    use ruda_core::tensor::{DType, TensorMetadata, data::{TensorData, Tolerance}};
    use half::{bf16, f16};
    use num_traits::Float;

    use ruda_core::tensor::host::HostTensor;

    // ============================================================================
    // Reference implementations (per-row, last-axis).
    //
    // These mirror the contract the fused kernel commits to: stable softmax via
    // (x - max), layer_norm via (x - mean) * inv(sqrt(var + eps)) with optional
    // affine. Written in plain Rust over f32/f64 slices so the tests avoid any
    // tensor-library dependency.
    // ============================================================================

    fn softmax_row<T: Float>(row_in: &[T], row_out: &mut [T]) {
        let max = row_in
            .iter()
            .copied()
            .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });
        let mut sum = T::zero();
        for (i, &x) in row_in.iter().enumerate() {
            let e = (x - max).exp();
            row_out[i] = e;
            sum = sum + e;
        }
        for v in row_out.iter_mut() {
            *v = *v / sum;
        }
    }

    fn softmax_last_ref<T: Float>(data: &[T], row_len: usize) -> Vec<T> {
        let mut out = vec![T::zero(); data.len()];
        for (i, o) in data.chunks(row_len).zip(out.chunks_mut(row_len)) {
            softmax_row(i, o);
        }
        out
    }

    fn layer_norm_row<T: Float>(
        row_in: &[T],
        gamma: &[T],
        beta: Option<&[T]>,
        eps: T,
        row_out: &mut [T],
    ) {
        let n = T::from(row_in.len()).unwrap();
        let mean = row_in.iter().copied().fold(T::zero(), |a, b| a + b) / n;
        let var = row_in
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |a, b| a + b)
            / n;
        let inv_std = T::one() / (var + eps).sqrt();
        for (i, &x) in row_in.iter().enumerate() {
            let normed = (x - mean) * inv_std;
            let scaled = normed * gamma[i];
            row_out[i] = match beta {
                Some(b) => scaled + b[i],
                None => scaled,
            };
        }
    }

    fn layer_norm_last_ref<T: Float>(
        data: &[T],
        gamma: &[T],
        beta: Option<&[T]>,
        eps: T,
        row_len: usize,
    ) -> Vec<T> {
        let mut out = vec![T::zero(); data.len()];
        for (i, o) in data.chunks(row_len).zip(out.chunks_mut(row_len)) {
            layer_norm_row(i, gamma, beta, eps, o);
        }
        out
    }

    // ============================================================================
    // Helpers: FlexTensor constructors for typed inputs.
    // ============================================================================

    fn flex_f32(data: Vec<f32>, shape: &[usize]) -> HostTensor {
        HostTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    fn flex_f64(data: Vec<f64>, shape: &[usize]) -> HostTensor {
        HostTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    fn flex_half<T: ruda_core::tensor::element::Element>(data: Vec<T>, shape: &[usize]) -> HostTensor {
        HostTensor::from_data(TensorData::new(data, shape.to_vec()))
    }

    // ============================================================================
    // layer_norm tests
    // ============================================================================

    #[test]
    fn test_layer_norm_2d_with_beta() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let gamma = flex_f32(vec![1.0; 4], &[4]);
        let beta = flex_f32(vec![0.0; 4], &[4]);
        let out = crate::activation::layer_norm(t, gamma, Some(beta), 1e-5);

        let expected: Vec<f32> = vec![
            -1.3416408, -0.4472136, 0.4472136, 1.3416408, -1.3416408, -0.4472136, 0.4472136,
            1.3416408,
        ];
        out.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_layer_norm_with_affine() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![2.0, 0.5, 1.0, 3.0], &[4]);
        let beta = flex_f32(vec![1.0, -1.0, 0.0, 2.0], &[4]);
        let out = crate::activation::layer_norm(t, gamma, Some(beta), 1e-5);

        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        // affine: [-1.6833, -1.2236, 0.4472, 6.0249]
        out.into_data().assert_approx_eq::<f32>(
            &TensorData::new(vec![-1.6833, -1.2236, 0.4472, 6.0249], vec![1, 4]),
            Tolerance::absolute(1e-3),
        );
    }

    #[test]
    fn test_layer_norm_no_beta() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0; 4], &[4]);
        let out = crate::activation::layer_norm(t, gamma, None, 1e-5);

        out.into_data().assert_approx_eq::<f32>(
            &TensorData::new(
                vec![-1.3416408, -0.4472136, 0.4472136, 1.3416408],
                vec![1, 4],
            ),
            Tolerance::absolute(1e-4),
        );
    }

    // ============================================================================
    // softmax SIMD / rayon boundary tests
    // ============================================================================

    #[test]
    fn test_softmax_simd_body_row() {
        // Row length 32 ensures the SIMD body runs on every supported target:
        // NEON (lanes=4), AVX2 (lanes=8), AVX-512 (lanes=16), SIMD128 (lanes=4).
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let expected = softmax_last_ref(&data, 32);
        let fused = crate::activation::softmax(flex_f32(data, &[1, 32]), 1);
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![1, 32]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_softmax_multi_chunk_rayon() {
        // 100 rows > ROWS_PER_TASK (64) triggers the rayon par_chunks path.
        let data: Vec<f32> = (0..100 * 16).map(|i| ((i % 17) as f32) * 0.05).collect();
        let expected = softmax_last_ref(&data, 16);
        let fused = crate::activation::softmax(flex_f32(data, &[100, 16]), 1);
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![100, 16]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_softmax_f64() {
        // Exercises softmax_last_dtype! + softmax_row_native f64 path.
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = softmax_last_ref(&data, 4);
        let fused = crate::activation::softmax(flex_f64(data, &[2, 4]), 1);
        fused.into_data().assert_approx_eq::<f64>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-10),
        );
    }

    #[test]
    fn test_softmax_f16() {
        // Exercises softmax_last_dtype! + softmax_row_half f16 path.
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
        let data: Vec<f16> = source.iter().map(|&x| f16::from_f32(x)).collect();
        let expected = softmax_last_ref(&data, 4);
        let fused = crate::activation::softmax(flex_half(data, &[2, 4]), 1);
        fused.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-2),
        );
    }

    #[test]
    fn test_softmax_bf16() {
        // Exercises softmax_last_dtype! + softmax_row_half bf16 path.
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5];
        let data: Vec<bf16> = source.iter().map(|&x| bf16::from_f32(x)).collect();
        let expected = softmax_last_ref(&data, 4);
        let fused = crate::activation::softmax(flex_half(data, &[2, 4]), 1);
        fused.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(5e-2),
        );
    }

    #[test]
    fn test_layer_norm_multi_chunk_rayon() {
        // 128 rows > ROWS_PER_TASK (64) triggers the rayon path.
        let data: Vec<f32> = (0..128 * 16).map(|i| ((i % 19) as f32) * 0.03).collect();
        let gamma_data: Vec<f32> = vec![1.0; 16];
        let beta_data: Vec<f32> = vec![0.0; 16];
        let expected = layer_norm_last_ref(&data, &gamma_data, Some(&beta_data), 1e-5f32, 16);
        let fused = crate::activation::layer_norm(
            flex_f32(data, &[128, 16]),
            flex_f32(gamma_data, &[16]),
            Some(flex_f32(beta_data, &[16])),
            1e-5,
        );
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![128, 16]),
            Tolerance::absolute(1e-4),
        );
    }

    #[test]
    fn test_softmax_empty_last_dim_returns_input() {
        // shape [2, 0]: empty last dim should round-trip unchanged instead
        // of producing NaN via 0/0.
        let t = flex_f32(Vec::<f32>::new(), &[2, 0]);
        let result = crate::activation::softmax(t, 1);
        assert_eq!(result.shape().as_slice(), &[2, 0]);
    }

    #[test]
    fn test_layer_norm_empty_last_dim_returns_input() {
        let t = flex_f32(Vec::<f32>::new(), &[3, 0]);
        let gamma = flex_f32(Vec::<f32>::new(), &[0]);
        let beta = flex_f32(Vec::<f32>::new(), &[0]);
        let result = crate::activation::layer_norm(t, gamma, Some(beta), 1e-5);
        assert_eq!(result.shape().as_slice(), &[3, 0]);
    }

    #[test]
    #[should_panic(expected = "gamma must be a 1-D tensor")]
    fn test_layer_norm_gamma_length_mismatch_panics() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0, 1.0, 1.0], &[3]);
        let _ = crate::activation::layer_norm(t, gamma, None, 1e-5);
    }

    #[test]
    #[should_panic(expected = "beta must be a 1-D tensor")]
    fn test_layer_norm_beta_length_mismatch_panics() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0, 1.0, 1.0, 1.0], &[4]);
        let beta = flex_f32(vec![0.0, 0.0, 0.0], &[3]);
        let _ = crate::activation::layer_norm(t, gamma, Some(beta), 1e-5);
    }

    #[test]
    #[should_panic(expected = "gamma must be a 1-D tensor")]
    fn test_layer_norm_gamma_rank_mismatch_panics() {
        // gamma [2, 4] has matching last-dim but rank 2, so the old last-dim
        // check alone would have accepted it and then indexed wrong storage.
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f32(vec![1.0; 8], &[2, 4]);
        let _ = crate::activation::layer_norm(t, gamma, None, 1e-5);
    }

    // Row length 17 leaves exactly one scalar-tail element after every common
    // SIMD width (NEON/SSE f32x4: body=16, tail=1; AVX2 f32x8: body=16, tail=1;
    // AVX-512 f32x16: body=16, tail=1). Row lengths that divide evenly by the
    // SIMD width skip the tail branch entirely, so a bug in the scalar tail
    // kernel would sail past CI without a test like this.
    #[test]
    fn test_softmax_simd_body_plus_scalar_tail() {
        let data: Vec<f32> = (0..34).map(|i| (i as f32 * 0.137) - 2.3).collect();
        let expected = softmax_last_ref(&data, 17);
        let fused = crate::activation::softmax(flex_f32(data, &[2, 17]), 1);
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![2, 17]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_layer_norm_simd_body_plus_scalar_tail() {
        let data: Vec<f32> = (0..34).map(|i| (i as f32 * 0.137) - 2.3).collect();
        let gamma_data: Vec<f32> = (0..17).map(|i| 1.0 + i as f32 * 0.05).collect();
        let beta_data: Vec<f32> = (0..17).map(|i| i as f32 * 0.01).collect();
        let expected = layer_norm_last_ref(&data, &gamma_data, Some(&beta_data), 1e-5f32, 17);
        let fused = crate::activation::layer_norm(
            flex_f32(data, &[2, 17]),
            flex_f32(gamma_data, &[17]),
            Some(flex_f32(beta_data, &[17])),
            1e-5,
        );
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected, vec![2, 17]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_layer_norm_f64_with_beta_multi_chunk() {
        // 80 rows > ROWS_PER_TASK (64) exercises the rayon multi-chunk f64 path.
        let d_model = 16;
        let n_rows = 80;
        let data: Vec<f64> = (0..n_rows * d_model)
            .map(|i| ((i % 13) as f64) * 0.07 - 0.3)
            .collect();
        let gamma_data: Vec<f64> = vec![0.9; d_model];
        let beta_data: Vec<f64> = vec![0.05; d_model];
        let eps = 1e-5f64;
        let expected = layer_norm_last_ref(&data, &gamma_data, Some(&beta_data), eps, d_model);
        let fused = crate::activation::layer_norm(
            flex_f64(data, &[n_rows, d_model]),
            flex_f64(gamma_data, &[d_model]),
            Some(flex_f64(beta_data, &[d_model])),
            eps,
        );
        fused.into_data().assert_approx_eq::<f64>(
            &TensorData::new(expected, vec![n_rows, d_model]),
            Tolerance::absolute(1e-10),
        );
    }

    #[test]
    fn test_layer_norm_f64_no_beta() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 1.5, -0.5];
        let gamma_data: Vec<f64> = vec![1.0; 4];
        let eps = 1e-5f64;
        let expected = layer_norm_last_ref(&data, &gamma_data, None, eps, 4);
        let fused = crate::activation::layer_norm(
            flex_f64(data, &[2, 4]),
            flex_f64(gamma_data, &[4]),
            None,
            eps,
        );
        fused.into_data().assert_approx_eq::<f64>(
            &TensorData::new(expected, vec![2, 4]),
            Tolerance::absolute(1e-10),
        );
    }

    // Shared body for f16/bf16 layer_norm tests. The fused half-precision
    // kernel casts to f32 internally, so the reference is computed in f32
    // and compared back against the half output with an f32 tolerance.
    fn check_layer_norm_half_precision<E>(from_f32: fn(f32) -> E, dtype: DType)
    where
        E: ruda_core::tensor::element::Element + Float,
    {
        let rows_f32: [f32; 12] = [
            1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0, 0.5, -0.5, 1.5, -1.5,
        ];
        let gamma_f32: [f32; 4] = [1.0, 0.5, 1.5, 1.0];
        let beta_f32: [f32; 4] = [0.1, -0.1, 0.0, 0.2];
        let eps = 1e-5f32;

        let expected_f32 = layer_norm_last_ref(&rows_f32, &gamma_f32, Some(&beta_f32), eps, 4);

        let data: Vec<E> = rows_f32.iter().map(|&x| from_f32(x)).collect();
        let gamma_data: Vec<E> = gamma_f32.iter().map(|&x| from_f32(x)).collect();
        let beta_data: Vec<E> = beta_f32.iter().map(|&x| from_f32(x)).collect();
        assert_eq!(E::dtype(), dtype);

        let fused = crate::activation::layer_norm(
            flex_half(data, &[3, 4]),
            flex_half(gamma_data, &[4]),
            Some(flex_half(beta_data, &[4])),
            eps as f64,
        );
        fused.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected_f32, vec![3, 4]),
            Tolerance::absolute(3e-2),
        );
    }

    #[test]
    fn test_layer_norm_f16_via_f32_cast() {
        check_layer_norm_half_precision::<f16>(f16::from_f32, DType::F16);
    }

    #[test]
    fn test_layer_norm_bf16_via_f32_cast() {
        check_layer_norm_half_precision::<bf16>(bf16::from_f32, DType::BF16);
    }

    #[test]
    #[should_panic(expected = "softmax dim")]
    fn test_softmax_dim_out_of_range_panics() {
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = crate::activation::softmax(t, 2);
    }

    #[test]
    #[should_panic(expected = "gamma dtype")]
    fn test_layer_norm_gamma_dtype_mismatch_panics() {
        // Input f32, gamma f64: layer_norm rejects the mismatch up front
        // rather than panicking later inside the storage-typed access.
        let t = flex_f32(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let gamma = flex_f64(vec![1.0; 4], &[4]);
        let _ = crate::activation::layer_norm(t, gamma, None, 1e-5);
    }
