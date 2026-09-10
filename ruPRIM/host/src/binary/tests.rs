    use super::*;
    use alloc::vec;
    use ruda_core::tensor::data::{TensorData, Tolerance};

    // ===================
    // F16 tests
    // ===================

    #[test]
    fn test_binary_add_f16() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = HostTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = HostTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<f16> = vec![6.0, 8.0, 10.0, 12.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    #[test]
    fn test_binary_mul_f16() {
        let a_vals: Vec<f16> = vec![2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = HostTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = HostTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let expected: Vec<f16> = vec![2.0, 6.0, 12.0, 20.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    #[test]
    fn test_binary_f16_transposed() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = HostTensor::from_data(TensorData::new(a_vals, vec![2, 2])).transpose(0, 1);
        let b = HostTensor::from_data(TensorData::new(b_vals, vec![2, 2])).transpose(0, 1);

        // a_t = [[1,3], [2,4]], b_t = [[10,30], [20,40]]
        // result = [[11,33], [22,44]]
        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<f16> = vec![11.0, 33.0, 22.0, 44.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_scalar_f16() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0].into_iter().map(f16::from_f32).collect();
        let a = HostTensor::from_data(TensorData::new(a_vals, vec![3]));

        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let expected: Vec<f16> = vec![11.0, 12.0, 13.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    // ===================
    // BF16 tests
    // ===================

    #[test]
    fn test_binary_add_bf16() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = HostTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = HostTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<bf16> = vec![6.0, 8.0, 10.0, 12.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_binary_mul_bf16() {
        let a_vals: Vec<bf16> = vec![2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = HostTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = HostTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let expected: Vec<bf16> = vec![2.0, 6.0, 12.0, 20.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_binary_bf16_transposed() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = HostTensor::from_data(TensorData::new(a_vals, vec![2, 2])).transpose(0, 1);
        let b = HostTensor::from_data(TensorData::new(b_vals, vec![2, 2])).transpose(0, 1);

        // a_t = [[1,3], [2,4]], b_t = [[10,30], [20,40]]
        // result = [[11,33], [22,44]]
        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<bf16> = vec![11.0, 33.0, 22.0, 44.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.5)),
        );
    }

    #[test]
    fn test_scalar_bf16() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let a = HostTensor::from_data(TensorData::new(a_vals, vec![3]));

        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let expected: Vec<bf16> = vec![11.0, 12.0, 13.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_scalar_f16_non_representable() {
        // 0.1 is not exactly representable in f16; verify the scalar is rounded
        // to f16 precision before the op (matching dtype semantics).
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0].into_iter().map(f16::from_f32).collect();
        let a = HostTensor::from_data(TensorData::new(a_vals, vec![3]));

        let s_f16 = f16::from_f32(0.1);
        let result = scalar_op(a, 0.1, |x, y| x * y, |x, y| x * y);
        let expected: Vec<f16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| f16::from_f32(f16::from_f32(v).to_f32() * s_f16.to_f32()))
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(f16::from_f32(0.001)),
        );
    }

    #[test]
    fn test_scalar_bf16_non_representable() {
        // 1.1 is not exactly representable in bf16; verify dtype rounding.
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let a = HostTensor::from_data(TensorData::new(a_vals, vec![3]));

        let s_bf16 = bf16::from_f32(1.1);
        let result = scalar_op(a, 1.1, |x, y| x * y, |x, y| x * y);
        let expected: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| bf16::from_f32(bf16::from_f32(v).to_f32() * s_bf16.to_f32()))
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(bf16::from_f32(0.01)),
        );
    }

    // ============================================================================
    // Broadcast binary-op fast paths
    // ============================================================================

    /// Shared-row broadcast: 1-D gamma reshaped + expanded, with the
    /// size-1 outer dim exemption in play.
    #[test]
    fn test_binary_shared_row_broadcast_f32() {
        let a = HostTensor::from_data(TensorData::new(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![1, 3, 4],
        ));
        // 1D gamma broadcast over rows.
        let gamma =
            HostTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]));
        let gamma_unsqueezed = gamma.reshape(Shape::from(vec![1, 1, 4]));
        let result = binary_op(
            a,
            gamma_unsqueezed,
            |a, b| a * b,
            |a, b| a * b,
            Some(BinaryOp::Mul),
        );
        let data = result.into_data();
        let expected = vec![
            10.0f32, 40.0, 90.0, 160.0, 50.0, 120.0, 210.0, 320.0, 90.0, 200.0, 330.0, 480.0,
        ];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    /// Per-row scalar broadcast: `[1, 3, 4] - mean_dim(-1)` shape,
    /// rhs expands to strides `[3, 1, 0]`.
    #[test]
    fn test_binary_per_row_scalar_broadcast_f32() {
        let a = HostTensor::from_data(TensorData::new(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
            vec![1, 3, 4],
        ));
        // Scalars shaped like `mean_dim(-1)` output.
        let mean = HostTensor::from_data(TensorData::new(vec![2.5f32, 25.0, 250.0], vec![1, 3, 1]));
        let result = binary_op(a, mean, |a, b| a - b, |a, b| a - b, Some(BinaryOp::Sub));
        let data = result.into_data();
        let expected = vec![
            -1.5f32, -0.5, 0.5, 1.5, -15.0, -5.0, 5.0, 15.0, -150.0, -50.0, 50.0, 150.0,
        ];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    /// Non-contig lhs + shared-row broadcast: must materialize lhs
    /// contiguous then dispatch to the shared-row kernel.
    #[test]
    fn test_binary_permuted_lhs_broadcast_rhs() {
        let a = HostTensor::from_data(TensorData::new(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 3, 4],
        ));
        let a_permuted = a.transpose(1, 2); // shape [2, 4, 3]
        let gamma = HostTensor::from_data(TensorData::new(vec![1.0f32, 10.0, 100.0], vec![3]));
        let gamma_expanded = gamma.reshape(Shape::from(vec![1, 1, 3]));

        let result = binary_op(
            a_permuted,
            gamma_expanded,
            |a, b| a * b,
            |a, b| a * b,
            Some(BinaryOp::Mul),
        );

        // Compare against a naive reference computed on the permuted
        // values.
        let reference: Vec<f32> = {
            // original[b, r, c] = b*12 + r*4 + c
            // permuted[b, c, r] = original[b, r, c]
            // result[b, c, r] = permuted[b, c, r] * gamma_row[r]
            let gamma_vals = [1.0f32, 10.0, 100.0];
            let mut out = Vec::with_capacity(24);
            for b in 0..2 {
                for c in 0..4 {
                    for r in 0..3 {
                        let orig_val = (b * 12 + r * 4 + c) as f32;
                        out.push(orig_val * gamma_vals[r]);
                    }
                }
            }
            out
        };

        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), reference.as_slice());
    }

    /// Non-contig lhs + per-row-scalar broadcast-sub.
    #[test]
    fn test_binary_permuted_lhs_per_row_scalar_sub() {
        let a = HostTensor::from_data(TensorData::new(
            (1..=24).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 3, 4],
        ));
        let a_permuted = a.transpose(1, 2); // shape [2, 4, 3]

        // Per-row scalar in the permuted layout: shape [2, 4, 1].
        let mean = HostTensor::from_data(TensorData::new(
            (0..8).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 4, 1],
        ));

        let result = binary_op(
            a_permuted,
            mean,
            |a, b| a - b,
            |a, b| a - b,
            Some(BinaryOp::Sub),
        );

        // Reference computation.
        let reference: Vec<f32> = {
            let mut out = Vec::with_capacity(24);
            for b in 0..2 {
                for c in 0..4 {
                    let mean_val = (b * 4 + c) as f32;
                    for r in 0..3 {
                        let orig_val = (b * 12 + r * 4 + c + 1) as f32;
                        out.push(orig_val - mean_val);
                    }
                }
            }
            out
        };

        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), reference.as_slice());
    }

    /// Exercise every `(op, pattern)` combination of the broadcast fast path:
    /// Add/Sub/Mul/Div crossed with SharedRow and PerRowScalar. The existing
    /// targeted tests only cover a subset, so a sign error in
    /// `div_shared_row_inplace_f32` or `add_per_row_scalar` would ship green.
    #[test]
    fn test_binary_broadcast_all_ops_and_patterns_f32() {
        fn build_shared() -> (HostTensor, HostTensor) {
            let a = HostTensor::from_data(TensorData::new(
                vec![4.0f32, 8.0, 12.0, 20.0, 30.0, 60.0],
                vec![2, 3],
            ));
            let b = HostTensor::from_data(TensorData::new(vec![2.0f32, 4.0, 3.0], vec![3]))
                .reshape(Shape::from(vec![1, 3]));
            (a, b)
        }
        fn build_perrow() -> (HostTensor, HostTensor) {
            let a = HostTensor::from_data(TensorData::new(
                vec![4.0f32, 8.0, 12.0, 20.0, 30.0, 60.0],
                vec![2, 3],
            ));
            let b = HostTensor::from_data(TensorData::new(vec![2.0f32, 5.0], vec![2, 1]));
            (a, b)
        }

        let run = |name: &str,
                   build: fn() -> (HostTensor, HostTensor),
                   simd_op: BinaryOp,
                   op_fn: fn(f32, f32) -> f32,
                   expected: &[f32]| {
            let (a, b) = build();
            let result = binary_op(
                a,
                b,
                op_fn,
                |x: f64, y: f64| op_fn(x as f32, y as f32) as f64,
                Some(simd_op),
            );
            let data = result.into_data();
            assert_eq!(
                data.as_slice::<f32>().unwrap(),
                expected,
                "case {name} produced wrong values"
            );
        };

        // SharedRow expected: lhs[i][j] OP rhs[j]
        run(
            "shared_add",
            build_shared,
            BinaryOp::Add,
            |a, b| a + b,
            &[6.0, 12.0, 15.0, 22.0, 34.0, 63.0],
        );
        run(
            "shared_sub",
            build_shared,
            BinaryOp::Sub,
            |a, b| a - b,
            &[2.0, 4.0, 9.0, 18.0, 26.0, 57.0],
        );
        run(
            "shared_mul",
            build_shared,
            BinaryOp::Mul,
            |a, b| a * b,
            &[8.0, 32.0, 36.0, 40.0, 120.0, 180.0],
        );
        run(
            "shared_div",
            build_shared,
            BinaryOp::Div,
            |a, b| a / b,
            &[2.0, 2.0, 4.0, 10.0, 7.5, 20.0],
        );
        // PerRowScalar expected: lhs[i][j] OP rhs[i]
        run(
            "perrow_add",
            build_perrow,
            BinaryOp::Add,
            |a, b| a + b,
            &[6.0, 10.0, 14.0, 25.0, 35.0, 65.0],
        );
        run(
            "perrow_sub",
            build_perrow,
            BinaryOp::Sub,
            |a, b| a - b,
            &[2.0, 6.0, 10.0, 15.0, 25.0, 55.0],
        );
        run(
            "perrow_mul",
            build_perrow,
            BinaryOp::Mul,
            |a, b| a * b,
            &[8.0, 16.0, 24.0, 100.0, 150.0, 300.0],
        );
        run(
            "perrow_div",
            build_perrow,
            BinaryOp::Div,
            |a, b| a / b,
            &[2.0, 4.0, 6.0, 4.0, 6.0, 12.0],
        );
    }

    /// Non-unique lhs: `apply_broadcast_pattern_f32` takes the allocating
    /// branch instead of writing in place. Clone the lhs so its Arc refcount
    /// is > 1, then run a broadcast op and verify the result matches the
    /// unique path. Without this test, a regression in the allocating branch
    /// would only fire on shared-lhs call sites which are rare in bench code.
    #[test]
    fn test_binary_broadcast_non_unique_lhs_f32() {
        let a = HostTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ));
        let _keep_alive = a.clone(); // bump Arc refcount so lhs is shared
        let b = HostTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0], vec![3]))
            .reshape(Shape::from(vec![1, 3]));
        let result = binary_op(a, b, |a, b| a + b, |a, b| a + b, Some(BinaryOp::Add));
        let data = result.into_data();
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[11.0f32, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    /// Fully-broadcast scalar: rhs strides all 0, PerRowScalar with
    /// empty outer walk, applies one scalar across the whole dst.
    #[test]
    fn test_binary_fully_broadcast_scalar_f32() {
        let a = HostTensor::from_data(TensorData::new(
            (0..12).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 2, 3],
        ));
        // 1-element tensor expanded to lhs's full shape. All strides
        // become 0.
        let scalar_tensor = HostTensor::from_data(TensorData::new(vec![100.0f32], [1]));
        let scalar_expanded = crate::expand::expand(scalar_tensor, Shape::from(vec![2, 2, 3]));
        // Sanity check: every stride is 0.
        assert!(scalar_expanded.layout().strides().iter().all(|&s| s == 0));

        let result = binary_op(
            a,
            scalar_expanded,
            |a, b| a + b,
            |a, b| a + b,
            Some(BinaryOp::Add),
        );

        let expected: Vec<f32> = (0..12).map(|i| i as f32 + 100.0).collect();
        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }
