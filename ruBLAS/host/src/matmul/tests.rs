    use alloc::{vec, vec::Vec};
    use ruda_core::tensor::data::TensorData;
    use half::{bf16, f16};

    use ruda_core::tensor::host::HostTensor;
    use super::{int_matmul, matmul};

    #[test]
    fn test_matmul_f64() {
        let lhs = HostTensor::from_data(TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], [2, 2]));
        let rhs = HostTensor::from_data(TensorData::new(vec![5.0f64, 6.0, 7.0, 8.0], [2, 2]));

        let result = matmul(lhs, rhs);
        let values: Vec<f64> = result.into_data().to_vec().unwrap();

        assert_eq!(values, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_f16() {
        let lhs_vals: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect();
        let rhs_vals: Vec<f16> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect();

        let lhs = HostTensor::from_data(TensorData::new(lhs_vals, [2, 2]));
        let rhs = HostTensor::from_data(TensorData::new(rhs_vals, [2, 2]));

        let result = matmul(lhs, rhs);
        let values: Vec<f16> = result.into_data().to_vec().unwrap();

        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (a, e) in values.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.1, "f16 matmul mismatch");
        }
    }

    #[test]
    fn test_matmul_bf16() {
        let lhs_vals: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect();
        let rhs_vals: Vec<bf16> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect();

        let lhs = HostTensor::from_data(TensorData::new(lhs_vals, [2, 2]));
        let rhs = HostTensor::from_data(TensorData::new(rhs_vals, [2, 2]));

        let result = matmul(lhs, rhs);
        let values: Vec<bf16> = result.into_data().to_vec().unwrap();

        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (a, e) in values.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5, "bf16 matmul mismatch");
        }
    }

    #[test]
    fn test_matmul_batched_transposed_f64() {
        // Non-contiguous (swap_dims) batched matmul on the F64 dtype path.
        let q_data = TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
        let k_data = TensorData::new(vec![1.0f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], [2, 2, 2]);

        let q = HostTensor::from_data(q_data.clone());
        let k = HostTensor::from_data(k_data.clone());
        let k_t = k.transpose(1, 2);
        let result = matmul(q, k_t);

        let q2 = HostTensor::from_data(q_data);
        let k2 = HostTensor::from_data(k_data)
            .transpose(1, 2)
            .to_contiguous();
        let expected = matmul(q2, k2);

        let values: Vec<f64> = result.into_data().to_vec().unwrap();
        let expected: Vec<f64> = expected.into_data().to_vec().unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn test_matmul_batched_transposed_f16() {
        // Non-contiguous (swap_dims) batched matmul on the F16 dtype path.
        let f = f16::from_f32;
        let q_data = TensorData::new(
            vec![
                f(1.0),
                f(2.0),
                f(3.0),
                f(4.0),
                f(5.0),
                f(6.0),
                f(7.0),
                f(8.0),
            ],
            [2, 2, 2],
        );
        let k_data = TensorData::new(
            vec![
                f(1.0),
                f(0.0),
                f(0.0),
                f(1.0),
                f(2.0),
                f(0.0),
                f(0.0),
                f(2.0),
            ],
            [2, 2, 2],
        );

        let q = HostTensor::from_data(q_data.clone());
        let k = HostTensor::from_data(k_data.clone());
        let k_t = k.transpose(1, 2);
        let result = matmul(q, k_t);

        let q2 = HostTensor::from_data(q_data);
        let k2 = HostTensor::from_data(k_data)
            .transpose(1, 2)
            .to_contiguous();
        let expected = matmul(q2, k2);

        let values: Vec<f16> = result.into_data().to_vec().unwrap();
        let expected: Vec<f16> = expected.into_data().to_vec().unwrap();
        assert_eq!(values, expected);
    }

    fn reference_i64(lhs: &[i64], rhs: &[i64], m: usize, k: usize, n: usize) -> Vec<i64> {
        let mut output = vec![0i64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i64;
                for contraction in 0..k {
                    sum = sum.wrapping_add(
                        lhs[i * k + contraction].wrapping_mul(rhs[contraction * n + j]),
                    );
                }
                output[i * n + j] = sum;
            }
        }
        output
    }

    #[test]
    fn test_int_matmul_i64_rectangular_wrapping() {
        let lhs_values = vec![i64::MAX, i64::MIN, -1, 0, 7, 2, -3, 11, i64::MAX, 5];
        let rhs_values: Vec<i64> = (0..35)
            .map(|i| (i as i64).wrapping_mul(i64::MAX / 3).wrapping_sub(17))
            .collect();
        let expected = reference_i64(&lhs_values, &rhs_values, 2, 5, 7);
        let lhs = HostTensor::from_data(TensorData::new(lhs_values, [2, 5]));
        let rhs = HostTensor::from_data(TensorData::new(rhs_values, [5, 7]));
        let result = int_matmul(lhs, rhs);
        assert_eq!(result.layout().shape().as_slice(), &[2, 7]);
        assert_eq!(result.storage::<i64>(), expected);
    }

    #[test]
    fn test_int_matmul_i64_broadcast_batches() {
        let lhs_values: Vec<i64> = (0..30).map(|i| i as i64 - 13).collect();
        let rhs_values: Vec<i64> = (0..140).map(|i| (i as i64 % 19) - 9).collect();
        let mut expected = Vec::new();
        for lhs_batch in 0..2 {
            for rhs_batch in 0..4 {
                expected.extend(reference_i64(
                    &lhs_values[lhs_batch * 15..(lhs_batch + 1) * 15],
                    &rhs_values[rhs_batch * 35..(rhs_batch + 1) * 35],
                    3,
                    5,
                    7,
                ));
            }
        }
        let lhs = HostTensor::from_data(TensorData::new(lhs_values, [2, 1, 3, 5]));
        let rhs = HostTensor::from_data(TensorData::new(rhs_values, [1, 4, 5, 7]));
        let result = int_matmul(lhs, rhs);
        assert_eq!(result.layout().shape().as_slice(), &[2, 4, 3, 7]);
        assert_eq!(result.storage::<i64>(), expected);
    }

    #[test]
    fn test_int_matmul_i64_transposed_and_offset_views() {
        let lhs = HostTensor::from_data(TensorData::new(vec![1i64, 2, 3, 4, 5, 6], [3, 2]))
            .transpose(0, 1);
        let rhs = HostTensor::from_data(TensorData::new((1i64..=16).collect(), [4, 4]))
            .narrow(0, 1, 3)
            .narrow(1, 1, 2);
        let result = int_matmul(lhs, rhs);
        assert_eq!(result.layout().shape().as_slice(), &[2, 2]);
        assert_eq!(result.storage::<i64>(), &[106, 115, 136, 148]);
    }

    macro_rules! integer_empty_matrix_tests {
        ($name:ident, $ty:ty) => {
            #[test]
            fn $name() {
                // Empty rows, columns and contraction axes, in both 2D and
                // batched calls. The zero-contraction result contains zeros.
                for batch in [None, Some(2)] {
                    for (m, k, n) in [(0, 3, 4), (2, 3, 0), (2, 0, 4), (0, 0, 0)] {
                        let mut lhs_shape = Vec::new();
                        let mut rhs_shape = Vec::new();
                        let mut output_shape = Vec::new();
                        if let Some(b) = batch {
                            lhs_shape.push(b);
                            rhs_shape.push(b);
                            output_shape.push(b);
                        }
                        lhs_shape.extend([m, k]);
                        rhs_shape.extend([k, n]);
                        output_shape.extend([m, n]);
                        let batches = batch.unwrap_or(1);
                        let lhs = HostTensor::from_data(TensorData::new(
                            vec![0 as $ty; batches * m * k], lhs_shape,
                        ));
                        let rhs = HostTensor::from_data(TensorData::new(
                            vec![0 as $ty; batches * k * n], rhs_shape,
                        ));
                        let result = int_matmul(lhs, rhs);
                        assert_eq!(result.layout().shape().as_slice(), output_shape.as_slice());
                        assert_eq!(result.storage::<$ty>(), vec![0 as $ty; batches * m * n]);
                    }
                }
            }
        };
    }

    integer_empty_matrix_tests!(test_int_matmul_i32_empty_dimensions, i32);
    integer_empty_matrix_tests!(test_int_matmul_i64_empty_dimensions, i64);

    macro_rules! empty_batch_broadcast_tests {
        ($name:ident, $ty:ty, $operation:ident) => {
            #[test]
            fn $name() {
                // Test zero against a singleton on either side and against
                // another zero. A zero-sized batch remains zero after broadcast.
                for (lhs_batches, rhs_batches) in [(0, 1), (1, 0), (0, 0)] {
                    let lhs = HostTensor::from_data(TensorData::new(
                        vec![1 as $ty; lhs_batches * 6], [lhs_batches, 2, 3],
                    ));
                    let rhs = HostTensor::from_data(TensorData::new(
                        vec![1 as $ty; rhs_batches * 12], [rhs_batches, 3, 4],
                    ));
                    let result = $operation(lhs, rhs);
                    assert_eq!(result.layout().shape().as_slice(), &[0, 2, 4]);
                    assert!(result.storage::<$ty>().is_empty());
                }
            }
        };
    }

    empty_batch_broadcast_tests!(test_int_matmul_i32_empty_batch_broadcast, i32, int_matmul);
    empty_batch_broadcast_tests!(test_int_matmul_i64_empty_batch_broadcast, i64, int_matmul);
    empty_batch_broadcast_tests!(test_matmul_f64_empty_batch_broadcast, f64, matmul);

    #[test]
    #[should_panic(expected = "batch dimensions not broadcastable")]
    fn test_int_matmul_rejects_incompatible_batches() {
        let lhs = HostTensor::from_data(TensorData::new(vec![1i64; 12], [2, 2, 3]));
        let rhs = HostTensor::from_data(TensorData::new(vec![1i64; 36], [3, 3, 4]));
        let _ = int_matmul(lhs, rhs);
    }

    #[test]
    #[should_panic(expected = "batch dimensions not broadcastable")]
    fn test_matmul_rejects_incompatible_batches() {
        let lhs = HostTensor::from_data(TensorData::new(vec![1f64; 12], [2, 2, 3]));
        let rhs = HostTensor::from_data(TensorData::new(vec![1f64; 36], [3, 3, 4]));
        let _ = matmul(lhs, rhs);
    }
