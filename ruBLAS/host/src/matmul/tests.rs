    use alloc::{vec, vec::Vec};
    use ruda_core::tensor::data::TensorData;
    use half::{bf16, f16};

    use ruda_core::tensor::host::HostTensor;
    use super::matmul;

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
