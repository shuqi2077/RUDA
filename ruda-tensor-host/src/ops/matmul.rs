pub use rublas_host::{int_matmul, matmul};

// ============================================================================
// Tests
// ============================================================================

// Tests kept here exercise flex-specific behavior of the matmul kernel:
// dtype-specific storage paths (F64, F16, BF16) that the generic
// FloatElem-parameterized backend-tests cannot reach. Plain contiguous
// F32/I32/I64 matmul, stride-through-matmul variants (transposed /
// swap_dims / broadcast-transposed), and other generic coverage have
// been migrated to ruda-backend-tests so they run against every backend.
// When adding new tests, keep them here only if they probe a flex
// dtype-storage path; otherwise add them to
// crates/ruda-backend-tests/tests/tensor/float/ops/matmul.rs.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use ruda_tensor::TensorData;
    use ruda_tensor::ops::FloatTensorOps;
    use ruda_tensor::{bf16, f16};

    use crate::{Host, HostTensor};

    #[test]
    fn test_matmul_f64() {
        let lhs = HostTensor::from_data(TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], [2, 2]));
        let rhs = HostTensor::from_data(TensorData::new(vec![5.0f64, 6.0, 7.0, 8.0], [2, 2]));

        let result = Host::float_matmul(lhs, rhs);
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

        let result = Host::float_matmul(lhs, rhs);
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

        let result = Host::float_matmul(lhs, rhs);
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
        let result = Host::float_matmul(q, k_t);

        let q2 = HostTensor::from_data(q_data);
        let k2 = HostTensor::from_data(k_data)
            .transpose(1, 2)
            .to_contiguous();
        let expected = Host::float_matmul(q2, k2);

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
        let result = Host::float_matmul(q, k_t);

        let q2 = HostTensor::from_data(q_data);
        let k2 = HostTensor::from_data(k_data)
            .transpose(1, 2)
            .to_contiguous();
        let expected = Host::float_matmul(q2, k2);

        let values: Vec<f16> = result.into_data().to_vec().unwrap();
        let expected: Vec<f16> = expected.into_data().to_vec().unwrap();
        assert_eq!(values, expected);
    }
}
