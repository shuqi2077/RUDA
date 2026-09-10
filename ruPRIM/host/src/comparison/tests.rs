    use super::*;
    use ruda_core::tensor::data::TensorData;

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> HostTensor {
        HostTensor::from_data(TensorData::new(data, vec![rows, cols]))
    }

    #[test]
    fn test_any_float_dim_transposed() {
        // [[0, 1], [0, 0]] transposed -> [[0, 0], [1, 0]]
        // any_dim along dim 1: [false, true] -> [0, 1]
        // Before the C2 fix, to_contiguous was called inside reduce_bool_dim_with
        // AFTER taking a reference to the original non-contiguous storage,
        // causing stale pointer reads.
        let tensor = tensor_2d(vec![0.0, 1.0, 0.0, 0.0], 2, 2);
        let transposed = tensor.transpose(0, 1);
        assert!(!transposed.is_contiguous());

        let result = any_float_dim(transposed, 1, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1]); // row 0: all zeros; row 1: has a 1
    }

    #[test]
    fn test_any_float_dim_narrowed() {
        // [0, 5, 0, 3, 0, 0] narrowed to [[5, 0], [3, 0]] (shape [2,2])
        // any_dim along dim 1: [true, true] -> [1, 1]
        let tensor = HostTensor::from_data(TensorData::new(
            vec![0.0f32, 5.0, 0.0, 3.0, 0.0, 0.0],
            [3, 2],
        ));
        let narrowed = tensor.narrow(0, 0, 2); // first 2 rows: [[0, 5], [0, 3]]
        let result = any_float_dim(narrowed, 1, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 1]);
    }
