    use super::*;
    use ruda_core::tensor::data::TensorData;

    #[test]
    fn test_gather_with_i32_indices() {
        let tensor = HostTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], [4]));
        let indices = HostTensor::from_data(TensorData::new(vec![3i32, 0, 2], [3]));

        let result = gather::<f32>(tensor, 0, indices);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![40.0, 10.0, 30.0]);
    }

    #[test]
    fn test_select_with_i32_indices() {
        let tensor = HostTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        ));
        let indices = HostTensor::from_data(TensorData::new(vec![2i32, 0], [2]));

        let result = select::<f32>(tensor, 0, indices);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![5.0, 6.0, 1.0, 2.0]);
    }

    /// Exercises the uninit buffer path (dim=0) and, when rayon is enabled,
    /// the chunked parallel path (output > 4 MB).
    #[test]
    fn test_select_2d_dim0_large() {
        let rows = 2048;
        let cols = 1024;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let tensor = HostTensor::from_data(TensorData::new(data.clone(), [rows, cols]));

        // Select every other row in reverse order.
        let idx: Vec<i64> = (0..rows as i64).rev().step_by(2).collect();
        let num_idx = idx.len();
        let indices = HostTensor::from_data(TensorData::new(idx.clone(), [num_idx]));

        let result = select::<f32>(tensor, 0, indices);
        assert_eq!(result.layout().shape().to_vec(), vec![num_idx, cols]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for (i, &row_idx) in idx.iter().enumerate() {
            let expected_start = row_idx as usize * cols;
            let actual = &out[i * cols..(i + 1) * cols];
            let expected = &data[expected_start..expected_start + cols];
            assert_eq!(
                actual, expected,
                "mismatch at output row {i} (src row {row_idx})"
            );
        }
    }
