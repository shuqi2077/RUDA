    use super::*;
    use ruda_core::tensor::data::TensorData;

    #[test]
    fn test_slice_basic() {
        // Create a 2x3 tensor: [[0, 1, 2], [3, 4, 5]]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = HostTensor::from_data(TensorData::new(data, [2, 3]));

        // Slice [0:1, 1:3] -> [[1, 2]]
        let slices = vec![Slice::new(0, Some(1), 1), Slice::new(1, Some(3), 1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![1, 2]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_slice_with_step() {
        // Create a 1D tensor: [0, 1, 2, 3, 4, 5]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = HostTensor::from_data(TensorData::new(data, [6]));

        // Slice [0:6:2] -> [0, 2, 4]
        let slices = vec![Slice::new(0, Some(6), 2)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_index() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = HostTensor::from_data(TensorData::new(data, [5]));

        // Slice [-3:] -> [2, 3, 4]
        let slices = vec![Slice::new(-3, None, 1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_step() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = HostTensor::from_data(TensorData::new(data, [5]));

        // Slice [0..;-1] -> [4, 3, 2, 1, 0] (reverse full range)
        // In Ruda's semantics: range selects elements, step determines order
        let slices = vec![Slice::new(0, None, -1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![5]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_slice_assign_1d() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = HostTensor::from_data(TensorData::new(data, [5]));

        // Assign [10, 11, 12] to positions [1:4]
        let value_data: Vec<f32> = vec![10.0, 11.0, 12.0];
        let value = HostTensor::from_data(TensorData::new(value_data, [3]));
        let slices = vec![Slice::new(1, Some(4), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0.0, 10.0, 11.0, 12.0, 4.0]);
    }

    #[test]
    fn test_slice_assign_2d() {
        // Create a 3x3 tensor
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = HostTensor::from_data(TensorData::new(data, [3, 3]));

        // Assign [[10, 11], [12, 13]] to [1:3, 1:3]
        let value_data: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0];
        let value = HostTensor::from_data(TensorData::new(value_data, [2, 2]));
        let slices = vec![Slice::new(1, Some(3), 1), Slice::new(1, Some(3), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 6.0, 12.0, 13.0,]
        );
    }

    #[test]
    fn test_slice_assign_2d_full_row() {
        // Create a 3x4 tensor
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [3, 4]));

        // Assign [100, 101, 102, 103] to row 1
        let value_data: Vec<f32> = vec![100.0, 101.0, 102.0, 103.0];
        let value = HostTensor::from_data(TensorData::new(value_data, [1, 4]));
        let slices = vec![Slice::new(1, Some(2), 1), Slice::new(0, None, 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![
                0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0, 8.0, 9.0, 10.0, 11.0,
            ]
        );
    }

    // Broadcast-scalar fast path tests: mimic what
    // `Tensor::slice_fill` produces (a 1-element source expanded to
    // the slice shape with all strides zero).

    fn broadcast_scalar_f32(value: f32, target_shape: &[usize]) -> HostTensor {
        let scalar_tensor = HostTensor::from_data(TensorData::new(vec![value], [1]));
        crate::expand::expand(scalar_tensor, Shape::from(target_shape.to_vec()))
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_1d_contiguous() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = HostTensor::from_data(TensorData::new(data, [5]));
        let value = broadcast_scalar_f32(7.0, &[3]);
        let slices = vec![Slice::new(1, Some(4), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0.0, 7.0, 7.0, 7.0, 4.0]);
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_2d_inner_contiguous() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [4, 4]));
        let value = broadcast_scalar_f32(-1.0, &[2, 2]);
        let slices = vec![Slice::new(1, Some(3), 1), Slice::new(1, Some(3), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, -1.0, -1.0, 7.0, 8.0, -1.0, -1.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ]
        );
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_3d_inner_contiguous() {
        // 3D case that matched the user-reported regression shape.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [2, 3, 4]));
        let value = broadcast_scalar_f32(9.0, &[1, 2, 2]);
        let slices = vec![
            Slice::new(0, Some(1), 1),
            Slice::new(0, Some(2), 1),
            Slice::new(1, Some(3), 1),
        ];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Region [0..1, 0..2, 1..3] fills positions (0, 0, 1), (0, 0, 2),
        // (0, 1, 1), (0, 1, 2) with 9.0. Linear indices: 1, 2, 5, 6.
        let mut expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
        for &i in &[1usize, 2, 5, 6] {
            expected[i] = 9.0;
        }
        assert_eq!(values, expected);
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_strided_fallback() {
        // Stepped slice: hits the strided fallback (not inner-contiguous).
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [10]));
        // Source needs to match slice_info length: s![0..10;2] selects
        // 5 positions (0, 2, 4, 6, 8), so the expand target is [5].
        let value = broadcast_scalar_f32(0.0, &[5]);
        let slices = vec![Slice::new(0, Some(10), 2)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![0.0, 1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]
        );
    }

    /// Broadcast-scalar fast path on a non-f32 dtype.
    #[test]
    fn test_slice_assign_broadcast_scalar_i64() {
        fn broadcast_scalar_i64(value: i64, target_shape: &[usize]) -> HostTensor {
            let scalar_tensor = HostTensor::from_data(TensorData::new(vec![value], [1]));
            crate::expand::expand(scalar_tensor, Shape::from(target_shape.to_vec()))
        }

        let data: Vec<i64> = (0..12).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [3, 4]));
        let value = broadcast_scalar_i64(-7, &[2, 2]);
        let slices = vec![Slice::new(0, Some(2), 1), Slice::new(1, Some(3), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Positions (0, 1), (0, 2), (1, 1), (1, 2): linear indices 1,
        // 2, 5, 6 get replaced by -7.
        assert_eq!(values, vec![0, -7, -7, 3, 4, -7, -7, 7, 8, 9, 10, 11]);
    }

    /// ND strided fallback path: 3D slice with a stepped inner dim.
    #[test]
    fn test_slice_assign_broadcast_scalar_nd_strided_fallback() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [2, 3, 4]));
        // Step 2 on the innermost dim means slice_info's last step != 1,
        // so inner_contiguous is false and we take the ND fallback.
        let value = broadcast_scalar_f32(9.0, &[2, 3, 2]);
        let slices = vec![
            Slice::new(0, Some(2), 1),
            Slice::new(0, Some(3), 1),
            Slice::new(0, Some(4), 2),
        ];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Step-2 on dim 2 picks indices 0 and 2, so within each
        // `[b, r, :]` row the 0 and 2 positions get 9.0.
        let mut expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
        for b in 0..2 {
            for r in 0..3 {
                for c in [0, 2] {
                    expected[b * 12 + r * 4 + c] = 9.0;
                }
            }
        }
        assert_eq!(values, expected);
    }

    /// 2D inner-contig branch with `row_step > 1`.
    #[test]
    fn test_slice_assign_broadcast_scalar_2d_stepped_rows() {
        let data: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [5, 5]));
        // Step-2 rows pick out rows 0, 2, 4 (3 rows). Slice the inner
        // dim as a contiguous range so the 2D-inner-contig branch with
        // row_step != 1 fires.
        let value = broadcast_scalar_f32(-1.0, &[3, 3]);
        let slices = vec![Slice::new(0, Some(5), 2), Slice::new(1, Some(4), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        let mut expected: Vec<f32> = (0..25).map(|i| i as f32).collect();
        for r in [0, 2, 4] {
            for c in 1..4 {
                expected[r * 5 + c] = -1.0;
            }
        }
        assert_eq!(values, expected);
    }
