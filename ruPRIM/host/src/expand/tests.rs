    use super::*;
    use ruda_core::tensor::data::TensorData;

    #[test]
    fn test_expand_1d_to_2d() {
        // [3] -> [2, 3]
        let tensor = HostTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let expanded = expand(tensor, Shape::new([2, 3]));

        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        assert_eq!(expanded.layout().strides(), &[0, 1]);
    }

    #[test]
    fn test_expand_broadcast_dim() {
        // [3, 1] -> [3, 4]
        let tensor = HostTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3, 1]));
        let expanded = expand(tensor, Shape::new([3, 4]));

        assert_eq!(expanded.layout().shape().to_vec(), vec![3, 4]);
        // Original strides for [3, 1] would be [1, 1]
        // After expand to [3, 4], stride for dim 1 becomes 0
        assert_eq!(expanded.layout().strides()[1], 0);
    }

    #[test]
    fn test_expand_same_shape() {
        // [2, 3] -> [2, 3] (no change)
        let tensor = HostTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let original_strides = tensor.layout().strides().to_vec();
        let expanded = expand(tensor, Shape::new([2, 3]));

        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        assert_eq!(expanded.layout().strides(), &original_strides);
    }

    // === Non-contiguous tensor tests ===

    #[test]
    fn test_expand_transposed() {
        // [[1, 2], [3, 4]] transposed -> [[1, 3], [2, 4]] with strides [1, 2]
        // Expand [2, 2] -> [3, 2, 2] by prepending dimension
        let tensor = HostTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let transposed = tensor.transpose(0, 1);
        assert!(!transposed.is_contiguous());
        assert_eq!(transposed.layout().strides(), &[1, 2]);

        let expanded = expand(transposed, Shape::new([3, 2, 2]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![3, 2, 2]);
        // New dim with stride 0, original strides preserved
        assert_eq!(expanded.layout().strides(), &[0, 1, 2]);

        // Verify content: should see same transposed values repeated 3 times
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        // [[1, 3], [2, 4]] repeated 3 times
        assert_eq!(
            data,
            vec![1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn test_expand_flipped_1d() {
        // [1, 2, 3] flipped -> [3, 2, 1] with negative stride
        // Expand [3] -> [2, 3]
        let tensor = HostTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let flipped = crate::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);

        let expanded = expand(flipped, Shape::new([2, 3]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        // Stride 0 for new broadcast dim, negative stride preserved
        assert_eq!(expanded.layout().strides()[0], 0);
        assert!(expanded.layout().strides()[1] < 0);

        // Verify content: [3, 2, 1] repeated twice
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_expand_flipped_2d_preserves_negative_stride() {
        // [[1, 2], [3, 4]] with axis 0 flipped -> [[3, 4], [1, 2]]
        // Shape [2, 2] with strides [-2, 1] (negative on axis 0)
        // Expand [2, 2] -> [3, 2, 2]
        let tensor = HostTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = crate::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);
        assert_eq!(flipped.layout().strides()[1], 1);

        let expanded = expand(flipped, Shape::new([3, 2, 2]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![3, 2, 2]);
        // Stride 0 for broadcast, negative stride preserved for axis 1, positive for axis 2
        assert_eq!(expanded.layout().strides()[0], 0);
        assert!(expanded.layout().strides()[1] < 0);
        assert_eq!(expanded.layout().strides()[2], 1);

        // Verify content
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        // [[3, 4], [1, 2]] repeated 3 times
        assert_eq!(
            data,
            vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_expand_narrowed_preserves_offset() {
        // [0, 1, 2, 3, 4] narrowed to [1, 2, 3] with offset 1
        // Expand [3] -> [2, 3]
        let tensor = HostTensor::from_data(TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], [5]));
        let narrowed = tensor.narrow(0, 1, 3);
        assert_eq!(narrowed.layout().start_offset(), 1);

        let expanded = expand(narrowed, Shape::new([2, 3]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        // Start offset preserved
        assert_eq!(expanded.layout().start_offset(), 1);

        // Verify content: [1, 2, 3] repeated twice
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "broadcasting cannot drop dimensions")]
    fn test_expand_to_fewer_dims_panics() {
        // Shape [2, 3, 4] expanded to target [2, 3] would silently drop
        // the trailing dim and produce an invalid layout; must panic.
        let tensor = HostTensor::from_data(TensorData::new(alloc::vec![0.0f32; 24], [2, 3, 4]));
        let _ = expand(tensor, Shape::new([2, 3]));
    }

    #[test]
    #[should_panic(expected = "broadcasting cannot drop dimensions")]
    fn test_expand_1d_to_scalar_panics() {
        // 1D -> 0D exercises the saturating_sub edge we removed: dim_diff
        // used to come out as 0 and the loop would iterate zero times,
        // silently producing a 0-dim layout over 1D data.
        let tensor = HostTensor::from_data(TensorData::new(alloc::vec![1.0f32, 2.0, 3.0], [3]));
        let _ = expand(tensor, Shape::from(alloc::vec::Vec::<usize>::new()));
    }

    #[test]
    fn test_broadcast_binary_with_flipped() {
        // One tensor flipped, broadcast to same shape
        // lhs: [1, 2, 3, 4] flipped -> [4, 3, 2, 1], shape [4]
        // rhs: [[1], [2]], shape [2, 1] -> broadcast to [2, 4]
        // After broadcast: lhs [2, 4], rhs [2, 4]
        let lhs = HostTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let lhs = crate::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let rhs = HostTensor::from_data(TensorData::new(vec![10.0f32, 20.0], [2, 1]));

        let (lhs_bc, rhs_bc) = broadcast_binary(lhs, rhs);
        assert_eq!(lhs_bc.layout().shape().to_vec(), vec![2, 4]);
        assert_eq!(rhs_bc.layout().shape().to_vec(), vec![2, 4]);

        // lhs should have stride 0 in dim 0 (broadcast), negative stride in dim 1 (from flip)
        assert_eq!(lhs_bc.layout().strides()[0], 0);
        assert!(lhs_bc.layout().strides()[1] < 0);

        // Verify lhs content: [4, 3, 2, 1] repeated twice
        let lhs_data: Vec<f32> = lhs_bc.into_data().to_vec().unwrap();
        assert_eq!(lhs_data, vec![4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0]);
    }
