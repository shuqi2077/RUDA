    use super::*;
    use ruda_core::tensor::data::TensorData;

    #[test]
    fn test_flip_is_zero_copy() {
        // Verify flip doesn't copy data by checking it shares the same underlying storage
        let tensor = HostTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let tensor_ptr = tensor.bytes().as_ptr();
        let flipped = flip(tensor, &[0]);
        let flipped_ptr = flipped.bytes().as_ptr();
        assert_eq!(
            tensor_ptr, flipped_ptr,
            "flip should share underlying storage"
        );
    }
