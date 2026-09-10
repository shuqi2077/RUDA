    use super::*;

    fn make_input_f32(batch: usize, channels: usize, height: usize, width: usize) -> HostTensor {
        let numel = batch * channels * height * width;
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![batch, channels, height, width])),
            DType::F32,
        )
    }

    #[test]
    fn test_nearest_upsample_2x() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![1, 1, 2, 2])),
            DType::F32,
        );

        let result = interpolate_nearest_f32(x, [4, 4], true);
        let output = result.storage::<f32>();

        assert_eq!(output.len(), 16);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[3], 2.0);
    }

    #[test]
    fn test_bilinear_upsample_2x() {
        let data = vec![0.0f32, 1.0, 1.0, 0.0];
        let x = HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(vec![1, 1, 2, 2])),
            DType::F32,
        );

        let result = interpolate_bilinear_f32(x, [4, 4], true);
        let output = result.storage::<f32>();

        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
        assert!((output[12] - 1.0).abs() < 1e-5);
        assert!((output[15] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_bicubic_basic() {
        let x = make_input_f32(1, 1, 4, 4);
        let result = interpolate_bicubic_f32(x, [8, 8], true);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 8, 8]);
    }

    #[test]
    fn test_downsample() {
        let x = make_input_f32(1, 1, 4, 4);
        let result = interpolate_nearest_f32(x, [2, 2], true);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_nearest_backward() {
        let x = make_input_f32(1, 1, 2, 2);
        let grad = HostTensor::new(
            Bytes::from_elems(vec![1.0f32; 16]),
            Layout::contiguous(Shape::from(vec![1, 1, 4, 4])),
            DType::F32,
        );

        let result = interpolate_nearest_backward_f32(x, grad, [4, 4], true);
        let output = result.storage::<f32>();

        assert_eq!(output.len(), 4);
        assert!((output[0] - 4.0).abs() < 1e-5);
    }
