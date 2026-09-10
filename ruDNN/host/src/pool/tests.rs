    use super::*;
    use ruda_core::tensor::data::TensorData;

    #[test]
    fn test_pool_output_size() {
        // Basic: input=4, kernel=2, padding=0, stride=2, dilation=1
        // output = (4 + 0 - 2) / 2 + 1 = 2
        assert_eq!(pool_output_size(4, 2, 0, 2, 1, false), 2);

        // With padding: input=4, kernel=2, padding=1, stride=2
        // output = (4 + 2 - 2) / 2 + 1 = 3
        assert_eq!(pool_output_size(4, 2, 1, 2, 1, false), 3);

        // With ceil mode: input=5, kernel=2, padding=0, stride=2
        // floor: (5 - 2) / 2 + 1 = 2
        // ceil: ceil(3/2) + 1 = 3
        assert_eq!(pool_output_size(5, 2, 0, 2, 1, false), 2);
        assert_eq!(pool_output_size(5, 2, 0, 2, 1, true), 3);

        // With dilation: input=7, kernel=2, dilation=2
        // effective_kernel = 2*(2-1)+1 = 3
        // output = (7 - 3) / 1 + 1 = 5
        assert_eq!(pool_output_size(7, 2, 0, 1, 2, false), 5);
    }

    #[test]
    fn test_avg_pool2d_count_include_pad() {
        // 3x3 input with padding 1, kernel 2x2, stride 2
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 3, 3]));

        // count_include_pad = true: divide by kernel size (4)
        let result_include = avg_pool2d_f32(x.clone(), [2, 2], [2, 2], [1, 1], true, false);
        let out_include: Vec<f32> = result_include.into_data().to_vec().unwrap();

        // count_include_pad = false: divide by actual count
        let result_exclude = avg_pool2d_f32(x, [2, 2], [2, 2], [1, 1], false, false);
        let out_exclude: Vec<f32> = result_exclude.into_data().to_vec().unwrap();

        // Corner position with padding: only 1 valid element
        // With count_include_pad: 1.0 / 4 = 0.25
        // Without: 1.0 / 1 = 1.0
        assert!((out_include[0] - 0.25).abs() < 1e-5);
        assert!((out_exclude[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_pool2d_f64() {
        let x_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = max_pool2d_f64(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(out, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_pool2d_f16() {
        let x_data: Vec<f16> = (1..=16).map(|x| f16::from_f32(x as f32)).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = max_pool2d_f16(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();

        assert!((out[0].to_f32() - 6.0).abs() < 0.1);
        assert!((out[1].to_f32() - 8.0).abs() < 0.1);
        assert!((out[2].to_f32() - 14.0).abs() < 0.1);
        assert!((out[3].to_f32() - 16.0).abs() < 0.1);
    }

    #[test]
    fn test_max_pool2d_bf16() {
        let x_data: Vec<bf16> = (1..=16).map(|x| bf16::from_f32(x as f32)).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = max_pool2d_bf16(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let out: Vec<bf16> = result.into_data().to_vec().unwrap();

        assert!((out[0].to_f32() - 6.0).abs() < 0.5);
        assert!((out[1].to_f32() - 8.0).abs() < 0.5);
    }

    #[test]
    fn test_max_pool_backward() {
        // Forward pass
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = HostTensor::from_data(TensorData::new(x_data.clone(), vec![1, 1, 4, 4]));
        let (_output, indices) =
            max_pool2d_with_indices_f32(x.clone(), [2, 2], [2, 2], [0, 0], [1, 1], false);

        // Backward pass: gradient of 1.0 for each output
        let grad = HostTensor::from_data(TensorData::new(vec![1.0f32; 4], vec![1, 1, 2, 2]));

        let x_grad = max_pool2d_backward_f32(x, grad, indices);
        let grad_data: Vec<f32> = x_grad.into_data().to_vec().unwrap();

        // Gradient should be 1.0 at max positions, 0.0 elsewhere
        // Max positions were: 5, 7, 13, 15 (0-indexed)
        assert_eq!(grad_data[5], 1.0);
        assert_eq!(grad_data[7], 1.0);
        assert_eq!(grad_data[13], 1.0);
        assert_eq!(grad_data[15], 1.0);
        assert_eq!(grad_data[0], 0.0);
    }

    #[test]
    fn test_avg_pool_backward() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        // Backward with gradient of 4.0 for each output (will distribute as 1.0 each)
        let grad = HostTensor::from_data(TensorData::new(vec![4.0f32; 4], vec![1, 1, 2, 2]));

        let x_grad = avg_pool2d_backward_f32(x, grad, [2, 2], [2, 2], [0, 0], false);
        let grad_data: Vec<f32> = x_grad.into_data().to_vec().unwrap();

        // Each position in input should receive grad/4 = 1.0
        assert!(grad_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_adaptive_avg_pool_backward() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        // Backward with gradient of 4.0 for each output element
        let grad = HostTensor::from_data(TensorData::new(vec![4.0f32; 4], vec![1, 1, 2, 2]));
        let x_grad = adaptive_avg_pool2d_backward_f32(x, grad);
        let grad_data: Vec<f32> = x_grad.into_data().to_vec().unwrap();

        // Each input position receives gradient from its output region
        assert!(grad_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    #[should_panic(expected = "kernel size must be > 0")]
    fn test_pool_output_size_zero_kernel_panics() {
        pool_output_size(4, 0, 0, 1, 1, false);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn test_pool_output_size_zero_stride_panics() {
        pool_output_size(4, 2, 0, 0, 1, false);
    }
