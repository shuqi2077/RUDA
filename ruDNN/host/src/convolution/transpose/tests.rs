    use super::*;
    use ruda_core::tensor::data::TensorData;

    #[test]
    fn test_conv_transpose2d_f64() {
        let x = HostTensor::from_data(TensorData::new(
            vec![1.0f64, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
        ));
        let w = HostTensor::from_data(TensorData::new(vec![1.0f64; 4], vec![1, 1, 2, 2]));
        let opts = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1);
        let result = conv_transpose2d_f64(x, w, None, &opts);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
    }

    #[test]
    fn test_conv_transpose2d_f16() {
        let x_data: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let w_data: Vec<f16> = [1.0f32; 4].iter().map(|&v| f16::from_f32(v)).collect();
        let x = HostTensor::from_data(TensorData::new(x_data, vec![1, 1, 2, 2]));
        let w = HostTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let opts = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1);
        let result = conv_transpose2d_f16(x, w, None, &opts);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();
        let expected = [1.0f32, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.1);
        }
    }
