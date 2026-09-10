use super::*;
use crate::qwen35::Norm;

#[test]
fn rms_norm_dtype_layout_tail_and_validation() {
    type B = Cuda<f32, i32>;
    let device = CudaDevice::default();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for rows in [1, 3] {
            for width in [1, 3, 31, 127, 128, 129, 256, 1024, 1025] {
                let values = (0..rows*width).map(|i| ((i*17%43) as f32-21.0)/7.0).collect::<Vec<_>>();
                let input = Tensor::<B, 2>::from_data(TensorData::new(values, [width, rows]), (&device, dtype)).swap_dims(0, 1);
                let actual_input = input.clone().cast(DType::F32).into_data().to_vec::<f32>().unwrap();
                let weights = (0..width).map(|i| 0.5+(i%7) as f32/8.0).collect::<Vec<_>>();
                let gamma = Tensor::<B, 1>::from_data(TensorData::new(weights.clone(), [width]), (&device, DType::F32));
                let output = rudnn::normalization::rms_norm(input.into_primitive().tensor(), gamma.into_primitive().tensor(), 1e-6).unwrap();
                assert_eq!(output.dtype, dtype);
                let output = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(output)).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
                for (row, actual) in actual_input.chunks_exact(width).zip(output.chunks_exact(width)) {
                    let variance = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / width as f64;
                    for ((&value, &weight), &observed) in row.iter().zip(&weights).zip(actual) {
                        let expected = (value as f64 / (variance+1e-6).sqrt() * weight as f64) as f32;
                        let tolerance = match dtype { DType::BF16 => 0.008, DType::F16 => 0.001, _ => 2e-6 };
                        assert!((observed-expected).abs() <= tolerance*expected.abs().max(1.0), "{dtype:?} rows={rows} width={width}: {observed} != {expected}");
                    }
                }
            }
        }
    }
    let input = Tensor::<B, 2>::from_data([[0f32, 0.], [f32::INFINITY, 1.], [f32::NAN, 1.]], (&device, DType::F32));
    let gamma = Tensor::<B, 1>::ones([2], (&device, DType::F32));
    let launch = |epsilon| rudnn::normalization::rms_norm(input.clone().into_primitive().tensor(), gamma.clone().into_primitive().tensor(), epsilon);
    let output = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(launch(1e-6).unwrap())).into_data().to_vec::<f32>().unwrap();
    assert_eq!(&output[..2], &[0.0, 0.0]);
    assert!(output[2].is_nan());
    assert_eq!(output[3], 0.0);
    assert!(output[4].is_nan() && output[5].is_nan());
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] { assert!(launch(epsilon).is_err()); }
    let wrong_gamma = Tensor::<B, 1>::ones([3], (&device, DType::F32));
    assert!(rudnn::normalization::rms_norm(input.into_primitive().tensor(), wrong_gamma.into_primitive().tensor(), 1e-6).is_err());
    let empty = Tensor::<B, 2>::empty([0, 2], (&device, DType::F32));
    let output = rudnn::normalization::rms_norm(empty.into_primitive().tensor(), gamma.clone().into_primitive().tensor(), 1e-6).unwrap();
    assert_eq!(output.meta.shape()[..], [0, 2]);
    let zero_width = Tensor::<B, 2>::empty([1, 0], (&device, DType::F32));
    assert!(rudnn::normalization::rms_norm(zero_width.into_primitive().tensor(), gamma.into_primitive().tensor(), 1e-6).is_err());
}

#[test]
#[ignore = "requires RUDA_QWEN35_POST_NORM_REFERENCE with original residual and RMSNorm intermediates"]
fn real_residual_normalization() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_POST_NORM_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    assert_eq!(meta["original_post_exact"], true);
    let shape: [usize; 3] = serde_json::from_value(meta["shape"].clone()).unwrap();
    let epsilon = meta["epsilon"].as_f64().unwrap();
    let device = CudaDevice::default();
    let input = |name| Tensor::<B, 3>::from_data(
        TensorData::new(floats(&reference.join(format!("{name}.f32"))), shape), (&device, DType::BF16));
    let weight = Tensor::<B, 1>::from_data(
        TensorData::new(floats(&reference.join("weight.f32")), [shape[2]]), (&device, DType::F32));
    let compare = |name, value: Tensor<B, 3>| {
        let actual = value.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = floats(&reference.join(format!("{name}.f32")));
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|x| x.is_finite()));
        let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        let max = actual.iter().zip(&expected).map(|(a, b)| (a-b).abs()).fold(0f32, f32::max);
        eprintln!("post norm {name}: different={different}/{} max_abs={max}", actual.len());
        different
    };
    let residual = input("embedded") + input("mixer");
    assert_eq!(compare("residual", residual.clone()), 0);
    let x = residual.clone().cast(DType::F32);
    let variance = x.clone().square().mean_dim(2);
    compare("variance", variance.clone());
    let inverse = (variance + epsilon).rsqrt();
    compare("inverse", inverse.clone());
    let normalized = x.clone() * inverse;
    compare("normalized", normalized.clone());
    compare("weighted", normalized * (weight.clone() + 1.0).unsqueeze::<3>());
    assert_eq!(compare("output", Norm { weight: weight.clone(), epsilon }.forward(residual).unwrap()), 0);
    let fixed_inverse = Tensor::<B, 3>::from_data(TensorData::new(
        floats(&reference.join("inverse.f32")), [shape[0], shape[1], 1]), (&device, DType::F32));
    let fixed_normalized = x * fixed_inverse;
    assert_eq!(compare("normalized", fixed_normalized.clone()), 0);
    assert_eq!(compare("output", (fixed_normalized * (weight + 1.0).unsqueeze::<3>()).cast(DType::BF16)), 0);
}
