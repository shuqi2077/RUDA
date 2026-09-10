use super::*;
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use std::{fs, path::PathBuf};

#[test]
fn linear_adds_bias_before_bf16_rounding() {
    type B = Cuda<bf16, i32>;
    let device = CudaDevice::default();
    let layer = Linear::<B> {
        weight: ruda_model::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![1f32, 1.], [2, 1]),
            (&device, DType::BF16),
        )),
        bias: Some(ruda_model::module::Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![-256f32], [1]),
            (&device, DType::BF16),
        ))),
    };
    let input = Tensor::from_data(
        TensorData::new(vec![256f32, 1.], [1, 2]),
        (&device, DType::BF16),
    );
    let output = linear(&layer, input)
        .unwrap()
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(output, [1.]);
}

#[test]
fn bf16_normalization_keeps_float32_statistics() {
    type B = Cuda<bf16, i32>;
    let device = CudaDevice::default();
    let mut norm = ruda_nn::LayerNormConfig::new(4)
        .with_epsilon(1e-6)
        .init::<B>(&device);
    norm.gamma = ruda_model::module::Param::from_tensor(Tensor::from_data(
        TensorData::new(vec![1f32; 4], [4]),
        (&device, DType::F32),
    ));
    norm.beta = Some(ruda_model::module::Param::from_tensor(Tensor::from_data(
        TensorData::new(vec![0f32; 4], [4]),
        (&device, DType::F32),
    )));
    let input = Tensor::from_data(
        TensorData::new(vec![256f32, 258., 260., 262.], [1, 4]),
        (&device, DType::BF16),
    );
    let output = normalize(&norm, input)
        .unwrap()
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(output, [-1.34375, -0.447265625, 0.447265625, 1.34375]);
}

#[test]
fn device_layer_norm_dtypes_widths_strides_and_optional_bias() {
    type B = Cuda<bf16, i32>;
    let device = CudaDevice::default();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let round = |x: f32| match dtype {
            DType::F16 => half::f16::from_f32(x).to_f32(),
            DType::BF16 => bf16::from_f32(x).to_f32(),
            _ => x,
        };
        for width in [1, 3, 4, 33, 768, 769, 3072] {
            let values: Vec<f32> = (0..width * 2)
                .map(|i| round(64.0 + (i % 17) as f32 * 0.25))
                .collect();
            let x = Tensor::<B, 2>::from_data(
                TensorData::new(values.clone(), [width, 2]),
                (&device, dtype),
            )
            .swap_dims(0, 1);
            let gamma = Tensor::<B, 1>::from_data(
                TensorData::new(vec![1.5f32; width], [width]),
                (&device, DType::F32),
            );
            let beta = Tensor::<B, 1>::from_data(
                TensorData::new(vec![0.25f32; width], [width]),
                (&device, DType::F32),
            );
            for with_bias in [false, true] {
                let result = rudnn::normalization::layer_norm(
                    x.clone().into_primitive().tensor(),
                    gamma.clone().into_primitive().tensor(),
                    with_bias.then(|| beta.clone().into_primitive().tensor()),
                    1e-6,
                )
                .unwrap();
                let result = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(result))
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                for row in 0..2 {
                    let xs: Vec<f64> = (0..width).map(|i| values[2 * i + row] as f64).collect();
                    let mean = xs.iter().sum::<f64>() / width as f64;
                    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / width as f64;
                    for (i, x) in xs.into_iter().enumerate() {
                        let expected = round(
                            (1.5 * (x - mean) / (var + 1e-6).sqrt()
                                + if with_bias { 0.25 } else { 0.0 })
                                as f32,
                        );
                        let tolerance = match dtype {
                            DType::BF16 => 0.0078125,
                            DType::F16 => 0.0009765625,
                            _ => 0.00003,
                        } * expected.abs().max(1.0);
                        assert!(
                            (result[row * width + i] - expected).abs() <= tolerance,
                            "{dtype:?} width={width} row={row} col={i}: {} != {expected}",
                            result[row * width + i]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn device_layer_norm_validates_affine_and_preserves_nan() {
    type B = Cuda<bf16, i32>;
    let device = CudaDevice::default();
    let x = Tensor::<B, 2>::from_data([[1f32, 2., 3., f32::NAN]], (&device, DType::BF16));
    let gamma = Tensor::<B, 1>::from_data([1f32; 4], (&device, DType::F32));
    for epsilon in [-1., 0., f32::INFINITY, f32::NAN] {
        assert!(
            rudnn::normalization::layer_norm(
                x.clone().into_primitive().tensor(),
                gamma.clone().into_primitive().tensor(),
                None,
                epsilon
            )
            .is_err()
        );
    }
    for bad in [gamma.clone().slice([0..3]), gamma.clone().cast(DType::BF16)] {
        assert!(
            rudnn::normalization::layer_norm(
                x.clone().into_primitive().tensor(),
                bad.into_primitive().tensor(),
                None,
                1e-6
            )
            .is_err()
        );
    }
    let output = rudnn::normalization::layer_norm(
        x.into_primitive().tensor(),
        gamma.into_primitive().tensor(),
        None,
        1e-6,
    )
    .unwrap();
    let output = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(output))
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(output.iter().all(|x| x.is_nan()));
}

#[test]
fn device_softmax_matches_independent_rows_and_special_values() {
    type B = Cuda<bf16, i32>;
    let device = CudaDevice::default();
    for width in [1, 3, 17, 32, 33, 256, 1025] {
        let values = (0..2 * width).map(|i| (i % 41) as f32 * 0.75 - 15.).collect::<Vec<_>>();
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(values.clone(), [width, 2]), (&device, DType::F32),
        ).transpose();
        let actual = rudnn::normalization::softmax_last_axis(input.into_primitive().tensor()).unwrap();
        let actual = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(actual)).into_data().to_vec::<f32>().unwrap();
        for row in 0..2 {
            let max = (0..width).map(|i| values[2*i+row] as f64).fold(f64::NEG_INFINITY, f64::max);
            let exp = (0..width).map(|i| (values[2*i+row] as f64 - max).exp()).collect::<Vec<_>>();
            let sum = exp.iter().sum::<f64>();
            for (i, value) in exp.iter().enumerate() {
                let expected = value / sum;
                assert!((actual[row*width+i] as f64 - expected).abs() <= 4e-6 * expected.max(1e-30));
            }
            assert!((actual[row*width..(row+1)*width].iter().sum::<f32>() - 1.).abs() < 2e-6);
        }
    }
    for values in [[f32::NAN, 0., 1.], [f32::INFINITY, 0., 1.], [f32::NEG_INFINITY; 3]] {
        let input = Tensor::<B, 2>::from_data([values], (&device, DType::F32));
        let actual = rudnn::normalization::softmax_last_axis(input.into_primitive().tensor()).unwrap();
        let actual = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(actual)).into_data().to_vec::<f32>().unwrap();
        assert!(actual.iter().all(|v| v.is_nan()));
    }
    let masked = Tensor::<B, 2>::from_data([[f32::NEG_INFINITY, 0., f32::NEG_INFINITY]], (&device, DType::F32));
    let actual = rudnn::normalization::softmax_last_axis(masked.into_primitive().tensor()).unwrap();
    assert_eq!(Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(actual)).into_data().to_vec::<f32>().unwrap(), [0., 1., 0.]);
    for shape in [[0, 3], [2, 0]] {
        let input = Tensor::<B, 2>::from_data(TensorData::new(Vec::<f32>::new(), shape), (&device, DType::F32));
        let result = rudnn::normalization::softmax_last_axis(input.into_primitive().tensor());
        assert_eq!(result.is_ok(), shape[1] != 0);
    }
    let wrong_dtype = Tensor::<B, 2>::from_data([[1f32, 2.]], (&device, DType::BF16));
    assert!(rudnn::normalization::softmax_last_axis(wrong_dtype.into_primitive().tensor()).is_err());
}

fn floats(path: PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn operator_error<B: Backend, const D: usize>(name: &str, actual: Tensor<B, D>, path: PathBuf) {
    let expected = floats(path);
    let actual = actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>();
    let norm = expected.iter().map(|&n| (n as f64).powi(2)).sum::<f64>();
    let count = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "isolated {name}: mismatches={count}/{} relative_l2={} max_abs={max}",
        actual.len(),
        (error / norm).sqrt()
    );
}

#[test]
#[ignore = "requires RUDA_QWEN35_MODEL and RUDA_QWEN35_VISION_REFERENCE"]
fn real_checkpoint_isolated_blocks() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_VISION_REFERENCE").unwrap());
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let grids: Vec<[usize; 3]> = serde_json::from_value(meta["grids"].clone()).unwrap();
    let shape: [usize; 2] = serde_json::from_value(meta["patch_shape"].clone()).unwrap();
    let device = CudaDevice::default();
    let loaded = load_huggingface_qwen35_vision::<B>(
        std::env::var("RUDA_QWEN35_MODEL").unwrap(), &device,
    ).unwrap();
    let c = loaded.model.config();
    let p = position::positions(&grids, c.spatial_merge_size, c.num_position_embeddings.isqrt());
    let (cos, sin) = rotary::<B>(p.coordinates, shape[0], c.hidden_size / c.num_heads, &device);
    operator_error("computed-cos", cos.clone(), reference.join("cos0.f32"));
    operator_error("computed-sin", sin.clone(), reference.join("sin0.f32"));
    for (index, block) in loaded.model.blocks.iter().enumerate() {
        let input = if index == 0 { "position".to_owned() } else { format!("block-{}", index - 1) };
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(floats(reference.join(format!("{input}.f32"))), [shape[0], c.hidden_size]),
            (&device, DType::BF16),
        );
        let out = block.forward(x, c.num_heads, cos.clone(), sin.clone(), &p.segments).unwrap();
        operator_error(&format!("block-{index}"), out, reference.join(format!("block-{index}.f32")));
    }
}

#[test]
#[ignore = "requires RUDA_QWEN35_MODEL and RUDA_QWEN35_VISION_REFERENCE"]
fn real_checkpoint_vision_reference() {
    type B = Cuda<bf16, i32>;
    let directory = std::env::var("RUDA_QWEN35_MODEL").expect("model directory");
    let reference =
        PathBuf::from(std::env::var("RUDA_QWEN35_VISION_REFERENCE").expect("reference directory"));
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let grids: Vec<[usize; 3]> = serde_json::from_value(meta["grids"].clone()).unwrap();
    let shape: [usize; 2] = serde_json::from_value(meta["patch_shape"].clone()).unwrap();
    let device = CudaDevice::default();
    let loaded = load_huggingface_qwen35_vision::<B>(directory, &device).unwrap();
    assert_eq!(
        loaded.report.applied_tensors,
        meta["tensors"].as_u64().unwrap() as usize
    );
    assert!(
        loaded
            .unloaded_tensors
            .iter()
            .all(|n| !n.starts_with("model.visual."))
    );
    if reference.join("norm0.f32").exists() {
        let input = |name: &str, width| {
            Tensor::<B, 2>::from_data(
                TensorData::new(
                    floats(reference.join(format!("{name}.f32"))),
                    [shape[0], width],
                ),
                (&device, DType::BF16),
            )
        };
        let block = &loaded.model.blocks[0];
        let width = loaded.model.config.hidden_size;
        let intermediate = loaded.model.config.intermediate_size;
        operator_error(
            "norm0",
            normalize(&block.norm1, input("position", width)).unwrap(),
            reference.join("norm0.f32"),
        );
        operator_error(
            "qkv0",
            linear(&block.qkv, input("norm0", width)).unwrap(),
            reference.join("qkv0.f32"),
        );
        operator_error(
            "norm2",
            normalize(&block.norm2, input("norm2-input", width)).unwrap(),
            reference.join("norm2.f32"),
        );
        operator_error(
            "fc1",
            linear(&block.fc1, input("norm2", width)).unwrap(),
            reference.join("fc1.f32"),
        );
        operator_error(
            "gelu0",
            gelu_approximate(input("fc1", intermediate).cast(DType::F32)).cast(DType::BF16),
            reference.join("gelu0.f32"),
        );
        operator_error(
            "fc2",
            linear(&block.fc2, input("gelu0", intermediate)).unwrap(),
            reference.join("fc2.f32"),
        );
        if reference.join("cos0.f32").exists() {
            let heads = loaded.model.config.num_heads;
            let d = width / heads;
            let rotary = |name: &str| {
                Tensor::<B, 4>::from_data(
                    TensorData::new(
                        floats(reference.join(format!("{name}.f32"))),
                        [1, shape[0], 1, d],
                    ),
                    (&device, DType::F32),
                )
            };
            let segments = position::positions(
                &grids,
                loaded.model.config.spatial_merge_size,
                loaded.model.config.num_position_embeddings.isqrt(),
            )
            .segments;
            let context = Block::<B>::attention(
                input("qkv0", 3 * width),
                heads,
                rotary("cos0"),
                rotary("sin0"),
                &segments,
                |segment, name, tensor| {
                    let name = format!("attn-{segment}-{name}");
                    operator_error(&name, tensor.clone(), reference.join(format!("{name}.f32")));
                },
            ).unwrap();
            operator_error("context0", context, reference.join("context0.f32"));
            operator_error(
                "proj0",
                linear(&block.proj, input("context0", width)).unwrap(),
                reference.join("proj0.f32"),
            );
        }
    }
    let patches = Tensor::<B, 2>::from_data(
        TensorData::new(floats(reference.join("patches.f32")), shape),
        (&device, DType::F32),
    );
    let output = loaded
        .model
        .forward_observed(patches.clone(), &grids, |stage, actual| {
            let name = match stage {
                0 => "patch".into(),
                1 => "position".into(),
                n => format!("block-{}", n - 2),
            };
            let path = reference.join(format!("{name}.f32"));
            if path.exists() {
                let expected = floats(path);
                let actual = actual
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                assert_eq!(actual.len(), expected.len());
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum::<f64>();
                let norm = expected.iter().map(|&n| (n as f64).powi(2)).sum::<f64>();
                let max = actual
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "vision trace {name}: relative_l2={} max_abs={max}",
                    (error / norm).sqrt()
                );
            }
        })
        .unwrap();
    let mut numerical_failures = Vec::new();
    for (name, actual) in [
        ("hidden", output.hidden_states),
        ("merged", output.merged_states.clone()),
    ] {
        let expected = floats(reference.join(format!("{name}.f32")));
        let actual = actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|n| n.is_finite()));
        let mut squared = 0.0f64;
        let mut norm = 0.0f64;
        let mut max_error = 0f32;
        for (&a, &b) in actual.iter().zip(&expected) {
            squared += (a as f64 - b as f64).powi(2);
            norm += (b as f64).powi(2);
            max_error = max_error.max((a - b).abs());
        }
        let relative_l2 = (squared / norm).sqrt();
        eprintln!("vision {name}: relative_l2={relative_l2} max_abs={max_error}");
        if !relative_l2.is_finite() || relative_l2 >= 0.02 {
            numerical_failures.push((name, relative_l2));
        }
    }
    assert!(
        numerical_failures.is_empty(),
        "BF16 vision reference relative L2 exceeded 2%: {numerical_failures:?}"
    );
    let c = loaded.model.config();
    let merge = c.spatial_merge_size * c.spatial_merge_size;
    let mut offset = 0;
    for grid in &grids {
        let n = grid[0] * grid[1] * grid[2];
        let separate = loaded
            .model
            .forward(
                patches.clone().slice([offset..offset + n, 0..shape[1]]),
                &[*grid],
            )
            .unwrap()
            .merged_states;
        let packed = output
            .merged_states
            .clone()
            .slice([offset / merge..(offset + n) / merge, 0..c.out_hidden_size]);
        let a = separate
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = packed.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let error = a
            .iter()
            .zip(b)
            .map(|(&a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("vision packed/separate offset={offset}: max_abs={error}");
        assert!(
            error <= 0.03125,
            "packed images must not attend across image boundaries"
        );
        offset += n;
    }
    assert!(loaded.model.forward(patches.clone(), &[]).is_err());
    assert!(loaded.model.forward(patches.clone(), &[[1, 3, 3]]).is_err());
    assert!(
        loaded
            .model
            .forward(patches, &[[usize::MAX, 2, 2]])
            .is_err()
    );
}
