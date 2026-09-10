use super::*;

#[test]
#[ignore = "requires RUDA_QWEN35_GATED_NORM_REFERENCE with fixed original core/gate/weight"]
fn real_core_gated_normalization() {
    type B = Cuda<bf16,i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_GATED_NORM_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    assert_eq!(meta["original_gated_exact"],true);
    let shape: [usize;4] = serde_json::from_value(meta["shape"].clone()).unwrap();
    let epsilon = meta["epsilon"].as_f64().unwrap();
    let weight_dtype = match meta["weight_dtype"].as_str().unwrap() {
        "torch.float32" => DType::F32,
        "torch.bfloat16" => DType::BF16,
        name => panic!("unexpected reference weight dtype: {name}"),
    };
    let device = CudaDevice::default();
    let input = |name| Tensor::<B,4>::from_data(
        TensorData::new(floats(&reference.join(format!("{name}.f32"))),shape),(&device,DType::F32));
    let weight = Tensor::<B,1>::from_data(TensorData::new(floats(&reference.join("weight.f32")),[shape[3]]),(&device,weight_dtype));
    let compare = |name, value: Tensor<B,4>| {
        let actual = value.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = floats(&reference.join(format!("{name}.f32")));
        assert_eq!(actual.len(),expected.len());
        assert!(actual.iter().all(|x| x.is_finite()));
        let different = actual.iter().zip(&expected).filter(|(a,b)| a != b).count();
        let max = actual.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
        eprintln!("gated norm {name}: different={different}/{} max_abs={max}",actual.len());
        different
    };
    let core = input("core");
    let gate = input("gate");
    let variance = core.clone().square().mean_dim(3);
    compare("variance",variance.clone());
    let inverse = (variance + epsilon).rsqrt();
    compare("inverse",inverse.clone());
    let normalized = (core.clone() * inverse).cast(DType::BF16);
    compare("normalized",normalized.clone());
    compare("weighted",normalized.cast(weight_dtype) * weight.clone().unsqueeze::<4>());
    compare("activated",silu(gate.clone()));
    let output = delta::Delta::gated_normalize(core,gate,weight,DType::BF16,epsilon).unwrap();
    assert_eq!(compare("output",output),0);
}
