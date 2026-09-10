use super::*;

#[test]
#[ignore = "requires a byte-validated original Delta layer trace and local model"]
fn real_delta_layer_components() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_DELTA_LAYER_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("layer.json")).unwrap()).unwrap();
    let index = meta["index"].as_u64().unwrap() as usize;
    let [batch, sequence, width]: [usize;3] = serde_json::from_value(meta["tensors"]["hidden"]["shape"].clone()).unwrap();
    let device = CudaDevice::default();
    let model = load_huggingface_qwen35_multimodal::<B>(std::env::var("RUDA_QWEN35_MODEL").unwrap(), &device).unwrap();
    let layer = &model.text.layers[index];
    let Mixer::Delta(delta) = &layer.mixer else { panic!("reference must identify a Delta layer") };
    let config = &model.text.config;
    assert_eq!(width, config.hidden_size);
    let values = config.linear_num_value_heads * config.linear_value_head_dim;
    let read = |name: &str, columns: usize, dtype| Tensor::<B,3>::from_data(
        TensorData::new(floats(&reference.join(format!("{name}.f32"))), [batch,sequence,columns]), (&device,dtype));
    let compare = |label: &str, name: &str, actual: Tensor<B,3>| {
        let actual = actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = floats(&reference.join(format!("{name}.f32")));
        assert_eq!(actual.len(),expected.len());
        assert!(actual.iter().all(|x| x.is_finite()));
        let different = actual.iter().zip(&expected).filter(|(a,b)| a!=b).count();
        let max_abs = actual.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
        let squared = actual.iter().zip(&expected).map(|(&a,&b)| (a as f64-b as f64).powi(2)).sum::<f64>();
        let norm = expected.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
        eprintln!("Delta layer {index} {label}: different={different}/{} max_abs={max_abs} relative_l2={}",actual.len(),(squared/norm).sqrt());
        different
    };
    let hidden = read("hidden",width,DType::BF16);
    let normalized = layer.input_norm.forward(hidden.clone()).unwrap();
    compare("input norm","input0",normalized.clone());
    for (name, value) in [("qkv0",delta.qkv.forward(normalized.clone())),
        ("a0",delta::Delta::gate_projection(&delta.a,normalized.clone()).unwrap()),
        ("b0",delta::Delta::gate_projection(&delta.b,normalized.clone()).unwrap()),
        ("z0",delta.z.forward(normalized.clone()))] {
        compare(name,name,value);
    }
    let update = delta.forward_observed(normalized,config,&mut None,&mut None,
        Some(&mut |name, value| { compare(name,name,value); })).unwrap();
    compare("mixer","mixer0",update.clone());
    let residual = hidden.clone()+update;
    let post = layer.post_norm.forward(residual.clone()).unwrap();
    compare("post norm","post0",post.clone());
    let mlp = layer.mlp.forward(post);
    compare("MLP","mlp0",mlp.clone());
    assert_eq!(compare("layer output","output",residual+mlp),0);

    let core_shape = [batch,sequence,config.linear_num_value_heads,config.linear_value_head_dim];
    let gated = delta::Delta::<B>::gated_normalize(read("core0",values,DType::F32).reshape(core_shape),
        read("z0",values,DType::F32).reshape(core_shape),delta.norm.clone(),DType::BF16,config.rms_norm_eps).unwrap();
    assert_eq!(compare("fixed core gated norm","gated0",gated.reshape([batch,sequence,values])),0);
    compare("fixed gated output projection","mixer0",delta.out.forward(read("gated0",values,DType::BF16)));
    compare("fixed mixer post norm","post0",layer.post_norm.forward(hidden+read("mixer0",width,DType::BF16)).unwrap());
    compare("fixed post MLP","mlp0",layer.mlp.forward(read("post0",width,DType::BF16)));
}
