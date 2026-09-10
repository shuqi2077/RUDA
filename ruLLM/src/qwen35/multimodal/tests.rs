use super::*;
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use std::{fs, path::PathBuf};

#[path = "chunk_tests.rs"]
mod chunk_tests;

#[path = "silu_tests.rs"]
mod silu_tests;

#[path = "attention_tests.rs"]
mod attention_tests;

#[path = "delta_layer_tests.rs"]
mod delta_layer_tests;

#[path = "gated_norm_tests.rs"]
mod gated_norm_tests;

#[path = "norm_tests.rs"]
mod norm_tests;

fn floats(path: &Path) -> Vec<f32> {
    fs::read(path).unwrap().chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[test]
fn generation_propagates_prefill_and_decode_errors() {
    type B = Cuda<bf16,i32>;
    struct FailsAt(usize);
    impl CausalModel<B> for FailsAt {
        type Cache = usize;
        fn new_cache(&self) -> usize { 0 }
        fn forward_cached_last(&self, _:Tensor<B,2,Int>,_:&mut usize) -> Tensor<B,3> {
            panic!("generation must call the fallible entry point")
        }
        fn try_forward_cached_last(&self,tokens:Tensor<B,2,Int>,cache:&mut usize) -> Result<Tensor<B,3>,GenerationError> {
            if *cache == self.0 { return Err(GenerationError(format!("failure at {}",self.0))); }
            *cache += 1;
            Ok(Tensor::zeros([1,1,2],(&tokens.device(),DType::BF16)))
        }
    }
    for step in [0,1] {
        let result = crate::generate_causal_greedy(&FailsAt(step),
            &crate::CausalModelLimits { vocab_size:2,max_sequence_length:3 },&[0],
            &crate::GreedyGenerationConfig { max_new_tokens:2,eos_token_ids:vec![] },&CudaDevice::default());
        assert_eq!(result.unwrap_err().0,format!("failure at {step}"));
    }
}

#[test]
#[ignore = "requires local Qwen3.5 model and fixed multimodal reference; native JPEG generation"]
fn real_checkpoint_native_image_generation() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ruda-dataset/tests/data/image_folder_coco");
    native_image_generation(&[fixtures.join("one_dot.jpg"),fixtures.join("two_dots_and_triangle.jpg")]);
}

#[test]
#[ignore = "requires local Qwen3.5 model, fixed multimodal reference and original RGB PNG fixtures"]
fn real_checkpoint_native_png_generation() {
    let fixtures = PathBuf::from(std::env::var("RUDA_QWEN35_PNG_REFERENCE").unwrap());
    native_image_generation(&[fixtures.join("image-0.png"),fixtures.join("image-1.png")]);
}

fn native_image_generation(images: &[PathBuf]) {
    type B = Cuda<bf16,i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_MULTIMODAL_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let prompt: Vec<i32> = serde_json::from_value(meta["prompt_ids"].clone()).unwrap();
    let expected: Vec<i32> = serde_json::from_value(meta["generated_token_ids"].clone()).unwrap();
    let model_dir = std::env::var("RUDA_QWEN35_MODEL").unwrap();
    let device = CudaDevice::default();
    let model = load_huggingface_qwen35_multimodal::<B>(&model_dir,&device).unwrap();
    let processor = Qwen35ImageProcessor::from_huggingface(&model_dir).unwrap();
    let unexpanded = prompt.iter().enumerate().filter_map(|(i,&id)| {
        (id != model.image_token_id || i == 0 || prompt[i-1] != id).then_some(id)
    }).collect::<Vec<_>>();
    let output = model.generate_image_files_greedy(&unexpanded,images,&processor,
        &crate::GreedyGenerationConfig { max_new_tokens:expected.len(),eos_token_ids:vec![model.text.config.eos_token_id] }).unwrap();
    eprintln!("native image generation tokens={:?}",output.generated_token_ids);
    assert_eq!(&output.token_ids[..prompt.len()],prompt);
    assert_eq!(output.generated_token_ids,expected);
    assert!(!output.stopped_on_eos);
}

#[test]
#[ignore = "diagnostic only; requires RUDA_QWEN35_MULTIMODAL_REFERENCE with layer traces"]
fn real_checkpoint_multimodal_layer_diagnosis() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_MULTIMODAL_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let prompt: Vec<i32> = serde_json::from_value(meta["prompt_ids"].clone()).unwrap();
    let grids: Vec<[usize;3]> = serde_json::from_value(meta["grids"].clone()).unwrap();
    let shape: [usize;2] = serde_json::from_value(meta["patch_shape"].clone()).unwrap();
    let device = CudaDevice::default();
    let model = load_huggingface_qwen35_multimodal::<B>(std::env::var("RUDA_QWEN35_MODEL").unwrap(), &device).unwrap();
    let compare = |name: &str, actual: Vec<f32>| {
        let expected = floats(&reference.join(format!("{name}.f32")));
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|v| v.is_finite()));
        let error = actual.iter().zip(&expected).map(|(&a,&b)| (a as f64-b as f64).powi(2)).sum::<f64>();
        let norm = expected.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
        let max = actual.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
        eprintln!("multimodal trace {name}: relative_l2={} max_abs={max}", (error/norm).sqrt());
    };
    let positions = position::image_positions(&[prompt.clone()],model.image_token_id,&grids,model.vision.config().spatial_merge_size).unwrap();
    let features = model.vision.forward(Tensor::<B,2>::from_data(
        TensorData::new(floats(&reference.join("patches.f32")),shape),(&device,DType::F32)),&grids).unwrap().merged_states;
    compare("visual", features.clone().cast(DType::F32).into_data().to_vec().unwrap());
    let width = model.text.config.hidden_size;
    let mut hidden = model.text.embedding.forward(Tensor::<B,2,Int>::from_data(TensorData::new(prompt.clone(),[1,prompt.len()]),&device));
    for span in positions.spans {
        let value = features.clone().slice([span.feature_start..span.feature_start+span.count,0..width]).cast(hidden.dtype()).reshape([1,span.count,width]);
        hidden = hidden.slice_assign([0..1,span.start..span.start+span.count,0..width],value);
    }
    compare("embedded",hidden.clone().cast(DType::F32).into_data().to_vec().unwrap());
    if reference.join("input0.f32").exists() {
        let layer = &model.text.layers[0];
        let normalized = layer.input_norm.forward(hidden.clone()).unwrap();
        compare("input0",normalized.clone().cast(DType::F32).into_data().to_vec().unwrap());
        if let Mixer::Delta(delta) = &layer.mixer {
            for (name, value) in [("qkv0",delta.qkv.forward(normalized.clone())),
                ("a0",delta::Delta::gate_projection(&delta.a,normalized.clone()).unwrap()),("b0",delta::Delta::gate_projection(&delta.b,normalized.clone()).unwrap()),
                ("z0",delta.z.forward(normalized.clone()))] {
                compare(name,value.cast(DType::F32).into_data().to_vec().unwrap());
            }
            let update = delta.forward_observed(normalized,&model.text.config,&mut None,&mut None,
                Some(&mut |name, value| {
                    if reference.join(format!("{name}.f32")).exists() {
                        compare(name,value.cast(DType::F32).into_data().to_vec().unwrap());
                    }
                })).unwrap();
            compare("mixer0",update.clone().cast(DType::F32).into_data().to_vec().unwrap());
            let post = layer.post_norm.forward(hidden.clone()+update).unwrap();
            compare("post0",post.clone().cast(DType::F32).into_data().to_vec().unwrap());
            compare("mlp0",layer.mlp.forward(post).cast(DType::F32).into_data().to_vec().unwrap());
        }
    }
    let (cos,sin) = attention::multimodal_rope::<B>(&model.text.config,&positions.coordinates,1,prompt.len(),hidden.dtype(),&device).unwrap();
    model.text.run_layers_observed(hidden,cos,sin,&mut model.text.new_cache(),|index,hidden| {
        compare(&format!("layer-{index}"),hidden.clone().cast(DType::F32).into_data().to_vec().unwrap());
    }).unwrap();
}

#[test]
#[ignore = "diagnostic only; each layer receives the fixed independent previous-layer output"]
fn real_checkpoint_isolated_layer_diagnosis() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_MULTIMODAL_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let prompt: Vec<i32> = serde_json::from_value(meta["prompt_ids"].clone()).unwrap();
    let grids: Vec<[usize;3]> = serde_json::from_value(meta["grids"].clone()).unwrap();
    let device = CudaDevice::default();
    let model = load_huggingface_qwen35_multimodal::<B>(std::env::var("RUDA_QWEN35_MODEL").unwrap(), &device).unwrap();
    let positions = position::image_positions(&[prompt.clone()], model.image_token_id, &grids, model.vision.config().spatial_merge_size).unwrap();
    let (cos, sin) = attention::multimodal_rope::<B>(&model.text.config, &positions.coordinates, 1, prompt.len(), DType::BF16, &device).unwrap();
    let mut cache = model.text.new_cache();
    for (index, (layer, state)) in model.text.layers.iter().zip(cache.layers.iter_mut()).enumerate() {
        let input_name = if index == 0 { "embedded".into() } else { format!("layer-{}", index-1) };
        let hidden = Tensor::<B,3>::from_data(TensorData::new(floats(&reference.join(format!("{input_name}.f32"))), [1, prompt.len(), model.text.config.hidden_size]), (&device, DType::BF16));
        let output = layer.forward(hidden, &model.text.config, cos.clone(), sin.clone(), state, 0).unwrap();
        let actual = output.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = floats(&reference.join(format!("layer-{index}.f32")));
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|x| x.is_finite()));
        let squared = actual.iter().zip(&expected).map(|(&a,&b)| (a as f64-b as f64).powi(2)).sum::<f64>();
        let norm = expected.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        let max = actual.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
        eprintln!("isolated layer {index}: relative_l2={} max_abs={max}", (squared/norm).sqrt());
    }
}

#[test]
#[ignore = "diagnostic only; fixed first-layer projections across matmul strategies"]
fn real_checkpoint_projection_diagnosis() {
    type B = Cuda<bf16, i32>;
    use rublas::tensor_matmul::{matmul, MatmulStrategy};
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_MULTIMODAL_REFERENCE").unwrap());
    let device = CudaDevice::default();
    let model = load_huggingface_qwen35_multimodal::<B>(std::env::var("RUDA_QWEN35_MODEL").unwrap(), &device).unwrap();
    let input = Tensor::<B,3>::from_data(TensorData::new(floats(&reference.join("input0.f32")),[1,148,1024]),(&device,DType::BF16));
    let Mixer::Delta(delta) = &model.text.layers[0].mixer else { panic!("reference requires first delta layer") };
    for (name,linear) in [("a0",&delta.a),("b0",&delta.b)] {
        let expected = floats(&reference.join(format!("{name}.f32")));
        let actual = delta::Delta::gate_projection(linear,input.clone()).unwrap().cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        assert_eq!(actual,expected,"precision-sensitive gate projection must match fixed reference");
        for (strategy_name,strategy) in [("default",MatmulStrategy::default()),("ruda",MatmulStrategy::Ruda),("naive",MatmulStrategy::Naive)] {
            let result = matmul(input.clone().into_primitive().tensor(),linear.weight.val().unsqueeze::<3>().into_primitive().tensor(),None,strategy,DType::BF16).unwrap();
            let actual = Tensor::<B,3>::from_primitive(TensorPrimitive::Float(result)).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
            let error = actual.iter().zip(&expected).map(|(&a,&b)| (a as f64-b as f64).powi(2)).sum::<f64>();
            let norm = expected.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
            let differing = actual.iter().zip(&expected).filter(|(a,b)| a!=b).count();
            let max = actual.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
            eprintln!("projection {name} {strategy_name}: differing={differing} relative_l2={} max_abs={max}",(error/norm).sqrt());
        }
        for (label,naive) in [("ruda",false),("naive",true)] {
            B::sync(&device).unwrap();
            let start = std::time::Instant::now();
            for _ in 0..20 {
                let strategy = if naive { MatmulStrategy::Naive } else { MatmulStrategy::Ruda };
                let _ = matmul(input.clone().into_primitive().tensor(),linear.weight.val().unsqueeze::<3>().into_primitive().tensor(),None,strategy,DType::BF16).unwrap();
            }
            B::sync(&device).unwrap();
            eprintln!("projection {name} {label}: 20 warm submissions+sync {:?}; includes host dispatch, not kernel-only time",start.elapsed());
        }
    }
}

#[test]
#[ignore = "requires RUDA_QWEN35_MODEL and RUDA_QWEN35_MULTIMODAL_REFERENCE"]
fn real_checkpoint_image_prefill_and_decode() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_MULTIMODAL_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let prompt: Vec<i32> = serde_json::from_value(meta["prompt_ids"].clone()).unwrap();
    let generated: Vec<i32> = serde_json::from_value(meta["generated_token_ids"].clone()).unwrap();
    let grids: Vec<[usize; 3]> = serde_json::from_value(meta["grids"].clone()).unwrap();
    let shape: [usize; 2] = serde_json::from_value(meta["patch_shape"].clone()).unwrap();
    let device = CudaDevice::default();
    let model = load_huggingface_qwen35_multimodal::<B>(std::env::var("RUDA_QWEN35_MODEL").unwrap(), &device).unwrap();
    let p = position::image_positions(&[prompt.clone()], model.image_token_id, &grids, model.vision.config().spatial_merge_size).unwrap();
    let expected_positions: Vec<Vec<usize>> = serde_json::from_value(meta["positions"].clone()).unwrap();
    for (i, coords) in p.coordinates.iter().enumerate() {
        for axis in 0..3 { assert_eq!(coords[axis], expected_positions[axis][i]); }
    }
    let patches = Tensor::<B, 2>::from_data(TensorData::new(floats(&reference.join("patches.f32")), shape), (&device, DType::F32));
    let mut cache = model.new_cache();
    let mut logits = model.prefill_images(&[prompt.clone()], patches.clone(), &grids, &mut cache).unwrap();
    assert_eq!(cache.next_positions(), &p.next);
    assert!(model.prefill_images(&[prompt.clone()], patches, &grids, &mut cache).is_err());
    for (step, &expected_token) in generated.iter().enumerate() {
        let actual = logits.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = floats(&reference.join(format!("logits-{step}.f32")));
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|x| x.is_finite()));
        let squared = actual.iter().zip(&expected).map(|(&a,&b)| (a as f64-b as f64).powi(2)).sum::<f64>();
        let norm = expected.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        let relative = (squared/norm).sqrt();
        let best = actual.iter().enumerate().fold(0, |best,(i,&x)| if x > actual[best] { i } else { best });
        eprintln!("multimodal step={step} argmax={best} expected={expected_token} relative_l2={relative}");
        assert_eq!(best as i32, expected_token);
        assert!(relative < 0.02, "multimodal logits relative L2 exceeded 2%");
        assert_eq!(cache.sequence_length(), prompt.len()+step);
        assert_eq!(cache.next_positions()[0], p.next[0]+step);
        if step+1 < generated.len() {
            logits = model.decode(Tensor::<B, 2, Int>::from_data([[expected_token]], &device), &mut cache).unwrap();
        } else { break; }
    }
}
