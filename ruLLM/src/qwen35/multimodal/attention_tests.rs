use super::*;
use ruda_tensor::api::activation::{sigmoid,softmax};

#[test]
#[ignore = "requires RUDA_QWEN35_ATTENTION_REFERENCE from the unchanged real multimodal reference"]
fn real_full_attention_components() {
    type B = Cuda<bf16,i32>;
    let directory = PathBuf::from(std::env::var("RUDA_QWEN35_ATTENTION_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(directory.join("attention.json")).unwrap()).unwrap();
    let device = CudaDevice::default();
    let read = |name: &str,dtype| {
        let shape: [usize;4] = serde_json::from_value(meta[name].clone()).unwrap();
        Tensor::<B,4>::from_data(TensorData::new(floats(&directory.join(format!("{name}.f32"))),shape),(&device,dtype))
    };
    let compare = |label: &str,name: &str,actual: Tensor<B,4>| {
        let values = actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = floats(&directory.join(format!("{name}.f32")));
        assert_eq!(values.len(),expected.len());
        assert!(values.iter().all(|v| v.is_finite()));
        let mismatches = values.iter().zip(&expected).filter(|(a,b)| a!=b).count();
        let max = values.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
        let error = values.iter().zip(&expected).map(|(&a,&b)| (a as f64-b as f64).powi(2)).sum::<f64>();
        let norm = expected.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
        eprintln!("attention {label}: mismatches={mismatches}/{} max_abs={max} relative_l2={}",values.len(),(error/norm).sqrt());
        mismatches
    };
    let q = read("query",DType::BF16);
    let k = read("key",DType::BF16);
    let v = read("value",DType::BF16);
    let [b,h,s,d] = q.dims();
    let repeats = h/k.dims()[1];
    let mask = (0..s).flat_map(|i| (0..s).map(move |j| if j<=i {0f32} else {bf16::MIN.to_f32()})).collect();
    let scores = q.matmul(attention::repeat_heads(k,repeats).swap_dims(2,3))*(d as f64).sqrt().recip()
        + Tensor::from_data(TensorData::new(mask,[1,1,s,s]),(&device,DType::BF16));
    compare("qk/scaling/mask","scores",scores);
    let scores = read("scores",DType::F32);
    let generic = softmax(scores.clone(),3);
    compare("generic softmax F32","softmax-f32",generic.clone());
    compare("generic softmax BF16","probabilities",generic.cast(DType::BF16));
    let native = attention::Attention::<B>::probabilities(scores).unwrap();
    assert_eq!(compare("production ruDNN softmax F32","softmax-f32",native.clone()),0);
    assert_eq!(compare("production ruDNN softmax BF16","probabilities",native.cast(DType::BF16)),0);
    let values = attention::repeat_heads(v,repeats);
    let core = read("probabilities",DType::BF16).matmul(values.clone()).swap_dims(1,2);
    compare("probabilities/value matmul","core",core.clone());
    let residue = attention::Attention::<B>::value_product(read("probabilities",DType::BF16),values.clone()).unwrap();
    assert_eq!(compare("residue-first probabilities/value matmul","core",residue.swap_dims(1,2)),0);
    let transposed = values.clone().swap_dims(2,3).matmul(read("probabilities",DType::BF16).swap_dims(2,3)).swap_dims(2,3).swap_dims(1,2);
    compare("transposed operand matmul","core",transposed);
    let f32_output = rublas::tensor_matmul::matmul(read("probabilities",DType::BF16).into_primitive().tensor(),
        values.clone().into_primitive().tensor(),None,rublas::tensor_matmul::MatmulStrategy::Ruda,DType::F32).unwrap();
    let f32_output = Tensor::<B,4>::from_primitive(TensorPrimitive::Float(f32_output)).swap_dims(1,2);
    compare("separate output cast","core",f32_output.clone().cast(DType::BF16));
    if let Ok(path) = std::env::var("RUDA_QWEN35_ATTENTION_ACCUM_OUTPUT") {
        let actual = core.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let expected = floats(&directory.join("core.f32"));
        let accum = f32_output.into_data().to_vec::<f32>().unwrap();
        let probabilities = floats(&directory.join("probabilities.f32"));
        let value = floats(&directory.join("value.f32"));
        let mut differences = Vec::new();
        for (index, (&observed, &reference)) in actual.iter().zip(&expected).enumerate() {
            if observed == reference { continue; }
            let feature = index % d;
            let head = index / d % h;
            let row = index / (d*h) % s;
            let batch = index / (d*h*s);
            let mut precise = 0f64;
            let mut sequential = 0f32;
            for k in 0..s {
                let left = probabilities[((batch*h+head)*s+row)*s+k];
                let right = value[((batch*(h/repeats)+head/repeats)*s+k)*d+feature];
                precise += left as f64 * right as f64;
                sequential = left.mul_add(right,sequential);
            }
            let record = serde_json::json!({"index":[batch,row,head,feature],
                "actual_bf16":observed,"expected_bf16":reference,"tensor_core_f32":accum[index],
                "tensor_core_f32_bits":accum[index].to_bits(),"sequential_f32":sequential,
                "diagnostic_f64_sum":precise});
            eprintln!("attention accumulation boundary: {record}");
            differences.push(record);
        }
        use std::io::Write;
        let mut output = fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
        output.write_all(&serde_json::to_vec_pretty(&differences).unwrap()).unwrap();
    }
    let naive = rublas::tensor_matmul::matmul(read("probabilities",DType::BF16).into_primitive().tensor(),
        values.clone().into_primitive().tensor(), None, rublas::tensor_matmul::MatmulStrategy::Naive,DType::BF16).unwrap();
    let naive = Tensor::<B,4>::from_primitive(TensorPrimitive::Float(naive)).swap_dims(1,2);
    compare("probabilities/value Naive matmul","core",naive);
    use rublas::kernel_ir::{definition::{MatmulElems,MatmulGlobalElems},launch::{Strategy,launch_ref}};
    use ruda_kernel::tiling::InputBinding;
    let lhs = read("probabilities",DType::BF16).into_primitive().tensor();
    let rhs = values.into_primitive().tensor();
    for (label,strategy) in [
        ("explicit MMA",Strategy::SimpleCyclicMma(Default::default())),
        ("tilewise MMA",Strategy::SimpleTilewiseMma(Default::default())),
        ("double-buffered MMA",Strategy::DoubleCyclicMma(Default::default())),
    ] {
        let output = rublas::tensor_matmul::init_matmul_output(&lhs,&rhs,DType::BF16);
        let mut elems = MatmulElems::from_globals(&MatmulGlobalElems {
            f32_math: Default::default(),
            lhs: DType::BF16.into(),rhs: DType::BF16.into(),out: DType::BF16.into(),
        });
        launch_ref(&strategy,&lhs.client,InputBinding::new(lhs.clone().binding(),DType::BF16.into()),
            InputBinding::new(rhs.clone().binding(),DType::BF16.into()),output.clone().binding(),&mut elems).unwrap();
        compare(label,"core",Tensor::<B,4>::from_primitive(TensorPrimitive::Float(output)).swap_dims(1,2));
    }
    let shape: [usize;3] = serde_json::from_value(meta["qgate"].clone()).unwrap();
    let qgate = Tensor::<B,3>::from_data(TensorData::new(floats(&directory.join("qgate.f32")),shape),(&device,DType::BF16));
    let gate = qgate.reshape([b,s,h,2*d]).slice([0..b,0..s,0..h,d..2*d]);
    compare("gate multiplication","gated",read("core",DType::BF16)*sigmoid(gate));

    let model_dir = PathBuf::from(std::env::var("RUDA_QWEN35_MODEL").unwrap());
    let raw: serde_json::Value = serde_json::from_slice(&fs::read(model_dir.join("config.json")).unwrap()).unwrap();
    let c: Qwen35TextConfig = serde_json::from_value(raw["text_config"].clone()).unwrap();
    let mut checkpoint = crate::huggingface::checkpoint::Checkpoint::open(&model_dir).unwrap();
    let mut w = crate::qwen35::loading::Weights::<B> { checkpoint: &mut checkpoint, device: &device };
    let prefix = "model.language_model.layers.3.self_attn";
    let layer = attention::Attention {
        q: w.linear(&format!("{prefix}.q_proj"), c.hidden_size, 2*h*d, c.attention_bias).unwrap(),
        k: w.linear(&format!("{prefix}.k_proj"), c.hidden_size, c.num_key_value_heads*d, c.attention_bias).unwrap(),
        v: w.linear(&format!("{prefix}.v_proj"), c.hidden_size, c.num_key_value_heads*d, c.attention_bias).unwrap(),
        out: w.linear(&format!("{prefix}.o_proj"), h*d, c.hidden_size, c.attention_bias).unwrap(),
        q_norm: crate::qwen35::Norm { weight: w.tensor(&format!("{prefix}.q_norm.weight"), [d]).unwrap(), epsilon: c.rms_norm_eps },
        k_norm: crate::qwen35::Norm { weight: w.tensor(&format!("{prefix}.k_norm.weight"), [d]).unwrap(), epsilon: c.rms_norm_eps },
    };
    let read3 = |name: &str| {
        let shape: [usize;3] = serde_json::from_value(meta[name].clone()).unwrap();
        Tensor::<B,3>::from_data(TensorData::new(floats(&directory.join(format!("{name}.f32"))), shape), (&device,DType::BF16))
    };
    let x = read3("input");
    let cos = read3("cos").unsqueeze_dim::<4>(1);
    let sin = read3("sin").unsqueeze_dim::<4>(1);
    let qgate = layer.q.forward(x.clone());
    assert_eq!(compare("actual q projection","qgate",qgate.clone().unsqueeze::<4>()),0);
    let query = layer.q_norm.forward(qgate.reshape([b,s,h,2*d]).slice([0..b,0..s,0..h,0..d])).unwrap().swap_dims(1,2);
    compare("actual query norm/rope","query",attention::rotate(query,cos.clone(),sin.clone()));
    let key = layer.k_norm.forward(layer.k.forward(x.clone()).reshape([b,s,c.num_key_value_heads,d])).unwrap().swap_dims(1,2);
    compare("actual key norm/rope","key",attention::rotate(key,cos.clone(),sin.clone()));
    assert_eq!(compare("actual value projection","value",layer.v.forward(x.clone()).reshape([b,s,c.num_key_value_heads,d]).swap_dims(1,2)),0);
    compare("fixed gated output projection","projected",layer.out.forward(read3("gated")).unsqueeze::<4>());
    assert_eq!(compare("actual full attention","projected",layer.forward(x,&c,cos,sin,&mut None,&mut None,0).unwrap().unsqueeze::<4>()),0);
}

#[test]
fn residue_first_matmul_shapes_dtypes_and_layouts() {
    type B = Cuda<bf16,i32>;
    let device = CudaDevice::default();
    let (b,m,n) = (2usize,3usize,5usize);
    for dtype in [DType::BF16,DType::F16] {
        for (case,k) in [1usize,16,17,31,32,33,148,164].into_iter().enumerate() {
            let left = (0..b*m*k).map(|i| ((i%9) as f32-4.0)/8.0).collect::<Vec<_>>();
            let right = (0..k*n).map(|i| ((i%7) as f32-3.0)/8.0).collect::<Vec<_>>();
            let mut lhs = Tensor::<B,4>::from_data(TensorData::new(left.clone(),[b,1,m,k]),(&device,dtype));
            let mut rhs = Tensor::<B,4>::from_data(TensorData::new(right.clone(),[1,1,k,n]),(&device,dtype));
            if case%2 != 0 {
                let mut transposed_left=Vec::new();
                for batch in 0..b { for col in 0..k { for row in 0..m { transposed_left.push(left[(batch*m+row)*k+col]); } } }
                let transposed_right=(0..n).flat_map(|col| (0..k).map({let right=&right; move |row| right[row*n+col]})).collect::<Vec<_>>();
                lhs=Tensor::<B,4>::from_data(TensorData::new(transposed_left,[b,1,k,m]),(&device,dtype)).swap_dims(2,3);
                rhs=Tensor::<B,4>::from_data(TensorData::new(transposed_right,[1,1,n,k]),(&device,dtype)).swap_dims(2,3);
            }
            for output_dtype in [dtype,DType::F32] {
                let output=rublas::tensor_matmul::matmul(lhs.clone().into_primitive().tensor(),rhs.clone().into_primitive().tensor(),None,
                    rublas::tensor_matmul::MatmulStrategy::CmmaResidueFirst,output_dtype).unwrap();
                let actual=Tensor::<B,4>::from_primitive(TensorPrimitive::Float(output));
                assert_eq!(actual.dims(),[b,1,m,n]);
                assert_eq!(actual.dtype(),output_dtype);
                let actual=actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
                for batch in 0..b { for row in 0..m { for col in 0..n {
                    let expected=(0..k).map(|t| left[(batch*m+row)*k+t]*right[t*n+col]).sum::<f32>();
                    let expected=match output_dtype {
                        DType::BF16=>bf16::from_f32(expected).to_f32(),
                        DType::F16=>half::f16::from_f32(expected).to_f32(),
                        _=>expected,
                    };
                    assert_eq!(actual[(batch*m+row)*n+col],expected,"{dtype:?}, {output_dtype:?}, K={k}, ({batch},{row},{col})");
                } } }
            }
        }
    }
    let lhs=Tensor::<B,4>::ones([1,1,3,4],(&device,DType::F32));
    let rhs=Tensor::<B,4>::ones([1,1,4,5],(&device,DType::BF16));
    assert!(rublas::tensor_matmul::matmul(lhs.into_primitive().tensor(),rhs.into_primitive().tensor(),None,
        rublas::tensor_matmul::MatmulStrategy::CmmaResidueFirst,DType::BF16).is_err());
}
