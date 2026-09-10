use super::*;
use rudnn::gated_delta::{GatedDeltaInput, chunk_gated_delta_rule};

#[test]
#[ignore = "requires RUDA_QWEN35_DECAY_REFERENCE with original real projection and checkpoint parameters"]
fn real_input_log_decay_matches_reference() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_DECAY_REFERENCE").unwrap());
    let device = CudaDevice::default();
    let a = Tensor::<B,3>::from_data(TensorData::new(floats(&reference.join("a.f32")),[1,148,16]),(&device,DType::BF16));
    let parameter = |name| Tensor::<B,1>::from_data(
        TensorData::new(floats(&reference.join(format!("{name}.f32"))),[16]),(&device,DType::F32));
    let actual = delta::Delta::log_decay(a,parameter("bias"),parameter("log-a"))
        .swap_dims(1,2).into_data().to_vec::<f32>().unwrap();
    let expected = floats(&reference.join("decay.f32"));
    let differing = actual.iter().zip(&expected).filter(|(a,b)| a != b).count();
    eprintln!("Real log decay: mismatches={differing}/{}",expected.len());
    assert_eq!(actual,expected);
}

#[test]
#[ignore = "requires RUDA_QWEN35_CHUNK_REFERENCE exported by pinned HF from the real first-layer trace"]
fn real_input_chunk_output_and_state() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_QWEN35_CHUNK_REFERENCE").unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let shape: [usize;4] = serde_json::from_value(meta["shape"].clone()).unwrap();
    let [batch, heads, sequence, kd] = shape;
    let vd = meta["value_dim"].as_u64().unwrap() as usize;
    let chunk = meta["chunk_size"].as_u64().unwrap() as usize;
    assert_eq!(meta["original_core_exact"], true);
    let device = CudaDevice::default();
    let read4 = |name, shape, dtype| Tensor::<B,4>::from_data(
        TensorData::new(floats(&reference.join(format!("{name}.f32"))), shape), (&device,dtype));
    let read3 = |name, dtype| Tensor::<B,3>::from_data(
        TensorData::new(floats(&reference.join(format!("{name}.f32"))), [batch,heads,sequence]), (&device,dtype));
    for continuation in [false, true] {
        let initial = if continuation { read4("initial", [batch,heads,kd,vd],DType::F32) }
            else { Tensor::zeros([batch,heads,kd,vd],(&device,DType::F32)) };
        let retained = initial.clone();
        let before = initial.clone().into_data().to_vec::<f32>().unwrap();
        let result = chunk_gated_delta_rule(GatedDeltaInput {
            query: read4("query",shape,DType::BF16).into_primitive().tensor(),
            key: read4("key",shape,DType::BF16).into_primitive().tensor(),
            value: read4("value",[batch,heads,sequence,vd],DType::BF16).into_primitive().tensor(),
            beta: read3("beta",DType::BF16).into_primitive().tensor(),
            log_decay: read3("decay",DType::F32).into_primitive().tensor(),
            initial_state: initial.into_primitive().tensor(),
            query_scale: (kd as f32).sqrt().recip(),
        },chunk).unwrap();
        for (name, actual, tolerance) in [
            (if continuation { "continued-output" } else { "output" },result.output,0.003),
            (if continuation { "continued-state" } else { "state" },result.final_state,0.0001),
        ] {
            let actual = Tensor::<B,4>::from_primitive(TensorPrimitive::Float(actual)).cast(DType::F32)
                .into_data().to_vec::<f32>().unwrap();
            let expected = floats(&reference.join(format!("{name}.f32")));
            assert_eq!(actual.len(),expected.len());
            assert!(actual.iter().all(|v| v.is_finite()));
            let err = actual.iter().zip(&expected).map(|(&a,&b)| (a as f64-b as f64).powi(2)).sum::<f64>();
            let norm = expected.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
            let relative = (err/norm).sqrt();
            eprintln!("HF real chunk {name}: relative_l2={relative}");
            assert!(relative < tolerance,"{name}: {relative} >= {tolerance}");
        }
        assert_eq!(retained.into_data().to_vec::<f32>().unwrap(),before);
    }
}
