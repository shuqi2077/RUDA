use super::*;
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use std::{fs, path::PathBuf};

type B = Cuda<bf16, i32>;

mod arithmetic_contract {
    use super::*;
    use ruda_kernel::dsl::prelude::*;
    use ruda_tensor::api::Tensor;

    #[ruda(launch)]
    fn separate_and_fused(input: &Array<f32>, output: &mut Array<f32>) {
        let a = input[0];
        let b = input[1];
        let c = input[2];
        let product = a * b;
        output[0] = product + c;
        output[1] = fma(a,b,c);
    }

    #[test]
    fn strict_multiply_add_preserves_rounding() {
        let device = CudaDevice::default();
        let a = 1.0f32 + f32::EPSILON;
        let b = 1.0f32 - f32::EPSILON;
        let input = Tensor::<B,1>::from_data(TensorData::new(vec![a,b,-1.0],[3]),(&device,DType::F32)).into_primitive().tensor();
        let output = Tensor::<B,1>::zeros([2],(&device,DType::F32)).into_primitive().tensor();
        separate_and_fused::launch::<ruda_driver_cuda::CudaRuntime>(&input.client,
            RudaCount::Static(1,1,1),RudaDim::new_1d(1),input.clone().into_array_arg(),output.clone().into_array_arg());
        let actual = Tensor::<B,1>::from_primitive(TensorPrimitive::Float(output)).into_data().to_vec::<f32>().unwrap();
        assert_eq!(actual,vec![(a*b)-1.0,a.mul_add(b,-1.0)],"strict multiply/add and explicit FMA must retain distinct rounding");
    }
}

#[test]
fn normalization_rsqrt_dtypes_layout_and_special_values() {
    let device = CudaDevice::default();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let values = (0..512).map(|i| 10f32.powf(-4.0 + 8.0*i as f32/511.0)).collect::<Vec<_>>();
        let input = Tensor::<B,2>::from_data(TensorData::new(values,[16,32]),(&device,dtype)).transpose();
        let rounded = input.clone().cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let result = input.rsqrt();
        assert_eq!(result.dims(),[32,16]);
        assert_eq!(result.dtype(),dtype);
        let actual = result.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        for (x,y) in rounded.into_iter().zip(actual) {
            let exact = (1.0/(x as f64).sqrt()) as f32;
            let expected = match dtype {
                DType::F16 => half::f16::from_f32(exact).to_f32(),
                DType::BF16 => bf16::from_f32(exact).to_f32(),
                _ => exact,
            };
            assert!((y-expected).abs() <= expected.abs()*2e-6, "{dtype:?} rsqrt({x}): {y} != {expected}");
        }
        let special = Tensor::<B,1>::from_data(TensorData::new(
            vec![0f32,-0.0,f32::INFINITY,-1.0,f32::NAN], [5]),(&device,dtype))
            .rsqrt().cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        assert_eq!(special[0],f32::INFINITY);
        assert_eq!(special[1],f32::NEG_INFINITY);
        assert_eq!(special[2],0.0);
        assert!(special[3].is_nan() && special[4].is_nan());
    }
}

#[test]
#[ignore = "requires RUDA_QWEN35_MODEL and RUDA_QWEN35_REFERENCE local acceptance inputs"]
fn real_checkpoint_reference_and_cached_continuation() {
    let directory = std::env::var("RUDA_QWEN35_MODEL").expect("model directory");
    let reference =
        PathBuf::from(std::env::var("RUDA_QWEN35_REFERENCE").expect("reference directory"));
    let expected: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.join("result.json")).unwrap()).unwrap();
    let prompt: Vec<i32> = serde_json::from_value(expected["prompt_ids"].clone()).unwrap();
    let generated: Vec<i32> =
        serde_json::from_value(expected["generated_token_ids"].clone()).unwrap();
    let device = CudaDevice::default();
    let loaded = load_huggingface_qwen35_text::<B>(directory, &device).unwrap();
    assert!(loaded.report.applied_tensors > 0);
    let model = loaded.model;
    let input = |ids: &[i32]| {
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), &device)
    };
    let mut cache = model.new_cache();
    let mut all = prompt.clone();
    for step in 0..generated.len() {
        let ids = if step == 0 {
            prompt.as_slice()
        } else {
            &generated[step - 1..step]
        };
        let logits = model.forward_cached_last(input(ids), &mut cache).unwrap();
        let [_, sequence, vocab] = logits.dims();
        let row = logits
            .slice([0..1, sequence - 1..sequence, 0..vocab])
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let file = if step == 0 {
            "prefill-logits.f32".into()
        } else {
            format!("decode-{}-logits.f32", step - 1)
        };
        let bytes = fs::read(reference.join(file)).unwrap();
        let expected_row = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(row.len(), expected_row.len());
        let max_error = row
            .iter()
            .zip(&expected_row)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rmse = (row
            .iter()
            .zip(&expected_row)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / row.len() as f64)
            .sqrt();
        let best = row
            .iter()
            .enumerate()
            .fold(0, |best, (i, v)| if *v > row[best] { i } else { best });
        eprintln!(
            "reference step={step} argmax={best} expected={} max_error={max_error} rmse={rmse}",
            generated[step]
        );
        assert!(row.iter().all(|n| n.is_finite()));
        assert_eq!(
            best as i32, generated[step],
            "independent reference greedy token at step {step}"
        );
        if step > 0 {
            all.push(generated[step - 1]);
        }
        assert_eq!(cache.sequence_length(), all.len());
        let mut fresh = model.new_cache();
        let full = model.forward_cached(input(&all), &mut fresh).unwrap();
        let s = all.len();
        let full = full
            .slice([0..1, s - 1..s, 0..vocab])
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let full_best = full
            .iter()
            .enumerate()
            .fold(0, |best, (i, v)| if *v > full[best] { i } else { best });
        assert_eq!(
            best, full_best,
            "cached versus full-context greedy token at step {step}"
        );
    }
}
