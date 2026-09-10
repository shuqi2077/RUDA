use super::*;
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};

#[test]
#[ignore = "requires local Qwen2.5-0.5B and the original eager generation reference"]
fn real_qwen05_checkpoint_generation() {
    type B = Cuda<bf16, i32>;
    let reference: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var("RUDA_QWEN05_REFERENCE").unwrap()).unwrap(),
    ).unwrap();
    let expected: Vec<i32> = serde_json::from_value(reference["eager-generated"].clone()).unwrap();
    assert_eq!(expected.len(), 8);
    let device = CudaDevice::default();
    let pipeline = load_huggingface_qwen2_pipeline::<B>(
        std::env::var("RUDA_QWEN05_MODEL").unwrap(), &device,
    ).unwrap();
    let prompt = pipeline.encode("The capital of France is", false).unwrap();
    let config = pipeline.loaded.config;
    let model = pipeline.loaded.model.into_packed_inference();
    let generation = crate::GreedyGenerationConfig {
        max_new_tokens: expected.len(),
        eos_token_ids: pipeline.loaded.default_eos_token_ids,
    };
    for run in 1..=2 {
        let output = crate::generate_greedy_packed_ruda(
            &model, &config, &prompt, &generation, &device,
        ).unwrap();
        eprintln!("Qwen2.5-0.5B run={run} tokens={:?}", output.generated_token_ids);
        assert_eq!(output.generated_token_ids, expected);
        assert_eq!(&output.token_ids[..prompt.len()], prompt);
        assert!(!output.stopped_on_eos);
    }
}
