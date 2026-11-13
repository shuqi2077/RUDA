//! End-to-end native selection inside the actual cached generation loop.
use ruda_tensor::api::{Int, Tensor, TensorData};
use ruda_tensor_device::cuda::{CudaDevice, CudaNative};
use rullm::{
    CausalModel, CausalModelLimits, SamplingConfig, SamplingGenerationConfig,
    generate_causal_sampled_device,
};
type B = CudaNative<f32, i32>;
struct Toy {
    device: CudaDevice,
}
impl CausalModel<B> for Toy {
    type Cache = usize;
    fn new_cache(&self) -> usize {
        0
    }
    fn forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut usize) -> Tensor<B, 3> {
        *cache += tokens.dims()[1];
        let mut logits = vec![f32::NEG_INFINITY; 3];
        logits[if *cache == 1 { 1 } else { 2 }] = 5.0;
        Tensor::from_data(TensorData::new(logits, [1, 1, 3]), &self.device)
    }
}
#[test]
fn native_sampler_drives_cache_decode_and_eos() {
    let device = CudaDevice::default();
    let model = Toy {
        device: device.clone(),
    };
    let config = SamplingGenerationConfig {
        max_new_tokens: 4,
        eos_token_ids: vec![2],
        sampling: SamplingConfig {
            seed: Some(19),
            ..Default::default()
        },
    };
    let out = generate_causal_sampled_device(
        &model,
        &CausalModelLimits {
            vocab_size: 3,
            max_sequence_length: 8,
        },
        &[0],
        &config,
        &device,
    )
    .unwrap();
    assert_eq!(out.generated_token_ids, vec![1, 2]);
    assert_eq!(out.token_ids, vec![0, 1, 2]);
    assert!(out.stopped_on_eos);
}
