use super::*;
use crate::{AwqBackend, SamplingConfig};
use half::f16;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_tensor::api::{Int, Tensor, TensorData};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

type TestBackend = AwqBackend<CudaRuntime>;
type StoredTensor = (String, Vec<usize>, Vec<u8>);
type StoredWeights = BTreeMap<String, StoredTensor>;
static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

mod text;

struct Fixture(PathBuf);

impl Fixture {
    fn new(weights: &StoredWeights, awq: bool, tied: bool) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ruda-awq-model-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let mut config = json!({
            "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
            "vocab_size": 24, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "max_position_embeddings": 16, "rms_norm_eps": 0.000001,
            "rope_theta": 1000000.0, "tie_word_embeddings": tied, "eos_token_id": 23
        });
        if awq {
            config["quantization_config"] = json!({"quant_method": "awq", "bits": 4, "group_size": 8, "zero_point": true, "version": "gemm", "modules_to_not_convert": ["lm_head", "o_proj"]});
        }
        std::fs::write(
            directory.join(CONFIG_FILE),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, (dtype, shape, bytes)) in weights {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.clone(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(directory.join(SINGLE_WEIGHTS_FILE), bytes).unwrap();
        Self(directory)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for entry in std::fs::read_dir(&self.0).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        std::fs::remove_dir(&self.0).unwrap();
    }
}

fn floats(shape: &[usize], values: impl IntoIterator<Item = f32>) -> StoredTensor {
    (
        "F16".into(),
        shape.to_vec(),
        values
            .into_iter()
            .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
    )
}

fn words(shape: &[usize], values: impl IntoIterator<Item = u32>) -> StoredTensor {
    (
        "I32".into(),
        shape.to_vec(),
        values.into_iter().flat_map(u32::to_le_bytes).collect(),
    )
}

fn model_weights() -> (StoredWeights, StoredWeights) {
    let mut dense = StoredWeights::new();
    dense.insert(
        "model.embed_tokens.weight".into(),
        floats(
            &[24, 16],
            (0..384).map(|index| ((index * 7 % 29) as f32 - 14.0) / 32.0),
        ),
    );
    for name in [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ] {
        dense.insert(
            name.into(),
            floats(&[16], (0..16).map(|index| 1.0 + index as f32 / 128.0)),
        );
    }
    dense.insert(
        "lm_head.weight".into(),
        floats(
            &[24, 16],
            (0..384).map(|index| ((index * 11 % 23) as f32 - 11.0) / 64.0),
        ),
    );
    let mut packed = dense.clone();
    for (projection_index, (suffix, k, n, bias)) in [
        ("self_attn.q_proj", 16, 16, true),
        ("self_attn.k_proj", 16, 8, true),
        ("self_attn.v_proj", 16, 8, true),
        ("self_attn.o_proj", 16, 16, false),
        ("mlp.gate_proj", 16, 32, false),
        ("mlp.up_proj", 16, 32, false),
        ("mlp.down_proj", 32, 16, false),
    ]
    .into_iter()
    .enumerate()
    {
        let prefix = format!("model.layers.0.{suffix}");
        let integer =
            |row: usize, column: usize| ((row * 3 + column * 5 + projection_index * 7) % 16) as u32;
        let zero =
            |group: usize, column: usize| ((group + column + projection_index) % 5 + 6) as u32;
        let scale = |group: usize, column: usize| ((group + column) % 3 + 1) as f32 / 128.0;
        let mut decoded = Vec::new();
        for column in 0..n {
            for row in 0..k {
                decoded.push(
                    (integer(row, column) as f32 - zero(row / 8, column) as f32)
                        * scale(row / 8, column),
                );
            }
        }
        let dense_weight = floats(&[n, k], decoded);
        dense.insert(format!("{prefix}.weight"), dense_weight.clone());
        if suffix == "self_attn.o_proj" {
            packed.insert(format!("{prefix}.weight"), dense_weight);
        } else {
            let pack = |row: usize, block: usize, zeros: bool| {
                [0, 2, 4, 6, 1, 3, 5, 7].into_iter().enumerate().fold(
                    0u32,
                    |word, (nibble, lane)| {
                        let column = block * 8 + lane;
                        word | (if zeros {
                            zero(row, column)
                        } else {
                            integer(row, column)
                        } << (4 * nibble))
                    },
                )
            };
            packed.insert(
                format!("{prefix}.qweight"),
                words(
                    &[k, n / 8],
                    (0..k).flat_map(|row| (0..n / 8).map(move |block| pack(row, block, false))),
                ),
            );
            packed.insert(
                format!("{prefix}.qzeros"),
                words(
                    &[k / 8, n / 8],
                    (0..k / 8).flat_map(|row| (0..n / 8).map(move |block| pack(row, block, true))),
                ),
            );
            packed.insert(
                format!("{prefix}.scales"),
                floats(
                    &[k / 8, n],
                    (0..k / 8).flat_map(|row| (0..n).map(move |column| scale(row, column))),
                ),
            );
        }
        if bias {
            let value = floats(&[n], (0..n).map(|column| (column as f32 - 4.0) / 128.0));
            dense.insert(format!("{prefix}.bias"), value.clone());
            packed.insert(format!("{prefix}.bias"), value);
        }
    }
    (packed, dense)
}

fn tokens(ids: &[i32], device: &CudaDevice) -> Tensor<TestBackend, 2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
}

fn compare(actual: Tensor<TestBackend, 3>, expected: Tensor<TestBackend, 3>) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.to_data().to_vec::<f16>().unwrap();
    let expected = expected.to_data().to_vec::<f16>().unwrap();
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let (actual, expected) = (actual.to_f32(), expected.to_f32());
        let tolerance = 8.0 * f16::EPSILON.to_f32() * expected.abs().max(1.0);
        assert!(
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
            "logit {index}: {actual} != {expected}, tolerance {tolerance}"
        );
    }
}

#[test]
fn full_awq_qwen_matches_dense_weights_for_prefill_and_cached_decode() {
    let (packed, dense) = model_weights();
    let packed_fixture = Fixture::new(&packed, true, false);
    let dense_fixture = Fixture::new(&dense, false, false);
    let device = CudaDevice::default();
    let loaded = load_huggingface_awq_qwen2::<CudaRuntime>(&packed_fixture.0, &device).unwrap();
    let reference = load_huggingface_qwen2::<TestBackend>(&dense_fixture.0, &device).unwrap();
    assert_eq!(loaded.report.applied_tensors, packed.len());
    compare(
        loaded.model.forward(tokens(&[1, 4, 9], &device)),
        reference.model.forward(tokens(&[1, 4, 9], &device)),
    );
    let mut cache = loaded.model.new_cache();
    let mut reference_cache = reference.model.new_cache();
    for ids in [&[1, 4][..], &[9][..], &[2, 6][..]] {
        compare(
            loaded
                .model
                .forward_cached(tokens(ids, &device), &mut cache),
            reference
                .model
                .forward_cached(tokens(ids, &device), &mut reference_cache),
        );
        assert_eq!(cache.position(), reference_cache.position());
    }
    assert_eq!(cache.position(), 5);
}

#[test]
fn awq_qwen_tied_embedding_and_last_logits_keep_original_cache_semantics() {
    let (packed, dense) = model_weights();
    let packed_fixture = Fixture::new(&packed, true, true);
    let dense_fixture = Fixture::new(&dense, false, true);
    let device = CudaDevice::default();
    let loaded = load_huggingface_awq_qwen2::<CudaRuntime>(&packed_fixture.0, &device).unwrap();
    let reference = load_huggingface_qwen2::<TestBackend>(&dense_fixture.0, &device).unwrap();
    let mut cache = loaded.model.new_cache();
    let mut reference_cache = reference.model.new_cache();
    compare(
        loaded
            .model
            .forward_cached_last(tokens(&[2, 5], &device), &mut cache),
        reference
            .model
            .forward_cached_last(tokens(&[2, 5], &device), &mut reference_cache),
    );
    assert_eq!(cache.position(), 2);
    assert!(loaded.report.tied_word_embeddings);
}

#[test]
fn awq_generation_reuses_seeded_sampling_and_length_validation() {
    let (packed, _) = model_weights();
    let fixture = Fixture::new(&packed, true, false);
    let device = CudaDevice::default();
    let loaded = load_huggingface_awq_qwen2::<CudaRuntime>(&fixture.0, &device).unwrap();
    let generation = SamplingGenerationConfig {
        max_new_tokens: 3,
        eos_token_ids: vec![],
        sampling: SamplingConfig {
            seed: Some(17),
            ..Default::default()
        },
    };
    let first = loaded
        .generate_tokens_sampled(&[1, 4], generation.clone(), &device)
        .unwrap();
    let second = loaded
        .generate_tokens_sampled(&[1, 4], generation, &device)
        .unwrap();
    assert_eq!(first, second);
    let empty = loaded
        .generate_tokens(
            &[1, 4],
            GreedyGenerationConfig {
                max_new_tokens: 0,
                eos_token_ids: vec![],
            },
            &device,
        )
        .unwrap();
    assert!(empty.generated_token_ids.is_empty());
    assert_eq!(empty.token_ids, vec![1, 4]);
    assert!(
        loaded
            .generate_tokens(&[], GreedyGenerationConfig::default(), &device)
            .is_err()
    );
}

#[test]
fn awq_model_loader_rejects_unconsumed_and_missing_weights() {
    let (mut packed, _) = model_weights();
    packed.insert("model.unused.weight".into(), floats(&[1], [1.0]));
    let fixture = Fixture::new(&packed, true, false);
    let device = CudaDevice::default();
    assert!(load_huggingface_awq_qwen2::<CudaRuntime>(&fixture.0, &device).is_err());
    packed.remove("model.unused.weight");
    packed.remove("model.layers.0.self_attn.k_proj.bias");
    let missing = Fixture::new(&packed, true, false);
    assert!(load_huggingface_awq_qwen2::<CudaRuntime>(&missing.0, &device).is_err());
    let (_, dense) = model_weights();
    let unquantized = Fixture::new(&dense, true, false);
    assert!(load_huggingface_awq_qwen2::<CudaRuntime>(&unquantized.0, &device).is_err());
}
