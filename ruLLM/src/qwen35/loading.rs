use super::*;
use crate::huggingface::checkpoint::Checkpoint;
use crate::{HuggingFaceLoadError, HuggingFaceLoadReport};
use ruda_model::module::Param;
use std::path::Path;

pub struct LoadedQwen35Text<B: Backend> {
    pub model: Qwen35TextModel<B>,
    pub report: HuggingFaceLoadReport,
    pub unloaded_tensors: Vec<String>,
}

pub(super) struct Weights<'a, B: Backend> {
    pub checkpoint: &'a mut Checkpoint,
    pub device: &'a B::Device,
}
impl<B: Backend> Weights<'_, B> {
    pub fn tensor<const D: usize>(
        &mut self,
        name: &str,
        shape: [usize; D],
    ) -> Result<Tensor<B, D>, HuggingFaceLoadError> {
        self.checkpoint.tensor(name, shape, self.device)
    }
    pub fn linear(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Linear<B>, HuggingFaceLoadError> {
        Ok(Linear {
            weight: Param::from_tensor(
                self.tensor(&format!("{prefix}.weight"), [output, input])?
                    .transpose(),
            ),
            bias: if bias {
                Some(Param::from_tensor(
                    self.tensor(&format!("{prefix}.bias"), [output])?,
                ))
            } else {
                None
            },
        })
    }
    fn norm(
        &mut self,
        prefix: &str,
        width: usize,
        epsilon: f64,
    ) -> Result<Norm<B>, HuggingFaceLoadError> {
        Ok(Norm {
            weight: self.tensor(&format!("{prefix}.weight"), [width])?,
            epsilon,
        })
    }
}

pub fn load_huggingface_qwen35_text<B: Backend>(
    directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen35Text<B>, HuggingFaceLoadError> {
    let dir = directory.as_ref();
    let config_path = dir.join("config.json");
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path).map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    )
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    if raw.get("quantization_config").is_some_and(|v| !v.is_null()) {
        return Err(HuggingFaceLoadError(
            "quantized Qwen3.5 loading is not implemented".into(),
        ));
    }
    let multimodal = raw.get("text_config").is_some();
    if multimodal && raw["model_type"] != "qwen3_5" {
        return Err(HuggingFaceLoadError(
            "expected Qwen3.5 configuration".into(),
        ));
    }
    let config: Qwen35TextConfig = serde_json::from_value(if multimodal {
        raw["text_config"].clone()
    } else {
        raw
    })
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    config.validate()?;
    let mut checkpoint = Checkpoint::open(dir)?;
    let mut w = Weights::<B> {
        checkpoint: &mut checkpoint,
        device,
    };
    let base = if multimodal {
        "model.language_model"
    } else {
        "model"
    };
    let embedding = Embedding {
        weight: Param::from_tensor(w.tensor(
            &format!("{base}.embed_tokens.weight"),
            [config.vocab_size, config.hidden_size],
        )?),
    };
    let mut layers = Vec::new();
    let c = &config;
    for (index, kind) in c.layer_types.iter().enumerate() {
        let prefix = format!("{base}.layers.{index}");
        let mixer = match kind {
            LayerType::FullAttention => {
                let p = format!("{prefix}.self_attn");
                let (q, kv) = (
                    c.num_attention_heads * c.head_dim,
                    c.num_key_value_heads * c.head_dim,
                );
                Mixer::Full(attention::Attention {
                    q: w.linear(
                        &format!("{p}.q_proj"),
                        c.hidden_size,
                        2 * q,
                        c.attention_bias,
                    )?,
                    k: w.linear(&format!("{p}.k_proj"), c.hidden_size, kv, c.attention_bias)?,
                    v: w.linear(&format!("{p}.v_proj"), c.hidden_size, kv, c.attention_bias)?,
                    out: w.linear(&format!("{p}.o_proj"), q, c.hidden_size, c.attention_bias)?,
                    q_norm: w.norm(&format!("{p}.q_norm"), c.head_dim, c.rms_norm_eps)?,
                    k_norm: w.norm(&format!("{p}.k_norm"), c.head_dim, c.rms_norm_eps)?,
                })
            }
            LayerType::LinearAttention => {
                let p = format!("{prefix}.linear_attn");
                let channels = 2 * c.linear_num_key_heads * c.linear_key_head_dim
                    + c.linear_num_value_heads * c.linear_value_head_dim;
                let values = c.linear_num_value_heads * c.linear_value_head_dim;
                Mixer::Delta(delta::Delta {
                    qkv: w.linear(&format!("{p}.in_proj_qkv"), c.hidden_size, channels, false)?,
                    z: w.linear(&format!("{p}.in_proj_z"), c.hidden_size, values, false)?,
                    a: w.linear(
                        &format!("{p}.in_proj_a"),
                        c.hidden_size,
                        c.linear_num_value_heads,
                        false,
                    )?,
                    b: w.linear(
                        &format!("{p}.in_proj_b"),
                        c.hidden_size,
                        c.linear_num_value_heads,
                        false,
                    )?,
                    out: w.linear(&format!("{p}.out_proj"), values, c.hidden_size, false)?,
                    conv: w.tensor(
                        &format!("{p}.conv1d.weight"),
                        [channels, 1, c.linear_conv_kernel_dim],
                    )?,
                    a_log: w.tensor(&format!("{p}.A_log"), [c.linear_num_value_heads])?,
                    dt_bias: w.tensor(&format!("{p}.dt_bias"), [c.linear_num_value_heads])?,
                    norm: w.tensor(&format!("{p}.norm.weight"), [c.linear_value_head_dim])?,
                })
            }
        };
        layers.push(Layer {
            mixer,
            input_norm: w.norm(
                &format!("{prefix}.input_layernorm"),
                c.hidden_size,
                c.rms_norm_eps,
            )?,
            post_norm: w.norm(
                &format!("{prefix}.post_attention_layernorm"),
                c.hidden_size,
                c.rms_norm_eps,
            )?,
            mlp: Mlp {
                gate: w.linear(
                    &format!("{prefix}.mlp.gate_proj"),
                    c.hidden_size,
                    c.intermediate_size,
                    false,
                )?,
                up: w.linear(
                    &format!("{prefix}.mlp.up_proj"),
                    c.hidden_size,
                    c.intermediate_size,
                    false,
                )?,
                down: w.linear(
                    &format!("{prefix}.mlp.down_proj"),
                    c.intermediate_size,
                    c.hidden_size,
                    false,
                )?,
            },
        });
    }
    let norm = w.norm(&format!("{base}.norm"), c.hidden_size, c.rms_norm_eps)?;
    let head = if c.tie_word_embeddings {
        if w.checkpoint.tensors.contains_key("lm_head.weight") {
            let supplied = w.tensor("lm_head.weight", [c.vocab_size, c.hidden_size])?;
            if supplied.into_data() != embedding.weight.val().into_data() {
                return Err(HuggingFaceLoadError(
                    "tied lm_head differs from embedding".into(),
                ));
            }
        }
        Linear {
            weight: Param::from_tensor(embedding.weight.val().transpose()),
            bias: None,
        }
    } else {
        w.linear("lm_head", c.hidden_size, c.vocab_size, false)?
    };
    let unloaded_tensors = checkpoint
        .tensors
        .keys()
        .filter(|n| !checkpoint.consumed.contains(*n))
        .cloned()
        .collect::<Vec<_>>();
    if unloaded_tensors.iter().any(|name| {
        !(name.starts_with("mtp.") || (multimodal && name.starts_with("model.visual.")))
    }) {
        return Err(HuggingFaceLoadError(format!(
            "unconsumed Qwen3.5 tensors: {unloaded_tensors:?}"
        )));
    }
    B::sync(device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    Ok(LoadedQwen35Text {
        report: HuggingFaceLoadReport {
            config_path,
            weight_files: checkpoint.files,
            applied_tensors: checkpoint.consumed.len(),
            tied_word_embeddings: c.tie_word_embeddings,
        },
        model: Qwen35TextModel {
            config,
            embedding,
            layers,
            norm,
            head,
        },
        unloaded_tensors,
    })
}
