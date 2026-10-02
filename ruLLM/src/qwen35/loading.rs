use super::*;
use crate::huggingface::checkpoint::Checkpoint;
use crate::{HuggingFaceLoadError, HuggingFaceLoadReport};
use crate::huggingface::AwqQuantizationConfig;
use ruda_model::module::Param;
use std::path::Path;

pub struct LoadedQwen35Text<B: Backend> {
    pub model: Qwen35TextModel<B>,
    pub report: HuggingFaceLoadReport,
    pub unloaded_tensors: Vec<String>,
    /// AWQ matrices expanded into dense model weights at load time.
    /// Nonzero does NOT mean that inference uses packed INT4 kernels.
    pub dequantized_awq_linears: usize,
    /// Projections retaining packed weights and using the ruBLAS AWQ kernel.
    pub packed_awq_linears: usize,
}

pub(super) type PackedLoader<B: Backend> = fn(&mut Checkpoint, &str, usize, usize,
    &AwqQuantizationConfig, DType, bool, &<B as ruda_tensor::BackendTypes>::Device)
    -> Result<Projection<B>, HuggingFaceLoadError>;

pub(super) struct Weights<'a, B: Backend> {
    pub checkpoint: &'a mut Checkpoint,
    pub device: &'a B::Device,
    pub quantization: Option<AwqQuantizationConfig>,
    pub quantized_dtype: DType,
    pub dequantized_awq_linears: usize,
    pub packed_awq_linears: usize,
    pub packed_loader: Option<PackedLoader<B>>,
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
        let packed = self.checkpoint.tensors.contains_key(&format!("{prefix}.qweight"));
        let weight = if packed {
            let config = self.quantization.as_ref().ok_or_else(|| HuggingFaceLoadError(
                format!("{prefix}: packed weights require an AWQ quantization_config")))?;
            if config.modules_to_not_convert.as_ref().is_some_and(|modules| modules.iter().any(
                |name| prefix == name || prefix.ends_with(&format!(".{name}")) ||
                    prefix.split('.').any(|part| part == name))) {
                return Err(HuggingFaceLoadError(format!("{prefix}: packed weights in an excluded module")));
            }
            let tensor = super::quantized_loading::load_awq_linear::<B>(self.checkpoint,
                prefix, input, output, config, self.quantized_dtype, self.device)?;
            self.dequantized_awq_linears += 1;
            tensor
        } else {
            self.tensor(&format!("{prefix}.weight"), [output, input])?
        };
        let dtype = weight.dtype();
        Ok(Linear {
            weight: Param::from_tensor(weight.transpose()),
            bias: if bias {
                Some(Param::from_tensor(
                    self.tensor(&format!("{prefix}.bias"), [output])?.cast(dtype),
                ))
            } else {
                None
            },
        })
    }
    /// Packed mode never allocates a full floating-point weight matrix.
    pub fn projection(&mut self, prefix: &str, input: usize, output: usize, bias: bool)
        -> Result<Projection<B>, HuggingFaceLoadError>
    {
        if self.checkpoint.tensors.contains_key(&format!("{prefix}.qweight")) {
            if let Some(load) = self.packed_loader {
                let config = self.quantization.as_ref().ok_or_else(|| HuggingFaceLoadError(
                    format!("{prefix}: packed weights require an AWQ quantization_config")))?;
                if config.modules_to_not_convert.as_ref().is_some_and(|modules| modules.iter().any(
                    |name| prefix == name || prefix.ends_with(&format!(".{name}")) ||
                        prefix.split('.').any(|part| part == name))) {
                    return Err(HuggingFaceLoadError(format!("{prefix}: packed weights in an excluded module")));
                }
                let result = load(self.checkpoint, prefix, input, output, config,
                    self.quantized_dtype, bias, self.device)?;
                self.packed_awq_linears += 1;
                return Ok(result);
            }
        }
        self.linear(prefix, input, output, bias).map(Projection::Dense)
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
    load_text(directory.as_ref(), device, None)
}

/// Load a text model with device-resident AWQ I32 words. Dense/excluded layers
/// remain dense. No conversion to a full floating weight matrix is performed.
pub fn load_huggingface_qwen35_text_packed<R, F, I, BT>(
    directory: impl AsRef<Path>, device: &R::Device,
) -> Result<LoadedQwen35Text<DeviceBackend<R, F, I, BT>>, HuggingFaceLoadError>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    load_text(directory.as_ref(), device, Some(super::quantized_loading::load_awq_projection::<R,F,I,BT>))
}

fn load_text<B: Backend>(directory: &Path, device: &B::Device, packed_loader: Option<PackedLoader<B>>)
    -> Result<LoadedQwen35Text<B>, HuggingFaceLoadError>
{
    let dir = directory;
    let config_path = dir.join("config.json");
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path).map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    )
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    // Accept metadata at the HF top level or in text_config, but never prefer
    // one conflicting declaration silently. Unsupported formats remain errors.
    let quantization = super::quantized_loading::configuration(&raw)?;
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
    let base = if multimodal { "model.language_model" } else { "model" };
    let quantized_dtype = checkpoint.tensors.get(&format!("{base}.embed_tokens.weight"))
        .map(|tensor| tensor.dtype)
        .ok_or_else(|| HuggingFaceLoadError("missing Qwen3.5 token embedding".into()))?;
    let mut w = Weights::<B> {
        checkpoint: &mut checkpoint,
        device,
        quantization,
        quantized_dtype,
        dequantized_awq_linears: 0,
        packed_awq_linears: 0,
        packed_loader,
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
                    q: w.projection(
                        &format!("{p}.q_proj"),
                        c.hidden_size,
                        2 * q,
                        c.attention_bias,
                    )?,
                    k: w.projection(&format!("{p}.k_proj"), c.hidden_size, kv, c.attention_bias)?,
                    v: w.projection(&format!("{p}.v_proj"), c.hidden_size, kv, c.attention_bias)?,
                    out: w.projection(&format!("{p}.o_proj"), q, c.hidden_size, c.attention_bias)?,
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
                    qkv: w.projection(&format!("{p}.in_proj_qkv"), c.hidden_size, channels, false)?,
                    z: w.projection(&format!("{p}.in_proj_z"), c.hidden_size, values, false)?,
                    a: w.projection(
                        &format!("{p}.in_proj_a"),
                        c.hidden_size,
                        c.linear_num_value_heads,
                        false,
                    )?,
                    b: w.projection(
                        &format!("{p}.in_proj_b"),
                        c.hidden_size,
                        c.linear_num_value_heads,
                        false,
                    )?,
                    out: w.projection(&format!("{p}.out_proj"), values, c.hidden_size, false)?,
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
                gate: w.projection(
                    &format!("{prefix}.mlp.gate_proj"),
                    c.hidden_size,
                    c.intermediate_size,
                    false,
                )?,
                up: w.projection(
                    &format!("{prefix}.mlp.up_proj"),
                    c.hidden_size,
                    c.intermediate_size,
                    false,
                )?,
                down: w.projection(
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
        Projection::Dense(Linear {
            weight: Param::from_tensor(embedding.weight.val().transpose()),
            bias: None,
        })
    } else {
        w.projection("lm_head", c.hidden_size, c.vocab_size, false)?
    };
    let dequantized_awq_linears = w.dequantized_awq_linears;
    let packed_awq_linears = w.packed_awq_linears;
    drop(w);
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
        dequantized_awq_linears,
        packed_awq_linears,
    })
}
