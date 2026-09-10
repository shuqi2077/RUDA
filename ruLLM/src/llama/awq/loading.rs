use super::*;
use crate::{AwqCheckpoint, HuggingFaceLoadError};
use std::collections::BTreeSet;

struct Weights<'a> {
    checkpoint: &'a AwqCheckpoint,
    consumed: BTreeSet<String>,
    packed_projections: usize,
}

impl Weights<'_> {
    fn validate(
        &mut self,
        name: &str,
        shape: impl Into<ruda_tensor::api::Shape>,
    ) -> Result<(), HuggingFaceLoadError> {
        let shape = shape.into();
        let (dtype, actual_shape) = self
            .checkpoint
            .tensor_info(name)
            .ok_or_else(|| HuggingFaceLoadError(format!("missing AWQ model tensor: {name}")))?;
        if dtype != DType::F16 || actual_shape != &shape {
            return Err(HuggingFaceLoadError(format!(
                "{name}: expected F16 {shape:?}, found {dtype:?} {actual_shape:?}"
            )));
        }
        self.consumed.insert(name.into());
        Ok(())
    }

    fn tensor<R: DeviceRuntime, const D: usize>(
        &mut self,
        name: &str,
        shape: [usize; D],
        device: &R::Device,
    ) -> Result<Tensor<AwqBackend<R>, D>, HuggingFaceLoadError>
    where
        R::Device: DeviceOps,
    {
        self.validate(name, shape)?;
        Ok(Tensor::from_data(
            self.checkpoint.tensor_data(name)?,
            (device, DType::F16),
        ))
    }

    fn projection<R: DeviceRuntime>(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
        device: &R::Device,
    ) -> Result<AwqProjection<R>, HuggingFaceLoadError>
    where
        R::Device: DeviceOps,
    {
        if self
            .checkpoint
            .tensor_info(&format!("{prefix}.qweight"))
            .is_some()
        {
            let linear = self
                .checkpoint
                .load_linear(prefix, input, output, bias, device)?;
            for suffix in ["qweight", "qzeros", "scales"] {
                self.consumed.insert(format!("{prefix}.{suffix}"));
            }
            if bias {
                self.consumed.insert(format!("{prefix}.bias"));
            }
            self.packed_projections += 1;
            Ok(AwqProjection::Packed(linear))
        } else {
            let weight =
                self.tensor::<R, 2>(&format!("{prefix}.weight"), [output, input], device)?;
            let bias = if bias {
                Some(Param::from_tensor(self.tensor::<R, 1>(
                    &format!("{prefix}.bias"),
                    [output],
                    device,
                )?))
            } else {
                None
            };
            Ok(AwqProjection::Dense(Linear {
                weight: Param::from_tensor(weight.transpose()),
                bias,
            }))
        }
    }

    fn norm<R: DeviceRuntime>(
        &mut self,
        prefix: &str,
        config: &LlamaConfig,
        device: &R::Device,
    ) -> Result<RmsNorm<AwqBackend<R>>, HuggingFaceLoadError>
    where
        R::Device: DeviceOps,
    {
        Ok(RmsNorm {
            gamma: Param::from_tensor(self.tensor::<R, 1>(
                &format!("{prefix}.weight"),
                [config.d_model],
                device,
            )?),
            epsilon: config.rms_norm_epsilon,
        })
    }
}

impl<R: DeviceRuntime> AwqLlamaForCausalLm<R>
where
    R::Device: DeviceOps,
{
    pub(crate) fn load_checkpoint(
        checkpoint: &AwqCheckpoint,
        config: &LlamaConfig,
        qwen2: bool,
        tied_embeddings: bool,
        device: &R::Device,
    ) -> Result<(Self, usize), HuggingFaceLoadError> {
        config
            .validate()
            .map_err(|error| HuggingFaceLoadError(error.to_string()))?;
        let mut weights = Weights {
            checkpoint,
            consumed: BTreeSet::new(),
            packed_projections: 0,
        };
        let embed_tokens = Embedding {
            weight: Param::from_tensor(weights.tensor::<R, 2>(
                "model.embed_tokens.weight",
                [config.vocab_size, config.d_model],
                device,
            )?),
        };
        let rope = if qwen2 {
            crate::llama::rope::qwen2_rope_with_dtype::<AwqBackend<R>>(config, device, DType::F16)
        } else {
            RotaryEncodingConfig::new(config.max_sequence_length, config.head_dimension())
                .with_theta(config.rope_theta)
                .init(device)
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let kv_dimension = config.num_kv_heads * config.head_dimension();
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let self_attn = attention::AwqAttention {
                q_proj: weights.projection::<R>(
                    &format!("{prefix}.self_attn.q_proj"),
                    config.d_model,
                    config.d_model,
                    qwen2,
                    device,
                )?,
                k_proj: weights.projection::<R>(
                    &format!("{prefix}.self_attn.k_proj"),
                    config.d_model,
                    kv_dimension,
                    qwen2,
                    device,
                )?,
                v_proj: weights.projection::<R>(
                    &format!("{prefix}.self_attn.v_proj"),
                    config.d_model,
                    kv_dimension,
                    qwen2,
                    device,
                )?,
                o_proj: weights.projection::<R>(
                    &format!("{prefix}.self_attn.o_proj"),
                    config.d_model,
                    config.d_model,
                    false,
                    device,
                )?,
                rope: rope.clone(),
                rotary_layout: if qwen2 {
                    RotaryLayout::HalfSplit
                } else {
                    RotaryLayout::Interleaved
                },
                num_query_heads: config.num_query_heads,
                num_kv_heads: config.num_kv_heads,
                head_dimension: config.head_dimension(),
            };
            let mlp = AwqFeedForward {
                gate_proj: weights.projection::<R>(
                    &format!("{prefix}.mlp.gate_proj"),
                    config.d_model,
                    config.d_ff,
                    false,
                    device,
                )?,
                up_proj: weights.projection::<R>(
                    &format!("{prefix}.mlp.up_proj"),
                    config.d_model,
                    config.d_ff,
                    false,
                    device,
                )?,
                down_proj: weights.projection::<R>(
                    &format!("{prefix}.mlp.down_proj"),
                    config.d_ff,
                    config.d_model,
                    false,
                    device,
                )?,
            };
            layers.push(AwqDecoderLayer {
                self_attn,
                mlp,
                input_layernorm: weights.norm::<R>(
                    &format!("{prefix}.input_layernorm"),
                    config,
                    device,
                )?,
                post_attention_layernorm: weights.norm::<R>(
                    &format!("{prefix}.post_attention_layernorm"),
                    config,
                    device,
                )?,
            });
        }
        let norm = weights.norm::<R>("model.norm", config, device)?;
        let lm_head = if tied_embeddings {
            if checkpoint.tensor_info("lm_head.weight").is_some() {
                weights.validate("lm_head.weight", [config.vocab_size, config.d_model])?;
            }
            AwqProjection::Dense(Linear {
                weight: Param::from_tensor(embed_tokens.weight.val().transpose().detach()),
                bias: None,
            })
        } else {
            weights.projection::<R>("lm_head", config.d_model, config.vocab_size, false, device)?
        };
        let unused = checkpoint
            .tensor_names()
            .filter(|name| !weights.consumed.contains(*name))
            .collect::<Vec<_>>();
        if !unused.is_empty() {
            return Err(HuggingFaceLoadError(format!(
                "unconsumed AWQ model tensors: {}",
                unused.join(", ")
            )));
        }
        if weights.packed_projections == 0 {
            return Err(HuggingFaceLoadError(
                "AWQ model contains no packed projections".into(),
            ));
        }
        AwqBackend::<R>::sync(device).map_err(|error| {
            HuggingFaceLoadError(format!("AWQ weight upload did not complete: {error}"))
        })?;
        Ok((
            Self {
                embed_tokens,
                layers,
                norm,
                lm_head,
                max_sequence_length: config.max_sequence_length,
            },
            weights.consumed.len(),
        ))
    }
}
