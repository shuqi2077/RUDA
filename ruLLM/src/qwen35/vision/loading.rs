use super::super::loading::Weights;
use super::*;
use crate::{HuggingFaceLoadError, HuggingFaceLoadReport, huggingface::checkpoint::Checkpoint};
use ruda_model::module::Param;
use ruda_nn::LayerNormConfig;
use std::path::Path;

pub struct LoadedQwen35Vision<B: Backend> {
    pub model: Qwen35VisionModel<B>,
    pub report: HuggingFaceLoadReport,
    pub unloaded_tensors: Vec<String>,
}

fn norm<B: Backend>(
    w: &mut Weights<'_, B>,
    prefix: &str,
    width: usize,
) -> Result<LayerNorm<B>, HuggingFaceLoadError> {
    let mut norm = LayerNormConfig::new(width)
        .with_epsilon(1e-6)
        .init(w.device);
    norm.gamma = Param::from_tensor(
        w.tensor(&format!("{prefix}.weight"), [width])?
            .cast(DType::F32),
    );
    norm.beta = Some(Param::from_tensor(
        w.tensor(&format!("{prefix}.bias"), [width])?
            .cast(DType::F32),
    ));
    Ok(norm)
}

pub fn load_huggingface_qwen35_vision<B: Backend>(
    directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen35Vision<B>, HuggingFaceLoadError> {
    let config_path = directory.as_ref().join("config.json");
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path).map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    )
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    if raw["model_type"] != "qwen3_5"
        || raw.get("quantization_config").is_some_and(|v| !v.is_null())
        || raw["vision_config"]
            .get("quantization_config")
            .is_some_and(|v| !v.is_null())
    {
        return Err(HuggingFaceLoadError(
            "expected unquantized Qwen3.5 vision checkpoint".into(),
        ));
    }
    let c: Qwen35VisionConfig = serde_json::from_value(raw["vision_config"].clone())
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    c.validate()?;
    let mut checkpoint = Checkpoint::open(directory.as_ref())?;
    let mut w = Weights::<B> {
        checkpoint: &mut checkpoint,
        device,
    };
    let base = "model.visual";
    let patch_weight = w.tensor(
        &format!("{base}.patch_embed.proj.weight"),
        [
            c.hidden_size,
            c.in_channels,
            c.temporal_patch_size,
            c.patch_size,
            c.patch_size,
        ],
    )?;
    let patch_bias = w.tensor(&format!("{base}.patch_embed.proj.bias"), [c.hidden_size])?;
    let position = Embedding {
        weight: Param::from_tensor(w.tensor(
            &format!("{base}.pos_embed.weight"),
            [c.num_position_embeddings, c.hidden_size],
        )?),
    };
    let mut blocks = Vec::with_capacity(c.depth);
    for index in 0..c.depth {
        let p = format!("{base}.blocks.{index}");
        blocks.push(Block {
            norm1: norm(&mut w, &format!("{p}.norm1"), c.hidden_size)?,
            norm2: norm(&mut w, &format!("{p}.norm2"), c.hidden_size)?,
            qkv: w.linear(
                &format!("{p}.attn.qkv"),
                c.hidden_size,
                3 * c.hidden_size,
                true,
            )?,
            proj: w.linear(
                &format!("{p}.attn.proj"),
                c.hidden_size,
                c.hidden_size,
                true,
            )?,
            fc1: w.linear(
                &format!("{p}.mlp.linear_fc1"),
                c.hidden_size,
                c.intermediate_size,
                true,
            )?,
            fc2: w.linear(
                &format!("{p}.mlp.linear_fc2"),
                c.intermediate_size,
                c.hidden_size,
                true,
            )?,
        });
    }
    let width = c.hidden_size * c.spatial_merge_size * c.spatial_merge_size;
    let model = Qwen35VisionModel {
        patch_weight,
        patch_bias,
        position,
        blocks,
        merger_norm: norm(&mut w, &format!("{base}.merger.norm"), c.hidden_size)?,
        merger_fc1: w.linear(&format!("{base}.merger.linear_fc1"), width, width, true)?,
        merger_fc2: w.linear(
            &format!("{base}.merger.linear_fc2"),
            width,
            c.out_hidden_size,
            true,
        )?,
        config: c,
    };
    let unloaded_tensors = checkpoint
        .tensors
        .keys()
        .filter(|n| !checkpoint.consumed.contains(*n))
        .cloned()
        .collect::<Vec<_>>();
    if unloaded_tensors.iter().any(|n| {
        !(n.starts_with("model.language_model.") || n.starts_with("mtp.") || n == "lm_head.weight")
    }) {
        return Err(HuggingFaceLoadError(format!(
            "unconsumed Qwen3.5 vision tensors: {unloaded_tensors:?}"
        )));
    }
    B::sync(device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    Ok(LoadedQwen35Vision {
        model,
        unloaded_tensors,
        report: HuggingFaceLoadReport {
            config_path,
            weight_files: checkpoint.files,
            applied_tensors: checkpoint.consumed.len(),
            tied_word_embeddings: false,
        },
    })
}
