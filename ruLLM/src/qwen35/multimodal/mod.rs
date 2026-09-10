mod position;
mod images;
mod generation;
#[cfg(all(test, feature = "nvidia"))]
mod tests;

use super::*;
use crate::HuggingFaceLoadError;
use ruda_core::device::Device;
use std::path::Path;

/// Image-prefill and text-decode model sharing the existing vision and text implementations.
pub struct Qwen35MultimodalModel<B: Backend> {
    pub text: Qwen35TextModel<B>,
    pub vision: Qwen35VisionModel<B>,
    image_token_id: i32,
    video_token_id: i32,
}

pub struct Qwen35MultimodalCache<B: Backend> {
    text: Qwen35Cache<B>,
    next_positions: Vec<usize>,
}

impl<B: Backend> Qwen35MultimodalCache<B> {
    pub fn sequence_length(&self) -> usize { self.text.sequence_length() }
    pub fn next_positions(&self) -> &[usize] { &self.next_positions }
}

pub fn load_huggingface_qwen35_multimodal<B: Backend>(
    directory: impl AsRef<Path>, device: &B::Device,
) -> Result<Qwen35MultimodalModel<B>, HuggingFaceLoadError> {
    let raw: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.as_ref().join("config.json")).map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    ).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let token = |name: &str| raw[name].as_i64().and_then(|id| i32::try_from(id).ok())
        .ok_or_else(|| HuggingFaceLoadError(format!("invalid {name}")));
    let image_token_id = token("image_token_id")?;
    let video_token_id = token("video_token_id")?;
    let text = load_huggingface_qwen35_text(&directory, device)?.model;
    let vision = load_huggingface_qwen35_vision(&directory, device)?.model;
    if vision.config().out_hidden_size != text.config.hidden_size
        || image_token_id == video_token_id
        || [image_token_id, video_token_id].iter().any(|&id| id < 0 || id as usize >= text.config.vocab_size)
    {
        return Err(HuggingFaceLoadError("incompatible Qwen3.5 text/vision configuration".into()));
    }
    Ok(Qwen35MultimodalModel { text, vision, image_token_id, video_token_id })
}

impl<B: Backend> Qwen35MultimodalModel<B> {
    pub fn new_cache(&self) -> Qwen35MultimodalCache<B> {
        Qwen35MultimodalCache { text: self.text.new_cache(), next_positions: Vec::new() }
    }
}

impl<R, F, I, BT> Qwen35MultimodalModel<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    /// Prefill equal-length, unpadded prompts with processor-packed image patches.
    /// Grids and features follow placeholder order across prompt rows. Returns last-token logits.
    pub fn prefill_images(
        &self,
        tokens: &[Vec<i32>],
        patches: Tensor<DeviceBackend<R, F, I, BT>, 2>,
        grids: &[[usize; 3]],
        cache: &mut Qwen35MultimodalCache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let sequence = tokens.first().map_or(0, Vec::len);
        if cache.sequence_length() != 0 || !cache.next_positions.is_empty()
            || sequence > self.text.config.max_position_embeddings
            || tokens.iter().flatten().any(|&id| id < 0 || id as usize >= self.text.config.vocab_size || id == self.video_token_id)
            || patches.device().to_id() != self.text.embedding.weight.val().device().to_id()
        {
            return Err(GenerationError("invalid image prefill tokens, device or nonempty cache".into()));
        }
        self.vision.config().grid_tokens(grids)?;
        let positions = position::image_positions(tokens, self.image_token_id, grids, self.vision.config().spatial_merge_size)?;
        let features = self.vision.forward(patches, grids)?.merged_states;
        let width = self.text.config.hidden_size;
        let device = self.text.embedding.weight.val().device();
        let input = Tensor::<DeviceBackend<R, F, I, BT>, 2, Int>::from_data(
            TensorData::new(tokens.iter().flatten().copied().collect::<Vec<_>>(), [tokens.len(), sequence]), &device,
        );
        let mut hidden = self.text.embedding.forward(input);
        for span in positions.spans {
            let image = features.clone().slice([span.feature_start..span.feature_start+span.count, 0..width])
                .cast(hidden.dtype()).reshape([1, span.count, width]);
            hidden = hidden.slice_assign([span.batch..span.batch+1, span.start..span.start+span.count, 0..width], image);
        }
        let logits = self.text.forward_embeddings_last(hidden, &positions.coordinates, &mut cache.text)?;
        cache.next_positions = positions.next;
        Ok(logits)
    }

    /// Continue an image-prefilled cache without re-encoding images. Supports multiple new tokens.
    pub fn decode(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut Qwen35MultimodalCache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        if cache.sequence_length() == 0 || batch != cache.next_positions.len() || sequence == 0
            || cache.next_positions.iter().any(|&p| p.checked_add(sequence).is_none_or(|n| n > self.text.config.max_position_embeddings))
            || tokens.device().to_id() != self.text.embedding.weight.val().device().to_id()
        {
            return Err(GenerationError("invalid Qwen3.5 multimodal continuation".into()));
        }
        let coordinates = cache.next_positions.iter()
            .flat_map(|&start| (start..start+sequence).map(|p| [p;3])).collect::<Vec<_>>();
        let hidden = self.text.embedding.forward(tokens);
        let logits = self.text.forward_embeddings_last(hidden, &coordinates, &mut cache.text)?;
        for position in &mut cache.next_positions { *position += sequence; }
        Ok(logits)
    }
}
