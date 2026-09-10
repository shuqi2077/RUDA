use super::*;

impl<B: Backend> Qwen35MultimodalModel<B> {
    /// Expand one image placeholder per image. Grids follow placeholder order across rows.
    pub fn expand_image_placeholders(
        &self, tokens: &[Vec<i32>], grids: &[[usize; 3]],
    ) -> Result<Vec<Vec<i32>>, GenerationError> {
        self.vision.config().grid_tokens(grids)?;
        if tokens.is_empty() || tokens.iter().any(Vec::is_empty) {
            return Err(GenerationError("image prompts must not be empty".into()));
        }
        let merge = self.vision.config().spatial_merge_size;
        let mut image = 0;
        let mut rows = Vec::with_capacity(tokens.len());
        for row in tokens {
            let mut expanded = Vec::new();
            for &token in row {
                if token < 0 || token as usize >= self.text.config.vocab_size || token == self.video_token_id {
                    return Err(GenerationError("invalid token in image prompt".into()));
                }
                let count = if token == self.image_token_id {
                    let &[t,h,w] = grids.get(image).ok_or_else(|| GenerationError("missing image for placeholder".into()))?;
                    if t != 1 { return Err(GenerationError("still-image prefill requires temporal grid 1".into())); }
                    image += 1;
                    (h / merge) * (w / merge)
                } else { 1 };
                if expanded.len().checked_add(count).is_none_or(|n| n > self.text.config.max_position_embeddings) {
                    return Err(GenerationError("image prompt exceeds model sequence capacity".into()));
                }
                expanded.extend(std::iter::repeat_n(token, count));
            }
            rows.push(expanded);
        }
        if image != grids.len() { return Err(GenerationError("image has no matching placeholder".into())); }
        if rows.iter().any(|row| row.len() != rows[0].len()) {
            return Err(GenerationError("image prefill requires equal-length unpadded rows".into()));
        }
        Ok(rows)
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
    /// Native RGB preprocessing and prefill. Each prompt contains one placeholder per image,
    /// not the already expanded patch placeholders accepted by `prefill_images`.
    pub fn prefill_rgb(
        &self,
        tokens: &[Vec<i32>],
        images: &[Qwen35RgbImage<'_>],
        processor: &Qwen35ImageProcessor,
        cache: &mut Qwen35MultimodalCache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        processor.validate_vision(self.vision.config())?;
        let prepared = processor.preprocess_rgb(images)?;
        let tokens = self.expand_image_placeholders(tokens, &prepared.grids)?;
        let patches = Tensor::<DeviceBackend<R, F, I, BT>, 2>::from_data(
            TensorData::new(prepared.patches, prepared.shape),
            (&self.text.embedding.weight.val().device(), DType::F32),
        );
        self.prefill_images(&tokens, patches, &prepared.grids, cache)
    }
}
