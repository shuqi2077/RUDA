use super::*;
use crate::{CausalModelLimits, GreedyGenerationConfig, TokenGenerationOutput, generate_causal_greedy};

struct ImageConditioned<'a, B: Backend> {
    model: &'a Qwen35MultimodalModel<B>,
    patches: Tensor<B, 2>,
    grids: Vec<[usize; 3]>,
}

impl<R, F, I, BT> CausalModel<DeviceBackend<R, F, I, BT>> for ImageConditioned<'_, DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    type Cache = Qwen35MultimodalCache<DeviceBackend<R, F, I, BT>>;

    fn new_cache(&self) -> Self::Cache { self.model.new_cache() }

    fn forward_cached_last(&self, tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>, cache: &mut Self::Cache)
        -> Tensor<DeviceBackend<R, F, I, BT>, 3>
    {
        self.try_forward_cached_last(tokens,cache).expect("Qwen3.5 image generation failed")
    }

    fn try_forward_cached_last(&self, tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>, cache: &mut Self::Cache)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError>
    {
        if cache.sequence_length() != 0 { return self.model.decode(tokens,cache); }
        let [batch, sequence] = tokens.dims();
        if batch != 1 || sequence == 0 { return Err(GenerationError("image generation requires a nonempty batch-one prompt".into())); }
        let ids = tokens.try_into_data().map_err(|e| GenerationError(e.to_string()))?
            .convert::<i32>().to_vec::<i32>().map_err(|e| GenerationError(e.to_string()))?;
        self.model.prefill_images(&[ids],self.patches.clone(),&self.grids,cache)
    }
}

impl<R, F, I, BT> Qwen35MultimodalModel<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    /// Generate from one prompt containing one placeholder per image, using the common greedy loop.
    pub fn generate_rgb_greedy(
        &self, tokens: &[i32], images: &[Qwen35RgbImage<'_>], processor: &Qwen35ImageProcessor,
        generation: &GreedyGenerationConfig,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        processor.validate_vision(self.vision.config())?;
        self.generate_prepared(tokens,processor.preprocess_rgb(images)?,generation)
    }

    /// Decode JPEG/PNG files and use the same native preprocessing, image prefill and text decode.
    pub fn generate_image_files_greedy<P: AsRef<Path>>(
        &self, tokens: &[i32], paths: &[P], processor: &Qwen35ImageProcessor,
        generation: &GreedyGenerationConfig,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        processor.validate_vision(self.vision.config())?;
        self.generate_prepared(tokens,processor.preprocess_files(paths)?,generation)
    }

    fn generate_prepared(
        &self, tokens: &[i32], prepared: Qwen35PreparedImages, generation: &GreedyGenerationConfig,
    ) -> Result<TokenGenerationOutput, GenerationError> {
        let rows = self.expand_image_placeholders(&[tokens.to_vec()],&prepared.grids)?;
        let device = self.text.embedding.weight.val().device();
        let conditioned = ImageConditioned { model:self, grids:prepared.grids,
            patches:Tensor::from_data(TensorData::new(prepared.patches,prepared.shape),(&device,DType::F32)) };
        let limits = CausalModelLimits { vocab_size:self.text.config.vocab_size,
            max_sequence_length:self.text.config.max_position_embeddings };
        generate_causal_greedy(&conditioned,&limits,&rows[0],generation,&device)
    }
}
