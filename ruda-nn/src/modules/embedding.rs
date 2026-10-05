
use ruda_model::config::Config;
use ruda_model::module::Initializer;
use ruda_model::module::Module;
use ruda_model::module::Param;
use ruda_model::module::{Content, DisplaySettings, ModuleDisplay};
use ruda_model::tensor::Int;
use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;

use ruda_model::tensor::module::embedding;

/// Configuration to create an [Embedding](Embedding) layer using the [init function](EmbeddingConfig::init).
#[derive(Config, Debug)]
pub struct EmbeddingConfig {
    /// The number of embedding vectors.
    pub n_embedding: usize,
    /// The size of each vector.
    pub d_model: usize,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::Normal{mean:0.0, std:1.0}")]
    pub initializer: Initializer,
}

/// Lookup table to store a fix number of vectors.
///
/// Should be created with [EmbeddingConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Embedding<B: Backend> {
    /// The learnable weights of the module of shape `[n_embedding, d_model]` initialized
    /// from a normal distribution `N(0, 1)`.
    pub weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> ModuleDisplay for Embedding<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [n_embedding, d_model] = self.weight.shape().dims();
        content
            .add("n_embedding", &n_embedding)
            .add("d_model", &d_model)
            .optional()
    }
}

impl EmbeddingConfig {
    /// Initialize a new [embedding](Embedding) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Embedding<B> {
        let weight = self
            .initializer
            .init([self.n_embedding, self.d_model], device);

        Embedding { weight }
    }
}

impl<B: Backend> Embedding<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See also [embedding](ruda_tensor::api::module::embedding).
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        embedding(self.weight.val(), input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use ruda_model::tensor::TensorData;
    use ruda_model::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn shared_embedding_preserves_dtype_and_accumulates_repeated_tokens() {
        use ruda_model::tensor::{DType, TensorPrimitive};
        use ruda_model::tensor::ops::embedding as shared;
        let device = Default::default();
        let indices = Tensor::<TestBackend, 2, Int>::from_data([[2, 1, 2], [0, 1, 2]], &device);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let weight = Tensor::<TestBackend, 2>::from_floats(
                [[1., 2.], [3., 4.], [5., 6.]], &device).cast(dtype);
            let output = Tensor::<TestBackend, 3>::from_primitive(TensorPrimitive::Float(
                shared::embedding::<TestBackend>(
                    weight.clone().into_primitive().tensor(), indices.clone().into_primitive())));
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims(), [2, 3, 2]);
            output.cast(DType::F32).to_data().assert_approx_eq::<f32>(
                &TensorData::from([[[5., 6.], [3., 4.], [5., 6.]], [[1., 2.], [3., 4.], [5., 6.]]]),
                Tolerance::absolute(0.));
            let grad = Tensor::<TestBackend, 3>::from_floats(
                [[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]], &device).cast(dtype);
            let grad = Tensor::<TestBackend, 2>::from_primitive(TensorPrimitive::Float(
                shared::embedding_backward::<TestBackend>(
                    weight.into_primitive().tensor(), grad.into_primitive().tensor(), indices.clone().into_primitive())));
            assert_eq!(grad.dtype(), dtype);
            grad.cast(DType::F32).to_data().assert_approx_eq::<f32>(
                &TensorData::from([[7., 8.], [12., 14.], [17., 20.]]), Tolerance::absolute(0.));
        }
    }

    #[test]
    fn initializer_zeros() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = EmbeddingConfig::new(5, 5).with_initializer(Initializer::Zeros);
        let embed = config.init::<TestBackend>(&Default::default());

        assert_eq!(config.initializer, Initializer::Zeros);
        embed.weight.to_data().assert_approx_eq::<FT>(
            &TensorData::zeros::<f32, _>(embed.weight.shape()),
            Tolerance::default(),
        );
    }

    #[test]
    fn display() {
        let config = EmbeddingConfig::new(100, 10);
        let embed = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{embed}"),
            "Embedding {n_embedding: 100, d_model: 10, params: 1000}"
        );
    }
}
