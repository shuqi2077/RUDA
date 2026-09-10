
use ruda_model::module::Module;
use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;

/// Applies the rectified linear unit function element-wise
/// See also [relu](ruda_tensor::api::activation::relu)
///
#[derive(Module, Clone, Debug, Default)]
pub struct Relu;

impl Relu {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        ruda_model::tensor::activation::relu(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Relu::new();

        assert_eq!(alloc::format!("{layer}"), "Relu");
    }
}
