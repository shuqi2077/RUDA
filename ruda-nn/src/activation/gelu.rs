
use ruda_model::module::Module;
use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;

/// Applies the Gaussian Error Linear Units function element-wise.
///
/// See also [gelu](ruda_tensor::api::activation::gelu)
///
/// When `approximate` is true, uses the tanh approximation:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
#[derive(Module, Clone, Debug, Default)]
pub struct Gelu {
    /// Whether to use tanh approximation.
    pub approximate: bool,
}

impl Gelu {
    /// Create the module with exact GELU.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create the module with tanh approximation.
    pub fn new_approximate() -> Self {
        Self { approximate: true }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if self.approximate {
            ruda_model::tensor::activation::gelu_approximate(input)
        } else {
            ruda_model::tensor::activation::gelu(input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use ruda_model::tensor::Tolerance;
    use ruda_model::tensor::ops::FloatElem;

    type FT = FloatElem<TestBackend>;

    #[test]
    fn shared_exact_gelu_backward_matches_original_host_derivative() {
        use ruda_model::tensor::{TensorPrimitive, ops::ActivationOps};
        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::from_floats([[-3., -1., 0.], [0.5, 1., 3.]], &device)
            .into_primitive().tensor();
        let upstream = Tensor::<TestBackend, 2>::from_floats([[0.25, -2., 3.], [0.125, 4., -0.5]], &device)
            .into_primitive().tensor();
        let expected = <TestBackend as ActivationOps<TestBackend>>::gelu_backward(input.clone(), upstream.clone());
        let actual = ruda_model::tensor::ops::gelu_backward_exact::<TestBackend>(input, upstream);
        Tensor::<TestBackend, 2>::from_primitive(TensorPrimitive::Float(actual)).to_data()
            .assert_approx_eq::<f32>(&Tensor::<TestBackend, 2>::from_primitive(TensorPrimitive::Float(expected)).to_data(),
                Tolerance::absolute(2e-6));
    }

    #[test]
    fn display() {
        let layer = Gelu::new();

        assert_eq!(alloc::format!("{layer}"), "Gelu {\n  approximate: false\n}");
    }

    #[test]
    fn forward_approximate() {
        let device = Default::default();
        let input =
            Tensor::<TestBackend, 2>::from_data([[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]], &device);

        let output = Gelu::new_approximate().forward(input);

        // PyTorch: torch.nn.functional.gelu(x, approximate="tanh")
        let expected = Tensor::<TestBackend, 2>::from_data(
            [
                [-0.1588079929, 0.0000000000, 0.8411920071],
                [0.3457140028, -0.1542859972, 1.9545977116],
            ],
            &device,
        );

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }
}
