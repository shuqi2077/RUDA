use ruda_model::{module::Module,tensor::{Tensor,backend::Backend}};

/// Stateless SiLU activation with native backend autodiff.
#[derive(Module,Clone,Debug,Default)]
pub struct Silu;

impl Silu {
    /// Construct the activation without parameters or a device allocation.
    pub fn new() -> Self { Self }

    /// Apply x*sigmoid(x), retaining input geometry and backend storage.
    pub fn forward<B: Backend,const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {
        ruda_model::tensor::activation::silu(input)
    }

    /// Explicit native SiLU training with FP32 low-precision VJP arithmetic.
    pub fn forward_native<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        ruda_model::tensor::activation::silu_native(input)
    }
}
