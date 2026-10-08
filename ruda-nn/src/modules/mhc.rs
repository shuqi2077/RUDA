//! Manifold-constrained hyper-connections (mHC), arXiv:2512.24880.
//!
//! Generic tensor composition, including autodiff and Module/Record support.
//! This is not a fused mHC kernel. Finite Sinkhorn iterations approximate the
//! column constraint; the final row normalization does not prove exact double
//! stochasticity for arbitrary logits.
use alloc::vec;
use ruda_model::config::Config;
use ruda_model::module::{Initializer, Module, Param};
use ruda_model::tensor::activation::{log_softmax, sigmoid};
use ruda_model::tensor::{backend::Backend, DType, Tensor, TensorData};

/// Configuration of an mHC residual connection, without a built-in branch.
#[derive(Config, Debug)]
pub struct MhcConfig {
    /// Width of one residual stream.
    pub width: usize,
    /// Number of residual streams.
    #[config(default = 4)]
    pub streams: usize,
    /// Fixed number of column-then-row normalizations.
    #[config(default = 20)]
    pub sinkhorn_iterations: usize,
    /// RMS-normalization epsilon.
    #[config(default = 1e-6)]
    pub epsilon: f64,
    /// Initial gain of the three dynamic projections.
    #[config(default = 0.01)]
    pub gate_init: f64,
}

/// Input-dependent mHC mappings for [batch, length, streams, width] state.
#[derive(Debug, Clone)]
pub struct MhcMappings<B: Backend> {
    /// Nonnegative read-in weights [batch, length, streams].
    pub pre: Tensor<B, 3>,
    /// Nonnegative write-out weights [batch, length, streams].
    pub post: Tensor<B, 3>,
    /// Residual mixing matrix [batch, length, streams, streams].
    pub residual: Tensor<B, 4>,
}

/// Trainable mHC connection. Wrap a branch without its own residual addition.
#[derive(Module, Debug)]
pub struct Mhc<B: Backend> {
    /// Combined projections [streams * width, streams * streams + 2 * streams].
    pub mapping: Param<Tensor<B, 2>>,
    /// Dynamic gains, in pre/post/residual order.
    pub alpha: Param<Tensor<B, 1>>,
    /// Static pre/post/residual logits, flattened in the same order.
    pub bias: Param<Tensor<B, 1>>,
    /// Per-stream feature width.
    pub width: usize,
    /// Number of streams.
    pub streams: usize,
    /// Fixed iteration count.
    pub sinkhorn_iterations: usize,
    /// Normalization epsilon.
    pub epsilon: f64,
}

impl MhcConfig {
    /// Initialize trainable mappings; panics for invalid dimensions/configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mhc<B> {
        assert!(self.width > 0 && self.streams > 0, "mHC dimensions must be positive");
        assert!(self.sinkhorn_iterations > 0, "mHC needs at least one Sinkhorn iteration");
        assert!(self.epsilon.is_finite() && self.epsilon > 0.0, "invalid mHC epsilon");
        assert!(self.gate_init.is_finite() && self.gate_init > 0.0, "invalid mHC gate gain");
        let input = self.streams.checked_mul(self.width).expect("mHC input size overflow");
        let twice = self.streams.checked_mul(2).expect("mHC stream size overflow");
        let output = self.streams.checked_mul(self.streams)
            .and_then(|n| n.checked_add(twice)).expect("mHC mapping size overflow");
        let mut bias = vec![0.0f32; output];
        let pre_bias = if self.streams == 1 { 0.0 } else {
            -num_traits::Float::ln((self.streams - 1) as f64) as f32
        };
        bias[..self.streams].fill(pre_bias);
        for stream in 0..self.streams {
            bias[twice + stream * self.streams + stream] = 2.0;
        }
        Mhc {
            mapping: Initializer::Normal { mean: 0.0,
                std: 1.0 / num_traits::Float::sqrt(input as f64) }.init([input, output], device),
            alpha: Initializer::Constant { value: self.gate_init }.init([3], device),
            bias: Param::from_tensor(Tensor::from_data(TensorData::new(bias, [output]), device)),
            width: self.width,
            streams: self.streams,
            sinkhorn_iterations: self.sinkhorn_iterations,
            epsilon: self.epsilon,
        }
    }
}

/// Log-domain Sinkhorn, columns first and rows last. Keeps FP64; otherwise FP32.
pub fn mhc_sinkhorn<B: Backend>(logits: Tensor<B, 4>, iterations: usize) -> Tensor<B, 4> {
    let dims = logits.dims();
    assert!(dims[2] > 0 && dims[2] == dims[3], "mHC requires square nonempty matrices");
    assert!(iterations > 0, "mHC needs a positive iteration count");
    let dtype = if logits.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
    let mut result = logits.cast(dtype);
    for _ in 0..iterations {
        result = log_softmax(result, 2);
        result = log_softmax(result, 3);
    }
    result.exp()
}

impl<B: Backend> Mhc<B> {
    fn check(&self, state: &Tensor<B, 4>) -> [usize; 4] {
        let dims = state.dims();
        assert!(dims[0] > 0 && dims[1] > 0, "mHC batch and length must be nonempty");
        assert_eq!(dims[2], self.streams, "mHC stream count mismatch");
        assert_eq!(dims[3], self.width, "mHC width mismatch");
        dims
    }

    /// Expand [batch, length, width] into independent residual streams.
    pub fn expand(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
        assert_eq!(input.dims()[2], self.width, "mHC width mismatch");
        input.unsqueeze_dim(2).repeat_dim(2, self.streams)
    }

    /// Average streams back to [batch, length, width].
    pub fn reduce(&self, state: Tensor<B, 4>) -> Tensor<B, 3> {
        self.check(&state);
        state.mean_dim(2).squeeze_dim(2)
    }

    /// Calculate dynamic input/output mappings and constrained residual mixing.
    pub fn mappings(&self, state: Tensor<B, 4>) -> MhcMappings<B> {
        let [batch, length, n, width] = self.check(&state);
        let dtype = if state.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
        let flat: Tensor<B, 3> = state.cast(dtype).reshape([batch, length, n * width]);
        let norm = (flat.clone().square().mean_dim(2) + self.epsilon).sqrt();
        let projection = (flat / norm).matmul(self.mapping.val().cast(dtype).unsqueeze());
        let alpha = self.alpha.val().cast(dtype);
        let bias = self.bias.val().cast(dtype);
        let pre = projection.clone().slice([0..batch, 0..length, 0..n])
            * alpha.clone().slice([0..1]).reshape([1, 1, 1])
            + bias.clone().slice([0..n]).reshape([1, 1, n]);
        let post = projection.clone().slice([0..batch, 0..length, n..2*n])
            * alpha.clone().slice([1..2]).reshape([1, 1, 1])
            + bias.clone().slice([n..2*n]).reshape([1, 1, n]);
        let residual = projection.slice([0..batch, 0..length, 2*n..2*n+n*n])
            * alpha.slice([2..3]).reshape([1, 1, 1])
            + bias.slice([2*n..2*n+n*n]).reshape([1, 1, n*n]);
        MhcMappings {
            pre: sigmoid(pre),
            post: sigmoid(post) * 2.0,
            residual: mhc_sinkhorn(residual.reshape([batch, length, n, n]), self.sinkhorn_iterations),
        }
    }

    /// Merge streams for a branch and retain mappings for the matching post step.
    pub fn pre(&self, state: Tensor<B, 4>) -> (Tensor<B, 3>, MhcMappings<B>) {
        let dtype = state.dtype();
        let mappings = self.mappings(state.clone());
        let merged = (state.cast(mappings.pre.dtype()) * mappings.pre.clone().unsqueeze_dim(3))
            .sum_dim(2).squeeze_dim(2).cast(dtype);
        (merged, mappings)
    }

    /// Mix the old streams and broadcast the residual-free branch output.
    pub fn post(&self, state: Tensor<B, 4>, branch: Tensor<B, 3>, mappings: MhcMappings<B>) -> Tensor<B, 4> {
        let [batch, length, _, width] = self.check(&state);
        assert_eq!(branch.dims(), [batch, length, width], "mHC branch output shape mismatch");
        let dtype = state.dtype();
        let work_dtype = mappings.residual.dtype();
        let residual = mappings.residual.matmul(state.cast(work_dtype));
        let update = mappings.post.unsqueeze_dim(3) * branch.cast(work_dtype).unsqueeze_dim(2);
        (residual + update).cast(dtype)
    }

    /// Run any residual-free branch; normal tensor autodiff reaches all mappings.
    pub fn forward<F>(&self, state: Tensor<B, 4>, branch: F) -> Tensor<B, 4>
    where F: FnOnce(Tensor<B, 3>) -> Tensor<B, 3> {
        let (merged, mappings) = self.pre(state.clone());
        self.post(state, branch(merged), mappings)
    }

    /// Compose an actual fallible native branch without replacing its original error.
    pub fn try_forward<F, E>(&self, state: Tensor<B, 4>, branch: F) -> Result<Tensor<B, 4>, E>
    where F: FnOnce(Tensor<B, 3>) -> Result<Tensor<B, 3>, E> {
        let (merged, mappings) = self.pre(state.clone());
        Ok(self.post(state, branch(merged)?, mappings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearConfig, TestAutodiffBackend, TestBackend};
    use ruda_model::tensor::{ops::FloatElem, Distribution, Tolerance};

    #[test]
    fn mhc_uniform_logits_are_doubly_stochastic() {
        let device = Default::default();
        let logits = Tensor::<TestBackend, 4>::zeros([1, 2, 3, 3], &device);
        let actual = mhc_sinkhorn(logits, 20);
        actual.sum_dim(2).to_data().assert_approx_eq::<FloatElem<TestBackend>>(
            &Tensor::<TestBackend, 4>::ones([1, 2, 1, 3], &device).to_data(), Tolerance::default());
    }

    #[test]
    fn mhc_expand_reduce_round_trip() {
        let device = Default::default();
        let layer = MhcConfig::new(3).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::random([2, 4, 3], Distribution::Default, &device);
        layer.reduce(layer.expand(input.clone())).to_data().assert_approx_eq::<FloatElem<TestBackend>>(
            &input.to_data(), Tolerance::default());
    }

    #[test]
    fn mhc_branch_and_mappings_receive_gradients() {
        let device = Default::default();
        let layer = MhcConfig::new(3).with_streams(2).init::<TestAutodiffBackend>(&device);
        let branch = LinearConfig::new(3, 3).init::<TestAutodiffBackend>(&device);
        let input = Tensor::<TestAutodiffBackend, 4>::random([2, 4, 2, 3], Distribution::Default, &device).require_grad();
        let output = layer.forward(input.clone(), |x| branch.forward(x));
        let gradients = output.square().mean().backward();
        assert!(input.grad(&gradients).is_some());
        assert!(layer.mapping.grad(&gradients).is_some());
        assert!(layer.alpha.grad(&gradients).is_some());
        assert!(layer.bias.grad(&gradients).is_some());
        assert!(branch.weight.grad(&gradients).is_some());
    }

    #[test]
    fn mhc_record_round_trip() {
        let device = Default::default();
        let layer = MhcConfig::new(3).init::<TestBackend>(&device);
        let record = layer.clone().into_record();
        let restored = MhcConfig::new(3).init::<TestBackend>(&device).load_record(record);
        let input = Tensor::<TestBackend, 4>::ones([1, 2, 4, 3], &device);
        layer.forward(input.clone(), |x| x).to_data().assert_approx_eq::<FloatElem<TestBackend>>(
            &restored.forward(input, |x| x).to_data(), Tolerance::default());
    }

    #[test]
    #[should_panic(expected = "mHC dimensions must be positive")]
    fn mhc_rejects_zero_streams() {
        MhcConfig::new(3).with_streams(0).init::<TestBackend>(&Default::default());
    }
}
