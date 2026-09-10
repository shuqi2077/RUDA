
use ruda_model::{module::AutodiffModule, record::Record};

use ruda_model::config::Config;
use ruda_model::tensor::{Tensor, backend::AutodiffBackend};
use ruda_model::tensor::{backend::Backend, ops::Device};
use serde::{Deserialize, Serialize};

use super::{
    SimpleOptimizer,
    adaptor::OptimizerAdaptor,
    decay::WeightDecayConfig,
    momentum::{Momentum, MomentumConfig, MomentumState},
};
use crate::LearningRate;
use ruda_model::tensor::DType;

mod error;
pub use error::MuonError;
mod grouped;
pub use grouped::{MuonAdamW, MuonAdamWConfig, MuonAdamWRecord};

/// Momentum convention. Checkpoint buffers are NOT interchangeable between modes.
#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuonMomentumMode {
    /// Preserve RUDA's existing SGD momentum (including first-step initialization).
    #[default]
    Sgd,
    /// Exponential moving average: m = beta*m + (1-beta)*g, starting at zero.
    /// Nesterov update is (1-beta)*g + beta*m. Dampening must be zero.
    Ema,
}

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Logical orientation used ONLY for shape-based learning-rate scaling.
/// The tensor itself is not reshaped and its update keeps the same geometry.
#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuonMatrixLayout {
    /// Interpret [rows, columns] as [outputs, inputs]; legacy behavior.
    #[default]
    AsStored,
    /// Tensor is [inputs, outputs], as in ruda-nn Linear. Use the reversed
    /// aspect ratio for Original LR scaling (the RMS rule is symmetric).
    InputOutput,
}

/// Learning rate adjustment method for Muon optimizer.
///
/// Muon adjusts the learning rate based on parameter shape to maintain consistent
/// RMS across rectangular matrices.
///
/// # References
///
/// - Original: [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/)
/// - Moonshot: [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)
#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustLrFn {
    /// Keller Jordan's original method: `lr * sqrt(max(1, A/B))`
    ///
    /// This scales the learning rate based on the aspect ratio of the weight matrix,
    /// ensuring that tall matrices (more rows than columns) get proportionally larger
    /// learning rates.
    ///
    /// # Example
    ///
    /// For a [1024, 512] matrix: `lr * sqrt(1024/512) = lr * 1.414`
    #[default]
    Original,

    /// Moonshot's method: `lr * 0.2 * sqrt(max(A, B))`
    ///
    /// This method is designed to match AdamW's RMS, allowing Muon to directly reuse
    /// learning rates and weight decay values tuned for AdamW without retuning.
    ///
    /// # Example
    ///
    /// For a [1024, 512] matrix: `lr * 0.2 * sqrt(1024) = lr * 6.4`
    MatchRmsAdamW,
}

impl AdjustLrFn {
    /// Calculate the learning rate adjustment ratio for a given parameter shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - Parameter shape (uses first two dimensions)
    ///
    /// # Returns
    ///
    /// Adjustment ratio to multiply with the base learning rate
    fn adjustment_ratio(&self, shape: &[usize]) -> f64 {
        if shape.len() < 2 {
            return 1.0;
        }

        let a = shape[0] as f64;
        let b = shape[1] as f64;

        match self {
            Self::Original => {
                // sqrt(max(1, A/B))
                let ratio = a / b;
                ratio.max(1.0).sqrt()
            }
            Self::MatchRmsAdamW => {
                // 0.2 * sqrt(max(A, B))
                0.2 * a.max(b).sqrt()
            }
        }
    }
}

/// Muon configuration.
///
/// Muon is an optimizer specifically designed for 2D parameters of neural network
/// hidden layers (weight matrices). Other parameters such as biases and embeddings
/// should be optimized using a standard method such as AdamW.
///
/// # Learning Rate Adjustment
///
/// Muon adjusts the learning rate based on parameter shape to maintain consistent
/// RMS across rectangular matrices. Two methods are available:
///
/// - **Original**: Uses `sqrt(max(1, A/B))` where A and B are the first two dimensions.
///   This is Keller Jordan's method and is the default.
///
/// - **MatchRmsAdamW**: Uses `0.2 * sqrt(max(A, B))`. This is Moonshot's method
///   designed to match AdamW's RMS, allowing direct reuse of AdamW hyperparameters.
///
/// # Example
///
/// ```ignore
/// use ruda_optim::{MuonConfig, AdjustLrFn};
///
/// // Using default (Original) method
/// let optimizer = MuonConfig::new().init();
///
/// // Using MatchRmsAdamW for AdamW-compatible hyperparameters
/// let optimizer = MuonConfig::new()
///     .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
///     .init();
/// ```
///
/// # References
///
/// - [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
/// - [Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982)
/// - [PyTorch Implementation](https://github.com/pytorch/pytorch/blob/main/torch/optim/muon.py)
/// - [Original Implementation](https://github.com/KellerJordan/Muon)
#[derive(Config, Debug)]
pub struct MuonConfig {
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,

    /// [Momentum](MomentumConfig) config.
    ///
    /// Muon always uses momentum. Default configuration:
    /// - momentum: 0.95
    /// - dampening: 0.0
    /// - nesterov: true
    #[config(default = "MomentumConfig { momentum: 0.95, dampening: 0.0, nesterov: true }")]
    momentum: MomentumConfig,

    /// Newton-Schulz iteration coefficients (a, b, c).
    ///
    /// These coefficients are selected to maximize the slope at zero for the
    /// quintic iteration. Default values are from Keller Jordan's implementation.
    #[config(default = "(3.4445, -4.775, 2.0315)")]
    ns_coefficients: (f32, f32, f32),

    /// Epsilon for numerical stability.
    #[config(default = 1e-7)]
    epsilon: f32,

    /// Number of Newton-Schulz iteration steps.
    #[config(default = 5)]
    ns_steps: usize,

    /// Learning rate adjustment method.
    ///
    /// Controls how the learning rate is adjusted based on parameter shape.
    /// See [`AdjustLrFn`] for available methods.
    #[config(default = "AdjustLrFn::Original")]
    adjust_lr_fn: AdjustLrFn,

    /// Legacy SGD momentum by default; choose Ema explicitly for the current
    /// reference implementation's buffer convention. Save this config with records.
    #[config(default = "MuonMomentumMode::Sgd")]
    momentum_mode: MuonMomentumMode,

    /// Scale before squaring to avoid FP32 Frobenius-norm overflow/underflow.
    /// Opt-in to preserve legacy rounding. Requires FP32 parameters and gradients.
    #[config(default = false)]
    stable_normalization: bool,

    /// Explicit matrix orientation; embeddings are not identified by this flag.
    #[config(default = "MuonMatrixLayout::AsStored")]
    matrix_layout: MuonMatrixLayout,
}

impl MuonConfig {
    /// Build a [`Muon`] from the config.
    pub fn build<B: Backend>(&self) -> Muon<B> {
        self.try_build().unwrap_or_else(|error| panic!("{error}"))
    }

    /// Validate host configuration without launching a device operation.
    pub fn validate(&self) -> Result<(), MuonError> {
        let beta = self.momentum.momentum;
        let dampening = self.momentum.dampening;
        if !beta.is_finite() || !(0.0..1.0).contains(&beta) {
            return Err(MuonError::InvalidConfig("momentum must be finite in [0, 1)"));
        }
        if !dampening.is_finite() || !(0.0..=1.0).contains(&dampening) {
            return Err(MuonError::InvalidConfig("dampening must be finite in [0, 1]"));
        }
        if self.momentum.nesterov && (beta == 0.0 || dampening != 0.0) {
            return Err(MuonError::InvalidConfig("Nesterov requires positive momentum and zero dampening"));
        }
        if self.momentum_mode == MuonMomentumMode::Ema && dampening != 0.0 {
            return Err(MuonError::InvalidConfig("EMA momentum requires zero dampening"));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(MuonError::InvalidConfig("epsilon must be finite and positive"));
        }
        if !(1..100).contains(&self.ns_steps) {
            return Err(MuonError::InvalidConfig("Newton-Schulz steps must be in 1..100"));
        }
        let (a, b, c) = self.ns_coefficients;
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return Err(MuonError::InvalidConfig("Newton-Schulz coefficients must be finite"));
        }
        if let Some(decay) = &self.weight_decay {
            if !decay.penalty.is_finite() || decay.penalty < 0.0 {
                return Err(MuonError::InvalidConfig("weight decay must be finite and nonnegative"));
            }
        }
        Ok(())
    }

    /// Build with explicit configuration errors instead of a panic.
    pub fn try_build<B: Backend>(&self) -> Result<Muon<B>, MuonError> {
        self.validate()?;
        let momentum = Momentum::new(&self.momentum);
        let weight_decay_penalty = self.weight_decay.as_ref().map(|wd| wd.penalty);

        Ok(Muon {
            momentum,
            ns_params: NewtonSchulzParams::new(self.ns_coefficients, self.ns_steps),
            weight_decay_penalty,
            epsilon: self.epsilon,
            adjust_lr_fn: self.adjust_lr_fn,
            momentum_mode: self.momentum_mode,
            momentum_beta: self.momentum.momentum,
            nesterov: self.momentum.nesterov,
            stable_normalization: self.stable_normalization,
            matrix_layout: self.matrix_layout,
        })
    }

    /// Fallible model optimizer initialization. Only pass matrix-only modules;
    /// use MuonAdamWConfig for a complete model with biases/embeddings.
    pub fn try_init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> Result<OptimizerAdaptor<Muon<B::InnerBackend>, M, B>, MuonError> {
        Ok(OptimizerAdaptor::from(self.try_build()?))
    }

    /// Initialize Muon optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer adaptor that can be used to optimize a module.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ruda_optim::{MuonConfig, AdjustLrFn, decay::WeightDecayConfig};
    ///
    /// // Basic configuration with default (Original) LR adjustment
    /// let optimizer = MuonConfig::new()
    ///     .with_weight_decay(Some(WeightDecayConfig::new(0.01)))
    ///     .init();
    ///
    /// // With AdamW-compatible settings using MatchRmsAdamW
    /// let optimizer = MuonConfig::new()
    ///     .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
    ///     .with_weight_decay(Some(WeightDecayConfig::new(0.1)))
    ///     .init();
    ///
    /// // Custom momentum and NS settings
    /// let optimizer = MuonConfig::new()
    ///     .with_momentum(MomentumConfig {
    ///         momentum: 0.9,
    ///         dampening: 0.1,
    ///         nesterov: false,
    ///     })
    ///     .with_ns_steps(7)
    ///     .init();
    /// ```
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<Muon<B::InnerBackend>, M, B> {
        OptimizerAdaptor::from(self.build())
    }
}

/// Parameters for Newton-Schulz orthogonalization.
#[derive(Clone, Copy)]
struct NewtonSchulzParams {
    a: f32,
    b: f32,
    c: f32,
    steps: usize,
}

impl NewtonSchulzParams {
    fn new(coefficients: (f32, f32, f32), steps: usize) -> Self {
        Self {
            a: coefficients.0,
            b: coefficients.1,
            c: coefficients.2,
            steps,
        }
    }
}

/// Muon optimizer.
///
/// Muon internally runs standard SGD-momentum, and then performs an orthogonalization
/// post-processing step using a finite quintic Newton-Schulz iteration. This is an
/// approximate spectral transformation, NOT an exact polar decomposition or a
/// guarantee that U*U^T is identity. This implementation computes in the tensor's
/// dtype; it does not silently cast to BF16 or create FP32 master parameters.
///
/// # Important Notes
///
/// 1. **Only for nonempty 2D parameters**: Muon is designed for hidden weight matrices. Use AdamW
///    or SGD for biases, embeddings, and layer norms.
///
/// 2. **Learning rate adjustment**: Muon automatically adjusts the learning rate based
///    on parameter shape. See [`AdjustLrFn`] for details.
///
/// 3. **Weight decay timing**: Unlike typical optimizers, Muon applies weight decay
///    AFTER orthogonalization but uses the original (unadjusted) learning rate for it.
#[derive(Clone)]
pub struct Muon<B: Backend> {
    momentum: Momentum<B>,
    ns_params: NewtonSchulzParams,
    weight_decay_penalty: Option<f32>,
    epsilon: f32,
    adjust_lr_fn: AdjustLrFn,
    momentum_mode: MuonMomentumMode,
    momentum_beta: f64,
    nesterov: bool,
    stable_normalization: bool,
    matrix_layout: MuonMatrixLayout,
}

impl<B: Backend> Muon<B> {
    /// Check tensor metadata before submitting kernels. Does not scan values for
    /// NaN/Inf. Unscale and validate gradients before calling (on every rank).
    pub fn validate_step<const D: usize>(
        &self, lr: LearningRate, tensor: &Tensor<B, D>, grad: &Tensor<B, D>,
        state: Option<&MuonState<B, D>>,
    ) -> Result<(), MuonError> {
        if D != 2 { return Err(MuonError::ExpectedMatrix { rank: D }); }
        if !lr.is_finite() || lr < 0.0 {
            return Err(MuonError::InvalidConfig("learning rate must be finite and nonnegative"));
        }
        let shape = tensor.shape();
        if shape.iter().any(|dim| *dim == 0) { return Err(MuonError::EmptyMatrix); }
        if shape != grad.shape() { return Err(MuonError::ShapeMismatch("gradient")); }
        if tensor.dtype() != grad.dtype() { return Err(MuonError::DTypeMismatch("gradient")); }
        if tensor.device() != grad.device() { return Err(MuonError::DeviceMismatch("gradient")); }
        if self.stable_normalization && tensor.dtype() != DType::F32 {
            return Err(MuonError::InvalidConfig("stable normalization requires FP32 tensors"));
        }
        if let Some(state) = state {
            let buffer = state.momentum.velocity();
            if shape != buffer.shape() { return Err(MuonError::ShapeMismatch("momentum")); }
            if tensor.dtype() != buffer.dtype() { return Err(MuonError::DTypeMismatch("momentum")); }
            if tensor.device() != buffer.device() { return Err(MuonError::DeviceMismatch("momentum")); }
        }
        let adjusted = self.adjust_lr(lr, &shape);
        let decay = lr * self.weight_decay_penalty.unwrap_or(0.0) as f64;
        if !adjusted.is_finite() || !decay.is_finite()
            || (tensor.dtype() == DType::F32 && (!(adjusted as f32).is_finite() || !(decay as f32).is_finite())) {
            return Err(MuonError::InvalidConfig("effective learning rate/decay overflows"));
        }
        Ok(())
    }

    /// Fallible metadata-checked update. Output and state may still be executing
    /// asynchronously. An accepted submission is not proof of device completion.
    ///
    /// Uses a complete 2D matrix, never a flattened mixed-parameter buffer or
    /// an arbitrary shard. Missing gradients must be skipped by the caller.
    pub fn try_step<const D: usize>(
        &self, lr: LearningRate, tensor: Tensor<B, D>, grad: Tensor<B, D>,
        state: Option<MuonState<B, D>>,
    ) -> Result<(Tensor<B, D>, Option<MuonState<B, D>>), MuonError> {
        self.validate_step(lr, &tensor, &grad, state.as_ref())?;
        let (update, momentum) = match self.momentum_mode {
            MuonMomentumMode::Sgd => self.momentum.transform(grad, state.map(|s| s.momentum)),
            MuonMomentumMode::Ema => {
                let beta = self.momentum_beta;
                let buffer = match state {
                    Some(s) => s.momentum.velocity().clone().mul_scalar(beta)
                        .add(grad.clone().mul_scalar(1.0 - beta)),
                    None => grad.clone().mul_scalar(1.0 - beta),
                };
                let update = if self.nesterov {
                    grad.mul_scalar(1.0 - beta).add(buffer.clone().mul_scalar(beta))
                } else { buffer.clone() };
                (update, MomentumState::new(buffer))
            }
        };
        let update = self.zeropower_via_newtonschulz(update);
        let adjusted_lr = self.adjust_lr(lr, &tensor.shape());
        let tensor = match self.weight_decay_penalty {
            Some(penalty) => tensor.mul_scalar(1.0 - lr * penalty as f64),
            None => tensor,
        };
        Ok((tensor - update.mul_scalar(adjusted_lr), Some(MuonState::new(momentum))))
    }

    /// Adjust learning rate based on parameter shape.
    ///
    /// # Arguments
    ///
    /// * `lr` - Base learning rate
    /// * `shape` - Parameter shape (uses first two dimensions)
    ///
    /// # Returns
    ///
    /// Adjusted learning rate
    ///
    /// ```ignore
    /// // For a [1024, 512] weight matrix with lr=0.01:
    /// // Original: 0.01 * sqrt(1024/512) = 0.01 * 1.414 = 0.01414
    /// // MatchRmsAdamW: 0.01 * 0.2 * sqrt(1024) = 0.01 * 0.2 * 32 = 0.064
    /// ```
    fn adjust_lr(&self, lr: LearningRate, shape: &[usize]) -> LearningRate {
        let ratio = match self.matrix_layout {
            MuonMatrixLayout::InputOutput if shape.len() == 2 =>
                self.adjust_lr_fn.adjustment_ratio(&[shape[1], shape[0]]),
            _ => self.adjust_lr_fn.adjustment_ratio(shape),
        };
        lr * ratio
    }

    /// Perform Newton-Schulz orthogonalization on a gradient tensor.
    ///
    /// This computes the zeroth power (orthogonalization) of the input matrix G
    /// using a quintic Newton-Schulz iteration.
    ///
    /// # Algorithm
    ///
    /// 1. Transpose if tall matrix (A > B)
    /// 2. Normalize: X = X / ||X||
    /// 3. For k steps:
    ///    - A = X @ X^T
    ///    - B = b*A + c*A^2
    ///    - X = a*X + B@X
    /// 4. Transpose back if needed
    ///
    /// # References
    ///
    /// - Original: https://github.com/KellerJordan/Muon/blob/master/muon.py
    /// - PyTorch: https://github.com/pytorch/pytorch/blob/main/torch/optim/muon.py
    fn zeropower_via_newtonschulz<const D: usize>(&self, g: Tensor<B, D>) -> Tensor<B, D> {
        let shape = g.shape();
        let dim_m2 = shape[D - 2];
        let dim_m1 = shape[D - 1];

        // Step 1: Transpose if tall matrix (more rows than columns)
        let (mut x, needs_transpose) = if dim_m2 > dim_m1 {
            (g.swap_dims(D - 2, D - 1), true)
        } else {
            (g, false)
        };

        // Step 2: Normalize by Frobenius norm
        // X = X / (||X|| + epsilon)
        if self.stable_normalization {
            // Equivalent in real arithmetic to x / max(||x||_F, eps), without
            // forming x*x at its original scale. A zero matrix stays zero.
            // The caller validates FP32; MIN_POSITIVE is not representable in FP16.
            let scale = x.clone().abs().max().clamp_min(f32::MIN_POSITIVE);
            let scaled = x.div(scale.clone().unsqueeze());
            let floor = scale.recip().mul_scalar(self.epsilon);
            let norm = scaled.clone().square().sum().sqrt().max_pair(floor);
            x = scaled.div(norm.unsqueeze());
        } else {
            // Preserve the existing RUDA normalization and dtype/rounding path.
            let norm = x.clone().powf_scalar(2.0).sum().sqrt()
                .clamp_min(self.epsilon).unsqueeze();
            x = x.div(norm);
        }

        // Step 3: Newton-Schulz iteration
        // This is the quintic iteration with coefficients (a, b, c)
        let NewtonSchulzParams { a, b, c, steps } = self.ns_params;

        for _ in 0..steps {
            // A = X @ X^T
            let x_t = x.clone().swap_dims(D - 2, D - 1);
            let a_matrix = x.clone().matmul(x_t);

            // B = b*A + c*A@A
            let a_squared = a_matrix.clone().matmul(a_matrix.clone());
            let b_matrix = a_matrix.mul_scalar(b).add(a_squared.mul_scalar(c));

            // X = a*X + B@X
            x = x.clone().mul_scalar(a).add(b_matrix.matmul(x.clone()));
        }

        // Step 4: Restore transpose if it was a tall matrix
        if needs_transpose {
            x = x.swap_dims(D - 2, D - 1);
        }

        x
    }
}

/// Muon state.
#[derive(Record, Clone, new)]
pub struct MuonState<B: Backend, const D: usize> {
    /// Current momentum state
    pub momentum: MomentumState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for Muon<B> {
    type State<const D: usize> = MuonState<B, D>;

    /// Perform a single Muon optimization step.
    ///
    /// # Algorithm
    ///
    /// 1. Apply momentum to gradient
    /// 2. Orthogonalize update via Newton-Schulz
    /// 3. Adjust learning rate based on parameter shape
    /// 4. Apply weight decay (using original lr)
    /// 5. Update parameter (using adjusted lr)
    ///
    /// # Notes
    ///
    /// Unlike typical optimizers, the weight decay and parameter update use
    /// different learning rates:
    /// - Weight decay uses the original `lr`
    /// - Parameter update uses the shape-adjusted `lr`
    ///
    /// # Panics
    /// This function will panic if the input tensors are not 2D.
    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        self.try_step(lr, tensor, grad, state)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

#[cfg(test)]
mod tests;
