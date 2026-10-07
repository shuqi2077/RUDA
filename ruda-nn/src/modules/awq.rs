use crate::{Dropout, DropoutConfig, Linear, LinearConfig, LoRALinearConfig};
use ruda_model::{
    module::{Initializer, Module, Param},
    tensor::{DType, FrozenAwqOps, Int, Tensor, TensorPrimitive, backend::Backend},
};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Frozen native AWQ projection retaining the original packed words and scales.
/// Construction accepts actual loaded values, not a dense matrix to requantize.
/// Its input derivative uses the same rounded coefficients as the packed forward.
#[derive(Module, Debug)]
pub struct FrozenAwqLinear<B: Backend> {
    /// I32 AWQ words with logical shape `[input, output / 8]`.
    pub qweight: Param<Tensor<B, 2, Int>>,
    /// I32 direct zero points with shape `[input / group_size, output / 8]`.
    pub qzeros: Param<Tensor<B, 2, Int>>,
    /// Original frozen FP16/BF16/FP32 scales, `[input / group_size, output]`.
    pub scales: Param<Tensor<B, 2>>,
    /// Optional original frozen bias, retaining the scale storage dtype.
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Explicit original checkpoint group size; no model-specific default.
    pub group_size: usize,
}

impl<B: Backend> FrozenAwqLinear<B> {
    /// Preserve loaded packed values, parameter IDs and source storage. Only
    /// floating base trainability is disabled; no dense weight is allocated.
    pub fn from_parameters(
        qweight: Param<Tensor<B, 2, Int>>, qzeros: Param<Tensor<B, 2, Int>>,
        scales: Param<Tensor<B, 2>>, bias: Option<Param<Tensor<B, 1>>>, group_size: usize,
    ) -> Self {
        let layer = Self { qweight, qzeros, scales, bias, group_size }.no_grad();
        layer.validate();
        layer
    }

    /// Check loaded shape/storage/device metadata without decoding packed values.
    pub fn validate(&self) {
        let weight = self.qweight.val();
        let zeros = self.qzeros.val();
        let scales = self.scales.val();
        let [input, packed_output] = weight.dims();
        let [groups, output] = scales.dims();
        assert!(input > 0 && output > 0 && self.group_size > 0, "AWQ dimensions/group must be positive");
        assert_eq!(input % self.group_size, 0, "AWQ input groups must be complete");
        assert_eq!(output % 8, 0, "AWQ output width must be divisible by eight");
        assert_eq!(packed_output, output / 8, "AWQ packed output width differs");
        assert_eq!(groups, input / self.group_size, "AWQ scale group count differs");
        assert_eq!(zeros.dims(), [groups, packed_output], "AWQ zero-point geometry differs");
        assert_eq!(weight.dtype(), DType::I32, "AWQ words must retain I32 storage");
        assert_eq!(zeros.dtype(), DType::I32, "AWQ zero points must retain I32 storage");
        assert!(matches!(scales.dtype(), DType::F16 | DType::BF16 | DType::F32), "AWQ scale storage is unsupported");
        assert!(!scales.is_require_grad(), "AWQ scales are frozen");
        assert!(weight.device() == zeros.device() && weight.device() == scales.device(), "AWQ devices differ");
        assert!(input.checked_mul(output).is_some_and(|n| n <= u32::MAX as usize), "AWQ matrix indexing overflows");
        if let Some(bias) = &self.bias {
            let bias = bias.val();
            assert_eq!(bias.dims(), [output], "AWQ bias width differs");
            assert_eq!(bias.dtype(), scales.dtype(), "AWQ bias/scale storage differs");
            assert!(bias.device() == weight.device(), "AWQ bias device differs");
            assert!(!bias.is_require_grad(), "AWQ bias is frozen");
        }
    }

    /// Actual source input/output widths, not the packed word count.
    pub fn dimensions(&self) -> [usize; 2] {
        [self.qweight.val().dims()[0], self.scales.val().dims()[1]]
    }
}

impl<B: FrozenAwqOps> FrozenAwqLinear<B> {
    /// Packed projection for arbitrary supported leading axes. Output retains
    /// activation storage and autograd reaches earlier trainable layers.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>, B::AwqError> {
        self.validate();
        B::frozen_awq_forward(
            input.into_primitive().tensor(), self.qweight.val().into_primitive(),
            self.qzeros.val().into_primitive(), self.scales.val().into_primitive().tensor(),
            self.bias.as_ref().map(|value| value.val().into_primitive().tensor()), self.group_size,
        ).map(|value| Tensor::from_primitive(TensorPrimitive::Float(value)))
    }
}

/// Trainable floating LoRA residual over an unchanged packed AWQ base.
/// The adapter dtype is independent of base scale dtype and activation dtype.
#[derive(Module, Debug)]
pub struct AwqLoRALinear<B: Backend> {
    /// Original frozen packed projection; never replaced by a floating matrix.
    pub base: FrozenAwqLinear<B>,
    /// Bias-free input-to-rank projection.
    pub adapter_a: Linear<B>,
    /// Bias-free rank-to-output projection.
    pub adapter_b: Linear<B>,
    /// Original caller-configured adapter-only dropout.
    pub dropout: Dropout,
    /// Explicit alpha/rank or alpha/sqrt(rank) multiplier.
    pub scale: f64,
}

impl LoRALinearConfig {
    /// Attach fresh trainable adapters with explicit floating storage and rsLoRA
    /// selection. The original packed base values and IDs remain unchanged.
    pub fn init_awq<B: Backend>(&self, base: FrozenAwqLinear<B>, adapter_dtype: DType, use_rslora: bool) -> AwqLoRALinear<B> {
        assert!(self.rank > 0 && self.alpha.is_finite(), "invalid AWQ adapter rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout), "adapter dropout must be in [0,1)");
        assert!(matches!(adapter_dtype, DType::F16 | DType::BF16 | DType::F32), "AWQ adapters require FP16/BF16/FP32 storage");
        base.validate();
        let [input, output] = base.dimensions();
        let device = base.qweight.val().device();
        let mut a = LinearConfig::new(input, self.rank).with_bias(false).init(&device);
        let mut b = LinearConfig::new(self.rank, output).with_bias(false).with_initializer(Initializer::Zeros).init(&device);
        a.weight = a.weight.map(|value| value.cast(adapter_dtype).detach().require_grad());
        b.weight = b.weight.map(|value| value.cast(adapter_dtype).detach().require_grad());
        self.from_awq_adapters(base, a, b, use_rslora)
    }

    /// Attach actual loaded A/B leaves without changing their IDs or values.
    /// No merge, requantization or independent base-scale gradient is implied.
    pub fn from_awq_adapters<B: Backend>(
        &self, base: FrozenAwqLinear<B>, adapter_a: Linear<B>, adapter_b: Linear<B>, use_rslora: bool,
    ) -> AwqLoRALinear<B> {
        assert!(self.rank > 0 && self.alpha.is_finite(), "invalid AWQ adapter rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout), "adapter dropout must be in [0,1)");
        base.validate();
        let [input, output] = base.dimensions();
        let device = base.qweight.val().device();
        let a = adapter_a.weight.val(); let b = adapter_b.weight.val();
        assert_eq!(a.dims(), [input, self.rank], "AWQ adapter A dimensions differ");
        assert_eq!(b.dims(), [self.rank, output], "AWQ adapter B dimensions differ");
        assert!(adapter_a.bias.is_none() && adapter_b.bias.is_none(), "AWQ adapters must be bias-free");
        assert!(a.device() == device && b.device() == device, "AWQ adapter/base devices differ");
        for dtype in [a.dtype(), b.dtype()] {
            assert!(matches!(dtype, DType::F16 | DType::BF16 | DType::F32), "unsupported AWQ adapter storage");
        }
        assert!(!B::ad_enabled(&device) || (a.is_require_grad() && b.is_require_grad()), "loaded AWQ adapters must be trainable");
        let denominator = if use_rslora { (self.rank as f64).sqrt() } else { self.rank as f64 };
        AwqLoRALinear { base, adapter_a, adapter_b, dropout: DropoutConfig::new(self.dropout).init(), scale: self.alpha / denominator }
    }
}

impl<B: FrozenAwqOps> AwqLoRALinear<B> {
    /// Original packed base plus scaled floating low-rank residual. Both paths
    /// retain their input derivatives, including through multiple adapted layers.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>, B::AwqError> {
        let base = self.base.forward(input.clone())?;
        let adapted = self.dropout.forward(input.cast(self.adapter_a.weight.val().dtype()));
        let hidden = self.adapter_a.forward(adapted).cast(self.adapter_b.weight.val().dtype());
        let update = self.adapter_b.forward(hidden).mul_scalar(self.scale);
        let dtype = base.dtype();
        Ok(base + update.cast(dtype))
    }
}
