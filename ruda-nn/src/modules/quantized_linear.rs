use crate::{Dropout, DropoutConfig, Linear, LinearConfig, LoRALinearConfig};
use ruda_model::{module::{Initializer, Module, Param, ParamId},
    tensor::{DType, FloatDType, Tensor, backend::Backend, quantization::QuantScheme}};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Frozen original packed `[output,input]` projection for any backend-supported QuantScheme.
/// INT2/4/8 and FP4/FP8 retain their actual codes, scales, block geometry and packing.
/// Computation uses the backend's quantized matmul, not a prebuilt full dense weight shadow.
#[derive(Module, Debug)]
pub struct QuantizedLinear<B: Backend> {
    /// Actual frozen packed weight, retaining its original parameter identity.
    pub weight: Param<Tensor<B, 2>>,
    /// Optional actual frozen bias.
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> QuantizedLinear<B> {
    /// Adopt existing packed parameter storage without decoding or requantizing it.
    pub fn from_parameters(weight: Param<Tensor<B, 2>>, bias: Option<Param<Tensor<B, 1>>>) -> Self {
        let layer = Self { weight: weight.map(|value| value.set_require_grad(false)),
            bias: bias.map(|value| value.map(|value| value.set_require_grad(false))) };
        layer.validate();
        layer
    }

    /// Check original packed geometry, floating bias and frozen flags without decoding values.
    pub fn validate(&self) {
        let value = self.weight.val();
        let [output, input] = value.dims();
        assert!(input > 0 && output > 0 && matches!(value.dtype(), DType::QFloat(_)), "quantized linear requires nonempty original packed weights");
        assert!(!value.is_require_grad(), "quantized linear base must remain frozen");
        if let Some(bias) = &self.bias {
            let bias = bias.val();
            assert_eq!(bias.dims(), [output], "quantized linear bias shape differs");
            assert_eq!(bias.device(), value.device(), "quantized linear bias device differs");
            assert!(matches!(bias.dtype(), DType::F16 | DType::BF16 | DType::F32), "quantized linear bias must use FP16/BF16/FP32");
            assert!(!bias.is_require_grad(), "quantized linear bias must remain frozen");
        }
    }

    /// Adopt actual packed tensors without temporarily creating a trainable dense surrogate.
    pub fn from_quantized(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        Self::from_parameters(Param::initialized(ParamId::new(), weight.set_require_grad(false)),
            bias.map(|value| Param::initialized(ParamId::new(), value.set_require_grad(false))))
    }

    /// Pack explicitly supplied floating weights using the chosen format and calibration arithmetic.
    /// No source/model inference or automatic format substitution is performed.
    pub fn from_float(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>, scheme: &QuantScheme,
        calibration_dtype: FloatDType) -> Self {
        let weight = weight.detach().set_require_grad(false).quantize_dynamic_with_precision(scheme, calibration_dtype);
        Self::from_quantized(weight, bias)
    }

    /// Pack an actual native floating Linear, preserving original weight/bias parameter IDs.
    /// Native `[input,output]` weights are transposed before applying the explicitly supplied
    /// block scheme in packed `[output,input]` coordinates; no geometry is inferred.
    pub fn from_linear(layer: Linear<B>, scheme: &QuantScheme, calibration_dtype: FloatDType) -> Self {
        let weight = layer.weight.map(|value| value.detach().set_require_grad(false).transpose()
            .quantize_dynamic_with_precision(scheme, calibration_dtype));
        Self::from_parameters(weight, layer.bias)
    }

    /// Project actual last-axis inputs using FP16/BF16/FP32 storage and the backend's packed matmul.
    /// Input gradients flow through the original quantized projection; base parameters remain frozen.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        assert!(D > 0, "quantized linear requires an input axis");
        assert!(matches!(input.dtype(), DType::F16 | DType::BF16 | DType::F32), "quantized linear compute requires FP16/BF16/FP32");
        let dtype = input.dtype();
        let weight = self.weight.val();
        let [output, width] = weight.dims();
        let mut shape = input.dims();
        assert_eq!(shape[D - 1], width, "quantized linear input width differs");
        assert_eq!(input.device(), weight.device(), "quantized linear input device differs");
        let rows = shape[..D - 1].iter().try_fold(1usize, |size, &extent| size.checked_mul(extent)).expect("quantized linear batch overflow");
        rows.checked_mul(output).expect("quantized linear output overflow");
        let mut value = input.reshape([rows, width]).matmul(weight.transpose()).dequantize_with_dtype(dtype.into());
        if let Some(bias) = &self.bias { value = value + bias.val().cast(dtype).reshape([1, output]); }
        shape[D - 1] = output;
        value.reshape(shape)
    }

    /// Explicit computation storage, independent of packed values and scale storage.
    pub fn forward_with_dtype<const D: usize>(&self, input: Tensor<B, D>, dtype: FloatDType) -> Tensor<B, D> {
        self.forward(input.cast(dtype))
    }
}

/// Trainable LoRA/RSLoRA residual on an original generic INT2/4/8 or FP4/FP8 packed base.
#[derive(Module, Debug)]
pub struct QuantizedLoRALinear<B: Backend> {
    /// Frozen original packed projection and optional bias.
    pub base: QuantizedLinear<B>,
    /// Actual trainable input-to-rank projection.
    pub adapter_a: Linear<B>,
    /// Actual trainable rank-to-output projection.
    pub adapter_b: Linear<B>,
    /// Original adapter-only dropout.
    pub dropout: Dropout,
    /// Explicit alpha/rank or alpha/sqrt(rank) multiplier.
    pub scale: f64,
}

impl LoRALinearConfig {
    /// Initialize only the adapter matrices; packed base values and IDs are unchanged.
    pub fn init_quantized<B: Backend>(&self, base: QuantizedLinear<B>, adapter_dtype: DType,
        use_rslora: bool) -> QuantizedLoRALinear<B> {
        assert!(self.rank > 0 && self.alpha.is_finite(), "invalid quantized LoRA rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout), "LoRA dropout must be in [0,1)");
        assert!(matches!(adapter_dtype, DType::F16 | DType::BF16 | DType::F32), "quantized LoRA adapter storage requires FP16/BF16/FP32");
        let weight = base.weight.val();
        let [output, input] = weight.dims();
        let device = weight.device();
        let mut adapter_a = LinearConfig::new(input, self.rank).with_bias(false).init(&device);
        let mut adapter_b = LinearConfig::new(self.rank, output).with_bias(false).with_initializer(Initializer::Zeros).init(&device);
        adapter_a.weight = adapter_a.weight.map(|value| value.cast(adapter_dtype).detach().require_grad());
        adapter_b.weight = adapter_b.weight.map(|value| value.cast(adapter_dtype).detach().require_grad());
        self.from_quantized_adapters(base, adapter_a, adapter_b, use_rslora)
    }

    /// Attach actual loaded adapters without replacing IDs, storage, values or packed base metadata.
    pub fn from_quantized_adapters<B: Backend>(&self, base: QuantizedLinear<B>, adapter_a: Linear<B>, adapter_b: Linear<B>,
        use_rslora: bool) -> QuantizedLoRALinear<B> {
        assert!(self.rank > 0 && self.alpha.is_finite(), "invalid quantized LoRA rank/alpha");
        assert!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout), "LoRA dropout must be in [0,1)");
        let weight = base.weight.val();
        let [output, input] = weight.dims();
        assert_eq!(adapter_a.weight.val().dims(), [input, self.rank], "quantized LoRA A shape differs");
        assert_eq!(adapter_b.weight.val().dims(), [self.rank, output], "quantized LoRA B shape differs");
        assert!(adapter_a.bias.is_none() && adapter_b.bias.is_none(), "quantized LoRA adapters must be bias-free");
        for value in [adapter_a.weight.val(), adapter_b.weight.val()] {
            assert_eq!(value.device(), weight.device(), "quantized LoRA adapter device differs");
            assert!(matches!(value.dtype(), DType::F16 | DType::BF16 | DType::F32), "quantized LoRA adapter precision unsupported");
            assert!(!B::ad_enabled(&value.device()) || value.is_require_grad(), "quantized LoRA adapters must be trainable");
        }
        let divisor = if use_rslora { (self.rank as f64).sqrt() } else { self.rank as f64 };
        QuantizedLoRALinear { base, adapter_a, adapter_b, dropout: DropoutConfig::new(self.dropout).init(), scale: self.alpha / divisor }
    }
}

impl<B: Backend> QuantizedLoRALinear<B> {
    /// Convert only A/B storage, retaining parameter IDs, trainability and the entire packed base.
    pub fn with_adapter_dtype(mut self, dtype: FloatDType) -> Self {
        assert!(matches!(dtype, FloatDType::F16 | FloatDType::BF16 | FloatDType::F32), "adapter precision requires FP16/BF16/FP32");
        self.adapter_a.weight = self.adapter_a.weight.map(|value| {
            let trainable = value.is_require_grad(); value.cast(dtype).detach().set_require_grad(trainable)
        });
        self.adapter_b.weight = self.adapter_b.weight.map(|value| {
            let trainable = value.is_require_grad(); value.cast(dtype).detach().set_require_grad(trainable)
        });
        self
    }

    /// Packed base output plus actual mixed-storage adapter update; output retains base activation storage.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let adapted = self.dropout.forward(input.clone().cast(self.adapter_a.weight.val().dtype()));
        let hidden = self.adapter_a.forward(adapted).cast(self.adapter_b.weight.val().dtype());
        let update = self.adapter_b.forward(hidden).mul_scalar(self.scale);
        let base = self.base.forward(input);
        let dtype = base.dtype();
        base + update.cast(dtype)
    }

    /// Explicit base activation arithmetic; adapter storage and packed metadata are unchanged.
    pub fn forward_with_dtype<const D: usize>(&self, input: Tensor<B, D>, dtype: FloatDType) -> Tensor<B, D> {
        self.forward(input.cast(dtype))
    }

    /// A/B-only continuation record; frozen INT2/4/8 or FP4/FP8 payload is not copied.
    pub fn adapter_record(&self, base_id: &str) -> Result<crate::LoRAAdapterRecord<B>, ruda_model::record::RecorderError> {
        crate::LoRAAdapterRecord::capture_quantized(self, base_id)
    }
}
