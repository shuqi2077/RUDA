use alloc::collections::BTreeSet;
use core::fmt;
use ruda_model::{
    module::{AutodiffModule, ModuleVisitor, Param, ParamId},
    tensor::{DType, FloatDType, Tensor, TensorMetadata, backend::AutodiffBackend},
};
use super::GradientsParams;

/// Invalid gradient metadata or an explicit scaling argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GradientTransformError {
    /// The multiplier/divisor is not finite or cannot be represented in the work dtype.
    InvalidScalar,
    /// The requested arithmetic dtype is not F32 or F64.
    InvalidDType,
    /// A gradient does not have its parameter's geometry or device.
    ParameterMismatch(ParamId),
    /// A gradient is quantized; arithmetic must not silently dequantize it.
    QuantizedGradient(ParamId),
    /// At least one registered gradient is not owned by the supplied module.
    UnknownParameters,
}

impl fmt::Display for GradientTransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScalar => f.write_str("gradient scale must be representable and finite; a divisor must be positive"),
            Self::InvalidDType => f.write_str("gradient work dtype must be F32 or F64"),
            Self::ParameterMismatch(id) => write!(f,"gradient geometry/device does not match parameter {id}"),
            Self::QuantizedGradient(id) => write!(f,"quantized gradient for parameter {id} cannot be scaled implicitly"),
            Self::UnknownParameters => f.write_str("gradient container includes parameters outside the supplied module"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GradientTransformError {}

pub(super) fn validate_work_dtype(dtype: FloatDType) -> Result<(), GradientTransformError> {
    if matches!(dtype,FloatDType::F32 | FloatDType::F64) { Ok(()) }
    else { Err(GradientTransformError::InvalidDType) }
}

pub(super) fn representable(value: f64, dtype: FloatDType) -> bool {
    value.is_finite() && (dtype == FloatDType::F64 || (value as f32).is_finite())
}

struct Metadata<'a> {
    gradients: &'a GradientsParams,
    seen: BTreeSet<ParamId>,
    matched: usize,
    error: Option<GradientTransformError>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for Metadata<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B,D>>) {
        if self.error.is_some() || !self.seen.insert(param.id) { return; }
        let Some(primitive) = self.gradients.container.get::<B::InnerBackend>(&param.id) else { return; };
        self.matched += 1;
        if matches!(primitive.dtype(),DType::QFloat(_)) {
            self.error = Some(GradientTransformError::QuantizedGradient(param.id));
            return;
        }
        if primitive.rank() != D {
            self.error = Some(GradientTransformError::ParameterMismatch(param.id));
            return;
        }
        let gradient = Tensor::<B::InnerBackend,D>::from_primitive(primitive);
        let parameter = param.val();
        if gradient.dims() != parameter.dims() || gradient.device() != parameter.device() {
            self.error = Some(GradientTransformError::ParameterMismatch(param.id));
        }
    }
}

impl GradientsParams {
    /// Check module membership, actual dimensions and device without reading values.
    /// Tied parameter IDs are counted once; absent gradients stay absent.
    pub fn validate_for<B: AutodiffBackend,M: AutodiffModule<B>>(
        &self, module: &M,
    ) -> Result<(),GradientTransformError> {
        let mut visitor = Metadata { gradients:self,seen:BTreeSet::new(),matched:0,error:None };
        module.visit(&mut visitor);
        if let Some(error) = visitor.error { return Err(error); }
        if visitor.matched != self.len() { return Err(GradientTransformError::UnknownParameters); }
        Ok(())
    }

    /// Copy handles and optionally cast all present module gradients to F32/F64.
    /// This does not modify parameter storage, IDs or the source container.
    pub fn cast_for<B: AutodiffBackend,M: AutodiffModule<B>>(
        &self, module: &M, dtype: FloatDType,
    ) -> Result<Self,GradientTransformError> {
        self.transformed::<B,M>(module,dtype,1.,false)
    }

    /// Multiply each present gradient in an explicit work dtype, once per tied ID.
    /// Negative and zero multipliers are intentional caller-selected transforms.
    pub fn scaled_for<B: AutodiffBackend,M: AutodiffModule<B>>(
        &self, module: &M, multiplier: f64, dtype: FloatDType,
    ) -> Result<Self,GradientTransformError> {
        self.transformed::<B,M>(module,dtype,multiplier,false)
    }

    /// Divide by a positive finite scale in F32/F64 without clearing the source.
    /// No clipping, nonfinite-step policy, optimizer update or dynamic scaling.
    pub fn unscaled_for<B: AutodiffBackend,M: AutodiffModule<B>>(
        &self, module: &M, divisor: f64, dtype: FloatDType,
    ) -> Result<Self,GradientTransformError> {
        self.transformed::<B,M>(module,dtype,divisor,true)
    }

    fn transformed<B: AutodiffBackend,M: AutodiffModule<B>>(
        &self, module: &M, dtype: FloatDType, scalar: f64, divide: bool,
    ) -> Result<Self,GradientTransformError> {
        validate_work_dtype(dtype)?;
        if !representable(scalar,dtype) || (divide && (scalar <= 0. ||
            (dtype == FloatDType::F32 && scalar as f32 == 0.))) {
            return Err(GradientTransformError::InvalidScalar);
        }
        self.validate_for::<B,M>(module)?;
        let mut visitor = Transform {source:self,result:Self::new(),seen:BTreeSet::new(),dtype,scalar,divide};
        module.visit(&mut visitor);
        Ok(visitor.result)
    }
}

struct Transform<'a> {
    source: &'a GradientsParams,
    result: GradientsParams,
    seen: BTreeSet<ParamId>,
    dtype: FloatDType,
    scalar: f64,
    divide: bool,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for Transform<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B,D>>) {
        if !self.seen.insert(param.id) { return; }
        let Some(gradient) = self.source.get::<B::InnerBackend,D>(param.id) else { return; };
        let gradient = gradient.cast(self.dtype);
        let value = if self.scalar == 1. { gradient }
            else if self.divide { gradient.div_scalar(self.scalar) }
            else { gradient.mul_scalar(self.scalar) };
        self.result.register::<B::InnerBackend,D>(param.id,value);
    }
}
