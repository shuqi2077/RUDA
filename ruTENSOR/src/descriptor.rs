use crate::{Error, Result, UnaryOp};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

/// A named tensor index. Names identify axes independently of their position.
pub type Mode = i32;

/// Extents and element strides, relative to the start of a tensor's buffer view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub(crate) extents: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) dtype: DType,
    elements: usize,
    bytes: usize,
}

pub(crate) fn product(values: &[usize]) -> Result<usize> {
    if values.contains(&0) { return Ok(0); }
    values.iter().try_fold(1usize, |n, &d| n.checked_mul(d).ok_or(Error::Overflow))
}

pub(crate) fn check_dtype(dtype: DType) -> Result<()> {
    match dtype {
        DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType(format!("{dtype:?}; use F16, BF16, F32 or F64"))),
    }
}

impl TensorDescriptor {
    pub fn new(extents: &[usize], strides: &[usize], dtype: DType) -> Result<Self> {
        check_dtype(dtype)?;
        if extents.len() != strides.len() {
            return Err(Error::InvalidDescriptor("extent and stride ranks differ".into()));
        }
        let elements = product(extents)?;
        let span = if elements == 0 { 0 } else {
            extents.iter().zip(strides).try_fold(1usize, |span, (&d, &s)| {
                span.checked_add((d - 1).checked_mul(s).ok_or(Error::Overflow)?)
                    .ok_or(Error::Overflow)
            })?
        };
        let bytes = span.checked_mul(dtype.size()).ok_or(Error::Overflow)?;
        Ok(Self { extents: extents.to_vec(), strides: strides.to_vec(), dtype, elements, bytes })
    }

    /// Packed row-major layout; an empty extent list denotes a scalar.
    pub fn contiguous(extents: &[usize], dtype: DType) -> Result<Self> {
        if extents.contains(&0) {
            return Self::new(extents, &vec![0; extents.len()], dtype);
        }
        let mut strides = vec![1; extents.len()];
        let mut stride = 1usize;
        for axis in (0..extents.len()).rev() {
            strides[axis] = stride;
            stride = stride.checked_mul(extents[axis].max(1)).ok_or(Error::Overflow)?;
        }
        Self::new(extents, &strides, dtype)
    }

    pub fn from_tensor<R: Runtime>(tensor: &RudaTensor<R>) -> Result<Self> {
        if tensor.qparams.is_some() {
            return Err(Error::UnsupportedDType("quantized tensor".into()));
        }
        let descriptor = Self::new(tensor.meta.shape(), tensor.meta.strides(), tensor.dtype)?;
        descriptor.check_buffer(tensor)?;
        Ok(descriptor)
    }

    pub fn extents(&self) -> &[usize] { &self.extents }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn rank(&self) -> usize { self.extents.len() }
    pub fn num_elements(&self) -> usize { self.elements }
    pub fn storage_bytes(&self) -> usize { self.bytes }

    /// Whether the axes form a writable, non-overlapping strided layout.
    pub fn is_nonoverlapping(&self) -> bool {
        if self.elements == 0 { return true; }
        let mut axes: Vec<_> = self.extents.iter().zip(&self.strides)
            .filter(|(d, _)| **d > 1).collect();
        axes.sort_by_key(|(_, s)| **s);
        let mut span = 1usize;
        for (&d, &s) in axes {
            if s < span { return false; }
            span += (d - 1) * s;
        }
        true
    }

    pub(crate) fn matches<R: Runtime>(&self, tensor: &RudaTensor<R>) -> bool {
        self.dtype == tensor.dtype && tensor.qparams.is_none()
            && self.extents.as_slice() == &tensor.meta.shape()[..]
            && self.strides.as_slice() == &tensor.meta.strides()[..]
    }

    pub(crate) fn check_buffer<R: Runtime>(&self, tensor: &RudaTensor<R>) -> Result<()> {
        if (tensor.handle.size() as u128) < self.bytes as u128 {
            return Err(Error::BufferTooSmall);
        }
        Ok(())
    }
}

/// Associates index names and a unary transform with a tensor descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperandDescriptor {
    pub(crate) tensor: TensorDescriptor,
    pub(crate) modes: Vec<Mode>,
    pub(crate) unary: UnaryOp,
}

impl OperandDescriptor {
    pub fn new(tensor: TensorDescriptor, modes: &[Mode]) -> Result<Self> {
        if tensor.rank() != modes.len() {
            return Err(Error::InvalidDescriptor("one mode is required for each axis".into()));
        }
        for (axis, mode) in modes.iter().enumerate() {
            for previous in 0..axis {
                if modes[previous] == *mode && tensor.extents[previous] != tensor.extents[axis] {
                    return Err(Error::InvalidDescriptor("repeated modes require equal diagonal extents".into()));
                }
            }
        }
        Ok(Self { tensor, modes: modes.to_vec(), unary: UnaryOp::Identity })
    }

    pub fn from_tensor<R: Runtime>(tensor: &RudaTensor<R>, modes: &[Mode]) -> Result<Self> {
        Self::new(TensorDescriptor::from_tensor(tensor)?, modes)
    }

    pub fn with_unary(mut self, unary: UnaryOp) -> Self { self.unary = unary; self }
    pub fn tensor(&self) -> &TensorDescriptor { &self.tensor }
    pub fn modes(&self) -> &[Mode] { &self.modes }
    pub fn unary(&self) -> UnaryOp { self.unary }
}
