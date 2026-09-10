use super::RuntimeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TensorDType { F16, BF16, F32, F64, I8, I4 }
impl TensorDType {
    pub const fn bits(self) -> u64 {
        match self { Self::F16 | Self::BF16 => 16, Self::F32 => 32, Self::F64 => 64, Self::I8 => 8, Self::I4 => 4 }
    }
    pub fn storage_bytes(self, elements: u64) -> Result<u64, RuntimeError> {
        let bits = elements.checked_mul(self.bits())
            .ok_or_else(|| RuntimeError::invalid("tensor bit count overflow"))?;
        Ok(bits / 8 + u64::from(bits % 8 != 0))
    }
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }
}

/// Positive-stride storage view. Zero strides (broadcast) and zero-sized
/// tensors are supported for storage accounting, not automatically for kernels.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset_elements: usize,
}
impl TensorLayout {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset_elements: usize) -> Result<Self, RuntimeError> {
        if shape.len() != strides.len() { return Err(RuntimeError::invalid("shape/stride rank mismatch")); }
        let value = Self { shape, strides, offset_elements };
        value.elements()?;
        value.storage_span_elements()?;
        Ok(value)
    }
    pub fn contiguous(shape: Vec<usize>) -> Result<Self, RuntimeError> {
        let mut stride = 1usize;
        let mut strides = vec![0; shape.len()];
        // Empty views access no elements. Avoid artificial overflow from the
        // dimensions on either side of a zero-sized dimension.
        if !shape.contains(&0) {
            for i in (0..shape.len()).rev() {
                strides[i] = stride;
                stride = stride.checked_mul(shape[i])
                    .ok_or_else(|| RuntimeError::invalid("contiguous stride overflow"))?;
            }
        }
        Self::new(shape, strides, 0)
    }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub const fn offset_elements(&self) -> usize { self.offset_elements }
    pub fn elements(&self) -> Result<usize, RuntimeError> {
        if self.shape.contains(&0) { return Ok(0); }
        self.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)
            .ok_or_else(|| RuntimeError::invalid("tensor element count overflow")))
    }
    /// Includes offset and holes; allocation bytes are not just product(shape).
    pub fn storage_span_elements(&self) -> Result<usize, RuntimeError> {
        if self.shape.contains(&0) { return Ok(0); }
        let last = self.shape.iter().zip(&self.strides).try_fold(self.offset_elements, |n, (&d, &s)| {
            (d - 1).checked_mul(s).and_then(|span| n.checked_add(span))
                .ok_or_else(|| RuntimeError::invalid("tensor storage span overflow"))
        })?;
        last.checked_add(1).ok_or_else(|| RuntimeError::invalid("tensor last offset overflow"))
    }
    pub fn storage_bytes(&self, dtype: TensorDType) -> Result<u64, RuntimeError> {
        dtype.storage_bytes(u64::try_from(self.storage_span_elements()?)
            .map_err(|_| RuntimeError::invalid("tensor storage span exceeds u64"))?)
    }
    pub fn is_contiguous(&self) -> bool {
        if self.shape.contains(&0) { return true; }
        let mut expected = 1usize;
        for (&d, &s) in self.shape.iter().zip(&self.strides).rev() {
            if d > 1 && s != expected { return false; }
            let Some(next) = expected.checked_mul(d) else { return false; };
            expected = next;
        }
        true
    }
    pub fn validate_storage(&self, dtype: TensorDType, available_bytes: u64) -> Result<(), RuntimeError> {
        if self.storage_bytes(dtype)? > available_bytes {
            return Err(RuntimeError::invalid("tensor view exceeds its storage allocation"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    pub batch: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_tokens: usize,
    pub cached_tokens: usize,
    pub head_dimension: usize,
}
impl AttentionShape {
    pub fn validate(self) -> Result<(), RuntimeError> {
        let values = [self.batch, self.query_heads, self.kv_heads, self.query_tokens,
            self.cached_tokens, self.head_dimension];
        if values.contains(&0) || self.query_heads % self.kv_heads != 0
            || self.query_tokens > self.cached_tokens {
            return Err(RuntimeError::invalid("invalid attention dimensions or grouped-query head ratio"));
        }
        TensorLayout::contiguous(vec![self.batch, self.query_heads, self.query_tokens, self.head_dimension])?;
        TensorLayout::contiguous(vec![self.batch, self.kv_heads, self.cached_tokens, self.head_dimension])?;
        Ok(())
    }
}
