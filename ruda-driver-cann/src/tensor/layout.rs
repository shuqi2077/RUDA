use crate::CannError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    I32 = 3,
    U8 = 4,
    I16 = 6,
    U16 = 7,
    U32 = 8,
    I64 = 9,
    U64 = 10,
    F64 = 11,
    Bool = 12,
    Complex64 = 16,
    Complex128 = 17,
    BF16 = 27,
}

impl DType {
    pub const fn bytes(self) -> usize {
        match self {
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub(crate) shape: Vec<i64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) dtype: DType,
    bytes: usize,
}

impl TensorLayout {
    pub fn contiguous(shape: &[i64], dtype: DType) -> Result<Self, CannError> {
        if shape.iter().any(|&d| d < 0) {
            return Err(invalid("negative tensor dimension"));
        }
        let mut strides = vec![0; shape.len()];
        let mut elements = 1i64;
        // Empty tensors have no reachable elements; their strides need not overflow.
        if shape.contains(&0) {
            elements = 0;
        } else {
            for i in (0..shape.len()).rev() {
                strides[i] = elements;
                elements = elements
                    .checked_mul(shape[i])
                    .ok_or_else(|| invalid("tensor element count overflow"))?;
            }
        }
        let bytes = usize::try_from(elements)
            .ok()
            .and_then(|n| n.checked_mul(dtype.bytes()))
            .ok_or_else(|| invalid("tensor byte count overflow"))?;
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            dtype,
            bytes,
        })
    }
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> CannError {
    CannError::InvalidTensor(message.into())
}

pub(crate) fn broadcast(a: &[i64], b: &[i64]) -> Result<Vec<i64>, CannError> {
    let rank = a.len().max(b.len());
    let mut result = vec![1; rank];
    for i in 0..rank {
        let x = a.len().checked_sub(i + 1).map_or(1, |j| a[j]);
        let y = b.len().checked_sub(i + 1).map_or(1, |j| b[j]);
        result[rank - i - 1] = if x == y || y == 1 {
            x
        } else if x == 1 {
            y
        } else {
            return Err(invalid("incompatible broadcast dimensions"));
        };
    }
    Ok(result)
}

pub(crate) fn matmul_shape(a: &[i64], b: &[i64]) -> Result<Vec<i64>, CannError> {
    if a.is_empty() || b.is_empty() {
        return Err(invalid("matmul requires non-scalar inputs"));
    }
    let k = if b.len() == 1 { b[0] } else { b[b.len() - 2] };
    if a[a.len() - 1] != k {
        return Err(invalid("matmul contraction dimensions differ"));
    }
    let mut out = broadcast(
        &a[..a.len().saturating_sub(2)],
        &b[..b.len().saturating_sub(2)],
    )?;
    if a.len() > 1 {
        out.push(a[a.len() - 2]);
    }
    if b.len() > 1 {
        out.push(b[b.len() - 1]);
    }
    Ok(out)
}

pub(crate) fn axis(dim: i64, rank: usize) -> Result<usize, CannError> {
    let rank = i64::try_from(rank).map_err(|_| invalid("rank overflow"))?;
    let dim = if dim < 0 {
        dim.checked_add(rank)
            .ok_or_else(|| invalid("axis overflow"))?
    } else {
        dim
    };
    if dim < 0 || dim >= rank {
        Err(invalid("axis out of range"))
    } else {
        Ok(dim as usize)
    }
}
