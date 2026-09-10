use alloc::borrow::Cow;
use crate::tensor::DType;
use half::{bf16, f16};

use crate::tensor::host::HostTensor;

/// The `DType` that matches `isize` on the current platform.
#[cfg(target_pointer_width = "64")]
pub const INDEX_DTYPE: DType = DType::I64;
/// The `DType` that matches `isize` on the current platform.
#[cfg(target_pointer_width = "32")]
pub const INDEX_DTYPE: DType = DType::I32;

/// Read a float tensor's storage as f32 values, regardless of source dtype.
/// Returns a borrowed slice for F32 (zero-copy) and an owned Vec for other
/// float dtypes (F64, F16, BF16).
///
/// Returns elements in underlying buffer order. Callers needing logical
/// iteration order must call `to_contiguous()` first.
///
/// # Panics
/// Panics if the tensor's dtype is not one of F32, F64, F16, or BF16.
pub fn float_storage_as_f32(tensor: &HostTensor) -> Cow<'_, [f32]> {
    match tensor.dtype() {
        DType::F32 => Cow::Borrowed(tensor.storage::<f32>()),
        DType::F64 => Cow::Owned(tensor.storage::<f64>().iter().map(|&x| x as f32).collect()),
        DType::F16 => Cow::Owned(
            tensor
                .storage::<f16>()
                .iter()
                .map(|x| f32::from(*x))
                .collect(),
        ),
        DType::BF16 => Cow::Owned(
            tensor
                .storage::<bf16>()
                .iter()
                .map(|x| f32::from(*x))
                .collect(),
        ),
        other => panic!("float_storage_as_f32: unsupported dtype {:?}", other),
    }
}

