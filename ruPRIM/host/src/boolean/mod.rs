use alloc::{vec, vec::Vec};
use ruda_core::tensor::{DType, Shape, host::HostTensor};

mod binary;
use binary::{BoolBinaryOp, bool_binary_op_simd};
mod indexing;
pub use indexing::{bool_argwhere, bool_select_or};

pub fn bool_equal(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    use ruda_core::tensor::host::strided_index::StridedIter;

    // Broadcast to a common shape before comparing. The contiguous fast
    // path below uses `zip`, which silently truncates to the shorter
    // operand; and the output shape is taken from lhs, so mismatched
    // operands would otherwise produce a result vec shorter than the
    // output layout claims.
    let (lhs, rhs) = crate::expand::broadcast_binary(lhs, rhs);

    let out_dtype = ruda_core::tensor::BoolDType::from(lhs.dtype());
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[u8] = lhs.bytes();
    let rhs_storage: &[u8] = rhs.bytes();

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| (a == b) as u8)
                .collect()
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| (lhs_storage[li] == rhs_storage[ri]) as u8)
                .collect()
        }
    };

    crate::comparison::make_bool_tensor(result, shape, out_dtype)
}

pub fn bool_not(mut tensor: HostTensor) -> HostTensor {
    use ruda_core::tensor::host::strided_index::StridedIter;

    debug_assert!(
        matches!(
            tensor.dtype(),
            DType::Bool(ruda_core::tensor::BoolStore::Native | ruda_core::tensor::BoolStore::U8)
        ),
        "bool_not: only Bool(Native) and Bool(U8) are supported, got {:?}",
        tensor.dtype()
    );

    // Fast path: in-place for unique, contiguous tensors at offset 0. This
    // preserves the input tensor's dtype tag implicitly (the in-place SIMD
    // ops flip bytes without touching the dtype tag).
    if tensor.is_unique()
        && tensor.layout().is_contiguous()
        && tensor.layout().start_offset() == 0
    {
        let storage = tensor.storage_mut::<u8>();
        crate::simd::bool_not_inplace_u8(storage);
        return tensor;
    }

    // Allocating path for shared, non-contiguous, or offset tensors:
    // preserve the input's bool dtype for the new tensor.
    let out_dtype = ruda_core::tensor::BoolDType::from(tensor.dtype());
    let shape = tensor.layout().shape().clone();
    let storage: &[u8] = tensor.bytes();

    let result: Vec<u8> = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let slice = &storage[start..end];
            let mut out = vec![0u8; slice.len()];
            crate::simd::bool_not_u8(slice, &mut out);
            out
        }
        None => StridedIter::new(tensor.layout())
            .map(|idx| (storage[idx] == 0) as u8)
            .collect(),
    };

    crate::comparison::make_bool_tensor(result, shape, out_dtype)
}

pub fn bool_and(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    bool_binary_op_simd(lhs, rhs, BoolBinaryOp::And)
}

pub fn bool_or(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    bool_binary_op_simd(lhs, rhs, BoolBinaryOp::Or)
}

pub fn bool_xor(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    bool_binary_op_simd(lhs, rhs, BoolBinaryOp::Xor)
}

pub fn bool_ones(
    shape: Shape,
    dtype: ruda_core::tensor::BoolDType,
) -> HostTensor {
    let num_elements = shape.num_elements();
    let data = vec![1u8; num_elements];
    crate::comparison::make_bool_tensor(data, shape, dtype)
}

pub fn bool_equal_elem(lhs: HostTensor, rhs: ruda_core::tensor::element::Scalar) -> HostTensor {
    use ruda_core::tensor::host::strided_index::StridedIter;

    let out_dtype = ruda_core::tensor::BoolDType::from(lhs.dtype());
    let shape = lhs.layout().shape().clone();
    let storage: &[u8] = lhs.bytes();
    let rhs_bool: bool = rhs.elem();
    let rhs_val = rhs_bool as u8;

    let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
        Some((start, end)) => storage[start..end]
            .iter()
            .map(|&v| (v == rhs_val) as u8)
            .collect(),
        None => StridedIter::new(lhs.layout())
            .map(|idx| (storage[idx] == rhs_val) as u8)
            .collect(),
    };

    crate::comparison::make_bool_tensor(result, shape, out_dtype)
}
