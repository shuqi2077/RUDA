use super::*;

/// Boolean binary operation type.
#[derive(Clone, Copy)]
pub(super) enum BoolBinaryOp {
    And,
    Or,
    Xor,
}

pub(super) fn bool_binary_op_simd(lhs: HostTensor, rhs: HostTensor, op: BoolBinaryOp) -> HostTensor {
    use ruda_core::tensor::host::strided_index::StridedIter;

    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "bool_binary_op: dtype mismatch");

    // Broadcast to a common shape before dispatching. The scalar/SIMD helpers
    // below assume equal-length operands; without this, mismatched shapes
    // either silently keep the lhs shape or OOB-panic inside the helpers.
    let (mut lhs, mut rhs) = crate::expand::broadcast_binary(lhs, rhs);

    // Preserve the input bool dtype (taken from lhs; rhs is assumed to match
    // in dtype, checked above).
    let out_dtype = ruda_core::tensor::BoolDType::from(lhs.dtype());
    let shape = lhs.layout().shape().clone();
    let l_offsets = lhs.layout().contiguous_offsets();
    let r_offsets = rhs.layout().contiguous_offsets();

    // Fast path 1: lhs is unique and contiguous at offset 0 -> in-place on lhs
    if lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (l_offsets, r_offsets)
    {
        let rhs_storage: &[u8] = rhs.bytes();
        let r_slice = &rhs_storage[r_start..r_end];
        let lhs_storage: &mut [u8] = lhs.storage_mut();
        let l_slice = &mut lhs_storage[..l_end];

        match op {
            BoolBinaryOp::And => crate::simd::bool_and_inplace_u8(l_slice, r_slice),
            BoolBinaryOp::Or => crate::simd::bool_or_inplace_u8(l_slice, r_slice),
            BoolBinaryOp::Xor => crate::simd::bool_xor_inplace_u8(l_slice, r_slice),
        }
        return lhs;
    }

    // Fast path 2: rhs is unique and contiguous at offset 0 -> in-place on rhs
    // (And/Or/Xor are commutative, so we can swap operands)
    if rhs.is_unique()
        && let (Some((l_start, l_end)), Some((0, r_end))) = (l_offsets, r_offsets)
    {
        let lhs_storage: &[u8] = lhs.bytes();
        let l_slice = &lhs_storage[l_start..l_end];
        let rhs_storage: &mut [u8] = rhs.storage_mut();
        let r_slice = &mut rhs_storage[..r_end];

        match op {
            BoolBinaryOp::And => crate::simd::bool_and_inplace_u8(r_slice, l_slice),
            BoolBinaryOp::Or => crate::simd::bool_or_inplace_u8(r_slice, l_slice),
            BoolBinaryOp::Xor => crate::simd::bool_xor_inplace_u8(r_slice, l_slice),
        }
        return rhs;
    }

    // Allocating path: neither tensor is suitable for in-place
    let lhs_storage: &[u8] = lhs.bytes();
    let rhs_storage: &[u8] = rhs.bytes();

    let result: Vec<u8> = match (l_offsets, r_offsets) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            let mut out = vec![0u8; l_slice.len()];
            match op {
                BoolBinaryOp::And => crate::simd::bool_and_u8(l_slice, r_slice, &mut out),
                BoolBinaryOp::Or => crate::simd::bool_or_u8(l_slice, r_slice, &mut out),
                BoolBinaryOp::Xor => crate::simd::bool_xor_u8(l_slice, r_slice, &mut out),
            }
            out
        }
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            match op {
                BoolBinaryOp::And => lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| lhs_storage[li] & rhs_storage[ri])
                    .collect(),
                BoolBinaryOp::Or => lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| lhs_storage[li] | rhs_storage[ri])
                    .collect(),
                BoolBinaryOp::Xor => lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| lhs_storage[li] ^ rhs_storage[ri])
                    .collect(),
            }
        }
    };

    crate::comparison::make_bool_tensor(result, shape, out_dtype)
}

