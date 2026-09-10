use super::*;

#[test]
fn contiguous_layout_preserves_scalar_empty_and_strides() {
    let matrix = TensorLayout::contiguous(&[2, 3, 4], DType::BF16).unwrap();
    assert_eq!(matrix.strides(), &[12, 4, 1]);
    assert_eq!(matrix.byte_len(), 48);
    assert_eq!(
        TensorLayout::contiguous(&[], DType::F64)
            .unwrap()
            .byte_len(),
        8
    );
    assert_eq!(
        TensorLayout::contiguous(&[i64::MAX, 0, i64::MAX], DType::F32)
            .unwrap()
            .byte_len(),
        0
    );
}

#[test]
fn invalid_layouts_fail_without_allocating() {
    assert!(TensorLayout::contiguous(&[-1], DType::F32).is_err());
    assert!(TensorLayout::contiguous(&[i64::MAX, 2], DType::F16).is_err());
    assert!(TensorLayout::contiguous(&[i64::MAX], DType::Complex128).is_err());
}

#[test]
fn fixed_width_dtypes_keep_their_storage_sizes() {
    for (dtype, size) in [
        (DType::F32, 4),
        (DType::F16, 2),
        (DType::BF16, 2),
        (DType::F64, 8),
        (DType::I64, 8),
        (DType::Bool, 1),
        (DType::Complex128, 16),
    ] {
        assert_eq!(
            TensorLayout::contiguous(&[7], dtype).unwrap().byte_len(),
            7 * size
        );
    }
}

#[test]
fn broadcast_includes_empty_dimensions() {
    assert_eq!(layout::broadcast(&[2, 1, 4], &[3, 4]).unwrap(), [2, 3, 4]);
    assert_eq!(layout::broadcast(&[0, 4], &[1, 4]).unwrap(), [0, 4]);
    assert_eq!(layout::broadcast(&[], &[3]).unwrap(), [3]);
    assert!(layout::broadcast(&[2, 4], &[3, 4]).is_err());
}

#[test]
fn matmul_shape_keeps_vector_and_batch_semantics() {
    for (a, b, out) in [
        (vec![3], vec![3], vec![]),
        (vec![2, 3], vec![3], vec![2]),
        (vec![3], vec![4, 3, 5], vec![4, 5]),
        (vec![7, 1, 2, 3], vec![4, 3, 5], vec![7, 4, 2, 5]),
        (vec![0, 3], vec![3, 4], vec![0, 4]),
    ] {
        assert_eq!(layout::matmul_shape(&a, &b).unwrap(), out);
    }
    assert!(layout::matmul_shape(&[], &[3]).is_err());
    assert!(layout::matmul_shape(&[2, 3], &[4, 2]).is_err());
    assert!(layout::matmul_shape(&[2, 3, 4], &[3, 4, 5]).is_err());
}

#[test]
fn negative_axes_and_invalid_axes_are_checked() {
    assert_eq!(layout::axis(-1, 3).unwrap(), 2);
    assert_eq!(layout::axis(-3, 3).unwrap(), 0);
    for axis in [i64::MIN, -4, 3, i64::MAX] {
        assert!(layout::axis(axis, 3).is_err());
    }
    assert!(layout::axis(0, 0).is_err());
}
