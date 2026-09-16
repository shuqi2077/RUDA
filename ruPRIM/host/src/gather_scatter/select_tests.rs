use super::*;

fn tensor<E: Element + Pod>(shape: &[usize], values: Vec<E>) -> HostTensor {
    HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(shape.to_vec())),
        E::dtype(),
    )
}

fn indices(values: &[i64]) -> HostTensor {
    tensor(&[values.len()], values.to_vec())
}

// Deliberately use coordinate decoding as an independent reference for the
// block-copy kernel. This only receives valid indices and contiguous data.
fn reference<E: Copy>(shape: &[usize], data: &[E], dim: usize, selected: &[i64]) -> Vec<E> {
    let mut output_shape = shape.to_vec();
    output_shape[dim] = selected.len();
    let output_size: usize = output_shape.iter().product();
    (0..output_size)
        .map(|mut output_offset| {
            let mut source_offset = 0;
            let mut source_stride = 1;
            for axis in (0..shape.len()).rev() {
                let coord = output_offset % output_shape[axis];
                output_offset /= output_shape[axis];
                let source_coord = if axis == dim {
                    selected[coord] as usize
                } else {
                    coord
                };
                source_offset += source_coord * source_stride;
                source_stride *= shape[axis];
            }
            data[source_offset]
        })
        .collect()
}

#[test]
fn all_axes_match_coordinate_reference() {
    for shape in [
        vec![5],
        vec![4, 5],
        vec![3, 4, 5],
        vec![2, 3, 4, 5],
        vec![2, 1, 3, 1],
    ] {
        let size: usize = shape.iter().product();
        let data: Vec<i64> = (0..size).map(|i| i as i64 - 13).collect();
        for dim in 0..shape.len() {
            let last = shape[dim] as i64 - 1;
            let selected = [last, 0, last, last / 2];
            let output = select::<i64>(tensor(&shape, data.clone()), dim, indices(&selected));
            let mut expected_shape = shape.clone();
            expected_shape[dim] = selected.len();
            assert_eq!(output.layout().shape().to_vec(), expected_shape);
            assert_eq!(output.dtype(), DType::I64);
            assert_eq!(
                output.storage::<i64>(),
                reference(&shape, &data, dim, &selected),
                "shape={shape:?}, dim={dim}"
            );
        }
    }
}

#[test]
fn all_integer_index_widths_are_accepted() {
    let source = tensor(&[2, 3, 2], (0..12).map(i64::from).collect());
    let expected = [4i64, 5, 0, 1, 4, 5, 10, 11, 6, 7, 10, 11];
    macro_rules! check {
        ($($index_type:ty),+ $(,)?) => {
            $(
                let selected = tensor(&[3], vec![2 as $index_type, 0, 2]);
                let output = select::<i64>(source.clone(), 1, selected);
                assert_eq!(output.storage::<i64>(), &expected);
            )+
        };
    }
    check!(i8, i16, i32, i64, u8, u16, u32, u64);
}

#[test]
fn copies_float_bits_without_arithmetic() {
    let bits = [0x8000_0000, 0x7fc0_0001, 0x7f80_0000, 0xff80_0000, 1, 0];
    let data: Vec<f32> = bits.iter().copied().map(f32::from_bits).collect();
    let output = select::<f32>(tensor(&[1, 2, 3], data), 2, indices(&[1, 0, 2, 1]));
    let actual: Vec<u32> = output.storage::<f32>().iter().map(|v| v.to_bits()).collect();
    assert_eq!(actual, vec![bits[1], bits[0], bits[2], bits[1], bits[4], bits[3], bits[5], bits[4]]);
    assert_eq!(output.dtype(), DType::F32);

    let data = vec![f64::from_bits(0x7ff8_0000_0000_0042), -0.0f64, f64::INFINITY, 1.0];
    let output = select::<f64>(tensor(&[2, 1, 2], data.clone()), 0, indices(&[1, 0, 1]));
    let expected = reference(&[2, 1, 2], &data, 0, &[1, 0, 1]);
    assert_eq!(
        output.storage::<f64>().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(output.dtype(), DType::F64);
}

#[test]
fn transposed_offset_tensor_and_strided_indices() {
    let source = tensor(&[3, 4, 5], (0..60).map(i64::from).collect())
        .transpose(0, 2)
        .narrow(1, 1, 2);
    let mut logical_data = Vec::new();
    for i in 0..5 {
        for j in 0..2 {
            for k in 0..3 {
                logical_data.push((k * 20 + (j + 1) * 5 + i) as i64);
            }
        }
    }
    let selected = HostTensor::new(
        Bytes::from_elems(vec![99i32, 2, 99, 0, 99, 2, 99]),
        Layout::new(Shape::from(vec![3]), vec![2], 1),
        DType::I32,
    );
    let output = select::<i64>(source.clone(), 2, selected);
    assert_eq!(output.storage::<i64>(), reference(&[5, 2, 3], &logical_data, 2, &[2, 0, 2]));
    let output = select::<i64>(source, 0, indices(&[4, 0, 4]));
    assert_eq!(output.storage::<i64>(), reference(&[5, 2, 3], &logical_data, 0, &[4, 0, 4]));
}

#[test]
fn empty_indices_preserve_rank_and_dtype() {
    for shape in [vec![0], vec![3, 0], vec![2, 0, 3], vec![2, 3, 4, 1]] {
        let data = vec![0i64; shape.iter().product()];
        for dim in 0..shape.len() {
            let output = select::<i64>(tensor(&shape, data.clone()), dim, indices(&[]));
            let mut expected_shape = shape.clone();
            expected_shape[dim] = 0;
            assert_eq!(output.layout().shape().to_vec(), expected_shape);
            assert!(output.storage::<i64>().is_empty());
            assert_eq!(output.dtype(), DType::I64);
        }
    }
}

#[test]
fn zero_outer_rows_preserve_lazy_index_checks() {
    for (shape, dim) in [(vec![0, 3], 1), (vec![0, 2, 0], 1), (vec![2, 0, 3], 2)] {
        let output = select::<i64>(tensor(&shape, vec![]), dim, indices(&[-1]));
        let mut expected_shape = shape;
        expected_shape[dim] = 1;
        assert_eq!(output.layout().shape().to_vec(), expected_shape);
        assert!(output.storage::<i64>().is_empty());
    }
}

#[test]
fn empty_trailing_slices_with_valid_indices() {
    for (shape, dim) in [(vec![3, 0], 0), (vec![3, 2, 0], 0), (vec![2, 3, 0], 1)] {
        let output = select::<i64>(tensor(&shape, vec![]), dim, indices(&[2, 0, 2]));
        assert!(output.storage::<i64>().is_empty());
    }
}

#[test]
#[should_panic(expected = "index -1 out of bounds")]
fn negative_last_axis_index_panics() {
    select::<i64>(tensor(&[2, 2, 2], vec![0i64; 8]), 2, indices(&[-1]));
}

#[test]
#[should_panic(expected = "index 3 out of bounds for dimension of size 3")]
fn out_of_range_middle_index_panics() {
    select::<i64>(tensor(&[2, 3, 2], vec![0i64; 12]), 1, indices(&[3]));
}

#[test]
#[should_panic(expected = "index -1 out of bounds")]
fn zero_width_rows_still_check_indices() {
    select::<i64>(tensor(&[3, 0], vec![]), 0, indices(&[-1]));
}

#[test]
#[should_panic(expected = "index -1 out of bounds")]
fn zero_width_middle_slices_still_check_indices() {
    select::<i64>(tensor(&[2, 3, 0], vec![]), 1, indices(&[-1]));
}

#[test]
#[should_panic(expected = "index 0 out of bounds for dimension of size 0")]
fn empty_selected_axis_with_nonempty_output_panics() {
    select::<i64>(tensor(&[2, 0, 3], vec![]), 1, indices(&[0]));
}

#[test]
#[should_panic(expected = "out of isize range")]
fn unsigned_index_conversion_checks_range_even_for_empty_output() {
    select::<i64>(tensor(&[0, 3], vec![]), 1, tensor(&[1], vec![u64::MAX]));
}

#[test]
#[should_panic(expected = "select: indices must be 1D")]
fn indices_rank_is_checked() {
    select::<i64>(tensor(&[2, 3, 2], vec![0i64; 12]), 1, tensor(&[1, 1], vec![0i64]));
}

#[test]
#[should_panic(expected = "dim 3 out of bounds for 3 dimensions")]
fn dimension_is_checked() {
    select::<i64>(tensor(&[2, 3, 2], vec![0i64; 12]), 3, indices(&[0]));
}

#[test]
fn large_outputs_cover_batched_copy_boundaries() {
    // Cases exceed 4 MiB and include partial final Rayon batches.
    for (shape, dim, count) in [
        (vec![129, 129, 129], 0, 65),
        (vec![33, 129, 513], 1, 79),
        (vec![129, 129, 129], 2, 65),
    ] {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let selected: Vec<i64> = (0..count).map(|i| (shape[dim] - 1 - i) as i64).collect();
        let expected = reference(&shape, &data, dim, &selected);
        let output = select::<f32>(tensor(&shape, data), dim, indices(&selected));
        assert_eq!(output.storage::<f32>(), expected, "shape={shape:?}, dim={dim}");
    }
}
