use super::*;
use core::cmp::Ordering;

/// Return the indices of the `k` largest values along `dim`, in descending order.
/// Equal values retain their original index order; NaNs precede finite values
/// and infinities, and positive and negative zero compare equal.
pub fn argtopk(tensor: HostTensor, dim: usize, k: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => argtopk_typed::<f32>(tensor, dim, k, compare_f32),
        DType::F64 => argtopk_typed::<f64>(tensor, dim, k, compare_f64),
        DType::F16 => argtopk_typed::<f16>(tensor, dim, k, |a, b| {
            compare_f32(&a.to_f32(), &b.to_f32())
        }),
        DType::BF16 => argtopk_typed::<bf16>(tensor, dim, k, |a, b| {
            compare_f32(&a.to_f32(), &b.to_f32())
        }),
        DType::I64 => argtopk_typed::<i64>(tensor, dim, k, Ord::cmp),
        DType::I32 => argtopk_typed::<i32>(tensor, dim, k, Ord::cmp),
        DType::I16 => argtopk_typed::<i16>(tensor, dim, k, Ord::cmp),
        DType::I8 => argtopk_typed::<i8>(tensor, dim, k, Ord::cmp),
        DType::U64 => argtopk_typed::<u64>(tensor, dim, k, Ord::cmp),
        DType::U32 => argtopk_typed::<u32>(tensor, dim, k, Ord::cmp),
        DType::U16 => argtopk_typed::<u16>(tensor, dim, k, Ord::cmp),
        DType::U8 => argtopk_typed::<u8>(tensor, dim, k, Ord::cmp),
        dtype => panic!("argtopk: unsupported dtype {dtype:?}"),
    }
}

fn compare_f32(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

fn compare_f64(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

fn argtopk_typed<E: Element + Pod + Copy + Sync>(
    tensor: HostTensor,
    dim: usize,
    k: usize,
    compare: fn(&E, &E) -> Ordering,
) -> HostTensor {
    let shape = tensor.layout().shape().clone();
    let empty = validate_sort_args(&shape, dim);
    let axis_len = shape[dim];
    assert!(k <= axis_len, "argtopk: k exceeds axis length");
    let mut output_shape = shape.clone();
    output_shape[dim] = k;
    if empty || k == 0 {
        return make_index_tensor(Vec::new(), output_shape);
    }

    let tensor = tensor.to_contiguous();
    let data: &[E] = tensor.storage();
    let inner: usize = shape[dim + 1..].iter().product();
    let output_chunk = k * inner;
    let mut indices = vec![0isize; output_shape.num_elements()];
    let fill_outer = |heap: &mut Vec<usize>, (outer, output): (usize, &mut [isize])| {
        for column in 0..inner {
            let base = outer * axis_len * inner + column;
            let order = |a: &usize, b: &usize| {
                compare(&data[base + *b * inner], &data[base + *a * inner]).then_with(|| a.cmp(b))
            };

            // A single result needs only one running index, with no scratch
            // allocation. Use the same ordering as the general selection path.
            if k == 1 {
                let best = (1..axis_len).fold(0, |best, candidate| {
                    if order(&candidate, &best) == Ordering::Less {
                        candidate
                    } else {
                        best
                    }
                });
                output[column] = best as isize;
                continue;
            }

            select_indices(heap, axis_len, k, &order);
            for (slot, &coordinate) in heap.iter().enumerate() {
                output[slot * inner + column] = coordinate as isize;
            }
        }
    };

    #[cfg(feature = "rayon")]
    if data.len() >= PARALLEL_THRESHOLD {
        indices
            .par_chunks_mut(output_chunk)
            .enumerate()
            .for_each_init(Vec::new, fill_outer);
        return make_index_tensor(indices, output_shape);
    }
    let mut heap = Vec::new();
    indices
        .chunks_mut(output_chunk)
        .enumerate()
        .for_each(|outer_and_output| fill_outer(&mut heap, outer_and_output));
    make_index_tensor(indices, output_shape)
}

/// Select and order the first `k` indices under `order`, reusing scratch space.
/// The caller handles empty selections and the allocation-free k=1 path.
fn select_indices(
    heap: &mut Vec<usize>,
    axis_len: usize,
    k: usize,
    order: &impl Fn(&usize, &usize) -> Ordering,
) {
    heap.clear();
    heap.extend(0..k);

    // With a full-axis selection there are no candidates to discard, so
    // constructing a heap would only add work ahead of the final sort.
    if k < axis_len {
        // Bottom-up heap construction is O(k), instead of O(k log k)
        // repeated insertion. The root is the worst retained candidate.
        for root in (0..k / 2).rev() {
            sift_down(heap, root, order);
        }
        for candidate in k..axis_len {
            if order(&candidate, &heap[0]) == Ordering::Less {
                heap[0] = candidate;
                sift_down(heap, 0, order);
            }
        }
    }
    heap.sort_unstable_by(order);
}

fn sift_down(heap: &mut [usize], mut root: usize, order: &impl Fn(&usize, &usize) -> Ordering) {
    while root < heap.len() / 2 {
        let left = root * 2 + 1;
        let right = left + 1;
        let child = if right < heap.len() && order(&heap[left], &heap[right]) == Ordering::Less {
            right
        } else {
            left
        };
        if order(&heap[root], &heap[child]) != Ordering::Less {
            break;
        }
        heap.swap(root, child);
        root = child;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use a full stable sort as the reference, independently gathering logical
    // input coordinates so layout conversion is also covered by these tests.
    fn check_against_sort<E: Element + Pod + Copy>(
        data: Vec<E>,
        layout: Layout,
        dim: usize,
        ks: &[usize],
        compare: fn(&E, &E) -> Ordering,
    ) {
        let shape = layout.shape().clone();
        let logical: Vec<E> = (0..shape.num_elements())
            .map(|mut flat| {
                let mut offset = layout.start_offset() as isize;
                for d in (0..shape.num_dims()).rev() {
                    offset += (flat % shape[d]) as isize * layout.strides()[d];
                    flat /= shape[d];
                }
                data[offset as usize]
            })
            .collect();
        let tensor = HostTensor::new(Bytes::from_elems(data), layout, E::dtype());
        let axis_len = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();
        for &k in ks {
            let actual = argtopk(tensor.clone(), dim, k);
            let mut expected_shape = shape.clone();
            expected_shape[dim] = k;
            let mut expected = vec![0isize; expected_shape.num_elements()];
            for row in 0..outer {
                for column in 0..inner {
                    let base = row * axis_len * inner + column;
                    let mut sorted: Vec<usize> = (0..axis_len).collect();
                    // Stability retains the smaller input index for equal
                    // values, including NaNs and signed zeros.
                    sorted.sort_by(|&a, &b| {
                        compare(
                            &logical[base + b * inner],
                            &logical[base + a * inner],
                        )
                    });
                    for (slot, &index) in sorted.iter().take(k).enumerate() {
                        expected[row * k * inner + slot * inner + column] = index as isize;
                    }
                }
            }
            assert_eq!(actual.dtype(), INDEX_DTYPE);
            assert_eq!(actual.layout().shape(), &expected_shape);
            assert!(actual.layout().is_contiguous());
            assert_eq!(
                actual.storage::<isize>(),
                expected.as_slice(),
                "dim={dim}, k={k}"
            );
        }
    }

    fn float_order(a: &f64, b: &f64) -> Ordering {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.partial_cmp(b).unwrap(),
        }
    }

    #[test]
    fn all_k_match_full_sort_for_odd_and_even_axes() {
        for axis_len in [1, 2, 3, 4, 7, 8, 9, 16, 17, 32, 33] {
            let shape = Shape::new([3, axis_len, 2]);
            let data: Vec<i32> = (0..shape.num_elements())
                .map(|i| ((i * 37 + i / 11) % 17) as i32 - 8)
                .collect();
            check_against_sort(
                data,
                Layout::contiguous(shape),
                1,
                &(0..=axis_len).collect::<Vec<_>>(),
                Ord::cmp,
            );
        }
    }

    #[test]
    fn ordered_and_tied_inputs_match_full_sort() {
        for data in [
            (0..129).collect::<Vec<i32>>(),
            (0..129).rev().collect(),
            vec![7i32; 129],
        ] {
            check_against_sort(
                data,
                Layout::contiguous(Shape::new([129])),
                0,
                &[0, 1, 2, 3, 64, 128, 129],
                Ord::cmp,
            );
        }
    }

    #[test]
    fn floats_preserve_nan_ties_signed_zeros_and_precision() {
        let data = vec![
            -0.0f64,
            0.0,
            f64::NAN,
            f64::NEG_INFINITY,
            1.0,
            1.0 + f64::EPSILON,
            -f64::NAN,
            f64::INFINITY,
            1.0,
        ];
        let ks: Vec<usize> = (0..=data.len()).collect();
        let layout = Layout::contiguous(Shape::new([data.len()]));
        check_against_sort(data.clone(), layout.clone(), 0, &ks, float_order);
        check_against_sort(
            data.iter().map(|&v| v as f32).collect(),
            layout,
            0,
            &ks,
            |a, b| float_order(&(*a as f64), &(*b as f64)),
        );

        let tensor = HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::new([9])),
            DType::F64,
        );
        assert_eq!(
            argtopk(tensor, 0, 9).storage::<isize>(),
            &[2, 6, 7, 5, 4, 8, 0, 1, 3]
        );
    }

    #[test]
    fn half_types_preserve_special_values() {
        let values = [
            f32::NAN,
            -0.0,
            0.0,
            4.0,
            -f32::NAN,
            f32::INFINITY,
            -2.0,
        ];
        let layout = Layout::contiguous(Shape::new([values.len()]));
        let ks: Vec<usize> = (0..=values.len()).collect();
        check_against_sort(
            values.iter().map(|&v| f16::from_f32(v)).collect(),
            layout.clone(),
            0,
            &ks,
            |a, b| float_order(&a.to_f64(), &b.to_f64()),
        );
        check_against_sort(
            values.iter().map(|&v| bf16::from_f32(v)).collect(),
            layout,
            0,
            &ks,
            |a, b| float_order(&a.to_f64(), &b.to_f64()),
        );
    }

    #[test]
    fn integer_types_preserve_extreme_values() {
        macro_rules! check_integer {
            ($($ty:ty),+ $(,)?) => {$(
                check_against_sort(
                    vec![<$ty>::MAX, 0, <$ty>::MIN, 1, <$ty>::MAX - 1, <$ty>::MAX],
                    Layout::contiguous(Shape::new([6])),
                    0,
                    &[0, 1, 2, 3, 5, 6],
                    Ord::cmp,
                );
            )+};
        }
        check_integer!(i8, i16, i32, i64, u8, u16, u32, u64);
    }

    #[test]
    fn transposed_narrowed_and_negative_stride_views() {
        let data: Vec<i32> = (0..60).map(|i| (i * 37) % 19 - 9).collect();
        let layouts = [
            Layout::contiguous(Shape::new([3, 4, 5])).transpose(0, 2),
            Layout::contiguous(Shape::new([3, 4, 5])).narrow(1, 1, 2),
            Layout::new(Shape::new([2, 5, 3]), vec![30, -6, 2], 24),
        ];
        for layout in layouts {
            for dim in 0..layout.num_dims() {
                let ks: Vec<usize> = (0..=layout.shape()[dim]).collect();
                check_against_sort(data.clone(), layout.clone(), dim, &ks, Ord::cmp);
            }
        }
    }

    #[test]
    fn empty_axes_and_zero_k_keep_shape_and_index_dtype() {
        for shape in [Shape::new([0, 5]), Shape::new([3, 0])] {
            for dim in 0..shape.num_dims() {
                let ks: Vec<usize> = (0..=shape[dim]).collect();
                check_against_sort(
                    Vec::<f32>::new(),
                    Layout::contiguous(shape.clone()),
                    dim,
                    &ks,
                    |a, b| float_order(&(*a as f64), &(*b as f64)),
                );
            }
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_nonlast_axis_matches_full_sort() {
        let axis_len = 129;
        let rows = PARALLEL_THRESHOLD / (axis_len * 2) + 1;
        let shape = Shape::new([rows, axis_len, 2]);
        let data: Vec<i32> = (0..shape.num_elements())
            .map(|i| ((i * 7919) % 997) as i32)
            .collect();
        check_against_sort(
            data,
            Layout::contiguous(shape),
            1,
            &[1, 7, 128, 129],
            Ord::cmp,
        );
    }

    #[test]
    #[should_panic(expected = "argtopk: k exceeds axis length")]
    fn oversized_k_is_rejected() {
        argtopk(
            HostTensor::new(
                Bytes::from_elems(vec![1i32, 2]),
                Layout::contiguous(Shape::new([2])),
                DType::I32,
            ),
            0,
            3,
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn invalid_dim_is_rejected_even_with_zero_k() {
        argtopk(
            HostTensor::new(
                Bytes::from_elems(vec![1i32, 2]),
                Layout::contiguous(Shape::new([2])),
                DType::I32,
            ),
            1,
            0,
        );
    }
}
