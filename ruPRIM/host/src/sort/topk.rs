use super::*;
use core::cmp::Ordering;

pub fn argtopk(tensor: HostTensor, dim: usize, k: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => argtopk_typed::<f32>(tensor, dim, k, compare_f32),
        DType::F64 => argtopk_typed::<f64>(tensor, dim, k, compare_f64),
        DType::F16 => argtopk_typed::<f16>(tensor, dim, k, |a, b| compare_f32(&a.to_f32(), &b.to_f32())),
        DType::BF16 => argtopk_typed::<bf16>(tensor, dim, k, |a, b| compare_f32(&a.to_f32(), &b.to_f32())),
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
    let fill_outer = |outer: usize, output: &mut [isize]| {
        let mut heap = Vec::with_capacity(k);
        for column in 0..inner {
            heap.clear();
            let base = outer * axis_len * inner + column;
            let order = |a: &usize, b: &usize| {
                compare(&data[base + *b * inner], &data[base + *a * inner])
                    .then_with(|| a.cmp(b))
            };
            for candidate in 0..axis_len {
                if heap.len() < k {
                    heap.push(candidate);
                    let mut child = heap.len() - 1;
                    while child > 0 {
                        let parent = (child - 1) / 2;
                        if order(&heap[parent], &heap[child]) != Ordering::Less {
                            break;
                        }
                        heap.swap(parent, child);
                        child = parent;
                    }
                } else if order(&candidate, &heap[0]) == Ordering::Less {
                    heap[0] = candidate;
                    sift_down(&mut heap, &order);
                }
            }
            heap.sort_unstable_by(order);
            for (slot, &coordinate) in heap.iter().enumerate() {
                output[slot * inner + column] = coordinate as isize;
            }
        }
    };

    #[cfg(feature = "rayon")]
    if data.len() >= PARALLEL_THRESHOLD {
        indices.par_chunks_mut(output_chunk).enumerate()
            .for_each(|(outer, output)| fill_outer(outer, output));
        return make_index_tensor(indices, output_shape);
    }
    for (outer, output) in indices.chunks_mut(output_chunk).enumerate() {
        fill_outer(outer, output);
    }
    make_index_tensor(indices, output_shape)
}

fn sift_down(heap: &mut [usize], order: &impl Fn(&usize, &usize) -> Ordering) {
    let mut root = 0;
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
