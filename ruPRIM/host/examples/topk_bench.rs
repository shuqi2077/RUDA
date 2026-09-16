//! Compare the public host argtopk API with the original heap implementation.
//!
//! Run with the same toolchain, features and thread count for both implementations:
//! `RAYON_NUM_THREADS=4 cargo run --release -p ruPRIM-host --example topk_bench`
//! Optional arguments: `--samples 9 --iterations 3 --filter last_small_k`.
//! CSV on stdout reports medians per operation; configuration goes to stderr.
//! Includes tensor cloning, contiguous conversion, allocation and output drop.
//! This measures CPU f32 argtopk, not GPU kernels or end-to-end model execution.

use ruda_core::{
    bytes::Bytes,
    tensor::{
        DType, Shape,
        element::Element,
        host::{HostTensor, Layout, dtype::INDEX_DTYPE},
    },
};
use std::{cmp::Ordering, hint::black_box, time::Instant};

#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use ruda_core::tensor::host::parallel::PARALLEL_THRESHOLD;

fn make_index_tensor(indices: Vec<isize>, shape: Shape) -> HostTensor {
    HostTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(shape),
        INDEX_DTYPE,
    )
}

fn validate_sort_args(shape: &Shape, dim: usize) -> bool {
    assert!(
        dim < shape.num_dims(),
        "sort: dim {} out of bounds for tensor with {} dimensions",
        dim,
        shape.num_dims()
    );
    let dim_size = shape[dim];
    assert!(
        dim_size <= isize::MAX as usize,
        "sort: dimension {} has size {} which exceeds isize::MAX",
        dim,
        dim_size
    );
    shape.num_elements() == 0
}

fn compare_f32(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

// Baseline from ac817b115e50b7e3ba26f61149647bdafdfd97cd:
// ruPRIM/host/src/sort/topk.rs, with dtype dispatch restricted to f32.
// Keep this algorithm, scratch allocation placement and output layout unchanged.
fn legacy_argtopk(tensor: HostTensor, dim: usize, k: usize) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => legacy_argtopk_typed::<f32>(tensor, dim, k, compare_f32),
        dtype => panic!("argtopk: unsupported benchmark dtype {dtype:?}"),
    }
}

fn legacy_argtopk_typed<E: Element + bytemuck::Pod + Copy + Sync>(
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
                    legacy_sift_down(&mut heap, &order);
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
        indices
            .par_chunks_mut(output_chunk)
            .enumerate()
            .for_each(|(outer, output)| fill_outer(outer, output));
        return make_index_tensor(indices, output_shape);
    }
    for (outer, output) in indices.chunks_mut(output_chunk).enumerate() {
        fill_outer(outer, output);
    }
    make_index_tensor(indices, output_shape)
}

fn legacy_sift_down(heap: &mut [usize], order: &impl Fn(&usize, &usize) -> Ordering) {
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

struct Case {
    name: &'static str,
    tensor: HostTensor,
    dim: usize,
    k: usize,
}

fn tensor(shape: &[usize]) -> HostTensor {
    let shape = Shape::from(shape.to_vec());
    let mut state = 0x12ab_34cdu32;
    let data: Vec<f32> = (0..shape.num_elements())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) % 1024) as f32 - 512.0
        })
        .collect();
    HostTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), DType::F32)
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let last = tensor(&[16, 4096]);
    for (name, k) in [
        ("last_top1", 1),
        ("last_small_k", 16),
        ("last_medium_k", 512),
        ("last_near_full", 3072),
        ("last_full", 4096),
    ] {
        cases.push(Case {
            name,
            tensor: last.clone(),
            dim: 1,
            k,
        });
    }
    let nonlast = tensor(&[4, 1024, 16]);
    for (name, k) in [
        ("nonlast_top1", 1),
        ("nonlast_small_k", 16),
        ("nonlast_near_full", 768),
        ("nonlast_full", 1024),
    ] {
        cases.push(Case {
            name,
            tensor: nonlast.clone(),
            dim: 1,
            k,
        });
    }
    let transposed = tensor(&[1024, 32]).transpose(0, 1);
    for (name, k) in [("transposed_small_k", 16), ("transposed_full", 1024)] {
        cases.push(Case {
            name,
            tensor: transposed.clone(),
            dim: 1,
            k,
        });
    }
    cases.push(Case {
        name: "many_short_rows",
        tensor: tensor(&[8192, 64]),
        dim: 1,
        k: 8,
    });
    cases.push(Case {
        name: "parallel_small_k",
        tensor: tensor(&[256, 2048]),
        dim: 1,
        k: 32,
    });
    cases
}

// Independent full-sort oracle verifies axis layout, NaNs, signed zeros and ties.
fn check(case: &Case) {
    let input = case.tensor.to_contiguous();
    let shape = input.layout().shape();
    let axis_len = shape[case.dim];
    let inner: usize = shape[case.dim + 1..].iter().product();
    let outer: usize = shape[..case.dim].iter().product();
    let data = input.storage::<f32>();
    let mut output_shape = shape.clone();
    output_shape[case.dim] = case.k;
    let mut expected = vec![0isize; output_shape.num_elements()];
    for row in 0..outer {
        for column in 0..inner {
            let base = row * axis_len * inner + column;
            let mut order: Vec<usize> = (0..axis_len).collect();
            order.sort_by(|&a, &b| {
                compare_f32(&data[base + b * inner], &data[base + a * inner])
                    .then_with(|| a.cmp(&b))
            });
            for (slot, &index) in order.iter().take(case.k).enumerate() {
                expected[(row * case.k + slot) * inner + column] = index as isize;
            }
        }
    }
    for (name, implementation) in [
        ("before", legacy_argtopk as Implementation),
        ("after", ruprim_host::sort::argtopk as Implementation),
    ] {
        let result = implementation(case.tensor.clone(), case.dim, case.k);
        assert_eq!(result.dtype(), INDEX_DTYPE, "{} {name}: dtype", case.name);
        assert_eq!(
            result.layout().shape(),
            &output_shape,
            "{} {name}: shape",
            case.name
        );
        assert_eq!(
            bytemuck::cast_slice::<u8, isize>(result.bytes()),
            expected.as_slice(),
            "{} {name}: indices",
            case.name
        );
    }
}

fn check_edge_cases() {
    let special = HostTensor::new(
        Bytes::from_elems(vec![
            f32::NAN, 2.0, 2.0, f32::INFINITY, -0.0, 0.0, f32::NEG_INFINITY, f32::NAN,
            -5.0, 7.0, 7.0, 4.0, f32::NAN, f32::INFINITY, f32::NAN, 4.0,
            -0.0, 0.0, -0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0, 1.0,
        ]),
        Layout::contiguous(Shape::new([3, 8])),
        DType::F32,
    );
    for k in 0..=8 {
        check(&Case {
            name: "special_values",
            tensor: special.clone(),
            dim: 1,
            k,
        });
        check(&Case {
            name: "special_transposed",
            tensor: special.transpose(0, 1),
            dim: 0,
            k,
        });
    }
    check(&Case {
        name: "empty_outer",
        tensor: tensor(&[0, 8]),
        dim: 1,
        k: 8,
    });
    check(&Case {
        name: "empty_axis",
        tensor: tensor(&[2, 0]),
        dim: 1,
        k: 0,
    });
}

type Implementation = fn(HostTensor, usize, usize) -> HostTensor;

fn measure(case: &Case, implementation: Implementation, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        drop(black_box(implementation(
            black_box(case.tensor.clone()),
            black_box(case.dim),
            black_box(case.k),
        )));
    }
    start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn main() {
    let mut samples = 9usize;
    let mut iterations = 3usize;
    let mut filter = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--samples" => {
                samples = args
                    .next()
                    .expect("--samples requires a number")
                    .parse()
                    .expect("invalid samples");
            }
            "--iterations" => {
                iterations = args
                    .next()
                    .expect("--iterations requires a number")
                    .parse()
                    .expect("invalid iterations");
            }
            "--filter" => filter = args.next().expect("--filter requires a substring"),
            _ => panic!("unknown argument {arg}; use --samples N --iterations N --filter NAME"),
        }
    }
    assert!(
        samples > 0 && iterations > 0,
        "samples and iterations must be positive"
    );
    let cases: Vec<Case> = cases()
        .into_iter()
        .filter(|case| case.name.contains(&filter))
        .collect();
    assert!(!cases.is_empty(), "filter matched no benchmark cases");
    eprintln!(
        "CPU f32; OS={}; arch={}; rayon={}; simd={}; samples={samples}; iterations={iterations}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        cfg!(feature = "rayon"),
        cfg!(feature = "simd")
    );
    #[cfg(feature = "rayon")]
    eprintln!("rayon_threads={}", rayon::current_num_threads());
    if cfg!(debug_assertions) {
        eprintln!("WARNING: use --release for meaningful timings");
    }
    check_edge_cases();
    for case in &cases {
        check(case);
    }
    eprintln!(
        "All {} timed cases plus edge cases passed both implementations against the full-sort oracle.",
        cases.len()
    );
    println!("case,shape,dim,k,iterations,samples,before_median_us,after_median_us,speedup");
    for case in &cases {
        // Warm both algorithms and the Rayon pool before collecting samples.
        for warmup in 0..4 {
            if warmup % 2 == 0 {
                measure(case, legacy_argtopk, 1);
                measure(case, ruprim_host::sort::argtopk, 1);
            } else {
                measure(case, ruprim_host::sort::argtopk, 1);
                measure(case, legacy_argtopk, 1);
            }
        }
        let mut before = Vec::with_capacity(samples);
        let mut after = Vec::with_capacity(samples);
        for sample in 0..samples {
            if sample % 2 == 0 {
                before.push(measure(case, legacy_argtopk, iterations));
                after.push(measure(case, ruprim_host::sort::argtopk, iterations));
            } else {
                after.push(measure(case, ruprim_host::sort::argtopk, iterations));
                before.push(measure(case, legacy_argtopk, iterations));
            }
        }
        let before = median(&mut before);
        let after = median(&mut after);
        let shape = case
            .tensor
            .layout()
            .shape()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "{},{shape},{},{},{iterations},{samples},{before:.3},{after:.3},{:.3}",
            case.name, case.dim, case.k, before / after
        );
    }
}
