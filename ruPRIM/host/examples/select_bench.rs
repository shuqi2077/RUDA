//! Compare public CPU f32 select with the ac817b1 indexing algorithm.
//!
//! Run: RAYON_NUM_THREADS=4 cargo run --release -p ruPRIM-host --example select_bench
//! Optional: --samples 9 --iterations 3 --filter last_axis
//! Includes tensor clones, contiguous conversion, allocation and output drop.
//! The old 2D row-copy baseline uses initialized storage instead of the old
//! unsound uninitialized Vec; its indexing and Rayon scheduling are unchanged.
//! All other old paths are unchanged. No GPU or model timing is performed.

use std::{borrow::Cow, hint::black_box, time::Instant};
use bytemuck::Pod;
use ruda_core::{
    bytes::Bytes,
    tensor::{DType, Shape, element::Element, host::{HostTensor, Layout}},
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Benchmarks use I64 indices only, preserving the original conversion behavior.
fn read_indices(tensor: &HostTensor) -> Cow<'_, [isize]> {
    assert_eq!(tensor.dtype(), DType::I64);
    #[cfg(target_pointer_width = "64")]
    { Cow::Borrowed(bytemuck::cast_slice(tensor.storage::<i64>())) }
    #[cfg(target_pointer_width = "32")]
    { Cow::Owned(tensor.storage::<i64>().iter().map(|&v| isize::try_from(v).unwrap()).collect()) }
}

#[inline(always)]
fn checked_index(raw: isize, dim_size: usize) -> usize {
    if raw < 0 || raw as usize >= dim_size {
        index_oob(raw, dim_size);
    }
    raw as usize
}

#[cold]
#[inline(never)]
fn index_oob(raw: isize, dim_size: usize) -> ! {
    panic!("index {raw} out of bounds for dimension of size {dim_size}");
}

#[inline]
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let ndims = dims.len();
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}


/// Select slices from tensor along a dimension using 1D indices.
///
/// Unlike gather, indices is 1D and selects entire slices.
/// For a 2D tensor with dim=0 and indices=[2, 0]:
/// output[0, :] = tensor[2, :]
/// output[1, :] = tensor[0, :]
fn legacy_select<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();

    let tensor_shape = tensor.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices.layout().num_dims(),
        1,
        "select: indices must be 1D, got {} dims",
        indices.layout().num_dims()
    );

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);
    let num_indices = indices_data.len();

    // Build output shape: replace dim with num_indices
    let mut output_dims = tensor_shape.to_vec();
    output_dims[dim] = num_indices;
    let output_shape = Shape::from(output_dims);

    // Use optimized 2D implementation with bulk copies
    if ndims == 2 {
        let result = select_2d::<E>(
            tensor_data,
            &indices_data,
            tensor_shape[0],
            tensor_shape[1],
            num_indices,
            dim,
        );
        let bytes = Bytes::from_elems(result);
        return HostTensor::new(bytes, Layout::contiguous(output_shape), E::dtype());
    }

    // General N-D case
    let tensor_strides: Vec<usize> = compute_strides(tensor_shape);
    let output_strides: Vec<usize> = compute_strides(&output_shape);
    let output_size = output_shape.num_elements();

    // Calculate slice size (elements after dim)
    let slice_size: usize = tensor_strides[dim];

    let select_dim_size = tensor_shape[dim];

    // If dim is the last dimension or we can use bulk copies
    if dim == ndims - 1 || slice_size == 1 {
        // Element-wise with parallelism
        #[cfg(feature = "rayon")]
        let result: Vec<E> = (0..output_size)
            .into_par_iter()
            .map(|out_idx| {
                let mut remaining = out_idx;
                let mut src_idx = 0;
                for d in 0..ndims {
                    let coord = remaining / output_strides[d];
                    remaining %= output_strides[d];
                    if d == dim {
                        let index_val = checked_index(indices_data[coord], select_dim_size);
                        src_idx += index_val * tensor_strides[d];
                    } else {
                        src_idx += coord * tensor_strides[d];
                    }
                }
                tensor_data[src_idx]
            })
            .collect();

        #[cfg(not(feature = "rayon"))]
        #[allow(clippy::needless_range_loop)]
        let result: Vec<E> = {
            let mut result = vec![E::default(); output_size];
            for out_idx in 0..output_size {
                let mut remaining = out_idx;
                let mut src_idx = 0;
                for d in 0..ndims {
                    let coord = remaining / output_strides[d];
                    remaining %= output_strides[d];
                    if d == dim {
                        let index_val = checked_index(indices_data[coord], select_dim_size);
                        src_idx += index_val * tensor_strides[d];
                    } else {
                        src_idx += coord * tensor_strides[d];
                    }
                }
                result[out_idx] = tensor_data[src_idx];
            }
            result
        };

        let bytes = Bytes::from_elems(result);
        return HostTensor::new(bytes, Layout::contiguous(output_shape), E::dtype());
    }

    // Use bulk copies for contiguous slices
    let mut result = vec![E::default(); output_size];

    // For each position in dimensions before `dim`
    let outer_count = if dim == 0 {
        1
    } else {
        tensor_shape[..dim].iter().product()
    };

    for outer in 0..outer_count {
        let outer_offset_tensor = outer * tensor_strides[if dim == 0 { 0 } else { dim - 1 }];
        let outer_offset_output = outer * output_strides[if dim == 0 { 0 } else { dim - 1 }];

        for (i, &idx) in indices_data.iter().enumerate() {
            let index_val = checked_index(idx, select_dim_size);
            let src_start = outer_offset_tensor + index_val * tensor_strides[dim];
            let dst_start = outer_offset_output + i * output_strides[dim];
            result[dst_start..dst_start + slice_size]
                .copy_from_slice(&tensor_data[src_start..src_start + slice_size]);
        }
    }

    let bytes = Bytes::from_elems(result);
    HostTensor::new(bytes, Layout::contiguous(output_shape), E::dtype())
}

/// Optimized 2D select with bulk row copies when dim=0.
#[inline]
fn select_2d<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor_data: &[E],
    indices_data: &[isize],
    tensor_rows: usize,
    tensor_cols: usize,
    num_indices: usize,
    dim: usize,
) -> Vec<E> {
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };
    let (output_rows, output_cols) = if dim == 0 {
        (num_indices, tensor_cols)
    } else {
        (tensor_rows, num_indices)
    };
    let output_size = output_rows * output_cols;

    // Minimum bytes of output before we consider rayon. Below this, a
    // single-threaded loop is faster because there is not enough work to
    // amortize the work-stealing dispatch overhead.
    #[cfg(feature = "rayon")]
    const PARALLEL_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;

    // Minimum elements per rayon task. Without batching, par_chunks_mut
    // creates one task per row (e.g. 512 single-row tasks of 4 KB each)
    // whose dispatch overhead dominates the actual copy.
    #[cfg(feature = "rayon")]
    const MIN_ELEMS_PER_TASK: usize = 64 * 1024;

    if dim == 0 {
        // Safe equivalent of the baseline's unsound uninitialized Vec.
        let mut result = vec![E::default(); output_size];

        #[cfg(feature = "rayon")]
        if output_size * size_of::<E>() >= PARALLEL_THRESHOLD_BYTES {
            // Batch multiple rows per rayon task so each task copies at
            // least MIN_ELEMS_PER_TASK elements.
            let rows_per_chunk = (MIN_ELEMS_PER_TASK / tensor_cols).max(1);
            let elems_per_chunk = rows_per_chunk * tensor_cols;
            result.par_chunks_mut(elems_per_chunk).enumerate().for_each(
                |(chunk_idx, dst_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    let chunk_rows = dst_chunk.len() / tensor_cols;
                    for i in 0..chunk_rows {
                        let src_row_idx = checked_index(indices_data[start_row + i], dim_size);
                        let src_start = src_row_idx * tensor_cols;
                        let dst_start = i * tensor_cols;
                        dst_chunk[dst_start..dst_start + tensor_cols]
                            .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
                    }
                },
            );
        } else {
            for (i, &idx) in indices_data.iter().enumerate() {
                let src_row_idx = checked_index(idx, dim_size);
                let src_start = src_row_idx * tensor_cols;
                let dst_start = i * tensor_cols;
                result[dst_start..dst_start + tensor_cols]
                    .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, &idx) in indices_data.iter().enumerate() {
                let src_row_idx = checked_index(idx, dim_size);
                let src_start = src_row_idx * tensor_cols;
                let dst_start = i * tensor_cols;
                result[dst_start..dst_start + tensor_cols]
                    .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
            }
        }

        result
    } else {
        // dim == 1: gather individual elements per row (not contiguous).
        // Zero-init is fine here since the inner loop is per-element anyway.
        let mut result = vec![E::default(); output_size];

        #[cfg(feature = "rayon")]
        if output_size * size_of::<E>() >= PARALLEL_THRESHOLD_BYTES {
            let rows_per_chunk = (MIN_ELEMS_PER_TASK / output_cols).max(1);
            let elems_per_chunk = rows_per_chunk * output_cols;
            result.par_chunks_mut(elems_per_chunk).enumerate().for_each(
                |(chunk_idx, dst_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    let chunk_rows = dst_chunk.len() / output_cols;
                    for r in 0..chunk_rows {
                        let row = start_row + r;
                        let dst_base = r * output_cols;
                        for (j, &idx) in indices_data.iter().enumerate() {
                            let src_col = checked_index(idx, dim_size);
                            dst_chunk[dst_base + j] = tensor_data[row * tensor_cols + src_col];
                        }
                    }
                },
            );
        } else {
            for row in 0..output_rows {
                for (j, &idx) in indices_data.iter().enumerate() {
                    let src_col = checked_index(idx, dim_size);
                    result[row * output_cols + j] = tensor_data[row * tensor_cols + src_col];
                }
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for row in 0..output_rows {
                for (j, &idx) in indices_data.iter().enumerate() {
                    let src_col = checked_index(idx, dim_size);
                    result[row * output_cols + j] = tensor_data[row * tensor_cols + src_col];
                }
            }
        }

        result
    }
}


struct Case {
    name: &'static str,
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
}

fn tensor(shape: &[usize]) -> HostTensor {
    let shape = Shape::from(shape.to_vec());
    let values: Vec<f32> = (0..shape.num_elements()).map(|i| (i % 8192) as f32 - 4096.0).collect();
    HostTensor::new(Bytes::from_elems(values), Layout::contiguous(shape), DType::F32)
}

fn selected(axis_size: usize, count: usize) -> HostTensor {
    let values: Vec<i64> = (0..count)
        .map(|i| ((axis_size - 1).wrapping_sub(i * 17) % axis_size) as i64)
        .collect();
    HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(vec![count])),
        DType::I64,
    )
}

fn case(name: &'static str, tensor: HostTensor, dim: usize, count: usize) -> Case {
    let indices = selected(tensor.layout().shape()[dim], count);
    Case { name, tensor, dim, indices }
}

fn cases() -> Vec<Case> {
    vec![
        case("vector", tensor(&[8192]), 0, 2048),
        case("rows_2d", tensor(&[2048, 1024]), 0, 1024),
        case("columns_2d", tensor(&[4096, 512]), 1, 256),
        case("rows_3d", tensor(&[128, 128, 128]), 0, 64),
        case("middle_axis_3d", tensor(&[128, 128, 128]), 1, 64),
        case("last_axis_3d", tensor(&[128, 128, 128]), 2, 64),
        case("last_axis_4d_small", tensor(&[8, 16, 64, 128]), 3, 64),
        case("last_axis_4d_large", tensor(&[16, 32, 64, 128]), 3, 96),
        // Expose the tradeoff: the new general kernel parallelizes outer rows,
        // so this case has only one task despite its large number of indices.
        case("last_axis_single_outer", tensor(&[1, 1, 2_097_152]), 2, 1_500_001),
        case("last_axis_transposed", tensor(&[128, 32, 256]).transpose(0, 2), 2, 64),
        case("middle_axis_offset", tensor(&[32, 128, 256]).narrow(1, 7, 96), 1, 64),
    ]
}

type Implementation = fn(HostTensor, usize, HostTensor) -> HostTensor;

fn check(case: &Case) {
    let tensor = case.tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let mut output_shape = shape.to_vec();
    let indices = case.indices.to_contiguous();
    let indices = indices.storage::<i64>();
    output_shape[case.dim] = indices.len();
    let data = tensor.storage::<f32>();
    let size: usize = output_shape.iter().product();
    // Coordinate-based oracle is independent from the optimized slice copies.
    let expected: Vec<u32> = (0..size).map(|mut index| {
        let mut source = 0;
        let mut stride = 1;
        for axis in (0..shape.num_dims()).rev() {
            let coord = index % output_shape[axis];
            index /= output_shape[axis];
            let source_coord = if axis == case.dim { indices[coord] as usize } else { coord };
            source += source_coord * stride;
            stride *= shape[axis];
        }
        data[source].to_bits()
    }).collect();
    for (name, implementation) in [
        ("before_safe", legacy_select::<f32> as Implementation),
        ("after", ruprim_host::gather_scatter::select::<f32> as Implementation),
    ] {
        let output = implementation(case.tensor.clone(), case.dim, case.indices.clone());
        assert_eq!(output.dtype(), DType::F32, "{} {name}: dtype", case.name);
        assert_eq!(output.layout().shape().to_vec(), output_shape, "{} {name}: shape", case.name);
        assert_eq!(
            output.storage::<f32>().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected,
            "{} {name}: values",
            case.name
        );
    }
}

fn measure(case: &Case, implementation: Implementation, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        drop(black_box(implementation(
            black_box(case.tensor.clone()),
            black_box(case.dim),
            black_box(case.indices.clone()),
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
            "--samples" => samples = args.next().expect("--samples requires N").parse().expect("invalid samples"),
            "--iterations" => iterations = args.next().expect("--iterations requires N").parse().expect("invalid iterations"),
            "--filter" => filter = args.next().expect("--filter requires text"),
            _ => panic!("unknown argument {arg}; use --samples N --iterations N --filter NAME"),
        }
    }
    assert!(samples > 0 && iterations > 0, "samples and iterations must be positive");
    let cases: Vec<Case> = cases().into_iter().filter(|case| case.name.contains(&filter)).collect();
    assert!(!cases.is_empty(), "filter matched no cases");
    eprintln!(
        "CPU f32, I64 indices; OS={}; arch={}; rayon={}; simd={}; samples={samples}; iterations={iterations}",
        std::env::consts::OS, std::env::consts::ARCH, cfg!(feature = "rayon"), cfg!(feature = "simd")
    );
    #[cfg(feature = "rayon")]
    eprintln!("rayon_threads={}", rayon::current_num_threads());
    if cfg!(debug_assertions) {
        eprintln!("WARNING: use --release for meaningful timings");
    }
    for case in &cases {
        check(case);
    }
    eprintln!("All {} cases passed both implementations against the coordinate oracle.", cases.len());
    println!("case,shape,dim,indices,iterations,samples,before_safe_median_us,after_median_us,speedup");
    let before_fn = legacy_select::<f32> as Implementation;
    let after_fn = ruprim_host::gather_scatter::select::<f32> as Implementation;
    for case in &cases {
        for warmup in 0..4 {
            let order = if warmup % 2 == 0 { [before_fn, after_fn] } else { [after_fn, before_fn] };
            for implementation in order {
                measure(case, implementation, 1);
            }
        }
        let mut before = Vec::with_capacity(samples);
        let mut after = Vec::with_capacity(samples);
        for sample in 0..samples {
            if sample % 2 == 0 {
                before.push(measure(case, before_fn, iterations));
                after.push(measure(case, after_fn, iterations));
            } else {
                after.push(measure(case, after_fn, iterations));
                before.push(measure(case, before_fn, iterations));
            }
        }
        let before = median(&mut before);
        let after = median(&mut after);
        let shape = case.tensor.layout().shape().iter().map(usize::to_string).collect::<Vec<_>>().join("x");
        println!(
            "{},{shape},{},{},{iterations},{samples},{before:.3},{after:.3},{:.3}",
            case.name, case.dim, case.indices.layout().num_elements(), before / after
        );
    }
}
