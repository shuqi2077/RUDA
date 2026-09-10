use super::*;

// ============================================================================
// Integer matmul (naive, with optional SIMD for i32)
// ============================================================================

/// Integer matrix multiplication dispatch.
pub fn int_matmul(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    assert_eq!(lhs.dtype(), rhs.dtype(), "int_matmul: dtype mismatch");

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    assert!(lhs_rank >= 2, "int_matmul requires at least 2D tensors");
    assert!(rhs_rank >= 2, "int_matmul requires at least 2D tensors");

    let k_lhs = lhs_shape[lhs_rank - 1];
    let k_rhs = rhs_shape[rhs_rank - 2];
    assert_eq!(k_lhs, k_rhs, "int_matmul: inner dimensions must match");

    match lhs.dtype() {
        DType::I32 => matmul_i32(lhs, rhs),
        DType::I64 => matmul_i64(lhs, rhs),
        _ => panic!("int_matmul: unsupported dtype {:?}", lhs.dtype()),
    }
}

/// i32 matmul using naive triple loop with SIMD dot product.
fn matmul_i32(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_i32(&lhs, &rhs)
    } else {
        matmul_batched_i32(lhs, rhs)
    }
}

/// 2D i32 matmul: [M, K] x [K, N] -> [M, N]
/// Transposes rhs to enable contiguous access for dot product.
fn matmul_2d_i32(lhs: &HostTensor, rhs: &HostTensor) -> HostTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    let lhs_data: &[i32] = lhs.storage();
    let rhs_data: &[i32] = rhs.storage();

    // Transpose rhs [K, N] -> [N, K] for contiguous column access
    let mut rhs_t = vec![0i32; k * n];
    for i in 0..k {
        for j in 0..n {
            rhs_t[j * k + i] = rhs_data[i * n + j];
        }
    }

    let mut output = vec![0i32; m * n];

    // Now both lhs rows and rhs columns (transposed rows) are contiguous
    for i in 0..m {
        let lhs_row = &lhs_data[i * k..(i + 1) * k];
        for j in 0..n {
            let rhs_col = &rhs_t[j * k..(j + 1) * k];
            output[i * n + j] = dot_i32(lhs_row, rhs_col);
        }
    }

    let out_shape = Shape::from(vec![m, n]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::I32,
    )
}

/// Dot product for i32 slices. Uses macerator SIMD when the `simd` feature is enabled.
#[inline]
fn dot_i32(a: &[i32], b: &[i32]) -> i32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        dot_i32_simd(a, b)
    }

    #[cfg(not(feature = "simd"))]
    {
        dot_i32_scalar(a, b)
    }
}

#[cfg(not(feature = "simd"))]
#[inline]
fn dot_i32_scalar(a: &[i32], b: &[i32]) -> i32 {
    let mut sum = 0i32;
    for i in 0..a.len() {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
    }
    sum
}

#[cfg(feature = "simd")]
#[macerator::with_simd]
fn dot_i32_simd<S: macerator::Simd>(a: &[i32], b: &[i32]) -> i32 {
    use macerator::{Scalar, VMulAdd, vload_unaligned};

    let lanes = i32::lanes::<S>();
    let len = a.len();
    let simd_len = len / lanes * lanes;
    let mut acc = 0i32.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        let va = unsafe { vload_unaligned(a.as_ptr().add(i)) };
        let vb = unsafe { vload_unaligned(b.as_ptr().add(i)) };
        acc = i32::vmul_add(va, vb, acc);
        i += lanes;
    }

    let mut sum = acc.reduce_add();
    while i < len {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
        i += 1;
    }
    sum
}

/// Batched i32 matmul: [B..., M, K] x [B..., K, N] -> [B..., M, N]
///
/// Uses naive triple-loop with SIMD dot product and batch-level parallelism.
fn matmul_batched_i32(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape[lhs_rank - 2];
    let k = lhs_shape[lhs_rank - 1];
    let n = rhs_shape[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape[..rhs_rank - 2].to_vec();

    let (broadcast_shape, lhs_strides, rhs_strides) = broadcast_batch_dims(&lhs_batch, &rhs_batch);

    let batch_size: usize = broadcast_shape.iter().product();
    let rhs_batch_size: usize = rhs_batch.iter().product();
    let lhs_matrix_size = checked_size(m, k);
    let rhs_matrix_size = checked_size(k, n);
    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let lhs_data: &[i32] = lhs.storage();
    let rhs_data: &[i32] = rhs.storage();

    // Transpose rhs per actual rhs batch: [B_rhs, K, N] -> [B_rhs, N, K]
    let mut rhs_transposed = vec![0i32; rhs_batch_size * n * k];
    for b in 0..rhs_batch_size {
        let src_offset = b * rhs_matrix_size;
        let dst_offset = b * n * k;
        for i in 0..k {
            for j in 0..n {
                rhs_transposed[dst_offset + j * k + i] = rhs_data[src_offset + i * n + j];
            }
        }
    }

    let mut output = vec![0i32; batch_size * out_matrix_size];

    let run_one = |b: usize, out_slice: &mut [i32]| {
        let lhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &lhs_strides);
        let rhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &rhs_strides);
        let lhs_offset = lhs_batch_idx * lhs_matrix_size;
        let rhs_t_offset = rhs_batch_idx * n * k;

        let lhs_slice = &lhs_data[lhs_offset..lhs_offset + lhs_matrix_size];
        let rhs_t_slice = &rhs_transposed[rhs_t_offset..rhs_t_offset + n * k];

        for i in 0..m {
            let lhs_row = &lhs_slice[i * k..(i + 1) * k];
            for j in 0..n {
                let rhs_col = &rhs_t_slice[j * k..(j + 1) * k];
                out_slice[i * n + j] = dot_i32(lhs_row, rhs_col);
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(out_matrix_size)
            .enumerate()
            .for_each(|(b, out_slice)| run_one(b, out_slice));
    }

    #[cfg(not(feature = "rayon"))]
    {
        for b in 0..batch_size {
            let offset = b * out_matrix_size;
            run_one(b, &mut output[offset..offset + out_matrix_size]);
        }
    }

    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::I32,
    )
}

/// i64 matmul using naive triple loop.
fn matmul_i64(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    if lhs_rank == 2 && rhs_rank == 2 {
        matmul_2d_i64(&lhs, &rhs)
    } else {
        matmul_batched_i64(lhs, rhs)
    }
}

/// 2D i64 matmul: [M, K] x [K, N] -> [M, N]
fn matmul_2d_i64(lhs: &HostTensor, rhs: &HostTensor) -> HostTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    let lhs_data: &[i64] = lhs.storage();
    let rhs_data: &[i64] = rhs.storage();

    let mut output = vec![0i64; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for l in 0..k {
                sum = sum.wrapping_add(lhs_data[i * k + l].wrapping_mul(rhs_data[l * n + j]));
            }
            output[i * n + j] = sum;
        }
    }

    let out_shape = Shape::from(vec![m, n]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::I64,
    )
}

/// Batched i64 matmul with broadcast support
fn matmul_batched_i64(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();
    let lhs_rank = lhs_shape.num_dims();
    let rhs_rank = rhs_shape.num_dims();

    let m = lhs_shape[lhs_rank - 2];
    let k = lhs_shape[lhs_rank - 1];
    let n = rhs_shape[rhs_rank - 1];

    let lhs_batch: Vec<usize> = lhs_shape[..lhs_rank - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape[..rhs_rank - 2].to_vec();

    // Compute broadcast batch dimensions
    let (broadcast_shape, lhs_strides, rhs_strides) = broadcast_batch_dims(&lhs_batch, &rhs_batch);

    let batch_size: usize = broadcast_shape.iter().product();
    let lhs_matrix_size = checked_size(m, k);
    let rhs_matrix_size = checked_size(k, n);
    let out_matrix_size = checked_size(m, n);

    let mut out_dims = broadcast_shape.clone();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);

    let lhs_data: &[i64] = lhs.storage();
    let rhs_data: &[i64] = rhs.storage();

    let mut output = vec![0i64; batch_size * out_matrix_size];

    for b in 0..batch_size {
        let lhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &lhs_strides);
        let rhs_batch_idx = batch_index_to_offset(b, &broadcast_shape, &rhs_strides);
        let lhs_offset = lhs_batch_idx * lhs_matrix_size;
        let rhs_offset = rhs_batch_idx * rhs_matrix_size;
        let out_offset = b * out_matrix_size;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0i64;
                for l in 0..k {
                    let lhs_idx = lhs_offset + i * k + l;
                    let rhs_idx = rhs_offset + l * n + j;
                    sum = sum.wrapping_add(lhs_data[lhs_idx].wrapping_mul(rhs_data[rhs_idx]));
                }
                output[out_offset + i * n + j] = sum;
            }
        }
    }

    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        DType::I64,
    )
}

