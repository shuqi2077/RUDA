use super::*;

#[allow(clippy::too_many_arguments)]
/// Process a single (batch, head) pair with flash attention.
///
/// Uses gemm for the two block matmuls per tile:
///   1. Score matmul: scores\[seq_q, tile_kv\] = Q @ K_tile^T
///   2. Value matmul: output += P @ V_tile (where P = exp(scores - max))
///
/// The online softmax (scale, mask, bias, exp, correction) is applied
/// row-by-row between the two gemm calls.
pub(super) fn flash_attention_head<T: FlashGemm>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    mask: Option<&[u8]>,
    bias: Option<&[T]>,
    p: &AttentionParams<T>,
    scratch: &mut ScratchBuffers<T>,
) {
    debug_assert_eq!(q.len(), p.seq_q * p.head_dim);
    debug_assert_eq!(k.len(), p.seq_kv * p.head_dim);
    debug_assert_eq!(v.len(), p.seq_kv * p.val_dim);
    debug_assert_eq!(output.len(), p.seq_q * p.val_dim);

    let neg_inf = T::neg_infinity();
    let AttentionParams {
        scale,
        softcap,
        causal_offset,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
    } = *p;

    // Reset scratch buffers for this (batch, head) pair
    let row_max = &mut scratch.row_max;
    row_max.fill(neg_inf);
    let row_sum = &mut scratch.row_sum;
    row_sum.fill(T::zero());
    let scores = &mut scratch.scores;

    let num_kv_tiles = seq_kv.div_ceil(TILE_KV);

    for tile_idx in 0..num_kv_tiles {
        let kv_start = tile_idx * TILE_KV;
        let kv_end = (kv_start + TILE_KV).min(seq_kv);
        let tile_kv = kv_end - kv_start;

        // Step 1: Score matmul via gemm
        // scores[seq_q, tile_kv] = Q[seq_q, head_dim] @ K_tile[tile_kv, head_dim]^T
        //
        // Q is row-major [seq_q, head_dim]: rs=head_dim, cs=1
        // K_tile^T: treat K[tile_kv, head_dim] row-major with swapped strides
        //   K row-major: rs=head_dim, cs=1 -> K^T: rs=1, cs=head_dim
        // scores row-major [seq_q, tile_kv]: rs=tile_kv, cs=1
        unsafe {
            T::block_gemm(BlockGemmArgs {
                m: seq_q,
                n: tile_kv,
                k: head_dim,
                dst: scores.as_mut_ptr(),
                dst_cs: 1,
                dst_rs: tile_kv as isize,
                read_dst: false,
                lhs: q.as_ptr(),
                lhs_cs: 1,
                lhs_rs: head_dim as isize,
                rhs: k.as_ptr().add(kv_start * head_dim),
                rhs_cs: head_dim as isize,
                rhs_rs: 1,
                alpha: T::zero(),
                beta: T::one(),
            });
        }

        // Step 2: Apply scale/softcap/mask/bias, online softmax, rescale output
        for qi in 0..seq_q {
            let score_row = &mut scores[qi * tile_kv..(qi + 1) * tile_kv];

            let mut tile_max = neg_inf;

            for (ki, score) in score_row.iter_mut().enumerate() {
                let kv_idx = kv_start + ki;
                let mut val = *score * scale;

                if let Some(cap) = softcap {
                    val = cap * (val / cap).tanh();
                }

                if let Some(m) = mask
                    && m[qi * seq_kv + kv_idx] != 0
                {
                    val = neg_inf;
                }

                if let Some(offset) = causal_offset
                    && (kv_idx as isize) > (qi as isize) + offset
                {
                    val = neg_inf;
                }

                if let Some(b) = bias {
                    val += b[qi * seq_kv + kv_idx];
                }

                *score = val;
                if val > tile_max {
                    tile_max = val;
                }
            }

            if tile_max == neg_inf {
                // All masked: zero out scores so gemm contributes nothing
                for score in score_row.iter_mut() {
                    *score = T::zero();
                }
                continue;
            }

            let new_max = if row_max[qi] > tile_max {
                row_max[qi]
            } else {
                tile_max
            };

            let mut tile_sum = T::zero();
            for score in score_row.iter_mut() {
                let e = (*score - new_max).exp();
                *score = e;
                tile_sum += e;
            }

            let correction = if row_max[qi] == neg_inf {
                T::zero()
            } else {
                (row_max[qi] - new_max).exp()
            };

            // Rescale existing output accumulator
            let out_row = &mut output[qi * val_dim..(qi + 1) * val_dim];
            for o in out_row.iter_mut() {
                *o = *o * correction;
            }

            row_sum[qi] = row_sum[qi] * correction + tile_sum;
            row_max[qi] = new_max;
        }

        // Step 3: Value matmul via gemm
        // output[seq_q, val_dim] += P[seq_q, tile_kv] @ V_tile[tile_kv, val_dim]
        //
        // P (scores buffer) row-major: rs=tile_kv, cs=1
        // V_tile row-major [tile_kv, val_dim]: rs=val_dim, cs=1
        // output row-major [seq_q, val_dim]: rs=val_dim, cs=1
        //
        // dst = 1.0 * dst + 1.0 * P @ V  (accumulate)
        unsafe {
            T::block_gemm(BlockGemmArgs {
                m: seq_q,
                n: val_dim,
                k: tile_kv,
                dst: output.as_mut_ptr(),
                dst_cs: 1,
                dst_rs: val_dim as isize,
                read_dst: true,
                lhs: scores.as_ptr(),
                lhs_cs: 1,
                lhs_rs: tile_kv as isize,
                rhs: v.as_ptr().add(kv_start * val_dim),
                rhs_cs: 1,
                rhs_rs: val_dim as isize,
                alpha: T::one(),
                beta: T::one(),
            });
        }
    }

    // Final normalization: output /= row_sum
    for qi in 0..seq_q {
        let sum = row_sum[qi];
        if sum > T::zero() {
            let inv_sum = T::one() / sum;
            let out_row = &mut output[qi * val_dim..(qi + 1) * val_dim];
            for o in out_row.iter_mut() {
                *o = *o * inv_sum;
            }
        }
        // sum == 0 means all positions masked; output stays zero
    }
}

