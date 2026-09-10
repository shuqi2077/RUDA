use super::*;

/// Naive attention: mathematically equivalent to flash but without KV tiling.
///
/// Materializes the full [seq_q, seq_kv] score matrix, applies scale/softcap/mask/causal/bias,
/// row-wise softmax, then a single output matmul. Two large gemm calls per (batch, head).
///
/// Faster than flash for short sequences. Also useful as a benchmarking baseline.
pub fn attention_naive(
    query: HostTensor,
    key: HostTensor,
    value: HostTensor,
    mask: Option<HostTensor>,
    attn_bias: Option<HostTensor>,
    options: AttentionModuleOptions,
) -> HostTensor {
    dispatch_attention_dtype!(
        query,
        key,
        value,
        mask,
        attn_bias,
        options,
        attention_naive_impl
    )
}

fn attention_naive_impl<T>(
    query: HostTensor,
    key: HostTensor,
    value: HostTensor,
    mask: Option<HostTensor>,
    attn_bias: Option<HostTensor>,
    options: AttentionModuleOptions,
) -> HostTensor
where
    T: FlashGemm + ruda_core::tensor::element::Element,
{
    if let Some(softcap) = options.softcap {
        assert!(softcap > 0.0, "softcap must be positive, got {softcap}");
    }

    let query = query.to_contiguous();
    let key = key.to_contiguous();
    let value = value.to_contiguous();

    let q_shape = query.layout().shape();
    let k_shape = key.layout().shape();
    let v_shape = value.layout().shape();
    assert!(q_shape.num_dims() == 4, "attention_naive: query must be 4D");
    assert!(k_shape.num_dims() == 4, "attention_naive: key must be 4D");
    assert!(v_shape.num_dims() == 4, "attention_naive: value must be 4D");

    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_q = q_shape[2];
    let head_dim = q_shape[3];
    assert!(head_dim > 0, "attention_naive: head_dim must be non-zero");

    let seq_kv = k_shape[2];
    let val_dim = v_shape[3];

    assert_eq!(k_shape[0], batch, "attention_naive: key batch mismatch");
    assert_eq!(k_shape[1], heads, "attention_naive: key heads mismatch");
    assert_eq!(
        k_shape[3], head_dim,
        "attention_naive: key head_dim mismatch"
    );
    assert_eq!(v_shape[0], batch, "attention_naive: value batch mismatch");
    assert_eq!(v_shape[1], heads, "attention_naive: value heads mismatch");
    assert_eq!(v_shape[2], seq_kv, "attention_naive: value seq_kv mismatch");

    let target = [batch, heads, seq_q, seq_kv];
    let mask_bcast = mask.map(|m| broadcast_attn_mask_bias(m, target, "mask"));
    let bias_bcast = attn_bias.map(|b| broadcast_attn_mask_bias(b, target, "bias"));

    let scale = T::from(
        options
            .scale
            .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt()),
    )
    .unwrap();
    let softcap: Option<T> = options.softcap.map(|s| T::from(s).unwrap());
    let causal_offset = if options.is_causal {
        Some(seq_kv as isize - seq_q as isize)
    } else {
        None
    };

    let q_data: &[T] = query.storage();
    let k_data: &[T] = key.storage();
    let v_data: &[T] = value.storage();
    let mask_data: Option<&[u8]> = mask_bcast.as_ref().map(|b| b.tensor.bytes());
    let bias_data: Option<&[T]> = bias_bcast.as_ref().map(|b| b.tensor.storage());
    let (mask_batch_step, mask_head_step) = mask_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step))
        .unwrap_or((0, 0));
    let (bias_batch_step, bias_head_step) = bias_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step))
        .unwrap_or((0, 0));

    let mut output = vec![T::zero(); batch * heads * seq_q * val_dim];
    let mut scores = vec![T::zero(); seq_q * seq_kv];

    let q_head_stride = seq_q * head_dim;
    let q_batch_stride = heads * q_head_stride;
    let k_head_stride = seq_kv * head_dim;
    let k_batch_stride = heads * k_head_stride;
    let v_head_stride = seq_kv * val_dim;
    let v_batch_stride = heads * v_head_stride;
    let o_head_stride = seq_q * val_dim;
    let o_batch_stride = heads * o_head_stride;
    let mask_tile_len = seq_q * seq_kv;

    let params = AttentionParams {
        scale,
        softcap,
        causal_offset,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
    };

    for b in 0..batch {
        for h in 0..heads {
            let q_off = b * q_batch_stride + h * q_head_stride;
            let k_off = b * k_batch_stride + h * k_head_stride;
            let v_off = b * v_batch_stride + h * v_head_stride;
            let o_off = b * o_batch_stride + h * o_head_stride;
            let mask_off = b * mask_batch_step + h * mask_head_step;
            let bias_off = b * bias_batch_step + h * bias_head_step;

            naive_attention_head(
                &q_data[q_off..q_off + q_head_stride],
                &k_data[k_off..k_off + k_head_stride],
                &v_data[v_off..v_off + v_head_stride],
                &mut output[o_off..o_off + o_head_stride],
                &mut scores,
                &params,
                (
                    mask_data.map(|m| &m[mask_off..mask_off + mask_tile_len]),
                    bias_data.map(|b| &b[bias_off..bias_off + mask_tile_len]),
                ),
            );
        }
    }

    let shape = ruda_core::tensor::Shape::from(vec![batch, heads, seq_q, val_dim]);
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        T::dtype(),
    )
}

/// Process a single (batch, head) pair with naive (non-tiled) attention.
///
/// 1. scores[seq_q, seq_kv] = Q @ K^T              (one gemm)
/// 2. Apply scale/softcap/mask/causal/bias + softmax (per-row)
/// 3. output[seq_q, val_dim] = scores @ V            (one gemm)
fn naive_attention_head<T: FlashGemm>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    scores: &mut [T],
    p: &AttentionParams<T>,
    mask_bias: (Option<&[u8]>, Option<&[T]>),
) {
    let (mask, bias) = mask_bias;
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

    // scores = Q @ K^T
    unsafe {
        T::block_gemm(BlockGemmArgs {
            m: seq_q,
            n: seq_kv,
            k: head_dim,
            dst: scores.as_mut_ptr(),
            dst_cs: 1,
            dst_rs: seq_kv as isize,
            read_dst: false,
            lhs: q.as_ptr(),
            lhs_cs: 1,
            lhs_rs: head_dim as isize,
            rhs: k.as_ptr(),
            rhs_cs: head_dim as isize,
            rhs_rs: 1,
            alpha: T::zero(),
            beta: T::one(),
        });
    }

    for qi in 0..seq_q {
        let row = &mut scores[qi * seq_kv..(qi + 1) * seq_kv];

        let mut row_max = neg_inf;
        for (ki, s) in row.iter_mut().enumerate() {
            let mut val = *s * scale;

            if let Some(cap) = softcap {
                val = cap * (val / cap).tanh();
            }

            if let Some(m) = mask
                && m[qi * seq_kv + ki] != 0
            {
                val = neg_inf;
            }

            if let Some(offset) = causal_offset
                && (ki as isize) > (qi as isize) + offset
            {
                val = neg_inf;
            }

            if let Some(b) = bias {
                val += b[qi * seq_kv + ki];
            }

            *s = val;
            if val > row_max {
                row_max = val;
            }
        }

        if row_max == neg_inf {
            row.fill(T::zero());
            continue;
        }

        let mut sum = T::zero();
        for s in row.iter_mut() {
            let e = (*s - row_max).exp();
            *s = e;
            sum += e;
        }

        let inv_sum = T::one() / sum;
        for s in row.iter_mut() {
            *s = *s * inv_sum;
        }
    }

    // output = softmax(scores) @ V
    unsafe {
        T::block_gemm(BlockGemmArgs {
            m: seq_q,
            n: val_dim,
            k: seq_kv,
            dst: output.as_mut_ptr(),
            dst_cs: 1,
            dst_rs: val_dim as isize,
            read_dst: false,
            lhs: scores.as_ptr(),
            lhs_cs: 1,
            lhs_rs: seq_kv as isize,
            rhs: v.as_ptr(),
            rhs_cs: 1,
            rhs_rs: val_dim as isize,
            alpha: T::zero(),
            beta: T::one(),
        });
    }
}

