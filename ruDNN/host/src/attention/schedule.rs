use super::*;

/// Flash attention: tiled computation that avoids materializing the full scores matrix.
///
/// Input shapes (all 4D):
///   query: \[batch, heads, seq_q, head_dim\]
///   key:   \[batch, heads, seq_kv, head_dim\]
///   value: \[batch, heads, seq_kv, val_dim\]
///   mask:  \[batch, heads, seq_q, seq_kv\] (optional, u8 where nonzero = masked out)
///   attn_bias: \[batch, heads, seq_q, seq_kv\] (optional)
///
/// Output: \[batch, heads, seq_q, val_dim\]
///
/// Algorithm per (batch, head):
///   For each KV tile of size TILE_KV:
///     1. Score matmul: scores\[seq_q, tile_kv\] = Q @ K_tile^T  (gemm)
///     2. Per query row: apply scale, softcap, mask, bias
///     3. Per query row: online softmax update (running max/sum, rescale accumulator)
///     4. Value matmul: output += P @ V_tile  (gemm)
///   Final: output\[qi\] /= row_sum\[qi\] for each query row
pub(super) fn attention_impl<T>(
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
    assert!(q_shape.num_dims() == 4, "attention: query must be 4D");
    assert!(k_shape.num_dims() == 4, "attention: key must be 4D");
    assert!(v_shape.num_dims() == 4, "attention: value must be 4D");

    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_q = q_shape[2];
    let head_dim = q_shape[3];
    assert!(head_dim > 0, "attention: head_dim must be non-zero");

    let seq_kv = k_shape[2];
    let val_dim = v_shape[3];

    assert_eq!(k_shape[0], batch, "attention: key batch mismatch");
    assert_eq!(k_shape[1], heads, "attention: key heads mismatch");
    assert_eq!(k_shape[3], head_dim, "attention: key head_dim mismatch");
    assert_eq!(v_shape[0], batch, "attention: value batch mismatch");
    assert_eq!(v_shape[1], heads, "attention: value heads mismatch");
    assert_eq!(v_shape[2], seq_kv, "attention: value seq_kv mismatch");

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

    // 4D strides for contiguous layout
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

    // Allocate scratch buffers once and reuse across all (batch, head) pairs
    let mut scratch = ScratchBuffers {
        row_max: vec![T::neg_infinity(); seq_q],
        row_sum: vec![T::zero(); seq_q],
        scores: vec![T::zero(); seq_q * TILE_KV],
    };

    for b in 0..batch {
        for h in 0..heads {
            let q_off = b * q_batch_stride + h * q_head_stride;
            let k_off = b * k_batch_stride + h * k_head_stride;
            let v_off = b * v_batch_stride + h * v_head_stride;
            let o_off = b * o_batch_stride + h * o_head_stride;
            let mask_off = b * mask_batch_step + h * mask_head_step;
            let bias_off = b * bias_batch_step + h * bias_head_step;

            flash_attention_head(
                &q_data[q_off..q_off + q_head_stride],
                &k_data[k_off..k_off + k_head_stride],
                &v_data[v_off..v_off + v_head_stride],
                &mut output[o_off..o_off + o_head_stride],
                mask_data.map(|m| &m[mask_off..mask_off + mask_tile_len]),
                bias_data.map(|b| &b[bias_off..bias_off + mask_tile_len]),
                &params,
                &mut scratch,
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

