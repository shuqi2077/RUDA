//! Model-independent attention tensor transformations for device backends.
//!
//! These are composition-level candidates, not CUDA/ROCm kernel implementations.
//! The caller supplies device-resident tensor operations; this module never reads
//! tensor elements, constructs CPU masks, or moves tensor data between devices.
//! Shape checks panic before launching the corresponding malformed operation.
//! The backend retains responsibility for dtype, device, and matmul validation.
//!
//! Masking, softmax, positional embeddings, cache ownership and model-specific
//! scaling deliberately remain outside this module. In particular, do NOT scale
//! MLA logits by the inverse square root of the compressed-cache rank.

/// Minimal operations on four-dimensional device tensors.
///
/// Function pointers avoid assuming an unverified RUDA import path or runtime
/// type. They dispatch existing tensor operations, not element-wise host loops.
/// Backend errors/panics retain that backend's behavior. `reshape` may allocate
/// for noncontiguous inputs; this interface makes no zero-copy guarantee.
pub struct TensorOps<T> {
    /// Return shape metadata, without downloading tensor contents.
    pub shape: fn(&T) -> [usize; 4],
    /// Reshape in logical row-major axis order.
    pub reshape: fn(T, [usize; 4]) -> T,
    /// Transpose the requested logical axes.
    pub swap_dims: fn(T, usize, usize) -> T,
    /// Batched matrix multiply with standard leading-axis broadcasting.
    pub matmul: fn(T, T) -> T,
    /// Element-wise addition with standard broadcasting.
    pub add: fn(T, T) -> T,
}

fn positive(shape: [usize; 4]) {
    assert!(shape.iter().all(|&n| n > 0), "attention dimensions must be positive");
    shape.into_iter().try_fold(1usize, usize::checked_mul)
        .expect("attention element count overflow");
}

fn fold_rows(sequence: usize, groups: usize) -> usize {
    sequence.checked_mul(groups).expect("GQA row count overflow")
}

fn qk_layout(query: [usize; 4], key: [usize; 4]) -> ([usize; 4], [usize; 4]) {
    positive(query);
    positive(key);
    let [batch, heads, sequence, dim] = query;
    let [key_batch, kv_heads, length, key_dim] = key;
    assert_eq!(batch, key_batch, "query/key batches must match");
    assert_eq!(dim, key_dim, "query/key feature dimensions must match");
    assert_eq!(heads % kv_heads, 0, "query heads must divide by KV heads");
    let rows = fold_rows(sequence, heads / kv_heads);
    let scores = [batch, heads, sequence, length];
    positive(scores);
    ([batch, kv_heads, rows, dim], scores)
}

fn pv_layout(probabilities: [usize; 4], value: [usize; 4]) -> ([usize; 4], [usize; 4]) {
    positive(probabilities);
    positive(value);
    let [batch, heads, sequence, length] = probabilities;
    let [value_batch, kv_heads, value_length, value_dim] = value;
    assert_eq!(batch, value_batch, "probability/value batches must match");
    assert_eq!(length, value_length, "probability/value lengths must match");
    assert_eq!(heads % kv_heads, 0, "query heads must divide by KV heads");
    let rows = fold_rows(sequence, heads / kv_heads);
    let output = [batch, heads, sequence, value_dim];
    positive(output);
    ([batch, kv_heads, rows, length], output)
}

/// Layout metadata retained for compatibility with the v4 algebra tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GqaLayout {
    /// Whether more than one query head shares each KV head.
    pub grouped: bool,
    /// Folded query shape [batch, KV heads, grouped rows, key dimension].
    pub query: [usize; 4],
    /// Unfolded score shape [batch, query heads, query length, key length].
    pub scores: [usize; 4],
    /// Folded probability shape consumed by the value product.
    pub probabilities: [usize; 4],
    /// Unfolded output shape.
    pub output: [usize; 4],
}

impl GqaLayout {
    /// Validate all three inputs and derive the two no-repeat matrix products.
    pub fn new(query: [usize; 4], key: [usize; 4], value: [usize; 4]) -> Self {
        let (folded_query, scores) = qk_layout(query, key);
        assert_eq!(key[1], value[1], "key/value heads must match");
        let (probabilities, output) = pv_layout(scores, value);
        Self { grouped: query[1] != key[1], query: folded_query, scores, probabilities, output }
    }
}

/// Compute raw GQA/MQA/MHA logits without repeating cached key heads.
///
/// Inputs are [B,Hq,S,D] and [B,Hkv,T,D], with head order
/// `(KV head, query head within its group)`. Returns [B,Hq,S,T].
/// This is NOT FlashAttention: the full score tensor is still materialized.
pub fn gqa_scores<T>(ops: &TensorOps<T>, query: T, key: T) -> T {
    let original = (ops.shape)(&query);
    let (folded, scores) = qk_layout(original, (ops.shape)(&key));
    let query = if folded == original { query } else { (ops.reshape)(query, folded) };
    let result = (ops.matmul)(query, (ops.swap_dims)(key, 2, 3));
    if (ops.shape)(&result) == scores { result } else { (ops.reshape)(result, scores) }
}

/// Apply attention probabilities without repeating value heads.
///
/// The supplied product callback preserves an adapter's existing accumulation,
/// cast, and error policy. Only the tensor layouts are changed.
pub fn gqa_value_product_with<T, E>(
    ops: &TensorOps<T>,
    probabilities: T,
    value: T,
    product: impl FnOnce(T, T) -> Result<T, E>,
) -> Result<T, E> {
    let original = (ops.shape)(&probabilities);
    let (folded, output) = pv_layout(original, (ops.shape)(&value));
    let probabilities = if folded == original { probabilities }
        else { (ops.reshape)(probabilities, folded) };
    let result = product(probabilities, value)?;
    Ok(if (ops.shape)(&result) == output { result } else { (ops.reshape)(result, output) })
}

/// Value product using the caller's default device matrix multiplication.
pub fn gqa_value_product<T>(ops: &TensorOps<T>, probabilities: T, value: T) -> T {
    let result: Result<T, core::convert::Infallible> =
        gqa_value_product_with(ops, probabilities, value, |p, v| Ok((ops.matmul)(p, v)));
    match result { Ok(tensor) => tensor, Err(never) => match never {} }
}

/// Absorb the non-positional MLA key expansion into the current queries.
///
/// `query` is [B,H,S,Dn]; the already prepared key projection is [1,H,Dn,R].
/// Returns [B,H,S,R]. Projection weights must already have the correct layout,
/// dtype, and dequantization; this is not a checkpoint loader.
pub fn mla_absorb_query<T>(ops: &TensorOps<T>, query: T, key_projection: T) -> T {
    let q = (ops.shape)(&query);
    let w = (ops.shape)(&key_projection);
    positive(q);
    positive(w);
    assert_eq!(w[0], 1, "MLA projection batch axis must be one");
    assert_eq!(w[1], q[1], "MLA projection heads must match queries");
    assert_eq!(w[2], q[3], "MLA key projection input dimension mismatch");
    positive([q[0], q[1], q[2], w[3]]);
    (ops.matmul)(query, key_projection)
}

/// Compute raw dense MLA scores directly from shared compressed history.
///
/// Query tensors: absorbed content [B,H,S,R], rotated position [B,H,S,P].
/// Cache tensors: normalized latent [B,1,T,R], rotated position [B,1,T,P].
/// RoPE must already be applied with the correct model convention. Returns
/// [B,H,S,T], BEFORE model-specific scaling, masks, bias, softcap and softmax.
/// Sparse MLA, padding, sliding windows and chunk positions are NOT inferred.
/// This composition can allocate two score-sized temporaries; use a dedicated
/// fused kernel, not this path unconditionally, for long-context prefill.
pub fn mla_scores<T>(
    ops: &TensorOps<T>, absorbed_query: T, position_query: T,
    latent_cache: T, position_cache: T,
) -> T {
    let q = (ops.shape)(&absorbed_query);
    let qp = (ops.shape)(&position_query);
    let c = (ops.shape)(&latent_cache);
    let kp = (ops.shape)(&position_cache);
    positive(q); positive(qp); positive(c); positive(kp);
    assert_eq!(&q[..3], &qp[..3], "MLA query batch/head/sequence must match");
    assert_eq!(c[1], 1, "MLA latent cache must remain shared across heads");
    assert_eq!(kp[1], 1, "MLA position cache must remain shared across heads");
    assert_eq!(c[2], kp[2], "MLA cache lengths must match");
    // Validate both products before either one is launched.
    qk_layout(q, c);
    qk_layout(qp, kp);
    let content = gqa_scores(ops, absorbed_query, latent_cache);
    let position = gqa_scores(ops, position_query, position_cache);
    (ops.add)(content, position)
}

/// Attend to compressed history before expanding the current output rows.
///
/// Probabilities [B,H,S,T], latent cache [B,1,T,R], value projection
/// [1,H,R,Dv] -> output [B,H,S,Dv]. No [B,H,T,Dv] historical values are formed.
/// Accumulation policy comes from `ops`; an adapter may supply F32 matmul/casts.
/// Floating-point operation order differs from full expansion and needs model
/// accuracy validation before enabling this path by default.
pub fn mla_value_product<T>(
    ops: &TensorOps<T>, probabilities: T, latent_cache: T, value_projection: T,
) -> T {
    let p = (ops.shape)(&probabilities);
    let c = (ops.shape)(&latent_cache);
    let w = (ops.shape)(&value_projection);
    positive(p); positive(c); positive(w);
    assert_eq!(c[1], 1, "MLA latent cache must remain shared across heads");
    pv_layout(p, c);
    assert_eq!(w[0], 1, "MLA projection batch axis must be one");
    assert_eq!(w[1], p[1], "MLA value projection heads must match probabilities");
    assert_eq!(w[2], c[3], "MLA value projection rank mismatch");
    positive([p[0], p[1], p[2], w[3]]);
    let context = gqa_value_product(ops, probabilities, latent_cache);
    (ops.matmul)(context, value_projection)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug)]
    struct Shape([usize; 4]);
    fn ops() -> TensorOps<Shape> {
        TensorOps {
            shape: |t| t.0,
            reshape: |t, d| {
                assert_eq!(t.0.iter().product::<usize>(), d.iter().product::<usize>()); Shape(d)
            },
            swap_dims: |mut t, a, b| { t.0.swap(a, b); t },
            matmul: |a, b| {
                assert_eq!(a.0[3], b.0[2]);
                assert!(a.0[0] == b.0[0] || a.0[0] == 1 || b.0[0] == 1);
                assert_eq!(a.0[1], b.0[1], "no repeated head broadcast allowed");
                Shape([a.0[0].max(b.0[0]), a.0[1], a.0[2], b.0[3]])
            },
            add: |a, b| { assert_eq!(a.0, b.0); a },
        }
    }
    #[test]
    fn grouped_attention_shapes() {
        for heads in [1, 2, 8] {
            for groups in [1, 2, 4] {
                let o = ops();
                let scores = gqa_scores(&o, Shape([2, heads*groups, 3, 16]), Shape([2, heads, 7, 16]));
                assert_eq!(scores.0, [2, heads*groups, 3, 7]);
                let result = gqa_value_product(&o, scores, Shape([2, heads, 7, 24]));
                assert_eq!(result.0, [2, heads*groups, 3, 24]);
            }
        }
    }
    #[test]
    fn mla_shapes_and_shared_cache() {
        let o = ops();
        let q = mla_absorb_query(&o, Shape([2, 8, 3, 16]), Shape([1, 8, 16, 32]));
        let scores = mla_scores(&o, q, Shape([2, 8, 3, 8]), Shape([2, 1, 65, 32]), Shape([2, 1, 65, 8]));
        let output = mla_value_product(&o, scores, Shape([2, 1, 65, 32]), Shape([1, 8, 32, 24]));
        assert_eq!(output.0, [2, 8, 3, 24]);
    }
    #[test]
    fn callback_error_is_not_replaced() {
        let result = gqa_value_product_with(&ops(), Shape([1, 8, 1, 5]), Shape([1, 2, 5, 7]), |_, _| Err("backend-error"));
        assert_eq!(result.unwrap_err(), "backend-error");
    }
    #[test]
    #[should_panic(expected="divide by KV")]
    fn invalid_head_ratio() { gqa_scores(&ops(), Shape([1, 7, 1, 4]), Shape([1, 2, 5, 4])); }
    #[test]
    #[should_panic(expected="positive")]
    fn empty_cache_rejected() { gqa_scores(&ops(), Shape([1, 8, 1, 4]), Shape([1, 2, 0, 4])); }
    #[test]
    #[should_panic(expected="feature dimensions")]
    fn mismatched_key_feature() { gqa_scores(&ops(), Shape([1, 8, 1, 4]), Shape([1, 2, 5, 8])); }
    #[test]
    #[should_panic(expected="batches must match")]
    fn mismatched_batch() { gqa_scores(&ops(), Shape([2, 8, 1, 4]), Shape([1, 2, 5, 4])); }
    #[test]
    #[should_panic(expected="lengths must match")]
    fn mismatched_value_length() { gqa_value_product(&ops(), Shape([1, 8, 1, 4]), Shape([1, 2, 5, 8])); }
    #[test]
    #[should_panic(expected="element count overflow")]
    fn overflow_rejected() { gqa_scores(&ops(), Shape([1, 8, usize::MAX, 4]), Shape([1, 2, 5, 4])); }
    #[test]
    #[should_panic(expected="must remain shared")]
    fn expanded_mla_cache_rejected() {
        mla_scores(&ops(), Shape([1, 8, 1, 16]), Shape([1, 8, 1, 8]), Shape([1, 8, 5, 16]), Shape([1, 1, 5, 8]));
    }
    #[test]
    #[should_panic(expected="rank mismatch")]
    fn invalid_mla_projection() {
        mla_value_product(&ops(), Shape([1, 8, 1, 5]), Shape([1, 1, 5, 16]), Shape([1, 8, 32, 8]));
    }
}
