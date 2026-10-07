//! Independent-document attention with explicit packed sequence geometry.
use alloc::{vec, vec::Vec};
use ruda_model::tensor::{Tensor, TensorData, Int, Bool, DType, backend::Backend};
use crate::Dropout;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Immutable cumulative token boundaries, including empty documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedSequenceLayout {
    boundaries: Vec<usize>,
}

impl PackedSequenceLayout {
    /// Describe exactly the supplied token payload; `[0]` represents no documents.
    pub fn new(boundaries: Vec<usize>, tokens: usize) -> Self {
        assert!(!boundaries.is_empty() && boundaries[0] == 0 && boundaries.last() == Some(&tokens), "boundaries must cover the actual token payload");
        assert!(boundaries.windows(2).all(|pair| pair[0] <= pair[1]), "document boundaries must be nondecreasing");
        Self { boundaries }
    }

    /// Number of logical documents, not the number of tokens.
    pub fn documents(&self) -> usize { self.boundaries.len() - 1 }

    /// Actual packed token count.
    pub fn tokens(&self) -> usize { *self.boundaries.last().unwrap() }

    /// Read the cumulative document boundaries without mutation.
    pub fn boundaries(&self) -> &[usize] { &self.boundaries }

    /// Maximum actual document length; zero for an empty payload.
    pub fn max_length(&self) -> usize {
        self.boundaries.windows(2).map(|pair| pair[1] - pair[0]).max().unwrap_or(0)
    }

    /// Reset token positions to zero at each document's start.
    /// The positions are metadata; no activations are downloaded from the device.
    pub fn positions<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
        let values: Vec<i64> = self.boundaries.windows(2).flat_map(|pair| 0..pair[1] - pair[0])
            .map(|position| i64::try_from(position).expect("document position exceeds integer range")).collect();
        Tensor::from_data(TensorData::new(values, [self.tokens()]), (device, DType::I64))
    }

    /// True only at the first actual token of each nonempty document.
    pub fn document_starts<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Bool> {
        let mut starts = vec![false; self.tokens()];
        for pair in self.boundaries.windows(2) { if pair[0] < pair[1] { starts[pair[0]] = true; } }
        Tensor::from_data(TensorData::new(starts, [self.tokens()]), device)
    }
}

/// Alignment of query positions against a document's key positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedCausalAlignment {
    /// Query zero attends at key position zero.
    UpperLeft,
    /// Query's final position aligns with the document's final key position.
    LowerRight,
}

/// Explicit attention policy, independent of a model family or checkpoint name.
#[derive(Clone, Copy, Debug)]
pub struct PackedAttentionOptions {
    /// Optional QK multiplier; None selects `1 / sqrt(key_features)`.
    pub scale: Option<f64>,
    /// Exclude future key positions after applying the explicit alignment.
    pub causal: bool,
    /// Position alignment for unequal query/key lengths, including KV-cache use.
    pub alignment: PackedCausalAlignment,
    /// Optional inclusive left/right distances; -1 means unbounded on that side.
    pub window: Option<(isize, isize)>,
}

impl Default for PackedAttentionOptions {
    fn default() -> Self {
        Self { scale: None, causal: false, alignment: PackedCausalAlignment::UpperLeft, window: None }
    }
}

fn connected_zero<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 1> {
    let mask = Tensor::<B, 3, Bool>::zeros(tensor.dims(), &tensor.device()).bool_not();
    tensor.cast(DType::F32).mask_fill(mask, 0).sum()
}

/// Packed MHA/GQA with independent documents and ordinary backend autodiff.
///
/// Q/K/V are `[total_tokens, heads, features]`; query heads are an integer multiple
/// of KV heads. Work and softmax statistics use FP32, and output retains Q dtype.
/// Each document uses backend tensor operations, with no global packed N-by-N
/// mask, CPU activation fallback or quantized-KV conversion. This is a per-document
/// dense implementation, not a fused or bounded-scratch FlashAttention kernel.
/// Optional dropout reuses the caller's native Dropout module/backend RNG.
pub fn packed_scaled_dot_product_attention<B: Backend>(
    query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>,
    query_layout: &PackedSequenceLayout, key_layout: &PackedSequenceLayout,
    options: PackedAttentionOptions, dropout: Option<&Dropout>,
) -> Tensor<B, 3> {
    let [queries, heads, features] = query.dims();
    let [keys, kv_heads, key_features] = key.dims();
    let [values, value_heads, value_features] = value.dims();
    assert!(heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && features > 0 && value_features > 0, "invalid attention head/feature geometry");
    assert_eq!((keys, kv_heads, features), (values, value_heads, key_features), "key/value or QK features differ");
    assert_eq!((query_layout.tokens(), key_layout.tokens()), (queries, keys), "packed metadata does not match payload lengths");
    assert_eq!(query_layout.documents(), key_layout.documents(), "query/key document counts differ");
    assert!(query.device() == key.device() && query.device() == value.device(), "attention operands must share a device");
    let dtype = query.dtype();
    assert!(matches!(dtype, DType::F32 | DType::F16 | DType::BF16) && key.dtype() == dtype && value.dtype() == dtype, "attention requires matching FP32/FP16/BF16 storage");
    let scale = options.scale.unwrap_or(1.0 / (features as f64).sqrt());
    assert!(scale.is_finite() && (scale as f32).is_finite(), "scale must be finite in FP32");
    if let Some((left, right)) = options.window { assert!(left >= -1 && right >= -1, "window distances must be nonnegative or -1"); }
    if let Some(dropout) = dropout { assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob), "invalid dropout probability"); }
    let device = query.device();
    if queries == 0 || dropout.is_some_and(|dropout|dropout.prob == 1.0 && B::ad_enabled(&device)) {
        let zero = connected_zero(query) + connected_zero(key) + connected_zero(value);
        return (Tensor::<B, 3>::zeros([queries, heads, value_features], (&device, DType::F32)) + zero.reshape([1, 1, 1])).cast(dtype);
    }
    let query = query.cast(DType::F32);
    let key = key.cast(DType::F32);
    let value = value.cast(DType::F32);
    let mut outputs = Vec::new();
    for (qrange, krange) in query_layout.boundaries.windows(2).zip(key_layout.boundaries.windows(2)) {
        let qlen = qrange[1] - qrange[0];
        let klen = krange[1] - krange[0];
        let q = query.clone().slice_dim(0, qrange[0]..qrange[1]).swap_dims(0, 1);
        let k = key.clone().slice_dim(0, krange[0]..krange[1]).swap_dims(0, 1);
        let v = value.clone().slice_dim(0, krange[0]..krange[1]).swap_dims(0, 1);
        if qlen == 0 || klen == 0 {
            let zero = connected_zero(q) + connected_zero(k) + connected_zero(v);
            outputs.push(Tensor::<B, 3>::zeros([qlen, heads, value_features], (&device, DType::F32)) + zero.reshape([1, 1, 1]));
            continue;
        }
        let groups = heads / kv_heads;
        let k = k.reshape([kv_heads, 1, klen, features]).repeat_dim(1, groups).reshape([heads, klen, features]);
        let v = v.reshape([kv_heads, 1, klen, value_features]).repeat_dim(1, groups).reshape([heads, klen, value_features]);
        let mut scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale);
        let qlen_i64 = i64::try_from(qlen).expect("query length exceeds integer range");
        let klen_i64 = i64::try_from(klen).expect("key length exceeds integer range");
        let offset = match options.alignment { PackedCausalAlignment::UpperLeft => 0, PackedCausalAlignment::LowerRight => klen_i64 - qlen_i64 };
        let rows = Tensor::<B, 1, Int>::arange(0..qlen_i64, (&device, DType::I64)).add_scalar(offset).reshape([qlen, 1]).repeat_dim(1, klen);
        let columns = Tensor::<B, 1, Int>::arange(0..klen_i64, (&device, DType::I64)).reshape([1, klen]).repeat_dim(0, qlen);
        let mut excluded = Tensor::<B, 2, Bool>::zeros([qlen, klen], &device);
        if options.causal { excluded = excluded.bool_or(columns.clone().greater(rows.clone())); }
        if let Some((left, right)) = options.window {
            if left >= 0 && (left as usize) < qlen.max(klen) { excluded = excluded.bool_or(columns.clone().lower(rows.clone().sub_scalar(left as i64))); }
            if right >= 0 && (right as usize) < qlen.max(klen) { excluded = excluded.bool_or(columns.greater(rows.add_scalar(right as i64))); }
        }
        let mask = excluded.unsqueeze_dim::<3>(0).repeat_dim(0, heads);
        let fully_masked = mask.clone().all_dim(2);
        scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        let maximum = scores.clone().max_dim(2).mask_fill(fully_masked.clone(), 0);
        let exponentials = (scores - maximum).exp();
        // Every unmasked finite row has a maximum exponential of exactly one.
        // A fully masked row instead has all zero exponentials and returns zero.
        let denominator = exponentials.clone().sum_dim(2).clamp_min(1);
        let mut weights = exponentials / denominator;
        if let Some(dropout) = dropout {
            weights = dropout.forward(weights);
        }
        outputs.push(weights.matmul(v).mask_fill(fully_masked.expand([heads,qlen,value_features]),0).swap_dims(0, 1));
    }
    Tensor::cat(outputs, 0).cast(dtype)
}
