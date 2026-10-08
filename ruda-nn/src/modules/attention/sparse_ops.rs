use alloc::vec;
use alloc::vec::Vec;
use ruda_model::{module::Module, tensor::{Bool, DType, FloatDType, Int, Tensor, TensorData, backend::Backend}};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

pub(super) fn work_dtype(dtype: DType) -> DType {
    assert!(matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64),
        "sparse attention requires floating-point storage");
    if dtype == DType::F64 { DType::F64 } else { DType::F32 }
}

pub(super) fn positions<B: Backend>(length: usize, start: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    let end = start.checked_add(length).expect("position range overflow");
    Tensor::arange(i64::try_from(start).expect("position exceeds I64")..
        i64::try_from(end).expect("position exceeds I64"), (device, DType::I64))
}

pub(super) fn valid_mask<B: Backend>(input: &Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 2, Bool> {
    let [batch, length, _] = input.dims();
    match valid {
        Some(valid) => {
            assert_eq!(valid.dims(), [batch, length], "valid mask must describe the actual input slots");
            assert_eq!(valid.device(), input.device(), "valid mask device differs");
            valid
        }
        None => Tensor::<B, 2, Bool>::zeros([batch, length], &input.device()).bool_not(),
    }
}

pub(super) fn rms<B: Backend, const D: usize>(input: Tensor<B, D>, eps: f64, weight: Option<Tensor<B, 1>>) -> Tensor<B, D> {
    let storage = input.dtype();
    let compute = work_dtype(storage);
    let input = input.cast(compute);
    let denominator = (input.clone().powf_scalar(2).mean_dim(D - 1) + eps).sqrt();
    let mut output = input / denominator;
    if let Some(weight) = weight {
        output = output * weight.cast(compute).unsqueeze::<D>();
    }
    output.cast(storage)
}

pub(super) fn masked_softmax<B: Backend, const D: usize>(scores: Tensor<B, D>, valid: Tensor<B, D, Bool>, dim: usize) -> Tensor<B, D> {
    assert_eq!(scores.device(), valid.device(), "softmax validity device differs");
    let shape = scores.dims();
    let compute = work_dtype(scores.dtype());
    if shape[dim] == 0 { return scores.mul_scalar(0); }
    let valid = valid.expand(shape);
    let inactive = valid.clone().any_dim(dim).bool_not();
    let scores = scores.mask_fill(valid.clone().bool_not(), f64::NEG_INFINITY)
        .mask_fill(inactive.expand(shape), 0);
    ruda_model::tensor::activation::softmax(scores, dim)
        * valid.cast::<FloatDType>(compute.into())
}

/// Gather [batch, queries, selected, features] without a broadcast key table.
/// Negative indices produce exact zero slots. Nonnegative indices must be less
/// than the supplied key count; native backend bounds rules apply.
pub fn sparse_gather_entries<B: Backend>(entries: Tensor<B, 3>, indices: Tensor<B, 3, Int>) -> Tensor<B, 4> {
    let [batch, keys, width] = entries.dims();
    let [index_batch, queries, count] = indices.dims();
    assert_eq!(index_batch, batch, "selected key batch differs");
    assert_eq!(indices.device(), entries.device(), "selected key device differs");
    assert_eq!(indices.dtype(), DType::I64, "selected key indices must be I64");
    let device = entries.device();
    let dtype = entries.dtype();
    if keys == 0 || batch == 0 || queries == 0 || count == 0 {
        return Tensor::zeros([batch, queries, count, width], (&device, dtype))
            + entries.sum().mul_scalar(0).reshape([1, 1, 1, 1]);
    }
    let valid = indices.clone().greater_equal_elem(0);
    let safe = indices.mask_fill(valid.clone().bool_not(), 0);
    let offsets = positions::<B>(batch, 0, &device).mul_scalar(i64::try_from(keys).expect("key stride exceeds I64"))
        .reshape([batch, 1, 1]);
    let ids = (safe + offsets).reshape([batch * queries * count]);
    let selected = entries.reshape([batch * keys, width]).select(0, ids)
        .reshape([batch, queries, count, width]);
    selected.mask_fill(valid.bool_not().reshape([batch, queries, count, 1])
        .expand([batch, queries, count, width]), 0)
}

/// Deterministic same-device top-k, ordered by score then smaller external ID.
/// Invalid slots carry score -infinity and ID -1. Scores must not contain NaN.
/// Selection uses O(k * keys) work and O(keys) live workspace per row, with no
/// full sort, host tensor readback, or autograd through integer decisions.
pub fn sparse_stable_topk<B: Backend>(scores: Tensor<B, 3>, k: usize,
    indices: Option<Tensor<B, 3, Int>>, valid: Option<Tensor<B, 3, Bool>>)
    -> (Tensor<B, 3>, Tensor<B, 3, Int>) {
    work_dtype(scores.dtype());
    let [batch, queries, keys] = scores.dims();
    let device = scores.device();
    let indices = indices.unwrap_or_else(|| positions::<B>(keys, 0, &device)
        .reshape([1, 1, keys]).expand([batch, queries, keys]));
    let valid = valid.unwrap_or_else(|| Tensor::<B, 3, Bool>::zeros([batch, queries, keys], &device).bool_not());
    assert_eq!(indices.dims(), scores.dims(), "top-k external IDs must match scores");
    assert_eq!(indices.dtype(), DType::I64, "top-k IDs must use I64");
    assert_eq!(indices.device(), device, "top-k IDs must share the score device");
    assert_eq!(valid.dims(), scores.dims(), "top-k validity must match scores");
    assert_eq!(valid.device(), device, "top-k validity must share the score device");
    let count = k.min(keys);
    if count == 0 {
        return (scores.slice_dim(2, 0..0), indices.slice_dim(2, 0..0));
    }
    let scores = scores.detach();
    let mut remaining = valid;
    let mut values = Vec::with_capacity(count);
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        let mut value = scores.clone();
        let mut index = indices.clone();
        let mut active = remaining.clone();
        let mut size = keys;
        while size > 1 {
            let pairs = size / 2;
            let paired_values = value.clone().slice_dim(2, 0..2 * pairs).reshape([batch, queries, pairs, 2]);
            let paired_indices = index.clone().slice_dim(2, 0..2 * pairs).reshape([batch, queries, pairs, 2]);
            let paired_valid = active.clone().slice_dim(2, 0..2 * pairs).reshape([batch, queries, pairs, 2]);
            let left = paired_values.clone().slice_dim(3, 0..1).reshape([batch, queries, pairs]);
            let right = paired_values.slice_dim(3, 1..2).reshape([batch, queries, pairs]);
            let il = paired_indices.clone().slice_dim(3, 0..1).reshape([batch, queries, pairs]);
            let ir = paired_indices.slice_dim(3, 1..2).reshape([batch, queries, pairs]);
            let vl = paired_valid.clone().slice_dim(3, 0..1).reshape([batch, queries, pairs]);
            let vr = paired_valid.slice_dim(3, 1..2).reshape([batch, queries, pairs]);
            let choose_left = vl.clone().bool_and(vr.clone().bool_not()
                .bool_or(left.clone().greater(right.clone()))
                .bool_or(left.clone().equal(right.clone()).bool_and(il.clone().lower_equal(ir.clone()))));
            let mut next_value = right.mask_where(choose_left.clone(), left);
            let mut next_index = ir.mask_where(choose_left, il);
            let mut next_valid = vl.bool_or(vr);
            if size % 2 != 0 {
                next_value = Tensor::cat(vec![next_value, value.slice_dim(2, size - 1..size)], 2);
                next_index = Tensor::cat(vec![next_index, index.slice_dim(2, size - 1..size)], 2);
                next_valid = Tensor::cat(vec![next_valid, active.slice_dim(2, size - 1..size)], 2);
            }
            value = next_value;
            index = next_index;
            active = next_valid;
            size = pairs + size % 2;
        }
        values.push(value.mask_fill(active.clone().bool_not(), f64::NEG_INFINITY));
        ids.push(index.clone().mask_fill(active.clone().bool_not(), -1));
        remaining = remaining.bool_and(active.expand([batch, queries, keys])
            .bool_and(indices.clone().equal(index.expand([batch, queries, keys]))).bool_not());
    }
    (Tensor::cat(values, 2), Tensor::cat(ids, 2))
}

/// KL distillation from detached nonnegative teacher mass on real selected slots.
/// Zero-mass/fully invalid rows contribute zero; only active rows enter the mean.
pub fn indexer_kl_loss<B: Backend>(scores: Tensor<B, 3>, teacher: Tensor<B, 3>,
    valid: Option<Tensor<B, 3, Bool>>) -> Tensor<B, 1> {
    let [batch, queries, keys] = scores.dims();
    assert_eq!(teacher.dims(), scores.dims(), "indexer teacher geometry differs");
    assert_eq!(teacher.device(), scores.device(), "indexer teacher device differs");
    work_dtype(teacher.dtype());
    let compute = work_dtype(scores.dtype());
    let valid = valid.unwrap_or_else(|| Tensor::<B, 3, Bool>::zeros(scores.dims(), &scores.device()).bool_not());
    assert_eq!(valid.dims(), scores.dims(), "indexer teacher validity differs");
    assert_eq!(valid.device(), scores.device(), "indexer teacher validity device differs");
    if keys == 0 || batch == 0 || queries == 0 { return scores.sum().mul_scalar(0); }
    let target = teacher.detach().cast(compute) * valid.clone().cast::<FloatDType>(compute.into());
    let mass = target.clone().sum_dim(2);
    let active = mass.clone().greater_elem(0).bool_and(valid.clone().any_dim(2));
    let target = target / mass.mask_fill(active.clone().bool_not(), 1);
    let logits = scores.cast(compute).mask_fill(valid.clone().bool_not(), f64::NEG_INFINITY)
        .mask_fill(active.clone().bool_not().expand([batch, queries, keys]), 0);
    let logp = ruda_model::tensor::activation::log_softmax(logits, 2)
        .mask_fill(valid.bool_and(active.clone().expand([batch, queries, keys])).bool_not(), 0);
    let logt = target.clone().mask_fill(target.clone().greater_elem(0).bool_not(), 1).log();
    (target * (logt - logp)).sum() / active.cast::<FloatDType>(compute.into()).sum().clamp_min(1)
}

/// Interleaved base RoPE applied to the trailing channels, independently of model family.
#[derive(Module, Debug)]
pub struct SparseRotaryEmbedding {
    /// Nonnegative even count of rotated trailing channels.
    pub rope_dim: usize,
    /// Positive finite base; frequencies retain FP32 storage before work-dtype promotion.
    pub base: f64,
}

impl SparseRotaryEmbedding {
    pub fn new(rope_dim: usize, base: f64) -> Self {
        assert!(rope_dim.is_multiple_of(2) && base.is_finite() && base > 0, "invalid rotary geometry/base");
        Self { rope_dim, base }
    }

    /// Rotate [batch, tokens, heads, width] with the supplied absolute positions.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>, pos: Tensor<B, 1, Int>, inverse: bool) -> Tensor<B, 4> {
        let [batch, tokens, heads, width] = input.dims();
        assert!(self.rope_dim <= width && self.rope_dim.is_multiple_of(2)
            && self.base.is_finite() && self.base > 0, "invalid rotary channel geometry");
        assert_eq!(pos.dims(), [tokens], "rotary position count differs");
        assert_eq!(pos.device(), input.device(), "rotary positions must share the input device");
        assert_eq!(pos.dtype(), DType::I64, "rotary positions require I64");
        let compute = work_dtype(input.dtype());
        if self.rope_dim == 0 { return input; }
        let pairs = self.rope_dim / 2;
        let frequencies: Vec<f32> = (0..pairs).map(|i| self.base.powf(-((2 * i) as f64) / self.rope_dim as f64) as f32).collect();
        let frequencies = Tensor::<B, 1>::from_data(TensorData::new(frequencies, [pairs]), &input.device()).cast(compute);
        let mut angles = pos.cast::<FloatDType>(compute.into()).reshape([tokens, 1]) * frequencies.reshape([1, pairs]);
        if inverse { angles = angles.neg(); }
        let cos = angles.clone().cos().reshape([1, tokens, 1, pairs]);
        let sin = angles.sin().reshape([1, tokens, 1, pairs]);
        let tail = input.clone().slice_dim(3, width - self.rope_dim..width).cast(compute)
            .reshape([batch, tokens, heads, pairs, 2]);
        let even = tail.clone().slice_dim(4, 0..1).reshape([batch, tokens, heads, pairs]);
        let odd = tail.slice_dim(4, 1..2).reshape([batch, tokens, heads, pairs]);
        let rotated = Tensor::<B, 4>::stack::<5>(vec![even.clone() * cos.clone() - odd.clone() * sin.clone(),
            even * sin + odd * cos], 4).reshape([batch, tokens, heads, self.rope_dim]).cast(input.dtype());
        if width == self.rope_dim { rotated }
        else { Tensor::cat(vec![input.slice_dim(3, 0..width - self.rope_dim), rotated], 3) }
    }

    /// Rotate shared [batch, tokens, width] key/value vectors.
    pub fn forward_shared<B: Backend>(&self, input: Tensor<B, 3>, pos: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        let [batch, tokens, width] = input.dims();
        self.forward(input.reshape([batch, tokens, 1, width]), pos, false).reshape([batch, tokens, width])
    }
}
