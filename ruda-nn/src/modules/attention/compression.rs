use alloc::vec;
use ruda_model::{config::Config, module::{Initializer, Module, Param},
    tensor::{Bool, Int, Tensor, backend::Backend}};
use crate::{Linear, LinearConfig};
use super::sparse_ops::{masked_softmax, rms, valid_mask, work_dtype};

/// Learned per-channel gated pooling over complete physical token blocks.
#[derive(Config, Debug)]
pub struct LearnedKVCompressorConfig {
    pub width: usize,
    pub head_dim: usize,
    pub ratio: usize,
    /// Two independently learned paths for previous/current blocks when enabled.
    #[config(default = false)]
    pub overlap: bool,
    #[config(default = 1e-6)]
    pub epsilon: f64,
}

/// Native differentiable KV compression with offset bias and RMS normalization.
#[derive(Module, Debug)]
pub struct LearnedKVCompressor<B: Backend> {
    pub value: Linear<B>,
    pub gate: Linear<B>,
    /// [ratio, streams * head_dim], with previous-path channels first.
    pub position_bias: Param<Tensor<B, 2>>,
    pub norm_weight: Param<Tensor<B, 1>>,
    pub width: usize,
    pub head_dim: usize,
    pub ratio: usize,
    pub overlap: bool,
    pub epsilon: f64,
}

/// Actual incomplete block and previous complete input block for incremental inference.
#[derive(Clone, Debug)]
pub struct CompressionState<B: Backend> {
    pub tail: Tensor<B, 3>,
    pub tail_valid: Tensor<B, 2, Bool>,
    pub previous: Tensor<B, 3>,
    pub previous_valid: Tensor<B, 2, Bool>,
}

impl<B: Backend> CompressionState<B> {
    /// Reorder or duplicate actual batch rows for beam expansion.
    pub fn reorder(&self, parents: Tensor<B, 1, Int>) -> Self {
        assert_eq!(parents.device(), self.tail.device(), "compression parents must share the cache device");
        Self {
            tail: self.tail.clone().select(0, parents.clone()),
            tail_valid: self.tail_valid.clone().select(0, parents.clone()),
            previous: self.previous.clone().select(0, parents.clone()),
            previous_valid: self.previous_valid.clone().select(0, parents),
        }
    }

    /// Actual pending physical slots, excluding the retained overlap block.
    pub fn pending_tokens(&self) -> usize { self.tail.dims()[1] }

    pub fn to_device(self, device: &B::Device) -> Self {
        Self { tail: self.tail.to_device(device), tail_valid: self.tail_valid.to_device(device),
            previous: self.previous.to_device(device), previous_valid: self.previous_valid.to_device(device) }
    }
}

impl LearnedKVCompressorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LearnedKVCompressor<B> {
        assert!(self.width > 0 && self.head_dim > 0 && self.ratio > 0
            && self.epsilon.is_finite() && self.epsilon > 0, "invalid compression configuration");
        let output = self.head_dim.checked_mul(if self.overlap { 2 } else { 1 }).expect("compression width overflow");
        LearnedKVCompressor::from_parts(
            LinearConfig::new(self.width, output).with_bias(false).init(device),
            LinearConfig::new(self.width, output).with_bias(false).init(device),
            Initializer::Zeros.init([self.ratio, output], device),
            Initializer::Ones.init([self.head_dim], device), self.overlap, self.epsilon)
    }
}

impl<B: Backend> LearnedKVCompressor<B> {
    /// Attach actual loaded linear/normalization leaves without replacing parameter IDs or mappers.
    pub fn from_parts(value: Linear<B>, gate: Linear<B>, position_bias: Param<Tensor<B, 2>>,
        norm_weight: Param<Tensor<B, 1>>, overlap: bool, epsilon: f64) -> Self {
        let [width, output] = value.weight.val().dims();
        let [ratio, bias_output] = position_bias.val().dims();
        let [head_dim] = norm_weight.val().dims();
        assert!(width > 0 && ratio > 0 && head_dim > 0 && epsilon.is_finite() && epsilon > 0,
            "invalid loaded compression geometry/epsilon");
        assert_eq!(output, head_dim.checked_mul(if overlap { 2 } else { 1 }).expect("compression width overflow"));
        assert_eq!(gate.weight.val().dims(), [width, output], "compression value/gate layouts differ");
        assert_eq!(bias_output, output, "compression offset bias width differs");
        assert!(value.bias.is_none() && gate.bias.is_none(), "compression projections must not add a linear bias");
        let device = value.weight.val().device();
        assert!(gate.weight.val().device() == device && position_bias.val().device() == device
            && norm_weight.val().device() == device, "compression parameters must share a device");
        Self { value, gate, position_bias, norm_weight, width, head_dim, ratio, overlap, epsilon }
    }

    fn validate(&self, input: &Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 2, Bool> {
        work_dtype(input.dtype());
        assert_eq!(input.dims()[2], self.width, "compression feature width differs");
        assert_eq!(input.device(), self.value.weight.val().device(), "compression input/parameter device differs");
        valid_mask(input, valid)
    }

    fn compress(&self, input: Tensor<B, 3>, valid: Tensor<B, 2, Bool>,
        previous: Option<(Tensor<B, 3>, Tensor<B, 2, Bool>)>) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let [batch, length, _] = input.dims();
        let blocks = length / self.ratio;
        let (r, d) = (self.ratio, self.head_dim);
        let device = input.device();
        let storage = input.dtype();
        let compute = work_dtype(storage);
        if blocks == 0 {
            return (Tensor::zeros([batch, 0, d], (&device, storage))
                + input.sum().mul_scalar(0).reshape([1, 1, 1]), valid.slice_dim(1, 0..0));
        }
        let complete = input.slice_dim(1, 0..blocks * r).cast(compute);
        let output = if self.overlap { 2 * d } else { d };
        let values = complete.clone().matmul(self.value.weight.val().cast(compute).unsqueeze::<3>())
            .reshape([batch, blocks, r, output]);
        let gates = complete.matmul(self.gate.weight.val().cast(compute).unsqueeze::<3>())
            .reshape([batch, blocks, r, output]) + self.position_bias.val().cast(compute).reshape([1, 1, r, output]);
        let mask = valid.slice_dim(1, 0..blocks * r).reshape([batch, blocks, r, 1]);
        let (values, gates, mask) = if self.overlap {
            let (pv, pg, pm) = if let Some((previous, previous_valid)) = previous {
                let previous = previous.cast(compute);
                let pv = previous.clone().matmul(self.value.weight.val().cast(compute)
                    .slice_dim(1, 0..d).unsqueeze::<3>()).reshape([batch, 1, r, d]);
                let pg = (previous.matmul(self.gate.weight.val().cast(compute).slice_dim(1, 0..d).unsqueeze::<3>())
                    + self.position_bias.val().cast(compute).slice_dim(1, 0..d).reshape([1, r, d]))
                    .reshape([batch, 1, r, d]);
                (pv, pg, previous_valid.reshape([batch, 1, r, 1]))
            } else {
                (Tensor::zeros([batch, 1, r, d], (&device, compute)),
                    Tensor::zeros([batch, 1, r, d], (&device, compute)),
                    Tensor::<B, 4, Bool>::zeros([batch, 1, r, 1], &device))
            };
            let prev_v = if blocks == 1 { pv } else {
                Tensor::cat(vec![pv, values.clone().slice([0..batch, 0..blocks - 1, 0..r, 0..d])], 1)
            };
            let prev_g = if blocks == 1 { pg } else {
                Tensor::cat(vec![pg, gates.clone().slice([0..batch, 0..blocks - 1, 0..r, 0..d])], 1)
            };
            let prev_m = if blocks == 1 { pm } else {
                Tensor::cat(vec![pm, mask.clone().slice_dim(1, 0..blocks - 1)], 1)
            };
            (Tensor::cat(vec![prev_v, values.slice_dim(3, d..2 * d)], 2),
                Tensor::cat(vec![prev_g, gates.slice_dim(3, d..2 * d)], 2),
                Tensor::cat(vec![prev_m, mask], 2))
        } else { (values, gates, mask) };
        let weights = masked_softmax(gates, mask.clone(), 2);
        let pooled = (values * weights).sum_dim(2).reshape([batch, blocks, d]);
        let result = rms(pooled, self.epsilon, Some(self.norm_weight.val().cast(compute))).cast(storage);
        (result, mask.any_dim(2).reshape([batch, blocks]))
    }

    /// Differentiable full-sequence pooling; an incomplete trailing block is never emitted.
    pub fn forward(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
        let valid = self.validate(&input, valid);
        self.compress(input, valid, None)
    }

    /// Incremental inference over every supplied token, retaining only the incomplete
    /// tail and one previous complete block. Use a backend with autograd disabled;
    /// full-history training derivatives belong to forward(), not detached cache state.
    pub fn append(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>, state: Option<&CompressionState<B>>)
        -> (Tensor<B, 3>, Tensor<B, 2, Bool>, CompressionState<B>) {
        assert!(!B::ad_enabled(&input.device()), "compression cache requires autograd-disabled inference");
        let valid = self.validate(&input, valid);
        let [batch, _, width] = input.dims();
        let device = input.device();
        let storage = input.dtype();
        let (data, masks, previous) = if let Some(state) = state {
            let tail = state.tail.dims()[1];
            assert!(tail < self.ratio, "compression cache contains an already complete tail");
            assert_eq!(state.tail.dims(), [batch, tail, width], "compression tail geometry differs");
            assert_eq!(state.previous.dims(), [batch, self.ratio, width], "compression overlap geometry differs");
            assert_eq!(state.tail_valid.dims(), [batch, tail], "compression tail validity differs");
            assert_eq!(state.previous_valid.dims(), [batch, self.ratio], "compression overlap validity differs");
            assert!(state.tail.device() == device && state.previous.device() == device
                && state.tail_valid.device() == device && state.previous_valid.device() == device,
                "compression cache device differs");
            assert!(state.tail.dtype() == storage && state.previous.dtype() == storage, "compression cache storage differs");
            (Tensor::cat(vec![state.tail.clone(), input], 1), Tensor::cat(vec![state.tail_valid.clone(), valid], 1),
                Some((state.previous.clone(), state.previous_valid.clone())))
        } else { (input, valid, None) };
        let length = data.dims()[1];
        let cutoff = length / self.ratio * self.ratio;
        let (result, result_valid) = self.compress(data.clone(), masks.clone(), previous.clone());
        let (previous, previous_valid) = if cutoff > 0 {
            (data.clone().slice_dim(1, cutoff - self.ratio..cutoff), masks.clone().slice_dim(1, cutoff - self.ratio..cutoff))
        } else if let Some(previous) = previous { previous }
        else { (Tensor::zeros([batch, self.ratio, width], (&device, storage)),
            Tensor::<B, 2, Bool>::zeros([batch, self.ratio], &device)) };
        // Native slice assignment into exact-size storage avoids retaining a prompt-sized view.
        let tail = data.slice_dim(1, cutoff..length);
        let tail_valid = masks.slice_dim(1, cutoff..length);
        let tail = if tail.dims()[1] == 0 { tail.detach() } else {
            Tensor::empty(tail.dims(), (&device, storage)).slice_assign([0..batch, 0..tail.dims()[1], 0..width], tail.detach())
        };
        let tail_valid = if tail_valid.dims()[1] == 0 { tail_valid } else {
            Tensor::<B, 2, Bool>::empty(tail_valid.dims(), (&device, tail_valid.dtype()))
                .slice_assign([0..batch, 0..tail_valid.dims()[1]], tail_valid)
        };
        let previous = Tensor::empty(previous.dims(), (&device, storage))
            .slice_assign([0..batch, 0..self.ratio, 0..width], previous.detach());
        let previous_valid = Tensor::<B, 2, Bool>::empty(previous_valid.dims(), (&device, previous_valid.dtype()))
            .slice_assign([0..batch, 0..self.ratio], previous_valid);
        (result.detach(), result_valid, CompressionState { tail, tail_valid, previous, previous_valid })
    }
}
