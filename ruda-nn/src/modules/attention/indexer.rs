use alloc::vec;
use alloc::vec::Vec;
use ruda_model::{config::Config, module::Module,
    tensor::{Bool, DType, Int, Tensor, backend::Backend}};
use crate::{Linear, LinearConfig, LayerNorm, LayerNormConfig};
use super::{SparseRotaryEmbedding, indexer_kl_loss, sparse_gather_entries, sparse_stable_topk};
use super::sparse_ops::{positions, work_dtype};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Trainable nonnegative-dot-product indexer with learned signed head weights.
#[derive(Config, Debug)]
pub struct LightningIndexerConfig {
    pub width: usize,
    #[config(default = 4)]
    pub num_heads: usize,
    #[config(default = 16)]
    pub head_dim: usize,
    #[config(default = "None")]
    pub query_dim: Option<usize>,
    #[config(default = 32)]
    pub topk: usize,
    #[config(default = 32)]
    pub query_chunk_size: usize,
    #[config(default = 128)]
    pub key_chunk_size: usize,
    /// Compressed keys are supplied by the caller instead of a token-key projection.
    #[config(default = false)]
    pub external_keys: bool,
    /// Train the indexer without sending its auxiliary gradient into main-model features.
    #[config(default = true)]
    pub detach_inputs: bool,
    #[config(default = 0)]
    pub rope_dim: usize,
    #[config(default = 10000.0)]
    pub rope_base: f64,
    #[config(default = 1e-6)]
    pub epsilon: f64,
}

/// Native score projections, optional token-key path, and chunk-bounded selection.
#[derive(Module, Debug)]
pub struct LightningIndexer<B: Backend> {
    pub query: Linear<B>,
    pub head_weight: Linear<B>,
    pub key: Option<Linear<B>>,
    pub key_norm: Option<LayerNorm<B>>,
    pub rotary: SparseRotaryEmbedding,
    pub width: usize,
    pub query_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub query_chunk_size: usize,
    pub key_chunk_size: usize,
    pub detach_inputs: bool,
}

/// Alias for the same native dynamic sparse attention indexer, not another scoring rule.
pub type DSAIndexer<B> = LightningIndexer<B>;

/// Optional actual visibility and absolute position metadata for streaming selection.
#[derive(Clone, Debug)]
pub struct IndexerMask<B: Backend> {
    /// [batch, queries, keys], True permits the selected edge.
    pub allowed: Option<Tensor<B, 3, Bool>>,
    pub key_valid: Option<Tensor<B, 2, Bool>>,
    pub query_valid: Option<Tensor<B, 2, Bool>>,
    /// [keys], absolute last token included in each key/compressed block.
    pub key_end_positions: Option<Tensor<B, 1, Int>>,
    /// [queries], absolute query token positions, defaulting to 0..queries.
    pub query_positions: Option<Tensor<B, 1, Int>>,
}

impl<B: Backend> Default for IndexerMask<B> {
    fn default() -> Self {
        Self { allowed: None, key_valid: None, query_valid: None, key_end_positions: None, query_positions: None }
    }
}

/// Selected integer IDs and recomputed differentiable scores on the same device.
#[derive(Clone, Debug)]
pub struct IndexerOutput<B: Backend> {
    pub indices: Tensor<B, 3, Int>,
    pub scores: Tensor<B, 3>,
    pub valid: Tensor<B, 3, Bool>,
}

impl LightningIndexerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LightningIndexer<B> {
        let query_dim = self.query_dim.unwrap_or(self.width);
        assert!(self.width > 0 && query_dim > 0 && self.num_heads > 0 && self.head_dim > 0
            && self.topk > 0 && self.query_chunk_size > 0 && self.key_chunk_size > 0 && self.epsilon.is_finite() && self.epsilon > 0.0,
            "invalid indexer dimensions/chunk budget/epsilon");
        let query_output = self.num_heads.checked_mul(self.head_dim).expect("indexer query width overflow");
        let query = LinearConfig::new(query_dim, query_output).with_bias(false).init(device);
        let head_weight = LinearConfig::new(self.width, self.num_heads).with_bias(false).init(device);
        let (key, key_norm) = if self.external_keys { (None, None) } else {
            (Some(LinearConfig::new(self.width, self.head_dim).with_bias(false).init(device)),
                Some(LayerNormConfig::new(self.head_dim).with_epsilon(self.epsilon).init(device)))
        };
        LightningIndexer::from_parts(query, head_weight, key, key_norm,
            SparseRotaryEmbedding::new(self.rope_dim, self.rope_base), self.topk,
            self.query_chunk_size, self.key_chunk_size, self.detach_inputs)
    }
}

impl<B: Backend> LightningIndexer<B> {
    /// Connect actual checkpoint projections and affine key normalization as-is.
    pub fn from_parts(query: Linear<B>, head_weight: Linear<B>, key: Option<Linear<B>>,
        key_norm: Option<LayerNorm<B>>, rotary: SparseRotaryEmbedding, topk: usize,
        query_chunk_size: usize, key_chunk_size: usize, detach_inputs: bool) -> Self {
        let [query_dim, query_output] = query.weight.val().dims();
        let [width, num_heads] = head_weight.weight.val().dims();
        assert!(width > 0 && query_dim > 0 && num_heads > 0 && query_output > 0
            && query_output.is_multiple_of(num_heads) && topk > 0 && query_chunk_size > 0 && key_chunk_size > 0,
            "invalid loaded indexer geometry/budget");
        let head_dim = query_output / num_heads;
        assert!(rotary.rope_dim <= head_dim, "indexer rotary width exceeds a head");
        assert!(query.bias.is_none() && head_weight.bias.is_none(), "indexer projections must be bias-free");
        let device = query.weight.val().device();
        assert_eq!(head_weight.weight.val().device(), device, "indexer projection devices differ");
        match (&key, &key_norm) {
            (Some(key), Some(norm)) => {
                assert_eq!(key.weight.val().dims(), [width, head_dim], "indexer token-key geometry differs");
                assert!(key.bias.is_none(), "indexer token-key projection must be bias-free");
                assert_eq!(norm.gamma.val().dims(), [head_dim], "indexer key normalization width differs");
                assert!(norm.epsilon().is_finite() && norm.epsilon() > 0.0, "invalid indexer key normalization epsilon");
                assert!(key.weight.val().device() == device && norm.gamma.val().device() == device
                    && norm.beta.as_ref().is_none_or(|beta| beta.val().device() == device),
                    "indexer key parameters must share a device");
            }
            (None, None) => {}
            _ => panic!("indexer token-key projection and normalization must be supplied together"),
        }
        Self { query, head_weight, key, key_norm, rotary, width, query_dim, num_heads, head_dim,
            topk, query_chunk_size, key_chunk_size, detach_inputs }
    }

    /// Prepare token keys; externally compressed keys bypass this path entirely.
    pub fn project_keys(&self, input: Tensor<B, 3>, pos: Option<Tensor<B, 1, Int>>) -> Tensor<B, 3> {
        assert_eq!(input.dims()[2], self.width, "indexer token feature width differs");
        assert_eq!(input.device(), self.query.weight.val().device(), "indexer token feature device differs");
        work_dtype(input.dtype());
        let key = self.key.as_ref().expect("external-key indexer has no token-key projection");
        let norm = self.key_norm.as_ref().expect("indexer token-key normalization missing");
        let input = if self.detach_inputs { input.detach() } else { input };
        let pos = pos.unwrap_or_else(|| positions::<B>(input.dims()[1], 0, &input.device()));
        self.rotary.forward_shared(norm.forward(key.forward(input)), pos)
    }

    fn queries(&self, input: Tensor<B, 3>, latent: Option<Tensor<B, 3>>, pos: Option<Tensor<B, 1, Int>>,
        selection: bool) -> (Tensor<B, 4>, Tensor<B, 3>) {
        let [batch, tokens, width] = input.dims();
        assert_eq!(width, self.width, "indexer feature width differs");
        assert_eq!(input.device(), self.query.weight.val().device(), "indexer feature/parameter device differs");
        let latent = latent.unwrap_or_else(|| input.clone());
        assert_eq!(latent.dims(), [batch, tokens, self.query_dim], "indexer query latent geometry differs");
        assert_eq!(latent.device(), input.device(), "indexer query latent device differs");
        work_dtype(latent.dtype());
        let compute = work_dtype(input.dtype());
        let pos = pos.unwrap_or_else(|| positions::<B>(tokens, 0, &input.device()));
        let input = if selection || self.detach_inputs { input.detach() } else { input };
        let latent = if selection || self.detach_inputs { latent.detach() } else { latent };
        let mut query_weight = self.query.weight.val();
        let mut head_weight = self.head_weight.weight.val();
        if selection { query_weight = query_weight.detach(); head_weight = head_weight.detach(); }
        let q = latent.cast(compute).matmul(query_weight.cast(compute).unsqueeze::<3>())
            .reshape([batch, tokens, self.num_heads, self.head_dim]);
        let scale = 1.0 / ((self.head_dim as f64) * (self.num_heads as f64)).sqrt();
        let weights = input.cast(compute).matmul(head_weight.cast(compute).unsqueeze::<3>()).mul_scalar(scale);
        (self.rotary.forward(q, pos, false), weights)
    }

    fn score(&self, query: Tensor<B, 4>, weights: Tensor<B, 3>, keys: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, queries, heads, width] = query.dims();
        let [key_batch, count, key_width] = keys.dims();
        assert_eq!((key_batch, key_width), (batch, width), "indexer key geometry differs");
        assert_eq!(keys.device(), query.device(), "indexer key device differs");
        work_dtype(keys.dtype());
        let compute = query.dtype();
        if count == 0 || queries == 0 || batch == 0 {
            return Tensor::zeros([batch, queries, count], (&query.device(), compute))
                + (query.sum() + weights.sum() + keys.cast(compute).sum()).mul_scalar(0).reshape([1, 1, 1]);
        }
        let dots = query.swap_dims(1, 2).matmul(keys.cast(compute).reshape([batch, 1, count, width]).swap_dims(2, 3));
        (ruda_model::tensor::activation::relu(dots) * weights.swap_dims(1, 2).reshape([batch, heads, queries, 1]))
            .sum_dim(1).reshape([batch, queries, count])
    }

    /// Dense differentiable indexer scores, used only when that cost is explicitly requested.
    pub fn scores(&self, input: Tensor<B, 3>, keys: Tensor<B, 3>, latent: Option<Tensor<B, 3>>,
        pos: Option<Tensor<B, 1, Int>>) -> Tensor<B, 3> {
        let (query, weights) = self.queries(input, latent, pos, false);
        self.score(query, weights, keys)
    }

    /// Recompute only selected scores with parameter gradients after integer top-k.
    pub fn selected_scores(&self, input: Tensor<B, 3>, keys: Tensor<B, 3>, indices: Tensor<B, 3, Int>,
        latent: Option<Tensor<B, 3>>, pos: Option<Tensor<B, 1, Int>>) -> Tensor<B, 3> {
        let (query, weights) = self.queries(input, latent, pos, false);
        let [batch, tokens, heads, width] = query.dims();
        assert_eq!(indices.dims()[..2], [batch, tokens], "indexer selected query geometry differs");
        assert_eq!(keys.dims()[2], width, "indexer selected key width differs");
        let count = indices.dims()[2];
        let entries = sparse_gather_entries(keys.cast(query.dtype()), indices);
        if count == 0 || tokens == 0 || batch == 0 {
            return Tensor::zeros([batch, tokens, count], (&query.device(), query.dtype()))
                + (query.sum() + weights.sum() + entries.sum()).mul_scalar(0).reshape([1, 1, 1]);
        }
        (ruda_model::tensor::activation::relu(query.matmul(entries.swap_dims(2, 3)))
            * weights.reshape([batch, tokens, heads, 1])).sum_dim(2).reshape([batch, tokens, count])
    }

    fn validate_mask(&self, input: &Tensor<B, 3>, keys: &Tensor<B, 3>, mask: &IndexerMask<B>) {
        let [batch, queries, width] = input.dims();
        assert_eq!(width, self.width, "indexer feature width differs");
        let count = keys.dims()[1];
        assert_eq!(keys.dims(), [batch, count, self.head_dim], "indexer prepared key geometry differs");
        assert_eq!(keys.device(), input.device(), "indexer prepared key device differs");
        if let Some(valid) = &mask.allowed {
            assert_eq!(valid.dims(), [batch, queries, count], "indexer allowed geometry differs");
            assert_eq!(valid.device(), input.device(), "indexer allowed device differs");
        }
        for (valid, shape) in [(&mask.query_valid, [batch, queries]), (&mask.key_valid, [batch, count])] {
            if let Some(valid) = valid {
                assert_eq!(valid.dims(), shape, "indexer token validity geometry differs");
                assert_eq!(valid.device(), input.device(), "indexer token validity device differs");
            }
        }
        for (pos, length) in [(&mask.query_positions, queries), (&mask.key_end_positions, count)] {
            if let Some(pos) = pos {
                assert_eq!(pos.dims(), [length], "indexer position metadata geometry differs");
                assert_eq!(pos.device(), input.device(), "indexer position metadata device differs");
                assert_eq!(pos.dtype(), DType::I64, "indexer absolute positions must use I64");
            }
        }
    }

    /// Stream query/key tiles and retain deterministic top-k, never a full Q-by-K score table.
    /// All supplied masks are conjoined. Key end positions enforce complete-block causality.
    pub fn select(&self, input: Tensor<B, 3>, keys: Tensor<B, 3>, latent: Option<Tensor<B, 3>>,
        mask: IndexerMask<B>) -> Tensor<B, 3, Int> {
        self.validate_mask(&input, &keys, &mask);
        let [batch, tokens, _] = input.dims();
        let count = keys.dims()[1];
        let device = input.device();
        let pos = mask.query_positions.clone().unwrap_or_else(|| positions::<B>(tokens, 0, &device));
        let (query, weights) = self.queries(input, latent, Some(pos.clone()), true);
        let keys = keys.detach();
        if batch == 0 || tokens == 0 || count == 0 || self.topk == 0 {
            return Tensor::empty([batch, tokens, self.topk.min(count)], (&device, DType::I64));
        }
        let mut chunks = Vec::new();
        for begin in (0..tokens).step_by(self.query_chunk_size) {
            let end = tokens.min(begin.saturating_add(self.query_chunk_size));
            let length = end - begin;
            let mut best = Tensor::<B, 3>::empty([batch, length, 0], (&device, query.dtype()));
            let mut ids = Tensor::<B, 3, Int>::empty([batch, length, 0], (&device, DType::I64));
            for start in (0..count).step_by(self.key_chunk_size) {
                let stop = count.min(start.saturating_add(self.key_chunk_size));
                let score = self.score(query.clone().slice_dim(1, begin..end), weights.clone().slice_dim(1, begin..end),
                    keys.clone().slice_dim(1, start..stop));
                let shape = [batch, length, stop - start];
                let mut valid = Tensor::<B, 3, Bool>::zeros(shape, &device).bool_not();
                if let Some(allowed) = &mask.allowed { valid = valid.bool_and(allowed.clone().slice([0..batch, begin..end, start..stop])); }
                if let Some(key_valid) = &mask.key_valid {
                    valid = valid.bool_and(key_valid.clone().slice_dim(1, start..stop).reshape([batch, 1, stop - start]).expand(shape));
                }
                if let Some(query_valid) = &mask.query_valid {
                    valid = valid.bool_and(query_valid.clone().slice_dim(1, begin..end).reshape([batch, length, 1]).expand(shape));
                }
                if let Some(ends) = &mask.key_end_positions {
                    let ends = ends.clone().slice_dim(0, start..stop).reshape([1, 1, stop - start]).expand(shape);
                    let rows = pos.clone().slice_dim(0, begin..end).reshape([1, length, 1]).expand(shape);
                    valid = valid.bool_and(ends.lower_equal(rows));
                }
                let next_ids = positions::<B>(stop - start, start, &device).reshape([1, 1, stop - start]).expand(shape);
                let combined_valid = Tensor::cat(vec![ids.clone().greater_equal_elem(0), valid], 2);
                let selected = sparse_stable_topk(Tensor::cat(vec![best, score], 2), self.topk,
                    Some(Tensor::cat(vec![ids, next_ids], 2)), Some(combined_valid));
                (best, ids) = selected;
            }
            chunks.push(ids);
        }
        Tensor::cat(chunks, 1)
    }

    /// Select prepared token/compressed keys and return their differentiable scores.
    /// With causal=true, absent key end metadata means token positions 0..keys,
    /// not inferred compression ratios or model-family rules.
    pub fn forward_prepared(&self, input: Tensor<B, 3>, keys: Tensor<B, 3>, latent: Option<Tensor<B, 3>>,
        mut mask: IndexerMask<B>, causal: bool) -> IndexerOutput<B> {
        if causal && mask.key_end_positions.is_none() {
            mask.key_end_positions = Some(positions::<B>(keys.dims()[1], 0, &input.device()));
        }
        let pos = mask.query_positions.clone();
        let indices = self.select(input.clone(), keys.clone(), latent.clone(), mask);
        let scores = self.selected_scores(input, keys, indices.clone(), latent, pos);
        let valid = indices.clone().greater_equal_elem(0);
        IndexerOutput { indices, scores, valid }
    }

    /// Token self/cross attention with the actual learned key projection.
    pub fn forward(&self, input: Tensor<B, 3>, key_states: Option<Tensor<B, 3>>,
        key_positions: Option<Tensor<B, 1, Int>>, latent: Option<Tensor<B, 3>>,
        mask: IndexerMask<B>, causal: bool) -> IndexerOutput<B> {
        let keys = self.project_keys(key_states.unwrap_or_else(|| input.clone()), key_positions);
        self.forward_prepared(input, keys, latent, mask, causal)
    }

    /// Dense teacher-mass distillation. Teacher targets are detached from the main graph.
    pub fn distillation_loss(&self, input: Tensor<B, 3>, keys: Tensor<B, 3>, teacher: Tensor<B, 3>,
        allowed: Option<Tensor<B, 3, Bool>>, latent: Option<Tensor<B, 3>>, pos: Option<Tensor<B, 1, Int>>) -> Tensor<B, 1> {
        indexer_kl_loss(self.scores(input, keys, latent, pos), teacher, allowed)
    }

    /// Distill only selected slots; invalid -1 indices contribute neither mass nor count.
    pub fn selected_distillation_loss(&self, input: Tensor<B, 3>, keys: Tensor<B, 3>, indices: Tensor<B, 3, Int>,
        teacher: Tensor<B, 3>, latent: Option<Tensor<B, 3>>, pos: Option<Tensor<B, 1, Int>>) -> Tensor<B, 1> {
        let valid = indices.clone().greater_equal_elem(0);
        indexer_kl_loss(self.selected_scores(input, keys, indices, latent, pos), teacher, Some(valid))
    }
}
