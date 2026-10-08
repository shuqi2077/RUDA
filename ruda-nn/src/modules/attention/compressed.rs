use alloc::vec;
use alloc::vec::Vec;
use ruda_model::{config::Config, module::{Initializer, Module, Param},
    tensor::{Bool, DType, FloatDType, Int, Tensor, backend::Backend}};
use crate::{Linear, LinearConfig};
use super::{CompressionState, LearnedKVCompressor, LearnedKVCompressorConfig, LightningIndexer,
    LightningIndexerConfig, IndexerMask, SparseRotaryEmbedding, sparse_gather_entries, indexer_kl_loss};
use super::sparse_ops::{masked_softmax, positions, rms, valid_mask, work_dtype};
use super::CompressedAttentionProjection;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Generic shared-KV compressed attention, with explicit architecture dimensions.
#[derive(Config, Debug)]
pub struct CompressedAttentionConfig {
    pub width: usize,
    pub num_heads: usize,
    /// CSA uses overlapping blocks and learned top-k; HCA attends all complete blocks.
    pub sparse: bool,
    #[config(default = "None")]
    pub head_dim: Option<usize>,
    #[config(default = 4)]
    pub compress_ratio: usize,
    #[config(default = 32)]
    pub topk: usize,
    #[config(default = 128)]
    pub window_size: usize,
    #[config(default = "None")]
    pub query_rank: Option<usize>,
    #[config(default = 4)]
    pub index_heads: usize,
    #[config(default = 16)]
    pub index_dim: usize,
    #[config(default = 0)]
    pub rope_dim: usize,
    #[config(default = 10000.0)]
    pub rope_base: f64,
    #[config(default = 1)]
    pub output_groups: usize,
    #[config(default = "None")]
    pub output_rank: Option<usize>,
    #[config(default = 32)]
    pub query_chunk_size: usize,
    #[config(default = 128)]
    pub key_chunk_size: usize,
    #[config(default = true)]
    pub attention_sink: bool,
    #[config(default = 1e-6)]
    pub epsilon: f64,
}

/// Actual checkpoint projections and learned compression/indexing parameters.
#[derive(Module, Debug)]
pub struct CompressedAttentionParts<B: Backend, P: Module<B> = Linear<B>> {
    pub query_down: P,
    pub query_up: P,
    pub query_norm: Param<Tensor<B, 1>>,
    pub local_kv: P,
    pub local_norm: Param<Tensor<B, 1>>,
    pub compressor: LearnedKVCompressor<B, P>,
    pub output_down: Vec<P>,
    pub output_up: P,
    pub sink: Option<Param<Tensor<B, 1>>>,
    pub indexer: Option<LightningIndexer<B, P>>,
    pub index_compressor: Option<LearnedKVCompressor<B, P>>,
}

/// Native CSA/HCA: local and compressed entries share one per-head softmax.
#[derive(Module, Debug)]
pub struct CompressedAttention<B: Backend, P: Module<B> = Linear<B>> {
    pub parts: CompressedAttentionParts<B, P>,
    pub rotary: SparseRotaryEmbedding,
    pub width: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub query_rank: usize,
    pub output_groups: usize,
    pub output_rank: usize,
    pub window_size: usize,
    pub query_chunk_size: usize,
    pub epsilon: f64,
}

/// Main attention output and independently trainable indexer distillation loss.
#[derive(Clone, Debug)]
pub struct CompressedAttentionOutput<B: Backend> {
    pub output: Tensor<B, 3>,
    pub indexer_loss: Tensor<B, 1>,
}

impl CompressedAttentionConfig {
    pub fn compressed_sparse(width: usize, num_heads: usize) -> Self { Self::new(width, num_heads, true) }

    pub fn heavily_compressed(width: usize, num_heads: usize) -> Self {
        Self::new(width, num_heads, false).with_compress_ratio(128)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> CompressedAttention<B> {
        assert!(self.width > 0 && self.num_heads > 0 && self.output_groups > 0
            && self.num_heads.is_multiple_of(self.output_groups), "invalid compressed attention head/group geometry");
        let head_dim = self.head_dim.unwrap_or_else(|| {
            assert!(self.width.is_multiple_of(self.num_heads), "implicit attention head width must divide residual width");
            self.width / self.num_heads
        });
        let query_rank = self.query_rank.unwrap_or(self.width);
        let output_rank = self.output_rank.unwrap_or(self.width / self.output_groups);
        assert!(head_dim > 0 && query_rank > 0 && output_rank > 0, "attention projection ranks must be positive");
        let query_output = self.num_heads.checked_mul(head_dim).expect("attention query width overflow");
        let grouped_output = self.output_groups.checked_mul(output_rank).expect("attention output rank overflow");
        let mut output_down = Vec::with_capacity(self.output_groups);
        for _ in 0..self.output_groups {
            output_down.push(LinearConfig::new(query_output / self.output_groups, output_rank).with_bias(false).init(device));
        }
        let (indexer, index_compressor) = if self.sparse {
            let indexer = LightningIndexerConfig::new(self.width).with_num_heads(self.index_heads).with_head_dim(self.index_dim)
                .with_query_dim(Some(query_rank)).with_topk(self.topk).with_query_chunk_size(self.query_chunk_size)
                .with_key_chunk_size(self.key_chunk_size).with_external_keys(true).with_rope_dim(self.rope_dim)
                .with_epsilon(self.epsilon).init(device);
            let compressor = LearnedKVCompressorConfig::new(self.width, self.index_dim, self.compress_ratio)
                .with_overlap(true).with_epsilon(self.epsilon).init(device);
            (Some(indexer), Some(compressor))
        } else { (None, None) };
        let parts = CompressedAttentionParts {
            query_down: LinearConfig::new(self.width, query_rank).with_bias(false).init(device),
            query_up: LinearConfig::new(query_rank, query_output).with_bias(false).init(device),
            query_norm: Initializer::Ones.init([query_rank], device),
            local_kv: LinearConfig::new(self.width, head_dim).with_bias(false).init(device),
            local_norm: Initializer::Ones.init([head_dim], device),
            compressor: LearnedKVCompressorConfig::new(self.width, head_dim, self.compress_ratio)
                .with_overlap(self.sparse).with_epsilon(self.epsilon).init(device),
            output_down,
            output_up: LinearConfig::new(grouped_output, self.width).with_bias(false).init(device),
            sink: if self.attention_sink { Some(Initializer::Zeros.init([self.num_heads], device)
                .map(|tensor: Tensor<B, 1>| tensor.cast(DType::F32))) } else { None },
            indexer, index_compressor,
        };
        CompressedAttention::from_parts(parts, SparseRotaryEmbedding::new(self.rope_dim, self.rope_base),
            self.window_size, self.query_chunk_size, self.epsilon)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> CompressedAttention<B, P> {
    /// Connect complete actual loaded leaves, retaining the original parameter identities.
    pub fn from_parts(parts: CompressedAttentionParts<B, P>, rotary: SparseRotaryEmbedding,
        window_size: usize, query_chunk_size: usize, epsilon: f64) -> Self {
        let [width, query_rank] = parts.query_down.dimensions();
        let [up_rank, query_output] = parts.query_up.dimensions();
        let [local_width, head_dim] = parts.local_kv.dimensions();
        let output_groups = parts.output_down.len();
        assert!(width > 0 && query_rank > 0 && head_dim > 0 && query_output > 0
            && query_output.is_multiple_of(head_dim) && output_groups > 0 && window_size > 0
            && query_chunk_size > 0 && epsilon.is_finite() && epsilon > 0.0, "invalid loaded compressed attention geometry");
        let num_heads = query_output / head_dim;
        assert!(num_heads.is_multiple_of(output_groups), "attention output groups must divide query heads");
        assert_eq!((up_rank, local_width), (query_rank, width), "compressed attention projection input widths differ");
        assert_eq!(parts.query_norm.val().dims(), [query_rank], "query normalization rank differs");
        assert_eq!(parts.local_norm.val().dims(), [head_dim], "local normalization rank differs");
        assert_eq!((parts.compressor.width, parts.compressor.head_dim), (width, head_dim), "attention compressor geometry differs");
        assert!(rotary.rope_dim <= head_dim, "attention rotary channels exceed a head");
        let output_rank = parts.output_down[0].dimensions()[1];
        assert!(output_rank > 0, "attention output rank must be positive");
        assert_eq!(parts.output_up.dimensions(), [output_groups.checked_mul(output_rank).expect("output rank overflow"), width],
            "attention output-up projection geometry differs");
        let device = parts.query_down.device();
        for layer in [&parts.query_down, &parts.query_up, &parts.local_kv, &parts.output_up] {
            assert!(!layer.has_bias(), "compressed attention projections must be bias-free");
            assert_eq!(layer.device(), device, "attention projection devices differ");
        }
        for layer in &parts.output_down {
            assert_eq!(layer.dimensions(), [query_output / output_groups, output_rank], "grouped output-down geometry differs");
            assert!(!layer.has_bias(), "compressed attention output-down must be bias-free");
            assert_eq!(layer.device(), device, "attention output-down device differs");
        }
        assert!(parts.query_norm.val().device() == device && parts.local_norm.val().device() == device
            && parts.compressor.value.device() == device, "attention normalization/compression device differs");
        if let Some(sink) = &parts.sink {
            assert_eq!(sink.val().dims(), [num_heads], "attention sink head count differs");
            assert_eq!(sink.val().device(), device, "attention sink device differs");
        }
        match (&parts.indexer, &parts.index_compressor) {
            (Some(indexer), Some(compressor)) => {
                assert!(parts.compressor.overlap && compressor.overlap, "CSA needs both overlapping compressors");
                assert!(indexer.key.is_none() && indexer.key_norm.is_none(), "CSA indexer consumes external compressed keys");
                assert_eq!((indexer.width, indexer.query_dim), (width, query_rank), "CSA indexer query geometry differs");
                assert_eq!((compressor.width, compressor.head_dim, compressor.ratio), (width, indexer.head_dim, parts.compressor.ratio),
                    "CSA key compressor geometry differs");
                assert!(indexer.query.device() == device && compressor.value.device() == device,
                    "CSA indexing device differs");
            }
            (None, None) => assert!(!parts.compressor.overlap, "HCA must use non-overlapping compression"),
            _ => panic!("compressed attention needs both indexer and index compressor or neither"),
        }
        Self { parts, rotary, width, num_heads, head_dim, query_rank, output_groups, output_rank,
            window_size, query_chunk_size, epsilon }
    }

    fn validate(&self, input: &Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 2, Bool> {
        let [batch, tokens, width] = input.dims();
        assert!(batch > 0 && tokens > 0 && width == self.width, "compressed attention needs nonempty actual input rows");
        work_dtype(input.dtype());
        assert_eq!(input.device(), self.parts.query_down.device(), "compressed attention input device differs");
        valid_mask(input, valid)
    }

    fn project(&self, input: Tensor<B, 3>, pos: Tensor<B, 1, Int>) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 3>) {
        let [batch, tokens, _] = input.dims();
        let latent = rms(self.parts.query_down.forward(input.clone()), self.epsilon, Some(self.parts.query_norm.val()));
        let query = self.parts.query_up.forward(latent.clone()).reshape([batch, tokens, self.num_heads, self.head_dim]);
        let query = self.rotary.forward(rms(query, self.epsilon, None).cast(input.dtype()), pos.clone(), false);
        let local = self.rotary.forward_shared(rms(self.parts.local_kv.forward(input.clone()), self.epsilon,
            Some(self.parts.local_norm.val())).cast(input.dtype()), pos);
        (latent, query, local)
    }

    fn attend(&self, input: Tensor<B, 3>, latent: Tensor<B, 3>, query: Tensor<B, 4>, pos: Tensor<B, 1, Int>,
        query_valid: Tensor<B, 2, Bool>, local: Tensor<B, 3>, local_valid: Tensor<B, 2, Bool>, local_start: usize,
        compressed: Tensor<B, 3>, compressed_valid: Tensor<B, 2, Bool>, index_keys: Option<Tensor<B, 3>>,
        return_aux: bool, indexer_warmup: bool) -> CompressedAttentionOutput<B> {
        let [batch, tokens, _] = input.dims();
        let device = input.device();
        let compute = work_dtype(input.dtype());
        let count = compressed.dims()[1];
        let ratio = i64::try_from(self.parts.compressor.ratio).expect("compression ratio exceeds I64");
        let ends = positions::<B>(count, 0, &device).mul_scalar(ratio).add_scalar(ratio - 1);
        let window = i64::try_from(self.window_size).expect("local window exceeds I64");
        let local_start = i64::try_from(local_start).expect("local position exceeds I64");
        let local_count = i64::try_from(local.dims()[1]).expect("local history exceeds I64");
        let mut chunks = Vec::new();
        let mut losses = Vec::new();
        let mut active_counts = Vec::new();
        for begin in (0..tokens).step_by(self.query_chunk_size) {
            let end = tokens.min(begin.saturating_add(self.query_chunk_size));
            let length = end - begin;
            let rows = pos.clone().slice_dim(0, begin..end);
            let qvalid = query_valid.clone().slice_dim(1, begin..end);
            let local_ids = rows.clone().reshape([length, 1]).sub_scalar(window - 1)
                + positions::<B>(self.window_size, 0, &device).reshape([1, self.window_size]);
            let local_ids = local_ids.sub_scalar(local_start);
            let local_slots = local_ids.clone().greater_equal_elem(0).bool_and(local_ids.clone().lower_elem(local_count));
            let local_ids = local_ids.mask_fill(local_slots.bool_not(), -1).reshape([1, length, self.window_size])
                .expand([batch, length, self.window_size]);
            let local_entries = sparse_gather_entries(local.clone(), local_ids.clone());
            let lmask = sparse_gather_entries(local_valid.clone().cast::<FloatDType>(input.dtype().into())
                .reshape([batch, local.dims()[1], 1]), local_ids).reshape([batch, length, self.window_size])
                .greater_elem(0).bool_and(qvalid.clone().reshape([batch, length, 1]).expand([batch, length, self.window_size]));
            let (ids, cmask) = if let Some(indexer) = &self.parts.indexer && !indexer_warmup {
                let ids = indexer.select(input.clone().slice_dim(1, begin..end),
                    index_keys.as_ref().expect("CSA compressed index keys missing").clone(),
                    Some(latent.clone().slice_dim(1, begin..end)), IndexerMask {
                        key_valid: Some(compressed_valid.clone()), query_valid: Some(qvalid.clone()),
                        key_end_positions: Some(ends.clone()), query_positions: Some(rows.clone()), allowed: None,
                    });
                let mask = ids.clone().greater_equal_elem(0);
                (ids, mask)
            } else {
                let shape = [batch, length, count];
                let mask = compressed_valid.clone().reshape([batch, 1, count]).expand(shape)
                    .bool_and(qvalid.clone().reshape([batch, length, 1]).expand(shape))
                    .bool_and(ends.clone().reshape([1, 1, count]).expand(shape)
                        .lower_equal(rows.clone().reshape([1, length, 1]).expand(shape)));
                let ids = positions::<B>(count, 0, &device).reshape([1, 1, count]).expand(shape).mask_fill(mask.clone().bool_not(), -1);
                (ids, mask)
            };
            let selected = sparse_gather_entries(compressed.clone(), ids.clone());
            let entries = Tensor::cat(vec![local_entries, selected], 2);
            let entry_count = entries.dims()[2];
            let mask = Tensor::cat(vec![lmask, cmask.clone()], 2);
            let mut scores = query.clone().slice_dim(1, begin..end).cast(compute)
                .matmul(entries.clone().cast(compute).swap_dims(2, 3)).mul_scalar(1.0 / (self.head_dim as f64).sqrt());
            let mut allowed = mask.reshape([batch, length, 1, entry_count]).expand([batch, length, self.num_heads, entry_count]);
            if let Some(sink) = &self.parts.sink {
                let sink = sink.val().cast(compute).reshape([1, 1, self.num_heads, 1]).expand([batch, length, self.num_heads, 1]);
                allowed = Tensor::cat(vec![allowed, Tensor::<B, 4, Bool>::zeros(sink.dims(), &device).bool_not()], 3);
                scores = Tensor::cat(vec![scores, sink], 3);
            }
            let mut probabilities = masked_softmax(scores, allowed, 3);
            if self.parts.sink.is_some() { probabilities = probabilities.slice_dim(3, 0..entry_count); }
            let attended = probabilities.clone().matmul(entries.cast(compute))
                * qvalid.clone().cast::<FloatDType>(compute.into()).reshape([batch, length, 1, 1]);
            chunks.push(self.rotary.forward(attended.cast(query.dtype()), rows.clone(), true));
            if return_aux && let Some(indexer) = &self.parts.indexer {
                let selected_count = ids.dims()[2];
                let teacher = probabilities.slice_dim(3, self.window_size..entry_count).sum_dim(2)
                    .reshape([batch, length, selected_count]);
                let active = teacher.clone().detach().sum_dim(2).greater_elem(0)
                    .bool_and(cmask.clone().any_dim(2)).cast::<FloatDType>(compute.into()).sum();
                let scores = indexer.selected_scores(input.clone().slice_dim(1, begin..end),
                    index_keys.as_ref().expect("CSA index keys missing").clone(), ids,
                    Some(latent.clone().slice_dim(1, begin..end)), Some(rows));
                losses.push(indexer_kl_loss(scores, teacher, Some(cmask)) * active.clone());
                active_counts.push(active);
            }
        }
        let attended = Tensor::cat(chunks, 1).reshape([batch, tokens, self.output_groups, self.num_heads / self.output_groups * self.head_dim]);
        let mut groups = Vec::with_capacity(self.output_groups);
        for (group, layer) in self.parts.output_down.iter().enumerate() {
            groups.push(layer.forward(attended.clone().slice_dim(2, group..group + 1)
                .reshape([batch, tokens, self.num_heads / self.output_groups * self.head_dim])));
        }
        let projected = self.parts.output_up.forward(Tensor::cat(groups, 2));
        let output = projected.clone() * query_valid.cast::<FloatDType>(projected.dtype().into()).reshape([batch, tokens, 1]);
        let indexer_loss = if losses.is_empty() {
            Tensor::zeros([1], (&device, compute))
        } else {
            Tensor::cat(losses, 0).sum() / Tensor::cat(active_counts, 0).sum().clamp_min(1)
        };
        CompressedAttentionOutput { output, indexer_loss }
    }

    /// Full-sequence causal CSA/HCA using native differentiable backend operations.
    pub fn forward(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        self.forward_options(input, valid, false, false).output
    }

    /// Main output plus indexer KL. Warm-up attends all causally visible compressed keys;
    /// backpropagating the loss alone leaves detached main-model features/targets untouched.
    pub fn forward_with_aux(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>,
        indexer_warmup: bool) -> CompressedAttentionOutput<B> {
        self.forward_options(input, valid, true, indexer_warmup)
    }

    fn forward_options(&self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>,
        return_aux: bool, indexer_warmup: bool) -> CompressedAttentionOutput<B> {
        let valid = self.validate(&input, valid);
        let pos = positions::<B>(input.dims()[1], 0, &input.device());
        let (latent, query, local) = self.project(input.clone(), pos.clone());
        let (compressed, compressed_valid) = self.parts.compressor.forward(input.clone(), Some(valid.clone()));
        let block_pos = positions::<B>(compressed.dims()[1], 0, &input.device())
            .mul_scalar(i64::try_from(self.parts.compressor.ratio).expect("compression ratio exceeds I64"));
        let compressed = self.rotary.forward_shared(compressed, block_pos.clone());
        let index_keys = self.parts.index_compressor.as_ref().map(|compressor| {
            let (keys, _) = compressor.forward(input.clone().detach(), Some(valid.clone()));
            self.parts.indexer.as_ref().expect("CSA indexer missing").rotary.forward_shared(keys, block_pos)
        });
        self.attend(input, latent, query, pos, valid.clone(), local, valid, 0,
            compressed, compressed_valid, index_keys, return_aux, indexer_warmup)
    }

    /// Bind incremental state to this immutable actual module. Rust borrowing prevents
    /// loading/updating/moving its parameters while the live inference session exists.
    pub fn inference_session(&self) -> CompressedAttentionSession<'_, B, P> {
        assert!(!B::ad_enabled(&self.parts.query_down.device()), "cached attention requires autograd-disabled inference");
        CompressedAttentionSession { module: self, cache: None }
    }
}

#[derive(Clone, Debug)]
struct CompressedHistory<B: Backend> {
    seen: usize,
    compressed: Tensor<B, 3>,
    compressed_valid: Tensor<B, 2, Bool>,
    index_keys: Option<Tensor<B, 3>>,
    local: Tensor<B, 3>,
    local_valid: Tensor<B, 2, Bool>,
    compression: CompressionState<B>,
    index_compression: Option<CompressionState<B>>,
}

/// Prefill/decode session borrowing one unchanged module, with no transferable stale cache.
#[derive(Debug)]
pub struct CompressedAttentionSession<'a, B: Backend, P: Module<B> = Linear<B>> {
    module: &'a CompressedAttention<B, P>,
    cache: Option<CompressedHistory<B>>,
}

impl<'a, B: Backend, P: CompressedAttentionProjection<B>> CompressedAttentionSession<'a, B, P> {
    pub fn position(&self) -> usize { self.cache.as_ref().map_or(0, |cache| cache.seen) }

    pub fn compressed_blocks(&self) -> usize { self.cache.as_ref().map_or(0, |cache| cache.compressed.dims()[1]) }

    /// Real physical raw slots retained by local and both compressor histories.
    pub fn retained_raw_tokens(&self) -> usize {
        self.cache.as_ref().map_or(0, |cache| cache.local.dims()[1] + cache.compression.tail.dims()[1]
            + cache.compression.previous.dims()[1] + cache.index_compression.as_ref().map_or(0,
                |index| index.tail.dims()[1] + index.previous.dims()[1]))
    }

    pub fn clear(&mut self) { self.cache = None; }

    /// Snapshot actual native histories for independent continuations or rollback.
    /// Tensor handles are shared immutably until subsequent functional operations;
    /// no host payload readback or prompt recomputation is involved.
    pub fn fork(&self) -> Self { Self { module: self.module, cache: self.cache.clone() } }

    /// Restore an actual snapshot from this exact still-borrowed module revision.
    /// Both sessions keep the module immutable throughout the snapshot lifetime.
    pub fn restore(&mut self, snapshot: Self) {
        assert!(core::ptr::eq(self.module, snapshot.module), "compressed snapshot belongs to another module");
        self.cache = snapshot.cache;
    }

    /// Logical retained native payload bytes, excluding allocator reserve/workspace.
    pub fn tensor_bytes(&self) -> usize {
        fn bytes<B: Backend, const D: usize, K: ruda_model::tensor::TensorKind<B> + ruda_model::tensor::BasicOps<B>>(
            tensor: &Tensor<B, D, K>) -> usize {
            tensor.dims().into_iter().try_fold(tensor.dtype().size(), |total, dimension| total.checked_mul(dimension))
                .expect("compressed cache logical byte count overflow")
        }
        fn compression_bytes<B: Backend>(state: &CompressionState<B>) -> usize {
            bytes(&state.tail) + bytes(&state.tail_valid) + bytes(&state.previous) + bytes(&state.previous_valid)
        }
        self.cache.as_ref().map_or(0, |cache| bytes(&cache.compressed) + bytes(&cache.compressed_valid)
            + cache.index_keys.as_ref().map_or(0, bytes) + bytes(&cache.local) + bytes(&cache.local_valid)
            + compression_bytes(&cache.compression) + cache.index_compression.as_ref().map_or(0, compression_bytes))
    }

    /// Reorder/duplicate every retained batch payload together for beam decoding.
    pub fn reorder(&mut self, parents: Tensor<B, 1, Int>) {
        if let Some(cache) = &mut self.cache {
            assert_eq!(parents.device(), cache.local.device(), "compressed cache beam parents device differs");
            cache.compressed = cache.compressed.clone().select(0, parents.clone());
            cache.compressed_valid = cache.compressed_valid.clone().select(0, parents.clone());
            cache.index_keys = cache.index_keys.as_ref().map(|keys| keys.clone().select(0, parents.clone()));
            cache.local = cache.local.clone().select(0, parents.clone());
            cache.local_valid = cache.local_valid.clone().select(0, parents.clone());
            cache.compression = cache.compression.reorder(parents.clone());
            cache.index_compression = cache.index_compression.as_ref().map(|state| state.reorder(parents.clone()));
        }
    }

    /// Process the complete supplied arbitrary-length chunk, preserving exactly the
    /// full-sequence complete-block/local-window visibility and absolute positions.
    pub fn forward(&mut self, input: Tensor<B, 3>, valid: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        assert!(!B::ad_enabled(&input.device()), "cached attention requires autograd-disabled inference");
        let valid = self.module.validate(&input, valid);
        let [batch, tokens, _] = input.dims();
        let start = self.position();
        let seen = start.checked_add(tokens).expect("compressed cache position overflow");
        let pos = positions::<B>(tokens, start, &input.device());
        if let Some(cache) = &self.cache {
            assert_eq!(cache.local.dims()[0], batch, "compressed cache batch changed without beam reorder");
            assert_eq!(cache.local.device(), input.device(), "compressed cache device changed");
            assert_eq!(cache.local.dtype(), input.dtype(), "compressed cache storage changed");
        }
        let (latent, query, local_new) = self.module.project(input.clone(), pos.clone());
        let (new_compressed, new_valid, compression) = self.module.parts.compressor.append(input.clone(), Some(valid.clone()),
            self.cache.as_ref().map(|cache| &cache.compression));
        let first = self.compressed_blocks();
        let block_pos = positions::<B>(new_compressed.dims()[1], first, &input.device())
            .mul_scalar(i64::try_from(self.module.parts.compressor.ratio).expect("compression ratio exceeds I64"));
        let new_compressed = self.module.rotary.forward_shared(new_compressed, block_pos.clone());
        let compressed = self.cache.as_ref().map_or_else(|| new_compressed.clone(),
            |cache| Tensor::cat(vec![cache.compressed.clone(), new_compressed.clone()], 1));
        let compressed_valid = self.cache.as_ref().map_or_else(|| new_valid.clone(),
            |cache| Tensor::cat(vec![cache.compressed_valid.clone(), new_valid.clone()], 1));
        let (index_keys, index_compression) = if let Some(compressor) = &self.module.parts.index_compressor {
            let (keys, _, state) = compressor.append(input.clone(), Some(valid.clone()),
                self.cache.as_ref().and_then(|cache| cache.index_compression.as_ref()));
            let keys = self.module.parts.indexer.as_ref().expect("CSA indexer missing").rotary.forward_shared(keys, block_pos);
            let keys = if let Some(cache) = &self.cache {
                Tensor::cat(vec![cache.index_keys.as_ref().expect("CSA cached index keys missing").clone(), keys], 1)
            } else { keys };
            (Some(keys), Some(state))
        } else { (None, None) };
        let local_start = start - self.cache.as_ref().map_or(0, |cache| cache.local.dims()[1]);
        let local = self.cache.as_ref().map_or_else(|| local_new.clone(),
            |cache| Tensor::cat(vec![cache.local.clone(), local_new.clone()], 1));
        let local_valid = self.cache.as_ref().map_or_else(|| valid.clone(),
            |cache| Tensor::cat(vec![cache.local_valid.clone(), valid.clone()], 1));
        let result = self.module.attend(input, latent, query, pos, valid, local.clone(), local_valid.clone(), local_start,
            compressed.clone(), compressed_valid.clone(), index_keys.clone(), false, false).output.detach();
        let keep = (self.module.window_size - 1).min(local.dims()[1]);
        let last = local.dims()[1] - keep;
        let local = local.slice_dim(1, last..last + keep).detach();
        let local_valid = local_valid.slice_dim(1, last..last + keep);
        let local = if keep == 0 { local } else {
            Tensor::empty(local.dims(), (&local.device(), local.dtype()))
                .slice_assign([0..batch, 0..keep, 0..self.module.head_dim], local)
        };
        let local_valid = if keep == 0 { local_valid } else {
            Tensor::<B, 2, Bool>::empty(local_valid.dims(), (&local_valid.device(), local_valid.dtype()))
                .slice_assign([0..batch, 0..keep], local_valid)
        };
        self.cache = Some(CompressedHistory { seen, compressed: compressed.detach(), compressed_valid,
            index_keys: index_keys.map(Tensor::detach), local, local_valid, compression, index_compression });
        result
    }
}
