use core::ops::Range;
use ruda_model::{module::Module, tensor::{DType, Tensor, backend::Backend}};
use crate::{Linear, LoRALinear, transformer::AdaptedProjection};
use super::{LearnedKVCompressor, LightningIndexer, CompressedAttention, CompressedAttentionParts};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KVCompressionProjectionRole { Value, Gate }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexerProjectionRole { Query, HeadWeight, Key }

/// Exact original projection roles; output group indices are physical group ordinals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressedAttentionProjectionRole {
    QueryDown, QueryUp, LocalKv, OutputDown(usize), OutputUp,
    KvValue, KvGate, IndexerQuery, IndexerHeadWeight, IndexerKey, IndexValue, IndexGate,
}

/// Actual linear projection used by native compressed attention and indexing.
/// The ordinary path retains the original implementation and storage policy;
/// the compute path promotes each actual base/A/B leaf independently without
/// merging weights, replacing IDs, or retaining selection-only parameter graphs.
pub trait CompressedAttentionProjection<B: Backend>: Module<B> {
    fn dimensions(&self) -> [usize; 2];
    fn device(&self) -> B::Device;
    fn has_bias(&self) -> bool;
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;

    /// Project only the declared output channel interval in the explicit compute dtype.
    /// Adapter dropout retains its original backend training-mode semantics.
    fn forward_in_compute(&self, input: Tensor<B, 3>, compute: DType, columns: Range<usize>,
        detach_parameters: bool) -> Tensor<B, 3>;
}

fn linear_compute<B: Backend>(layer: &Linear<B>, input: Tensor<B, 3>, compute: DType,
    columns: Range<usize>, detach: bool) -> Tensor<B, 3> {
    let [input_width, output_width] = layer.weight.val().dims();
    assert!(matches!(compute, DType::F32 | DType::F64), "compressed projection compute must be FP32/FP64");
    assert!(columns.start <= columns.end && columns.end <= output_width, "compressed projection channel interval exceeds actual output");
    assert_eq!(input.dims()[2], input_width, "compressed projection input width differs");
    assert_eq!(input.device(), layer.weight.val().device(), "compressed projection input device differs");
    let mut weight = layer.weight.val();
    if detach { weight = weight.detach(); }
    let mut output = input.cast(compute).matmul(weight.cast(compute).slice_dim(1, columns.clone()).unsqueeze::<3>());
    if let Some(bias) = &layer.bias {
        let mut bias = bias.val();
        if detach { bias = bias.detach(); }
        output = output + bias.cast(compute).slice_dim(0, columns).unsqueeze::<3>();
    }
    output
}

impl<B: Backend> CompressedAttentionProjection<B> for Linear<B> {
    fn dimensions(&self) -> [usize; 2] { self.weight.val().dims() }
    fn device(&self) -> B::Device { self.weight.val().device() }
    fn has_bias(&self) -> bool { self.bias.is_some() }
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> { self.forward(input) }
    fn forward_in_compute(&self, input: Tensor<B, 3>, compute: DType, columns: Range<usize>, detach: bool) -> Tensor<B, 3> {
        linear_compute(self, input, compute, columns, detach)
    }
}

impl<B: Backend> CompressedAttentionProjection<B> for LoRALinear<B> {
    fn dimensions(&self) -> [usize; 2] { self.base.weight.val().dims() }
    fn device(&self) -> B::Device { self.base.weight.val().device() }
    fn has_bias(&self) -> bool { self.base.bias.is_some() || self.adapter_a.bias.is_some() || self.adapter_b.bias.is_some() }
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> { self.forward(input) }
    fn forward_in_compute(&self, input: Tensor<B, 3>, compute: DType, columns: Range<usize>, detach: bool) -> Tensor<B, 3> {
        let base = linear_compute(&self.base, input.clone(), compute, columns.clone(), detach);
        let adapted = self.dropout.forward(input.cast(compute));
        let rank = self.adapter_a.weight.val().dims()[1];
        let hidden = linear_compute(&self.adapter_a, adapted, compute, 0..rank, detach);
        let update = linear_compute(&self.adapter_b, hidden, compute, columns, detach).mul_scalar(self.scale);
        base + update
    }
}

impl<B: Backend> CompressedAttentionProjection<B> for AdaptedProjection<B> {
    fn dimensions(&self) -> [usize; 2] {
        match self { Self::Dense(layer) => CompressedAttentionProjection::dimensions(layer),
            Self::LoRA(layer) => CompressedAttentionProjection::dimensions(layer) }
    }
    fn device(&self) -> B::Device {
        match self { Self::Dense(layer) => CompressedAttentionProjection::device(layer),
            Self::LoRA(layer) => CompressedAttentionProjection::device(layer) }
    }
    fn has_bias(&self) -> bool {
        match self { Self::Dense(layer) => CompressedAttentionProjection::has_bias(layer),
            Self::LoRA(layer) => CompressedAttentionProjection::has_bias(layer) }
    }
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> { self.forward(input) }
    fn forward_in_compute(&self, input: Tensor<B, 3>, compute: DType, columns: Range<usize>, detach: bool) -> Tensor<B, 3> {
        match self { Self::Dense(layer) => layer.forward_in_compute(input, compute, columns, detach),
            Self::LoRA(layer) => layer.forward_in_compute(input, compute, columns, detach) }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> LearnedKVCompressor<B, P> {
    /// Visit real value/gate modules, without accessing activation or parameter values on the host.
    pub fn visit_projections(&self, mut visitor: impl FnMut(KVCompressionProjectionRole, &P)) {
        visitor(KVCompressionProjectionRole::Value, &self.value);
        visitor(KVCompressionProjectionRole::Gate, &self.gate);
    }

    /// Transform only actual projections; position/norm leaves and original overlap semantics stay intact.
    pub fn map_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(KVCompressionProjectionRole, P) -> Q) -> LearnedKVCompressor<B, Q> {
        LearnedKVCompressor::from_parts(mapper(KVCompressionProjectionRole::Value, self.value),
            mapper(KVCompressionProjectionRole::Gate, self.gate), self.position_bias, self.norm_weight, self.overlap, self.epsilon)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> LightningIndexer<B, P> {
    pub fn visit_projections(&self, mut visitor: impl FnMut(IndexerProjectionRole, &P)) {
        visitor(IndexerProjectionRole::Query, &self.query);
        visitor(IndexerProjectionRole::HeadWeight, &self.head_weight);
        if let Some(key) = &self.key { visitor(IndexerProjectionRole::Key, key); }
    }

    /// Preserve actual loaded key normalization, geometry, RoPE and chunk/gradient policies.
    pub fn map_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(IndexerProjectionRole, P) -> Q) -> LightningIndexer<B, Q> {
        let query = mapper(IndexerProjectionRole::Query, self.query);
        let head_weight = mapper(IndexerProjectionRole::HeadWeight, self.head_weight);
        let key = self.key.map(|key| mapper(IndexerProjectionRole::Key, key));
        LightningIndexer::from_parts(query, head_weight, key, self.key_norm, self.rotary, self.topk,
            self.query_chunk_size, self.key_chunk_size, self.detach_inputs)
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> CompressedAttention<B, P> {
    /// Traverse only roles that actually exist in this CSA/HCA layer.
    pub fn visit_projections(&self, mut visitor: impl FnMut(CompressedAttentionProjectionRole, &P)) {
        use CompressedAttentionProjectionRole as Role;
        visitor(Role::QueryDown, &self.parts.query_down);
        visitor(Role::QueryUp, &self.parts.query_up);
        visitor(Role::LocalKv, &self.parts.local_kv);
        self.parts.compressor.visit_projections(|role, projection| visitor(match role {
            KVCompressionProjectionRole::Value => Role::KvValue, KVCompressionProjectionRole::Gate => Role::KvGate,
        }, projection));
        for (group, projection) in self.parts.output_down.iter().enumerate() { visitor(Role::OutputDown(group), projection); }
        visitor(Role::OutputUp, &self.parts.output_up);
        if let Some(indexer) = &self.parts.indexer {
            indexer.visit_projections(|role, projection| visitor(match role {
                IndexerProjectionRole::Query => Role::IndexerQuery, IndexerProjectionRole::HeadWeight => Role::IndexerHeadWeight,
                IndexerProjectionRole::Key => Role::IndexerKey,
            }, projection));
        }
        if let Some(compressor) = &self.parts.index_compressor {
            compressor.visit_projections(|role, projection| visitor(match role {
                KVCompressionProjectionRole::Value => Role::IndexValue, KVCompressionProjectionRole::Gate => Role::IndexGate,
            }, projection));
        }
    }

    /// Consume this exact layer and map original projections into a native dense/LoRA graph.
    /// Main/compressed/index paths stay distinct; no parameter merge, guessed target name,
    /// new compressor, reinitialized sink, or alteration of complete-block visibility occurs.
    pub fn map_projections<Q: CompressedAttentionProjection<B>>(self,
        mut mapper: impl FnMut(CompressedAttentionProjectionRole, P) -> Q) -> CompressedAttention<B, Q> {
        use CompressedAttentionProjectionRole as Role;
        let parts = self.parts;
        let query_down = mapper(Role::QueryDown, parts.query_down);
        let query_up = mapper(Role::QueryUp, parts.query_up);
        let local_kv = mapper(Role::LocalKv, parts.local_kv);
        let compressor = parts.compressor.map_projections(|role, projection| mapper(match role {
            KVCompressionProjectionRole::Value => Role::KvValue, KVCompressionProjectionRole::Gate => Role::KvGate,
        }, projection));
        let output_down = parts.output_down.into_iter().enumerate().map(|(group, projection)| mapper(Role::OutputDown(group), projection)).collect();
        let output_up = mapper(Role::OutputUp, parts.output_up);
        let indexer = parts.indexer.map(|indexer| indexer.map_projections(|role, projection| mapper(match role {
            IndexerProjectionRole::Query => Role::IndexerQuery, IndexerProjectionRole::HeadWeight => Role::IndexerHeadWeight,
            IndexerProjectionRole::Key => Role::IndexerKey,
        }, projection)));
        let index_compressor = parts.index_compressor.map(|compressor| compressor.map_projections(|role, projection| mapper(match role {
            KVCompressionProjectionRole::Value => Role::IndexValue, KVCompressionProjectionRole::Gate => Role::IndexGate,
        }, projection)));
        CompressedAttention::from_parts(CompressedAttentionParts {
            query_down, query_up, query_norm: parts.query_norm, local_kv, local_norm: parts.local_norm, compressor,
            output_down, output_up, sink: parts.sink, indexer, index_compressor,
        }, self.rotary, self.window_size, self.query_chunk_size, self.epsilon)
    }
}
