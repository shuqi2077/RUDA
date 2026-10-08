use alloc::{collections::{BTreeMap, BTreeSet}, format, string::String, vec::Vec};
use ruda_model::{module::{Module, ModuleVisitor, Param, ParamId},
    record::{Record, Recorder, RecorderError, PrecisionSettings}, tensor::{Tensor, backend::Backend}};
use crate::{Embedding, LoRALinear, LoRAAdapterRecord, LoRAAdapterSchema,
    attention::CompressedAttentionProjectionRole};
use super::{AdaptedProjection, HybridTiedEmbeddingAdapter, MhcTransformerBlock, MhcTransformerProjectionRole,
    HybridAttentionBackbone, HybridAttentionLanguageModel, HybridAttentionHead, HybridAttentionAdapterTarget};

/// Actual A/B-only state for native compressed/mHC blocks, backbones or language models.
pub struct HybridAttentionAdapterRecord<B: Backend> {
    version: u32,
    base_id: String,
    scope: String,
    pub(super) entries: Vec<(String, LoRAAdapterRecord<B>)>,
}

impl<B: Backend> Record<B> for HybridAttentionAdapterRecord<B> {
    type Item<S: PrecisionSettings> = (u32, String, String, Vec<(String, <LoRAAdapterRecord<B> as Record<B>>::Item<S>)>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version, self.base_id, self.scope, self.entries.into_iter().map(|(path, entry)| (path, entry.into_item::<S>())).collect())
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self { version: item.0, base_id: item.1, scope: item.2, entries: item.3.into_iter()
            .map(|(path, entry)| (path, LoRAAdapterRecord::<B>::from_item::<S>(entry, device))).collect() }
    }
}

fn invalid(reason: &str) -> RecorderError { RecorderError::Unknown(format!("Invalid native hybrid adapter record: {reason}")) }

pub(super) fn block_path(role: MhcTransformerProjectionRole) -> String {
    match role {
        MhcTransformerProjectionRole::Gate => "gate".into(),
        MhcTransformerProjectionRole::Up => "up".into(),
        MhcTransformerProjectionRole::Down => "down".into(),
        MhcTransformerProjectionRole::Attention(role) => {
            use CompressedAttentionProjectionRole as Role;
            let suffix = match role {
                Role::QueryDown => "query_down".into(), Role::QueryUp => "query_up".into(), Role::LocalKv => "local_kv".into(),
                Role::OutputDown(group) => format!("output_down.{group}"), Role::OutputUp => "output_up".into(),
                Role::KvValue => "compressor.value".into(), Role::KvGate => "compressor.gate".into(),
                Role::IndexerQuery => "indexer.query".into(), Role::IndexerHeadWeight => "indexer.head_weight".into(),
                Role::IndexerKey => "indexer.key".into(), Role::IndexValue => "index_compressor.value".into(),
                Role::IndexGate => "index_compressor.gate".into(),
            };
            format!("attention.parts.{suffix}")
        }
    }
}

fn model_path(target: HybridAttentionAdapterTarget) -> String {
    match target { HybridAttentionAdapterTarget::Layer(layer, role) => format!("layers.{layer}.{}", block_path(role)),
        HybridAttentionAdapterTarget::Head => "head".into() }
}

type Candidates<'a, B> = Vec<(String, &'a LoRALinear<B>)>;
type Tied<'a, B> = Option<(&'a HybridTiedEmbeddingAdapter<B>, &'a Embedding<B>)>;

struct AdapterOnly { adapter_ids: BTreeSet<ParamId>, other_trainable: bool }
impl<B: Backend> ModuleVisitor<B> for AdapterOnly {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if param.val().is_require_grad() && !self.adapter_ids.contains(&param.id) { self.other_trainable = true; }
    }
}

fn adapter_only<B: Backend, M: Module<B>>(module: &M, candidates: &Candidates<'_, B>, tied: Tied<'_, B>) -> Result<(), RecorderError> {
    let mut visitor = AdapterOnly { adapter_ids: BTreeSet::new(), other_trainable: false };
    for (_, layer) in candidates {
        visitor.adapter_ids.insert(layer.adapter_a.weight.id);
        visitor.adapter_ids.insert(layer.adapter_b.weight.id);
    }
    if let Some((layer, _)) = tied {
        visitor.adapter_ids.insert(layer.adapter_a.weight.id);
        visitor.adapter_ids.insert(layer.adapter_b.weight.id);
    }
    module.visit(&mut visitor);
    if visitor.other_trainable { return Err(invalid("trainable non-adapter leaves require a full model checkpoint")); }
    Ok(())
}

impl<B: Backend> HybridTiedEmbeddingAdapter<B> {
    fn schema(&self, embedding: &Embedding<B>, base_id: &str) -> Result<LoRAAdapterSchema, RecorderError> {
        if base_id.is_empty() { return Err(invalid("complete frozen base identity is required")); }
        let weight = embedding.weight.val();
        let [vocab, width] = weight.dims();
        let a = self.adapter_a.weight.val();
        let b = self.adapter_b.weight.val();
        let rank = a.dims()[1];
        if vocab == 0 || width == 0 || rank == 0 || !weight.dtype().is_float() || weight.is_require_grad()
            || a.dims() != [width, rank] || b.dims() != [rank, vocab]
            || a.device() != weight.device() || b.device() != weight.device() || !a.dtype().is_float() || !b.dtype().is_float()
            || self.adapter_a.bias.is_some() || self.adapter_b.bias.is_some() {
            return Err(invalid("shared frozen embedding or actual tied-head A/B geometry/storage/device differs"));
        }
        if !self.scale.is_finite() || !self.dropout.prob.is_finite() || !(0.0..1.0).contains(&self.dropout.prob) {
            return Err(invalid("tied-head forward scale/dropout differs"));
        }
        Ok(LoRAAdapterSchema { version: 1, base_id: base_id.into(), base_shape: [width, vocab], base_dtype: weight.dtype(),
            base_bias_dtype: None, rank, scale: self.scale, dropout: self.dropout.prob, trainable: [a.is_require_grad(), b.is_require_grad()] })
    }

    fn validate_record(&self, embedding: &Embedding<B>, record: &LoRAAdapterRecord<B>, base_id: &str) -> Result<(), RecorderError> {
        let actual = self.schema(embedding, base_id)?;
        if record.schema != actual || record.schema.scale.to_bits() != actual.scale.to_bits()
            || record.schema.dropout.to_bits() != actual.dropout.to_bits() {
            return Err(invalid("tied-head frozen base, rank, precision, flags or forward configuration differs"));
        }
        Ok(())
    }

    /// Retain only real head A/B parameters; the shared embedding is never serialized twice.
    pub fn adapter_record(&self, embedding: &Embedding<B>, base_id: &str) -> Result<LoRAAdapterRecord<B>, RecorderError> {
        LoRAAdapterRecord::capture_adapters(self.schema(embedding, base_id)?, &self.adapter_a, &self.adapter_b)
    }

    pub fn restore_adapter(self, embedding: &Embedding<B>, record: LoRAAdapterRecord<B>, base_id: &str) -> Result<Self, RecorderError> {
        self.validate_record(embedding, &record, base_id)?;
        let schema = record.schema.clone();
        let (adapter_a, adapter_b) = record.restore_adapters(self.adapter_a, self.adapter_b, &embedding.weight.val().device())?;
        let restored = Self { adapter_a, adapter_b, dropout: self.dropout, scale: self.scale };
        let actual = restored.schema(embedding, base_id)?;
        if schema != actual || schema.scale.to_bits() != actual.scale.to_bits() || schema.dropout.to_bits() != actual.dropout.to_bits() {
            return Err(invalid("restored tied-head continuation contract differs"));
        }
        Ok(restored)
    }
}

impl<B: Backend> HybridAttentionAdapterRecord<B> {
    pub(super) fn capture<M: Module<B>>(module: &M, base_id: &str, scope: &str, candidates: Candidates<'_, B>, tied: Tied<'_, B>) -> Result<Self, RecorderError> {
        if base_id.is_empty() { return Err(invalid("complete frozen base identity is required")); }
        adapter_only(module, &candidates, tied)?;
        let mut entries = Vec::new();
        for (path, layer) in candidates { entries.push((path, layer.adapter_record(base_id)?)); }
        if let Some((adapter, embedding)) = tied { entries.push(("head.tied_update".into(), adapter.adapter_record(embedding, base_id)?)); }
        if entries.is_empty() { return Err(invalid("no actual adapters exist in this module")); }
        Ok(Self { version: 1, base_id: base_id.into(), scope: scope.into(), entries })
    }

    pub(super) fn validate<M: Module<B>>(&self, module: &M, base_id: &str, scope: &str, candidates: Candidates<'_, B>, tied: Tied<'_, B>) -> Result<(), RecorderError> {
        if self.version != 1 || self.base_id != base_id || base_id.is_empty() || self.scope != scope {
            return Err(invalid("version, complete frozen base identity or module scope differs"));
        }
        adapter_only(module, &candidates, tied)?;
        let stored: BTreeMap<_, _> = self.entries.iter().map(|(path, entry)| (path.as_str(), entry)).collect();
        if stored.len() != self.entries.len() || stored.len() != candidates.len() + usize::from(tied.is_some()) {
            return Err(invalid("actual adapter target set differs or stored targets repeat"));
        }
        for (path, layer) in candidates {
            let entry = stored.get(path.as_str()).ok_or_else(|| invalid("stored adapter target differs"))?;
            entry.schema.validate_for(layer, base_id)?;
        }
        if let Some((adapter, embedding)) = tied {
            adapter.validate_record(embedding, stored.get("head.tied_update").ok_or_else(|| invalid("tied head target missing"))?, base_id)?;
        }
        Ok(())
    }

    pub fn targets(&self) -> impl Iterator<Item = &str> { self.entries.iter().map(|(path, _)| path.as_str()) }

    pub fn save<R: Recorder<B>>(self, recorder: &R, args: R::RecordArgs) -> Result<R::RecordOutput, RecorderError> { recorder.record(self, args) }
    pub fn load<R: Recorder<B>>(recorder: &R, args: R::LoadArgs, device: &B::Device) -> Result<Self, RecorderError> { recorder.load(args, device) }

    pub fn capture_block(block: &MhcTransformerBlock<B, AdaptedProjection<B>>, base_id: &str) -> Result<Self, RecorderError> {
        let mut entries = Vec::new();
        block.visit_projections(|role, projection| if let AdaptedProjection::LoRA(layer) = projection { entries.push((block_path(role), layer)); });
        Self::capture(block, base_id, "block", entries, None)
    }

    pub fn capture_backbone(backbone: &HybridAttentionBackbone<B, AdaptedProjection<B>>, base_id: &str) -> Result<Self, RecorderError> {
        let mut entries = Vec::new();
        backbone.visit_projections(|target, projection| if let AdaptedProjection::LoRA(layer) = projection { entries.push((model_path(target), layer)); });
        Self::capture(backbone, base_id, "backbone", entries, None)
    }

    pub fn capture_model(model: &HybridAttentionLanguageModel<B, AdaptedProjection<B>>, base_id: &str) -> Result<Self, RecorderError> {
        let mut entries = Vec::new();
        model.visit_projections(|target, projection| if let AdaptedProjection::LoRA(layer) = projection { entries.push((model_path(target), layer)); });
        let tied = match &model.head { HybridAttentionHead::TiedEmbeddingLoRA(adapter) => Some((adapter, &model.backbone.embedding)), _ => None };
        Self::capture(model, base_id, "model", entries, tied)
    }

    /// Check every declared target/continuation contract before loading any A/B leaf.
    pub fn validate_block(&self, block: &MhcTransformerBlock<B, AdaptedProjection<B>>, base_id: &str) -> Result<(), RecorderError> {
        let mut entries = Vec::new();
        block.visit_projections(|role, projection| if let AdaptedProjection::LoRA(layer) = projection { entries.push((block_path(role), layer)); });
        self.validate(block, base_id, "block", entries, None)
    }

    pub fn validate_backbone(&self, backbone: &HybridAttentionBackbone<B, AdaptedProjection<B>>, base_id: &str) -> Result<(), RecorderError> {
        let mut entries = Vec::new();
        backbone.visit_projections(|target, projection| if let AdaptedProjection::LoRA(layer) = projection { entries.push((model_path(target), layer)); });
        self.validate(backbone, base_id, "backbone", entries, None)
    }

    pub fn validate_model(&self, model: &HybridAttentionLanguageModel<B, AdaptedProjection<B>>, base_id: &str) -> Result<(), RecorderError> {
        let mut entries = Vec::new();
        model.visit_projections(|target, projection| if let AdaptedProjection::LoRA(layer) = projection { entries.push((model_path(target), layer)); });
        let tied = match &model.head { HybridAttentionHead::TiedEmbeddingLoRA(adapter) => Some((adapter, &model.backbone.embedding)), _ => None };
        self.validate(model, base_id, "model", entries, tied)
    }

    pub fn restore_block(self, block: MhcTransformerBlock<B, AdaptedProjection<B>>, base_id: &str)
        -> Result<MhcTransformerBlock<B, AdaptedProjection<B>>, RecorderError> {
        self.validate_block(&block, base_id)?;
        let mut entries: BTreeMap<_, _> = self.entries.into_iter().collect();
        block.try_map_projections(|role, projection| restore_projection(projection, &block_path(role), &mut entries, base_id))
    }

    pub fn restore_backbone(self, backbone: HybridAttentionBackbone<B, AdaptedProjection<B>>, base_id: &str)
        -> Result<HybridAttentionBackbone<B, AdaptedProjection<B>>, RecorderError> {
        self.validate_backbone(&backbone, base_id)?;
        let mut entries: BTreeMap<_, _> = self.entries.into_iter().collect();
        backbone.try_map_projections(|target, projection| restore_projection(projection, &model_path(target), &mut entries, base_id))
    }

    pub fn restore_model(self, model: HybridAttentionLanguageModel<B, AdaptedProjection<B>>, base_id: &str)
        -> Result<HybridAttentionLanguageModel<B, AdaptedProjection<B>>, RecorderError> {
        self.validate_model(&model, base_id)?;
        let mut entries: BTreeMap<_, _> = self.entries.into_iter().collect();
        let mut restored = model.try_map_projections(|target, projection| restore_projection(projection, &model_path(target), &mut entries, base_id))?;
        if let HybridAttentionHead::TiedEmbeddingLoRA(adapter) = restored.head {
            let entry = entries.remove("head.tied_update").ok_or_else(|| invalid("tied head record missing"))?;
            restored.head = HybridAttentionHead::TiedEmbeddingLoRA(adapter.restore_adapter(&restored.backbone.embedding, entry, base_id)?);
        }
        if !entries.is_empty() { return Err(invalid("unconsumed native adapter targets")); }
        Ok(restored)
    }
}

pub(super) fn restore_projection<B: Backend>(projection: AdaptedProjection<B>, path: &str,
    entries: &mut BTreeMap<String, LoRAAdapterRecord<B>>, base_id: &str) -> Result<AdaptedProjection<B>, RecorderError> {
    match projection {
        AdaptedProjection::Dense(layer) => Ok(AdaptedProjection::Dense(layer)),
        AdaptedProjection::LoRA(layer) => {
            let entry = entries.remove(path).ok_or_else(|| invalid("projection record missing"))?;
            Ok(AdaptedProjection::LoRA(entry.restore_into(layer, base_id)?))
        }
    }
}
