use alloc::{collections::BTreeMap, format, string::String, vec::Vec};
use ruda_model::{record::RecorderError, tensor::backend::Backend};
use crate::{LoRALinear, attention::CompressedAttentionProjectionRole};
use super::{HybridAttentionAdapterRecord, AdaptedProjection, MhcResidualStack, MhcResidualModel,
    MhcBranchProjectionMap, MhcBranchProjectionRole, MhcTransformerProjectionRole, FeedForwardAdapterTarget};
use super::compressed_adapter_record::{block_path, restore_projection};

fn attention_path(layer: usize, role: CompressedAttentionProjectionRole) -> String {
    format!("layers.{layer}.{}", block_path(MhcTransformerProjectionRole::Attention(role)))
}
fn feed_role(role: FeedForwardAdapterTarget) -> &'static str {
    match role { FeedForwardAdapterTarget::Up => "up", FeedForwardAdapterTarget::Gate => "gate", FeedForwardAdapterTarget::Down => "down" }
}
fn branch_path(layer: usize, role: MhcBranchProjectionRole) -> String {
    let path = match role { MhcBranchProjectionRole::FeedForward(role) => feed_role(role).into(),
        MhcBranchProjectionRole::Router => "routed.router".into(), MhcBranchProjectionRole::Shared(role) => format!("shared.{}", feed_role(role)) };
    format!("layers.{layer}.feed_forward.{path}")
}
type Candidates<'a, B> = Vec<(String, &'a LoRALinear<B>)>;

fn stack_candidates<B: Backend, F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
    stack: &MhcResidualStack<B, AdaptedProjection<B>, F>) -> Candidates<'_, B> {
    let mut entries = Vec::new();
    stack.visit_attention_projections(|target, projection| {
        if let AdaptedProjection::LoRA(layer) = projection { entries.push((attention_path(target.layer, target.role), layer)); }
    });
    stack.visit_branch_projections(|target, projection| {
        if let AdaptedProjection::LoRA(layer) = projection { entries.push((branch_path(target.layer, target.role), layer)); }
    });
    entries
}

fn model_candidates<B: Backend, F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
    model: &MhcResidualModel<B, AdaptedProjection<B>, F, AdaptedProjection<B>>) -> Candidates<'_, B> {
    let mut entries = stack_candidates(&model.stack).into_iter().map(|(path, layer)| (format!("stack.{path}"), layer)).collect::<Vec<_>>();
    if let AdaptedProjection::LoRA(layer) = &model.head.projection { entries.push(("head.projection".into(), layer)); }
    entries
}

impl<B: Backend> HybridAttentionAdapterRecord<B> {
    /// Real attention/router/shared/dense A/B only; original expert cubes and every other model leaf must be frozen.
    /// Trainable expert adapters or mHC/norm leaves require the complete native model-state record instead.
    pub fn capture_residual_stack<F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
        stack: &MhcResidualStack<B, AdaptedProjection<B>, F>, base_id: &str) -> Result<Self, RecorderError> {
        Self::capture(stack, base_id, "mhc_residual_stack", stack_candidates(stack), None)
    }

    pub fn capture_residual_model<F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
        model: &MhcResidualModel<B, AdaptedProjection<B>, F, AdaptedProjection<B>>, base_id: &str) -> Result<Self, RecorderError> {
        Self::capture(model, base_id, "mhc_residual_model", model_candidates(model), None)
    }

    /// Validate the complete exact target/schema set before replacing any A/B leaf.
    pub fn validate_residual_stack<F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
        &self, stack: &MhcResidualStack<B, AdaptedProjection<B>, F>, base_id: &str) -> Result<(), RecorderError> {
        self.validate(stack, base_id, "mhc_residual_stack", stack_candidates(stack), None)
    }

    pub fn validate_residual_model<F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
        &self, model: &MhcResidualModel<B, AdaptedProjection<B>, F, AdaptedProjection<B>>, base_id: &str) -> Result<(), RecorderError> {
        self.validate(model, base_id, "mhc_residual_model", model_candidates(model), None)
    }

    pub fn restore_residual_stack<F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
        self, stack: MhcResidualStack<B, AdaptedProjection<B>, F>, base_id: &str)
        -> Result<MhcResidualStack<B, AdaptedProjection<B>, F::Mapped<AdaptedProjection<B>>>, RecorderError> {
        self.validate_residual_stack(&stack, base_id)?;
        let mut entries: BTreeMap<_, _> = self.entries.into_iter().collect();
        let stack = stack.try_map_attention_projections(|target, projection|
            restore_projection(projection, &attention_path(target.layer, target.role), &mut entries, base_id))?;
        let stack = stack.try_map_branch_projections(|target, projection|
            restore_projection(projection, &branch_path(target.layer, target.role), &mut entries, base_id))?;
        if !entries.is_empty() { return Err(RecorderError::Unknown("Unconsumed mHC residual adapter targets".into())); }
        Ok(stack)
    }

    /// Reuse original A/B IDs, native dtype restoration and per-leaf save/load mappers without serializing any base.
    pub fn restore_residual_model<F: MhcBranchProjectionMap<B, Projection = AdaptedProjection<B>>>(
        self, model: MhcResidualModel<B, AdaptedProjection<B>, F, AdaptedProjection<B>>, base_id: &str)
        -> Result<MhcResidualModel<B, AdaptedProjection<B>, F::Mapped<AdaptedProjection<B>>, AdaptedProjection<B>>, RecorderError> {
        self.validate_residual_model(&model, base_id)?;
        let mut entries: BTreeMap<_, _> = self.entries.into_iter().collect();
        let model = model.try_map_attention_projections(|target, projection|
            restore_projection(projection, &format!("stack.{}", attention_path(target.layer, target.role)), &mut entries, base_id))?;
        let model = model.try_map_branch_projections(|target, projection|
            restore_projection(projection, &format!("stack.{}", branch_path(target.layer, target.role)), &mut entries, base_id))?;
        let model = model.try_map_head(|projection| restore_projection(projection, "head.projection", &mut entries, base_id))?;
        if !entries.is_empty() { return Err(RecorderError::Unknown("Unconsumed mHC residual model adapter targets".into())); }
        Ok(model)
    }
}
