use super::{ExpertAdapterTarget,ExpertAdapterRecordBase,ExpertLoRAAdapterSchema,ExpertLoRAAdapterRecord,PackedExpertLoRA,
    FloatingExpertLoRA,AdaptedExpertProjection,FloatingExpertProjection,AdaptedPackedSwiGluExperts,AdaptedFloatingSwiGluExperts,
    SelectablePackedExperts,MixedAdaptedExperts,FrozenPackedSwiGluExperts,FrozenNf4SwiGluExperts,FrozenExpertGeometry};
use alloc::vec::Vec;
use ruda_model::{module::Param,record::RecorderError,tensor::{Tensor,backend::Backend}};

/// Borrow the actual floating or packed expert adapter without retaining a base snapshot.
pub enum ExpertAdapterProjectionRef<'a,B:Backend> {
    /// Real A/B on an original immutable NF4/AWQ projection.
    Packed(&'a PackedExpertLoRA<B>),
    /// Real A/B on an original frozen floating cube.
    Floating(&'a FloatingExpertLoRA<B>),
}
impl<'a,B:Backend> ExpertAdapterProjectionRef<'a,B> {
    /// Actual original A, then B parameter leaves and identities.
    pub fn parameters(&self) -> [&'a Param<Tensor<B,3>>;2] {match self {
        Self::Packed(layer)=>[&layer.adapter_a.weight,&layer.adapter_b.weight],
        Self::Floating(layer)=>[&layer.adapter_a.weight,&layer.adapter_b.weight]}}
    /// Capture the actual base/adapter metadata without numeric tensor readback.
    pub fn schema(&self,base_id:&str) -> Result<ExpertLoRAAdapterSchema,RecorderError> {match self {
        Self::Packed(layer)=>ExpertLoRAAdapterSchema::capture(layer,base_id),
        Self::Floating(layer)=>ExpertLoRAAdapterSchema::capture(layer,base_id)}}
    /// Match exact source configuration, including multiplier/dropout floating-point bit patterns.
    pub fn validate_schema(&self,schema:&ExpertLoRAAdapterSchema,base_id:&str) -> Result<(),RecorderError> {match self {
        Self::Packed(layer)=>schema.validate_for(layer,base_id),Self::Floating(layer)=>schema.validate_for(layer,base_id)}}
    /// Capture only this projection's actual A/B values through the existing native record.
    pub fn adapter_record(&self,base_id:&str) -> Result<ExpertLoRAAdapterRecord<B>,RecorderError> {match self {
        Self::Packed(layer)=>layer.adapter_record(base_id),Self::Floating(layer)=>layer.adapter_record(base_id)}}
}
/// Typed adapter mapper; original unselected projections never enter this mapper.
pub trait ExpertAdapterMapper<B:Backend> {
    /// Map a present gate/up/down adapter, preserving its actual source base representation.
    fn map<Base:ExpertAdapterRecordBase<B>>(&mut self,role:ExpertAdapterTarget,layer:PackedExpertLoRA<B,Base>)
        -> Result<PackedExpertLoRA<B,Base>,RecorderError>;
}
/// Actual expert adapter access, independent of floating/NF4/AWQ source representation.
pub trait ExpertAdapterProjections<B:Backend>:FrozenExpertGeometry<B> {
    /// Present actual source roles in gate/up/down order; absent adapters are not synthesized.
    fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)>;
    /// Consume only actual adapter projections, retaining unselected original matrices and their flags.
    fn map_expert_adapters<M:ExpertAdapterMapper<B>>(self,mapper:&mut M) -> Result<Self,RecorderError>;
}
macro_rules! adapter_fields {
    ($experts:ident,$projection:ident,$reference:ident) => {
        impl<B:Backend> ExpertAdapterProjections<B> for $experts<B> {
            fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {
                let mut result=Vec::new();for (role,projection) in [(ExpertAdapterTarget::Gate,&self.gate),
                    (ExpertAdapterTarget::Up,&self.up),(ExpertAdapterTarget::Down,&self.down)] {
                    if let $projection::LoRA(layer)=projection {result.push((role,ExpertAdapterProjectionRef::$reference(layer)));}}
                result
            }
            fn map_expert_adapters<M:ExpertAdapterMapper<B>>(mut self,mapper:&mut M) -> Result<Self,RecorderError> {
                self.gate=match self.gate {$projection::LoRA(layer)=>$projection::LoRA(mapper.map(ExpertAdapterTarget::Gate,layer)?),original=>original};
                self.up=match self.up {$projection::LoRA(layer)=>$projection::LoRA(mapper.map(ExpertAdapterTarget::Up,layer)?),original=>original};
                self.down=match self.down {$projection::LoRA(layer)=>$projection::LoRA(mapper.map(ExpertAdapterTarget::Down,layer)?),original=>original};
                self.validate();Ok(self)
            }
        }
    };
}
adapter_fields!(AdaptedPackedSwiGluExperts,AdaptedExpertProjection,Packed);
adapter_fields!(AdaptedFloatingSwiGluExperts,FloatingExpertProjection,Floating);
macro_rules! original_experts {
    ($experts:ident) => {
        impl<B:Backend> ExpertAdapterProjections<B> for $experts<B> {
            fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {Vec::new()}
            fn map_expert_adapters<M:ExpertAdapterMapper<B>>(self,_mapper:&mut M) -> Result<Self,RecorderError> {Ok(self)}
        }
    };
}
original_experts!(FrozenPackedSwiGluExperts);
original_experts!(FrozenNf4SwiGluExperts);
impl<B:Backend> ExpertAdapterProjections<B> for SelectablePackedExperts<B> {
    fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {match self {
        Self::Original(_)=>Vec::new(),Self::Adapted(value)=>value.expert_adapter_projections()}}
    fn map_expert_adapters<M:ExpertAdapterMapper<B>>(self,mapper:&mut M) -> Result<Self,RecorderError> {match self {
        Self::Original(value)=>Ok(Self::Original(value)),Self::Adapted(value)=>Ok(Self::Adapted(value.map_expert_adapters(mapper)?))}}
}
impl<B:Backend,E:ExpertAdapterProjections<B>> ExpertAdapterProjections<B> for MixedAdaptedExperts<B,E> {
    fn expert_adapter_projections(&self) -> Vec<(ExpertAdapterTarget,ExpertAdapterProjectionRef<'_,B>)> {match self {
        Self::Original(value)=>value.expert_adapter_projections(),Self::Packed(value)=>value.expert_adapter_projections(),
        Self::Floating(value)=>value.expert_adapter_projections()}}
    fn map_expert_adapters<M:ExpertAdapterMapper<B>>(self,mapper:&mut M) -> Result<Self,RecorderError> {match self {
        Self::Original(value)=>Ok(Self::Original(value.map_expert_adapters(mapper)?)),
        Self::Packed(value)=>Ok(Self::Packed(value.map_expert_adapters(mapper)?)),
        Self::Floating(value)=>Ok(Self::Floating(value.map_expert_adapters(mapper)?))}}
}
