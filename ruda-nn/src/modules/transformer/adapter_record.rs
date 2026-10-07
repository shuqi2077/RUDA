use alloc::{collections::{BTreeMap,BTreeSet},format,string::String,vec,vec::Vec};
use ruda_model::{
    module::{Module,ModuleVisitor,Param,ParamId},
    record::{PrecisionSettings,Record,Recorder,RecorderError},
    tensor::{Tensor,backend::Backend},
};
use crate::{LoRALinear,LoRAAdapterRecord};
use super::{AdaptedProjection,AdaptedTransformerBlock};

/// A/B-only state of every selected projection in one actual native block.
/// Exact target paths bind equal-shaped query/key or gate/up adapters separately.
/// Caller base_id identifies the complete frozen architecture/weights/configuration.
pub struct TransformerAdapterRecord<B: Backend> {
    version: u32,
    base_id: String,
    entries: Vec<(String,LoRAAdapterRecord<B>)>,
}

impl<B: Backend> Record<B> for TransformerAdapterRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,Vec<(String,<LoRAAdapterRecord<B> as Record<B>>::Item<S>)>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.base_id,self.entries.into_iter().map(|(name,record)|(name,record.into_item::<S>())).collect())
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,base_id:item.1,entries:item.2.into_iter().map(|(name,record)|
            (name,LoRAAdapterRecord::<B>::from_item::<S>(record,device))).collect()}
    }
}

fn invalid(reason: &str) -> RecorderError {
    RecorderError::Unknown(format!("Invalid native transformer adapter record: {reason}"))
}

fn projections<B: Backend>(block: &AdaptedTransformerBlock<B>) -> Vec<(&'static str,&LoRALinear<B>)> {
    let mut result = Vec::new();
    let candidates = vec![
        ("attention.query",Some(&block.attention.query)),("attention.key",Some(&block.attention.key)),
        ("attention.value",Some(&block.attention.value)),("attention.output",Some(&block.attention.output)),
        ("feed_forward.up",Some(&block.feed_forward.up)),("feed_forward.gate",block.feed_forward.gate.as_ref()),
        ("feed_forward.down",Some(&block.feed_forward.down)),
    ];
    for (name,projection) in candidates {
        if let Some(AdaptedProjection::LoRA(layer)) = projection { result.push((name,layer)); }
    }
    result
}

struct AdapterOnly {
    adapter_ids: BTreeSet<ParamId>,
    unknown_trainable: bool,
}
impl<B: Backend> ModuleVisitor<B> for AdapterOnly {
    fn visit_float<const D: usize>(&mut self,param: &Param<Tensor<B,D>>) {
        if param.val().is_require_grad() && !self.adapter_ids.contains(&param.id) { self.unknown_trainable = true; }
    }
}

fn check_adapter_only<B: Backend>(block: &AdaptedTransformerBlock<B>) -> Result<(),RecorderError> {
    let mut visitor = AdapterOnly {adapter_ids:BTreeSet::new(),unknown_trainable:false};
    for (_,layer) in projections(block) {
        visitor.adapter_ids.insert(layer.adapter_a.weight.id);
        visitor.adapter_ids.insert(layer.adapter_b.weight.id);
    }
    block.visit(&mut visitor);
    if visitor.unknown_trainable { return Err(invalid("trainable non-adapter parameters require a full model checkpoint")); }
    Ok(())
}

impl<B: Backend> TransformerAdapterRecord<B> {
    /// Capture all actual selected A/B tensors, rejecting omitted trainable state.
    /// Frozen base, norms and activation weights are not copied or serialized.
    pub fn capture(block: &AdaptedTransformerBlock<B>,base_id: &str) -> Result<Self,RecorderError> {
        check_adapter_only(block)?;
        let mut entries = Vec::new();
        for (name,layer) in projections(block) { entries.push((name.into(),layer.adapter_record(base_id)?)); }
        if entries.is_empty() { return Err(invalid("block has no actual adapters")); }
        Ok(Self {version:1,base_id:base_id.into(),entries})
    }

    /// Exact stored projection paths in capture order, without tensor readback.
    pub fn targets(&self) -> impl Iterator<Item=&str> { self.entries.iter().map(|(name,_)|name.as_str()) }

    /// Validate all projection contracts before changing any adapter values.
    pub fn validate_for(&self,block: &AdaptedTransformerBlock<B>,base_id: &str) -> Result<(),RecorderError> {
        if self.version != 1 || self.base_id != base_id || base_id.is_empty() {
            return Err(invalid("format version or complete frozen base identity differs"));
        }
        check_adapter_only(block)?;
        let expected = projections(block);
        if self.entries.len() != expected.len() || expected.is_empty() { return Err(invalid("selected projection set differs")); }
        let mut seen = BTreeSet::new();
        for (name,record) in &self.entries {
            if !seen.insert(name.as_str()) { return Err(invalid("duplicate adapter target")); }
            let (_,layer) = expected.iter().find(|(path,_)|*path == name)
                .ok_or_else(||invalid("unknown or no-longer-adapted projection"))?;
            record.schema.validate_for(layer,base_id)?;
        }
        Ok(())
    }

    /// Write only actual adapters and target/base metadata through a native recorder.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Read the stored A/B tensors on the selected device, without creating a base.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore original adapter IDs/dtypes into the actual prepared block.
    /// Every dense projection/norm/activation and its base parameter IDs remain intact.
    pub fn restore_into(self,mut block: AdaptedTransformerBlock<B>,base_id: &str)
        -> Result<AdaptedTransformerBlock<B>,RecorderError> {
        self.validate_for(&block,base_id)?;
        let mut entries: BTreeMap<_,_> = self.entries.into_iter().collect();
        block.attention.query = restore_projection(block.attention.query,"attention.query",&mut entries,base_id)?;
        block.attention.key = restore_projection(block.attention.key,"attention.key",&mut entries,base_id)?;
        block.attention.value = restore_projection(block.attention.value,"attention.value",&mut entries,base_id)?;
        block.attention.output = restore_projection(block.attention.output,"attention.output",&mut entries,base_id)?;
        block.feed_forward.up = restore_projection(block.feed_forward.up,"feed_forward.up",&mut entries,base_id)?;
        block.feed_forward.gate = block.feed_forward.gate.map(|gate|
            restore_projection(gate,"feed_forward.gate",&mut entries,base_id)).transpose()?;
        block.feed_forward.down = restore_projection(block.feed_forward.down,"feed_forward.down",&mut entries,base_id)?;
        if !entries.is_empty() { return Err(invalid("unconsumed adapter targets")); }
        Ok(block)
    }
}

pub(super) fn restore_projection<B: Backend>(projection: AdaptedProjection<B>,path: &str,
    entries: &mut BTreeMap<String,LoRAAdapterRecord<B>>,base_id: &str) -> Result<AdaptedProjection<B>,RecorderError> {
    match projection {
        AdaptedProjection::Dense(layer) => Ok(AdaptedProjection::Dense(layer)),
        AdaptedProjection::LoRA(layer) => {
            let record = entries.remove(path).ok_or_else(||invalid("missing projection record"))?;
            Ok(AdaptedProjection::LoRA(record.restore_into(layer,base_id)?))
        }
    }
}

impl<B: Backend> AdaptedTransformerBlock<B> {
    /// Export only actual A/B weights, with explicit complete frozen-model identity.
    pub fn adapter_record(&self,base_id: &str) -> Result<TransformerAdapterRecord<B>,RecorderError> {
        TransformerAdapterRecord::capture(self,base_id)
    }
}
