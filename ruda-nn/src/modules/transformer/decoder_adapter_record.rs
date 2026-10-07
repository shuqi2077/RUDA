use alloc::{collections::{BTreeMap,BTreeSet},format,string::String,vec::Vec};
use ruda_model::{module::{Module,ModuleVisitor,Param,ParamId},
    record::{PrecisionSettings,Record,Recorder,RecorderError},tensor::{Tensor,backend::Backend}};
use crate::{LoRALinear,LoRAAdapterRecord};
use super::{AdaptedEncoderDecoderLayer,AdaptedEncoderDecoderStack,AdaptedStackLayer,
    DecoderCrossAttention,AdaptedProjection,AdaptedGroupedQueryAttention};
use super::adapter_record::restore_projection;

const SELF_PATHS: [&str;4] = ["self_attention.query","self_attention.key","self_attention.value","self_attention.output"];
const CROSS_PATHS: [&str;4] = ["cross_attention.query","cross_attention.key","cross_attention.value","cross_attention.output"];
const FFN_PATHS: [&str;3] = ["feed_forward.up","feed_forward.gate","feed_forward.down"];

/// Actual native encoder-decoder A/B state, without frozen base or memory payload tensors.
/// Caller base_id identifies the complete original architecture, configuration and weights.
pub struct EncoderDecoderAdapterRecord<B: Backend> {
    version: u32,
    base_id: String,
    layers: usize,
    entries: Vec<(usize,String,LoRAAdapterRecord<B>)>,
}

impl<B: Backend> Record<B> for EncoderDecoderAdapterRecord<B> {
    type Item<S: PrecisionSettings> = (u32,String,usize,Vec<(usize,String,<LoRAAdapterRecord<B> as Record<B>>::Item<S>)>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.base_id,self.layers,self.entries.into_iter().map(|(index,path,record)|
            (index,path,record.into_item::<S>())).collect())
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,device: &B::Device) -> Self {
        Self {version:item.0,base_id:item.1,layers:item.2,entries:item.3.into_iter().map(|(index,path,record)|
            (index,path,LoRAAdapterRecord::<B>::from_item::<S>(record,device))).collect()}
    }
}

fn invalid(reason: &str) -> RecorderError {
    RecorderError::Unknown(format!("Invalid native encoder-decoder adapter record: {reason}"))
}

fn push_adapter<'a,B: Backend>(result: &mut Vec<(&'static str,&'a LoRALinear<B>)>,path: &'static str,
    projection: Option<&'a AdaptedProjection<B>>) {
    if let Some(AdaptedProjection::LoRA(layer)) = projection { result.push((path,layer)); }
}

fn attention_adapters<'a,B: Backend>(result: &mut Vec<(&'static str,&'a LoRALinear<B>)>,
    attention: &'a AdaptedGroupedQueryAttention<B>,paths: [&'static str;4]) {
    push_adapter(result,paths[0],Some(&attention.query));
    push_adapter(result,paths[1],Some(&attention.key));
    push_adapter(result,paths[2],Some(&attention.value));
    push_adapter(result,paths[3],Some(&attention.output));
}

fn projections<B: Backend>(layer: &AdaptedEncoderDecoderLayer<B>) -> Vec<(&'static str,&LoRALinear<B>)> {
    let mut result = Vec::new();
    if let AdaptedStackLayer::Adapted(block) = &layer.backbone {
        attention_adapters(&mut result,&block.attention,SELF_PATHS);
        push_adapter(&mut result,FFN_PATHS[0],Some(&block.feed_forward.up));
        push_adapter(&mut result,FFN_PATHS[1],block.feed_forward.gate.as_ref());
        push_adapter(&mut result,FFN_PATHS[2],Some(&block.feed_forward.down));
    }
    if let DecoderCrossAttention::Adapted(block) = &layer.cross_attention {
        attention_adapters(&mut result,&block.attention,CROSS_PATHS);
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

fn check_adapter_only<B: Backend>(stack: &AdaptedEncoderDecoderStack<B>) -> Result<(),RecorderError> {
    let mut visitor = AdapterOnly {adapter_ids:BTreeSet::new(),unknown_trainable:false};
    for layer in &stack.layers {
        for (_,adapter) in projections(layer) {
            visitor.adapter_ids.insert(adapter.adapter_a.weight.id);
            visitor.adapter_ids.insert(adapter.adapter_b.weight.id);
        }
    }
    stack.visit(&mut visitor);
    if visitor.unknown_trainable { return Err(invalid("trainable non-adapter parameters require a full model checkpoint")); }
    Ok(())
}

impl<B: Backend> EncoderDecoderAdapterRecord<B> {
    /// Capture every actual adapter, including independent self/cross targets and ranks.
    /// Reject omitted trainable dense layers, norms, biases and activation parameters.
    pub fn capture(stack: &AdaptedEncoderDecoderStack<B>,base_id: &str) -> Result<Self,RecorderError> {
        if base_id.is_empty() { return Err(invalid("complete frozen-base identity must be supplied explicitly")); }
        check_adapter_only(stack)?;
        let mut entries = Vec::new();
        for (index,layer) in stack.layers.iter().enumerate() {
            for (path,adapter) in projections(layer) { entries.push((index,path.into(),adapter.adapter_record(base_id)?)); }
        }
        if entries.is_empty() { return Err(invalid("encoder-decoder stack has no actual adapters")); }
        Ok(Self {version:1,base_id:base_id.into(),layers:stack.layers.len(),entries})
    }

    /// Exact original layer indices and stage-qualified native projection paths.
    pub fn targets(&self) -> impl Iterator<Item=(usize,&str)> {
        self.entries.iter().map(|(index,path,_)|(*index,path.as_str()))
    }

    /// Validate every actual index/target/base contract before replacing any parameters.
    pub fn validate_for(&self,stack: &AdaptedEncoderDecoderStack<B>,base_id: &str) -> Result<(),RecorderError> {
        if self.version != 1 || self.base_id != base_id || base_id.is_empty() || self.layers != stack.layers.len() {
            return Err(invalid("version, complete frozen base or actual layer count differs"));
        }
        check_adapter_only(stack)?;
        let mut expected = BTreeMap::new();
        for (index,layer) in stack.layers.iter().enumerate() {
            for (path,adapter) in projections(layer) { expected.insert((index,path),adapter); }
        }
        if self.entries.len() != expected.len() || expected.is_empty() { return Err(invalid("actual decoder adapter target set differs")); }
        let mut seen = BTreeSet::new();
        for (index,path,record) in &self.entries {
            let target = (*index,path.as_str());
            if !seen.insert(target) { return Err(invalid("duplicated layer/projection target")); }
            let adapter = expected.get(&target).ok_or_else(||invalid("unknown or no-longer-adapted layer/projection target"))?;
            record.schema.validate_for(adapter,base_id)?;
        }
        Ok(())
    }

    /// Save only actual A/B tensors and exact decoder/base continuation metadata.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {
        recorder.record(self,args)
    }

    /// Load native adapter records onto an explicit device without reconstructing a base.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {
        recorder.load(args,device)
    }

    /// Restore exact A/B IDs/dtypes while retaining every frozen original decoder stage.
    /// Compatible optimizer/scheduler/pending gradients can then be restored through
    /// ruda-optim's ModelStateTrainingRecord using this actual model-state record.
    pub fn restore_into(self,stack: AdaptedEncoderDecoderStack<B>,base_id: &str)
        -> Result<AdaptedEncoderDecoderStack<B>,RecorderError> {
        self.validate_for(&stack,base_id)?;
        let mut records: BTreeMap<usize,BTreeMap<String,LoRAAdapterRecord<B>>> = BTreeMap::new();
        for (index,path,record) in self.entries { records.entry(index).or_default().insert(path,record); }
        let mut layers = Vec::with_capacity(stack.layers.len());
        for (index,layer) in stack.layers.into_iter().enumerate() {
            layers.push(restore_layer(layer,records.remove(&index).unwrap_or_default(),base_id)?);
        }
        if !records.is_empty() { return Err(invalid("unconsumed stored decoder layer")); }
        Ok(AdaptedEncoderDecoderStack {layers})
    }
}

fn restore_attention<B: Backend>(mut attention: AdaptedGroupedQueryAttention<B>,paths: [&str;4],
    entries: &mut BTreeMap<String,LoRAAdapterRecord<B>>,base_id: &str) -> Result<AdaptedGroupedQueryAttention<B>,RecorderError> {
    attention.query = restore_projection(attention.query,paths[0],entries,base_id)?;
    attention.key = restore_projection(attention.key,paths[1],entries,base_id)?;
    attention.value = restore_projection(attention.value,paths[2],entries,base_id)?;
    attention.output = restore_projection(attention.output,paths[3],entries,base_id)?;
    Ok(attention)
}

fn restore_layer<B: Backend>(mut layer: AdaptedEncoderDecoderLayer<B>,mut entries: BTreeMap<String,LoRAAdapterRecord<B>>,
    base_id: &str) -> Result<AdaptedEncoderDecoderLayer<B>,RecorderError> {
    layer.backbone = match layer.backbone {
        AdaptedStackLayer::Dense(block)=>AdaptedStackLayer::Dense(block),
        AdaptedStackLayer::Adapted(mut block)=>{
            block.attention = restore_attention(block.attention,SELF_PATHS,&mut entries,base_id)?;
            block.feed_forward.up = restore_projection(block.feed_forward.up,FFN_PATHS[0],&mut entries,base_id)?;
            block.feed_forward.gate = block.feed_forward.gate.map(|gate|
                restore_projection(gate,FFN_PATHS[1],&mut entries,base_id)).transpose()?;
            block.feed_forward.down = restore_projection(block.feed_forward.down,FFN_PATHS[2],&mut entries,base_id)?;
            AdaptedStackLayer::Adapted(block)
        },
    };
    layer.cross_attention = match layer.cross_attention {
        DecoderCrossAttention::Dense(block)=>DecoderCrossAttention::Dense(block),
        DecoderCrossAttention::Adapted(mut block)=>{
            block.attention = restore_attention(block.attention,CROSS_PATHS,&mut entries,base_id)?;
            DecoderCrossAttention::Adapted(block)
        },
    };
    if !entries.is_empty() { return Err(invalid("unconsumed stored decoder projection")); }
    Ok(layer)
}

impl<B: Backend> AdaptedEncoderDecoderStack<B> {
    /// Export actual self/cross/FFN adapters with an explicit complete frozen-model identity.
    pub fn adapter_record(&self,base_id: &str) -> Result<EncoderDecoderAdapterRecord<B>,RecorderError> {
        EncoderDecoderAdapterRecord::capture(self,base_id)
    }
}
