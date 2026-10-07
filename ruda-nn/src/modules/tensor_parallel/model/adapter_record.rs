use super::*;
use alloc::{collections::BTreeMap,format,string::String,vec::Vec};
use crate::{Linear,transformer::{AdaptedProjection,StackAdapterRecord,TransformerHeadAdapterRecord}};
use super::super::VocabParallelHeadAdapterRecord;
use ruda_model::{module::{ModuleVisitor,Param,ParamId},record::{PrecisionSettings,Record,Recorder,RecorderError}};

type VocabularyPlacement = (Vec<usize>,usize,usize);

fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid parallel model adapter record: {reason}"))}

fn placement<B:Backend>(model:&TensorParallelTransformerModel<B>,input:&VocabParallelLossLayout,input_rank:usize,
    output:&VocabParallelLossLayout,output_rank:usize) -> Result<(VocabularyPlacement,VocabularyPlacement),RecorderError> {
    if input_rank >= input.world_size() || output_rank >= output.world_size() {return Err(invalid("actual vocabulary rank is outside its layout"));}
    let token = &model.embeddings.token;
    if token.vocabulary_size != input.vocabulary_size() || token.vocabulary_start != input.interval(input_rank).start
        || token.local.weight.val().dims()[0] != input.interval(input_rank).len() || model.head.local_classes() != output.interval(output_rank).len() {
        return Err(invalid("actual input/output vocabulary placement differs"));
    }
    let entry = |layout:&VocabParallelLossLayout,rank|((0..layout.world_size()).map(|owner|layout.interval(owner).len()).collect(),layout.vocabulary_size(),rank);
    Ok((entry(input,input_rank),entry(output,output_rank)))
}

fn frozen<B:Backend,M:Module<B>>(module:&M) -> Result<(),RecorderError> {
    struct Inspect {trainable:bool}
    impl<B:Backend> ModuleVisitor<B> for Inspect {
        fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {self.trainable |= param.val().is_require_grad();}
    }
    let mut inspect = Inspect {trainable:false};module.visit(&mut inspect);
    if inspect.trainable {Err(invalid("omitted trainable input/norm/dense state requires a full model record"))} else {Ok(())}
}

fn adapted_backbone<B:Backend>(model:&TensorParallelTransformerModel<B>) -> bool {
    model.backbone.layers.iter().any(|layer|matches!(layer,TensorParallelAdaptedStackLayer::Adapted(_)))
}

fn each_adapter<B:Backend,F>(model:&TensorParallelTransformerModel<B>,mut visit:F)
    where F:FnMut(&Linear<B>) {
    let mut projection = |projection:&AdaptedProjection<B>| {
        if let AdaptedProjection::LoRA(layer) = projection {visit(&layer.adapter_a);visit(&layer.adapter_b);}
    };
    for layer in &model.backbone.layers {
        if let TensorParallelAdaptedStackLayer::Adapted(block) = layer {
            let attention = &block.attention.local;let feed = &block.feed_forward.local;
            for candidate in [&attention.query,&attention.key,&attention.value,&attention.output,&feed.up,&feed.down] {projection(candidate);}
            if let Some(gate) = &feed.gate {projection(gate);}
        }
    }
    match &model.head {
        TensorParallelOutputHead::AdaptedLinear(head)=>{visit(&head.local.projection.adapter_a);visit(&head.local.projection.adapter_b);},
        TensorParallelOutputHead::AdaptedVocabulary(head)=>{visit(&head.projection.adapter_a);visit(&head.projection.adapter_b);},
        _=>{},
    }
}

fn aliases<B:Backend>(model:&TensorParallelTransformerModel<B>) -> Result<Vec<usize>,RecorderError> {
    let mut seen:BTreeMap<(ParamId,bool),(usize,Tensor<B,2>)> = BTreeMap::new();
    let mut aliases = Vec::new();let mut error = None;
    each_adapter(model,|adapter| {
        let value = adapter.weight.val();let key = (adapter.weight.id,value.is_require_grad());let index = aliases.len();
        if let Some((canonical,previous)) = seen.get(&key) {
            if previous.dims() != value.dims() || previous.dtype() != value.dtype() || previous.device() != value.device() {
                error = Some(invalid("shared adapter IDs have incompatible geometry/storage/device"));
            }
            aliases.push(*canonical);
        } else {seen.insert(key,(index,value));aliases.push(index);}
    });
    if let Some(error) = error {Err(error)} else {Ok(aliases)}
}

fn rejoin_adapters<B:Backend>(mut model:TensorParallelTransformerModel<B>,expected:&[usize]) -> Result<TensorParallelTransformerModel<B>,RecorderError> {
    if aliases(&model)?.as_slice() != expected {return Err(invalid("restored adapter identities no longer match original parameter sharing"));}
    let mut values = Vec::new();each_adapter(&model,|adapter|values.push(adapter.weight.val()));
    let mut index = 0usize;
    let mut join = |adapter:&mut Linear<B>| {
        let value = values[expected[index]].clone();adapter.weight = adapter.weight.clone().map(|_|value);index += 1;
    };
    let mut projection = |projection:&mut AdaptedProjection<B>| {
        if let AdaptedProjection::LoRA(layer) = projection {join(&mut layer.adapter_a);join(&mut layer.adapter_b);}
    };
    for layer in &mut model.backbone.layers {
        if let TensorParallelAdaptedStackLayer::Adapted(block) = layer {
            let attention = &mut block.attention.local;let feed = &mut block.feed_forward.local;
            for candidate in [&mut attention.query,&mut attention.key,&mut attention.value,&mut attention.output,&mut feed.up,&mut feed.down] {projection(candidate);}
            if let Some(gate) = &mut feed.gate {projection(gate);}
        }
    }
    match &mut model.head {
        TensorParallelOutputHead::AdaptedLinear(head)=>{join(&mut head.local.projection.adapter_a);join(&mut head.local.projection.adapter_b);},
        TensorParallelOutputHead::AdaptedVocabulary(head)=>{join(&mut head.projection.adapter_a);join(&mut head.projection.adapter_b);},
        _=>{},
    }
    Ok(model)
}

/// Complete actual model A/B-only record with exact physical vocabulary placement.
/// Native layer/head records retain schemas, original adapter IDs/dtypes and target paths.
/// The caller's partition identity binds actual backbone head/FFN placement and forward configuration.
pub struct TensorParallelModelAdapterRecord<B:Backend> {
    version:u32,
    base_id:String,
    partition_id:String,
    input:VocabularyPlacement,
    output:VocabularyPlacement,
    aliases:Vec<usize>,
    backbone:Option<StackAdapterRecord<B>>,
    linear_head:Option<TransformerHeadAdapterRecord<B>>,
    vocabulary_head:Option<VocabParallelHeadAdapterRecord<B>>,
}

impl<B:Backend> Record<B> for TensorParallelModelAdapterRecord<B> {
    type Item<S:PrecisionSettings> = (u32,String,String,VocabularyPlacement,VocabularyPlacement,Vec<usize>,
        Option<<StackAdapterRecord<B> as Record<B>>::Item<S>>,Option<<TransformerHeadAdapterRecord<B> as Record<B>>::Item<S>>,
        Option<<VocabParallelHeadAdapterRecord<B> as Record<B>>::Item<S>>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        (self.version,self.base_id,self.partition_id,self.input,self.output,self.aliases,self.backbone.map(|record|record.into_item::<S>()),
            self.linear_head.map(|record|record.into_item::<S>()),self.vocabulary_head.map(|record|record.into_item::<S>()))
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,base_id:item.1,partition_id:item.2,input:item.3,output:item.4,aliases:item.5,
            backbone:item.6.map(|record|StackAdapterRecord::<B>::from_item::<S>(record,device)),
            linear_head:item.7.map(|record|TransformerHeadAdapterRecord::<B>::from_item::<S>(record,device)),
            vocabulary_head:item.8.map(|record|VocabParallelHeadAdapterRecord::<B>::from_item::<S>(record,device))}
    }
}

impl<B:Backend> TensorParallelModelAdapterRecord<B> {
    /// Capture all actual selected A/B modules without retaining frozen input/base/norm tensors.
    /// Nothing is frozen automatically: any omitted trainable parameter rejects adapter-only capture.
    pub fn capture(model:&TensorParallelTransformerModel<B>,base_id:&str,partition_id:&str,input:&VocabParallelLossLayout,input_rank:usize,
        output:&VocabParallelLossLayout,output_rank:usize) -> Result<Self,RecorderError> {
        if base_id.is_empty() || partition_id.is_empty() {return Err(invalid("complete frozen base and backbone partition identities are required"));}
        let (input,output_placement) = placement(model,input,input_rank,output,output_rank)?;
        frozen(&model.embeddings)?;frozen(&model.final_normalization)?;
        let backbone = if adapted_backbone(model) {Some(model.backbone.adapter_record(base_id)?)} else {frozen(&model.backbone)?;None};
        let (linear_head,vocabulary_head) = match &model.head {
            TensorParallelOutputHead::AdaptedLinear(head)=>(Some(head.adapter_record(base_id)?),None),
            TensorParallelOutputHead::AdaptedVocabulary(head)=>(None,Some(head.adapter_record(output,output_rank,base_id)?)),
            _=>{frozen(&model.head)?;(None,None)},
        };
        if backbone.is_none() && linear_head.is_none() && vocabulary_head.is_none() {return Err(invalid("model has no actual adapters"));}
        Ok(Self {version:1,base_id:base_id.into(),partition_id:partition_id.into(),input,output:output_placement,aliases:aliases(model)?,
            backbone,linear_head,vocabulary_head})
    }

    /// Check every native target/base/placement contract before replacing any adapter values.
    /// Sharing is compared by actual alias topology, not session-specific freshly initialized IDs.
    pub fn validate_for(&self,model:&TensorParallelTransformerModel<B>,base_id:&str,partition_id:&str,input:&VocabParallelLossLayout,input_rank:usize,
        output:&VocabParallelLossLayout,output_rank:usize) -> Result<(),RecorderError> {
        if self.version != 1 || base_id.is_empty() || partition_id.is_empty() || self.base_id != base_id || self.partition_id != partition_id {
            return Err(invalid("format or complete original base/partition identity differs"));
        }
        let actual = placement(model,input,input_rank,output,output_rank)?;
        if self.input != actual.0 || self.output != actual.1 {return Err(invalid("physical input/output vocabulary rank/layout differs"));}
        frozen(&model.embeddings)?;frozen(&model.final_normalization)?;
        match (&self.backbone,adapted_backbone(model)) {
            (Some(record),true)=>record.validate_for(&model.backbone.clone().into_local_stack(),base_id)?,
            (None,false)=>frozen(&model.backbone)?,
            _=>return Err(invalid("actual backbone adapter selection differs")),
        }
        match (&model.head,&self.linear_head,&self.vocabulary_head) {
            (TensorParallelOutputHead::AdaptedLinear(head),Some(record),None)=>record.validate_for(&head.local,base_id)?,
            (TensorParallelOutputHead::AdaptedVocabulary(head),None,Some(record))=>record.validate_for(head,output,output_rank,base_id)?,
            (TensorParallelOutputHead::Linear(_)|TensorParallelOutputHead::Vocabulary(_),None,None)=>frozen(&model.head)?,
            _=>return Err(invalid("actual output-head kind or adapter selection differs")),
        }
        if self.aliases.is_empty() || self.aliases != aliases(model)? {return Err(invalid("actual adapter parameter-sharing topology differs"));}
        Ok(())
    }

    /// Native adapter-only serialization; recorder precision still controls stored floating values.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}

    /// Load native actual A/B state onto the explicit device, without constructing a full base.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}

    /// Restore original A/B state and shared adapter leaves, retaining all frozen model components.
    /// The original embedding/head base tie is untouched; optimizer/gradient state is restored separately.
    pub fn restore_into(self,mut model:TensorParallelTransformerModel<B>,base_id:&str,partition_id:&str,input:&VocabParallelLossLayout,input_rank:usize,
        output:&VocabParallelLossLayout,output_rank:usize) -> Result<TensorParallelTransformerModel<B>,RecorderError> {
        self.validate_for(&model,base_id,partition_id,input,input_rank,output,output_rank)?;
        if let Some(record) = self.backbone {model.backbone = model.backbone.restore_adapter_record(record,base_id)?;}
        model.head = match (model.head,self.linear_head,self.vocabulary_head) {
            (TensorParallelOutputHead::AdaptedLinear(head),Some(record),None)=>TensorParallelOutputHead::AdaptedLinear(head.restore_adapter(record,base_id)?),
            (TensorParallelOutputHead::AdaptedVocabulary(head),None,Some(record))=>TensorParallelOutputHead::AdaptedVocabulary(record.restore_into(head,output,output_rank,base_id)?),
            (head,None,None)=>head,
            _=>return Err(invalid("unconsumed native head adapter state")),
        };
        rejoin_adapters(model,&self.aliases)
    }
}

impl<B:Backend> TensorParallelTransformerModel<B> {
    /// Capture the complete local model's native A/B-only state at an actual caller-owned boundary.
    pub fn adapter_record(&self,base_id:&str,partition_id:&str,input:&VocabParallelLossLayout,input_rank:usize,
        output:&VocabParallelLossLayout,output_rank:usize) -> Result<TensorParallelModelAdapterRecord<B>,RecorderError> {
        TensorParallelModelAdapterRecord::capture(self,base_id,partition_id,input,input_rank,output,output_rank)
    }

    /// Restore the exact original selected adapters without rebuilding input/base/norm modules.
    pub fn restore_adapter_record(self,record:TensorParallelModelAdapterRecord<B>,base_id:&str,partition_id:&str,input:&VocabParallelLossLayout,
        input_rank:usize,output:&VocabParallelLossLayout,output_rank:usize) -> Result<Self,RecorderError> {
        record.restore_into(self,base_id,partition_id,input,input_rank,output,output_rank)
    }
}
