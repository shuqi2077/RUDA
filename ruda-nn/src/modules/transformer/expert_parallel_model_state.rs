use alloc::{collections::{BTreeMap,BTreeSet},format,string::String,vec::Vec};
use ruda_model::{module::{Module,ModuleVisitor,ModuleMapper,Param,ParamId},record::{Record,PrecisionSettings,Recorder,RecorderError},
    serde::{Serialize,Deserialize},tensor::{Tensor,Int,Bool,DType,TensorData,TensorPrimitive,read_sync,backend::Backend}};
use crate::expert_parallel::ExpertParallelGeometry;
use super::{TransformerProjectionShape,ExpertParallelTransformerModel,ExpertParallelTransformerLayer,ExpertAdapterOwnershipEntry};

/// Actual source parameter kind, independent of the recorder's default integer/float element type.
#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub enum ExpertModelParameterKind {Float,Integer,Boolean}
/// Exact source path, original identity, storage and flag for one actual model parameter occurrence.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub struct ExpertModelParameterBinding {
    /// Actual module field/container path; no model-family name mapping is inferred.
    pub path:Vec<String>,
    /// Original saved parameter identity, including repeated tied occurrences.
    pub id:u64,
    /// Actual original logical axes, including empty expert dimensions.
    pub shape:Vec<usize>,
    /// Original native source dtype, not serialization work precision.
    pub dtype:DType,
    /// Actual floating/integer/boolean parameter kind.
    pub kind:ExpertModelParameterKind,
    /// Actual original floating training flag; integer/boolean parameters are untracked.
    pub trainable:bool,
}
type BindingKey=(u64,ExpertModelParameterKind,bool);
fn key(binding:&ExpertModelParameterBinding) -> BindingKey {(binding.id,binding.kind,binding.trainable)}
fn invalid(reason:&str) -> RecorderError {RecorderError::Unknown(format!("Invalid expert-owned model state: {reason}"))}
enum StoredValue<B:Backend> {
    Float(B::FloatTensorPrimitive),Integer(B::IntTensorPrimitive),Boolean(B::BoolTensorPrimitive),
}
impl<B:Backend> StoredValue<B> {
    fn into_data(self) -> TensorData {
        read_sync(self.into_data_async()).expect("native expert model state readback failed")
    }
    async fn into_data_async(self) -> Result<TensorData,RecorderError> {
        (match self {
            Self::Float(value)=>B::float_into_data(value).await,
            Self::Integer(value)=>B::int_into_data(value).await,
            Self::Boolean(value)=>B::bool_into_data(value).await,
        }).map_err(|error|invalid(&format!("native parameter readback failed: {error:?}")))
    }
    fn from_data(kind:ExpertModelParameterKind,data:TensorData,device:&B::Device) -> Self {match kind {
        ExpertModelParameterKind::Float=>Self::Float(B::float_from_data(data,device)),
        ExpertModelParameterKind::Integer=>Self::Integer(B::int_from_data(data,device)),
        ExpertModelParameterKind::Boolean=>Self::Boolean(B::bool_from_data(data,device)),
    }}
}
struct Capture<B:Backend> {
    path:Vec<String>,bindings:Vec<ExpertModelParameterBinding>,values:Vec<StoredValue<B>>,with_values:bool,
    aliases:BTreeMap<BindingKey,(Vec<usize>,DType,B::Device)>,paths:BTreeSet<Vec<String>>,error:Option<RecorderError>,
}
impl<B:Backend> Capture<B> {
    fn new(with_values:bool) -> Self {Self {path:Vec::new(),bindings:Vec::new(),values:Vec::new(),with_values,
        aliases:BTreeMap::new(),paths:BTreeSet::new(),error:None}}
    fn binding(&mut self,id:ParamId,shape:Vec<usize>,dtype:DType,kind:ExpertModelParameterKind,trainable:bool,device:B::Device) {
        let binding=ExpertModelParameterBinding {path:self.path.clone(),id:id.val(),shape:shape.clone(),dtype,kind,trainable};
        if !self.paths.insert(binding.path.clone()) {self.error=Some(invalid("duplicate native module parameter path"));}
        if let Some((previous,storage,resident))=self.aliases.get(&key(&binding)) {
            if previous!=&shape || storage!=&dtype || resident!=&device {self.error=Some(invalid("tied source parameter shape/storage/device differs"));}
        } else {self.aliases.insert(key(&binding),(shape,dtype,device));}
        self.bindings.push(binding);
    }
}
impl<B:Backend> ModuleVisitor<B> for Capture<B> {
    fn enter_module(&mut self,name:&str,_kind:&str) {self.path.push(name.into());}
    fn exit_module(&mut self,_name:&str,_kind:&str) {self.path.pop();}
    fn visit_float<const D:usize>(&mut self,parameter:&Param<Tensor<B,D>>) {
        let value=parameter.val();self.binding(parameter.id,value.dims().to_vec(),value.dtype(),ExpertModelParameterKind::Float,value.is_require_grad(),value.device());
        if self.with_values {self.values.push(StoredValue::Float(parameter.transform_for_save().val().detach().into_primitive().tensor()));}
    }
    fn visit_int<const D:usize>(&mut self,parameter:&Param<Tensor<B,D,Int>>) {
        let value=parameter.val();self.binding(parameter.id,value.dims().to_vec(),value.dtype(),ExpertModelParameterKind::Integer,false,value.device());
        if self.with_values {self.values.push(StoredValue::Integer(parameter.transform_for_save().val().into_primitive()));}
    }
    fn visit_bool<const D:usize>(&mut self,parameter:&Param<Tensor<B,D,Bool>>) {
        let value=parameter.val();self.binding(parameter.id,value.dims().to_vec(),value.dtype(),ExpertModelParameterKind::Boolean,false,value.device());
        if self.with_values {self.values.push(StoredValue::Boolean(parameter.transform_for_save().val().into_primitive()));}
    }
}
fn ownership<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>>(model:&ExpertParallelTransformerModel<B,P,E>) -> Vec<ExpertAdapterOwnershipEntry> {
    let mut result=Vec::new();for (index,layer) in model.layers.iter().enumerate() {if let ExpertParallelTransformerLayer::Parallel(block)=layer {
        result.push(ExpertAdapterOwnershipEntry {layer:index,prefix:block.routed.experts.ownership().prefix().to_vec(),rank:block.routed.experts.rank()});
    }}result
}
/// Complete actual rank-local model parameter state, with original IDs/dtypes/flags and explicit expert ownership.
/// Prepared architecture/options are bound by the caller's exact contract ID; they are not inferred or overwritten.
/// Native TensorData preserves saved payload storage, including U8 NF4 bytes and I32 AWQ words, regardless of recorder precision.
/// Optimizer, scheduler, pending gradients, data position and RNG use matching separate records at the same boundary.
pub struct ExpertParallelModelStateRecord<B:Backend> {
    version:u32,contract_id:String,layers:usize,ownership:Vec<ExpertAdapterOwnershipEntry>,
    bindings:Vec<ExpertModelParameterBinding>,values:Vec<StoredValue<B>>,
}
/// Backend-independent exact native payload archive produced by asynchronous checkpoint readback.
/// Carries actual rank-local parameters only; no gathered global weights or decoded packed bases.
#[derive(Clone,Serialize,Deserialize)]
#[serde(crate="ruda_model::serde")]
pub struct ExpertParallelModelSnapshot {
    version:u32,contract_id:String,layers:usize,ownership:Vec<ExpertAdapterOwnershipEntry>,
    bindings:Vec<ExpertModelParameterBinding>,values:Vec<(ExpertModelParameterKind,TensorData)>,
}
impl<B:Backend> Record<B> for ExpertParallelModelSnapshot {
    type Item<S:PrecisionSettings>=Self;
    fn into_item<S:PrecisionSettings>(self) -> Self {self}
    fn from_item<S:PrecisionSettings>(item:Self,_device:&B::Device) -> Self {item}
}
impl ExpertParallelModelSnapshot {
    /// Actual saved complete expert-world intervals and rank for each owned layer.
    pub fn ownership(&self) -> &[ExpertAdapterOwnershipEntry] {&self.ownership}
    /// Actual original native paths, source IDs, storage and flags, without uploading any payloads.
    pub fn bindings(&self) -> &[ExpertModelParameterBinding] {&self.bindings}
    /// Actual per-alias payload count; original parameter save/load mappers remain independent.
    pub fn parameter_occurrences(&self) -> usize {self.bindings.len()}
    /// Upload original native payload storage on an explicitly selected backend/device for checked restoration.
    pub fn into_record<B:Backend>(self,device:&B::Device) -> ExpertParallelModelStateRecord<B> {
        ExpertParallelModelStateRecord {version:self.version,contract_id:self.contract_id,layers:self.layers,ownership:self.ownership,bindings:self.bindings,
            values:self.values.into_iter().map(|(kind,data)|StoredValue::from_data(kind,data,device)).collect()}
    }
}
impl<B:Backend> Record<B> for ExpertParallelModelStateRecord<B> {
    type Item<S:PrecisionSettings>=(u32,String,usize,Vec<ExpertAdapterOwnershipEntry>,Vec<ExpertModelParameterBinding>,Vec<(ExpertModelParameterKind,TensorData)>);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {
        let values=self.bindings.iter().zip(self.values).map(|(binding,value)|(binding.kind,value.into_data())).collect();
        (self.version,self.contract_id,self.layers,self.ownership,self.bindings,values)
    }
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,contract_id:item.1,layers:item.2,ownership:item.3,bindings:item.4,
            values:item.5.into_iter().map(|(kind,data)|StoredValue::from_data(kind,data,device)).collect()}
    }
}
impl<B:Backend> ExpertParallelModelStateRecord<B> {
    /// Snapshot only actual resident owned/non-expert parameters, applying each original parameter's save mapper.
    /// Capture and serialize at one common completed boundary, without concurrent parameter updates.
    /// Does not gather global expert cubes, decode quantized bases or capture activation/collective history.
    pub fn capture<P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>>(model:&ExpertParallelTransformerModel<B,P,E>,contract_id:&str) -> Result<Self,RecorderError> {
        if contract_id.is_empty() {return Err(invalid("exact prepared architecture/operator contract ID is required"));}
        let mut capture=Capture::new(true);model.visit(&mut capture);if let Some(error)=capture.error {return Err(error);}
        Ok(Self {version:1,contract_id:contract_id.into(),layers:model.layers.len(),ownership:ownership(model),bindings:capture.bindings,values:capture.values})
    }
    /// Actual original complete expert-world prefix/rank for every owned layer.
    pub fn ownership(&self) -> &[ExpertAdapterOwnershipEntry] {&self.ownership}
    /// Exact actual native parameter paths, source identities, logical axes, storage and flags.
    pub fn bindings(&self) -> &[ExpertModelParameterBinding] {&self.bindings}
    /// Actual stored parameter occurrences, retaining separate per-alias save/load mapper payloads.
    pub fn parameter_occurrences(&self) -> usize {self.bindings.len()}
    /// Await actual native payload reads without a blocking-future requirement, retaining every original stored dtype.
    /// The resulting archive can be the caller-selected model state of ModelStateTrainingRecord.
    pub async fn into_snapshot(self) -> Result<ExpertParallelModelSnapshot,RecorderError> {
        if self.bindings.len()!=self.values.len() {return Err(invalid("native payload count differs from parameter bindings"));}
        let mut values=Vec::with_capacity(self.values.len());
        for (binding,value) in self.bindings.iter().zip(self.values) {values.push((binding.kind,value.into_data_async().await?));}
        Ok(ExpertParallelModelSnapshot {version:self.version,contract_id:self.contract_id,layers:self.layers,ownership:self.ownership,bindings:self.bindings,values})
    }
    /// Validate original complete topology and actual prepared parameter/alias contracts before replacing any values.
    /// Fresh prepared IDs may differ; saved IDs become authoritative only when restoration succeeds.
    pub fn validate_for<P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>>(&self,model:&ExpertParallelTransformerModel<B,P,E>,contract_id:&str) -> Result<(),RecorderError> {
        if self.version!=1 || contract_id.is_empty() || self.contract_id!=contract_id || self.layers!=model.layers.len()
            || self.ownership!=ownership(model) || self.bindings.len()!=self.values.len() {return Err(invalid("version, architecture contract, model layers or original expert ownership differs"));}
        let mut capture=Capture::new(false);model.visit(&mut capture);if let Some(error)=capture.error {return Err(error);}
        if capture.bindings.len()!=self.bindings.len() {return Err(invalid("actual parameter path set differs"));}
        let mut source_to_target=BTreeMap::new();let mut target_to_source=BTreeMap::new();
        for (saved,target) in self.bindings.iter().zip(capture.bindings) {
            if saved.path!=target.path || saved.shape!=target.shape || saved.dtype!=target.dtype || saved.kind!=target.kind || saved.trainable!=target.trainable {
                return Err(invalid("actual parameter path/shape/storage/training contract differs"));}
            let source=key(saved);let destination=key(&target);
            if source_to_target.insert(source,destination).is_some_and(|previous|previous!=destination)
                || target_to_source.insert(destination,source).is_some_and(|previous|previous!=source) {return Err(invalid("original shared/frozen parameter alias topology differs"));}
        }
        Ok(())
    }
    /// Save original native payload storage with an existing recorder, without widening packed bytes or narrowing floating values.
    pub fn save<R:Recorder<B>>(self,recorder:&R,args:R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load actual native parameter payloads on the selected device; no topology or missing source weights are fabricated.
    pub fn load<R:Recorder<B>>(recorder:&R,args:R::LoadArgs,device:&B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}
    /// Restore original IDs/native storage and one canonical resumed leaf per source ID/flag, preserving per-alias mappers.
    /// Can serve as the model state of ruda-optim's ModelStateTrainingRecord without an additional dependency.
    pub fn restore_into<P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>>(self,model:ExpertParallelTransformerModel<B,P,E>,contract_id:&str)
        -> Result<ExpertParallelTransformerModel<B,P,E>,RecorderError> {
        self.validate_for(&model,contract_id)?;
        let entries=self.bindings.into_iter().zip(self.values).map(|(binding,value)|(binding.path.clone(),(binding,value))).collect();
        let mut restore=Restore {path:Vec::new(),entries,canonical:BTreeMap::new(),error:None,seen:BTreeSet::new()};let model=model.map(&mut restore);
        if let Some(error)=restore.error {return Err(error);}
        if restore.seen.len()!=restore.entries.len() {return Err(invalid("model mapper omitted an actual parameter path"));}Ok(model)
    }
}
struct Restore<B:Backend> {
    path:Vec<String>,entries:BTreeMap<Vec<String>,(ExpertModelParameterBinding,StoredValue<B>)>,
    canonical:BTreeMap<BindingKey,StoredValue<B>>,error:Option<RecorderError>,seen:BTreeSet<Vec<String>>,
}
impl<B:Backend> Restore<B> {
    fn binding(&mut self,kind:ExpertModelParameterKind) -> Option<ExpertModelParameterBinding> {
        let Some((binding,_))=self.entries.get(&self.path) else {self.error=Some(invalid("missing validated parameter path"));return None;};
        if binding.kind!=kind {self.error=Some(invalid("parameter mapper changed the validated kind"));return None;}
        self.seen.insert(self.path.clone());Some(binding.clone())
    }
}
impl<B:Backend> ModuleMapper<B> for Restore<B> {
    fn enter_module(&mut self,name:&str,_kind:&str) {self.path.push(name.into());}
    fn exit_module(&mut self,_name:&str,_kind:&str) {self.path.pop();}
    fn map_float<const D:usize>(&mut self,parameter:Param<Tensor<B,D>>) -> Param<Tensor<B,D>> {
        let Some(binding)=self.binding(ExpertModelParameterKind::Float) else {return parameter;};
        let StoredValue::Float(value)=&self.entries[&self.path].1 else {self.error=Some(invalid("saved floating payload kind differs"));return parameter;};
        let loaded=parameter.transform_for_load(Tensor::from_primitive(TensorPrimitive::Float(value.clone())),ParamId::from(binding.id));
        let (id,value,mapper)=loaded.consume();let value=value.detach().set_require_grad(binding.trainable);
        if value.dims().to_vec()!=binding.shape || value.dtype()!=binding.dtype || value.is_require_grad()!=binding.trainable {
            self.error=Some(invalid("floating load mapper changed source shape/storage/flags"));return Param::from_mapped_value(id,value,mapper);}
        let primitive=match self.canonical.get(&key(&binding)) {
            Some(StoredValue::Float(previous))=>previous.clone(),Some(_)=>unreachable!("validated floating canonical kind"),
            None=>{let primitive=value.into_primitive().tensor();self.canonical.insert(key(&binding),StoredValue::Float(primitive.clone()));primitive},
        };Param::from_mapped_value(id,Tensor::from_primitive(TensorPrimitive::Float(primitive)),mapper)
    }
    fn map_int<const D:usize>(&mut self,parameter:Param<Tensor<B,D,Int>>) -> Param<Tensor<B,D,Int>> {
        let Some(binding)=self.binding(ExpertModelParameterKind::Integer) else {return parameter;};
        let StoredValue::Integer(value)=&self.entries[&self.path].1 else {self.error=Some(invalid("saved integer payload kind differs"));return parameter;};
        let loaded=parameter.transform_for_load(Tensor::from_primitive(value.clone()),ParamId::from(binding.id));let (id,value,mapper)=loaded.consume();
        if value.dims().to_vec()!=binding.shape || value.dtype()!=binding.dtype {self.error=Some(invalid("integer load mapper changed source shape/storage"));return Param::from_mapped_value(id,value,mapper);}
        let primitive=match self.canonical.get(&key(&binding)) {
            Some(StoredValue::Integer(previous))=>previous.clone(),Some(_)=>unreachable!("validated integer canonical kind"),
            None=>{let primitive=value.into_primitive();self.canonical.insert(key(&binding),StoredValue::Integer(primitive.clone()));primitive},
        };Param::from_mapped_value(id,Tensor::from_primitive(primitive),mapper)
    }
    fn map_bool<const D:usize>(&mut self,parameter:Param<Tensor<B,D,Bool>>) -> Param<Tensor<B,D,Bool>> {
        let Some(binding)=self.binding(ExpertModelParameterKind::Boolean) else {return parameter;};
        let StoredValue::Boolean(value)=&self.entries[&self.path].1 else {self.error=Some(invalid("saved boolean payload kind differs"));return parameter;};
        let loaded=parameter.transform_for_load(Tensor::from_primitive(value.clone()),ParamId::from(binding.id));let (id,value,mapper)=loaded.consume();
        if value.dims().to_vec()!=binding.shape || value.dtype()!=binding.dtype {self.error=Some(invalid("boolean load mapper changed source shape/storage"));return Param::from_mapped_value(id,value,mapper);}
        let primitive=match self.canonical.get(&key(&binding)) {
            Some(StoredValue::Boolean(previous))=>previous.clone(),Some(_)=>unreachable!("validated boolean canonical kind"),
            None=>{let primitive=value.into_primitive();self.canonical.insert(key(&binding),StoredValue::Boolean(primitive.clone()));primitive},
        };Param::from_mapped_value(id,Tensor::from_primitive(primitive),mapper)
    }
}
impl<B:Backend,P:TransformerProjectionShape<B>,E:ExpertParallelGeometry<B>> ExpertParallelTransformerModel<B,P,E> {
    /// Export complete actual local parameter state with native storage and original expert ownership.
    pub fn expert_model_state_record(&self,contract_id:&str) -> Result<ExpertParallelModelStateRecord<B>,RecorderError> {
        ExpertParallelModelStateRecord::capture(self,contract_id)
    }
}
