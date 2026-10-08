use super::*;
use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use core::fmt;
use ruda_model::{module::ModuleVisitor,record::{Record,PrecisionSettings}};

/// Original group binding/continuation or explicit learning-rate argument error.
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum GroupedOptimizerError {
    Configuration(&'static str),
    UnknownParameter(u64),
    State(&'static str),
}
impl fmt::Display for GroupedOptimizerError {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Configuration(value)=>write!(f,"native optimizer groups: {value}"),
            Self::UnknownParameter(id)=>write!(f,"unknown selected native floating parameter {id}"),
            Self::State(value)=>write!(f,"native optimizer group record: {value}")}
    }
}
impl core::error::Error for GroupedOptimizerError {}

/// Actual original per-group, rank-tagged optimizer histories and fixed ID routing.
/// Numerical configurations/clippers remain the caller-prepared original groups.
#[derive(Clone)]
pub struct GroupedOptimizerAdaptorRecord<O,B>
where B:AutodiffBackend,O:SimpleOptimizer<B::InnerBackend> {
    version:u32,
    routes:Vec<(u64,usize)>,
    records:Vec<HashMap<ParamId,AdaptorRecord<O,B>>>,
}
impl<O,B> Record<B> for GroupedOptimizerAdaptorRecord<O,B>
where B:AutodiffBackend,O:SimpleOptimizer<B::InnerBackend> {
    type Item<P:PrecisionSettings>=(u32,Vec<(u64,usize)>,Vec<<HashMap<ParamId,AdaptorRecord<O,B>> as Record<B>>::Item<P>>);
    fn into_item<P:PrecisionSettings>(self) -> Self::Item<P> {
        (self.version,self.routes,self.records.into_iter().map(|record|record.into_item::<P>()).collect())
    }
    fn from_item<P:PrecisionSettings>(item:Self::Item<P>,device:&B::Device) -> Self {
        Self {version:item.0,routes:item.1,records:item.2.into_iter()
            .map(|record|HashMap::<ParamId,AdaptorRecord<O,B>>::from_item::<P>(record,device)).collect()}
    }
}

/// Explicit ordinary native optimizer groups on arbitrary original model leaves.
/// `OptimizerChoice` permits heterogeneous algorithms/master wrappers, including
/// full-matrix Muon, without treating matrix updates as coordinate fragments.
/// Every selected ID belongs to one group; unselected lazy/packed/frozen state is
/// not materialized by group binding. Each group's ORIGINAL mapper handles device
/// placement, record rank, ties, requires-grad flags and distributed identities.
/// No gradient reduction, objective normalization, scheduler or backward replay.
#[derive(Clone)]
pub struct GroupedOptimizerAdaptor<O,M,B>
where B:AutodiffBackend,M:AutodiffModule<B>,O:SimpleOptimizer<B::InnerBackend> {
    groups:Vec<OptimizerAdaptor<O,M,B>>,
    routes:BTreeMap<ParamId,usize>,
    selection:Vec<Vec<ParamId>>,
}

impl<O,M,B> GroupedOptimizerAdaptor<O,M,B>
where B:AutodiffBackend,M:AutodiffModule<B>,O:SimpleOptimizer<B::InnerBackend> {
    /// Bind original prepared groups and explicit parameter IDs, without changing
    /// their existing histories or loading values from unselected model leaves.
    /// Partial selection is intentional; no role or default group is inferred.
    pub fn new(module:&M,groups:Vec<OptimizerAdaptor<O,M,B>>,routes:&[(ParamId,usize)]) -> Result<Self,GroupedOptimizerError> {
        if groups.is_empty() {return Err(GroupedOptimizerError::Configuration("at least one original native optimizer group required"));}
        let mut routing=BTreeMap::new();let mut selection=alloc::vec![Vec::new();groups.len()];
        for &(id,index) in routes {
            if index>=groups.len() || routing.insert(id,index).is_some() {
                return Err(GroupedOptimizerError::Configuration("duplicate canonical parameter route or unknown group"));
            }
            selection[index].push(id);
        }
        for ids in &mut selection {ids.sort();}
        let result=Self {groups,routes:routing,selection};result.validate_model(module)?;
        result.validate_histories(&result.groups.iter().map(|group|&group.records).collect::<Vec<_>>())?;
        Ok(result)
    }
    pub fn groups(&self) -> &[OptimizerAdaptor<O,M,B>] {&self.groups}
    pub fn group_for(&self,parameter:ParamId) -> Option<usize> {self.routes.get(&parameter).copied()}
    pub fn state_parameter_count(&self) -> usize {self.groups.iter().map(|group|group.records.len()).sum()}
    fn route_record(&self) -> Vec<(u64,usize)> {self.routes.iter().map(|(id,index)|(id.val(),*index)).collect()}

    /// Check selected float identity membership without reading any tensor value.
    /// Frozen selected identities remain valid bindings, with untouched history.
    pub fn validate_model(&self,module:&M) -> Result<(),GroupedOptimizerError> {
        struct Ids {found:BTreeSet<ParamId>}
        impl<B:AutodiffBackend> ModuleVisitor<B> for Ids {
            fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {self.found.insert(param.id);}
        }
        let mut ids=Ids {found:BTreeSet::new()};module.visit(&mut ids);
        for id in self.routes.keys() {if !ids.found.contains(id) {return Err(GroupedOptimizerError::UnknownParameter(id.val()));}}
        Ok(())
    }
    fn validate_histories(&self,records:&[&HashMap<ParamId,AdaptorRecord<O,B>>]) -> Result<(),GroupedOptimizerError> {
        if records.len()!=self.groups.len() {return Err(GroupedOptimizerError::State("original group count differs"));}
        for (index,records) in records.iter().enumerate() {
            if records.keys().any(|id|self.routes.get(id)!=Some(&index)) {
                return Err(GroupedOptimizerError::State("native history belongs to another or unselected parameter group"));
            }
        }
        Ok(())
    }
    fn step_with(&mut self,rates:&[LearningRate],mut module:M,mut gradients:GradAdaptor)
        -> Result<(M,GradAdaptor),GroupedOptimizerError> {
        if rates.len()!=self.groups.len() || rates.iter().any(|rate|!rate.is_finite() || *rate<0.0) {
            return Err(GroupedOptimizerError::Configuration("one finite nonnegative learning rate per original group required"));
        }
        self.validate_model(&module)?;
        for ((group,selection),rate) in self.groups.iter_mut().zip(&self.selection).zip(rates) {
            let mut mapper=SimpleOptimizerMapper::<B,O>::new(&group.optim,&mut group.records,&mut gradients,*rate,group.grad_clipping.as_ref());
            mapper.selection=Some(selection);module=module.map(&mut mapper);
        }
        Ok((module,gradients))
    }
    /// Independent original rates/options; return every derivative not consumed
    /// by selected trainable leaves. Backend/numerical errors retain the original
    /// optimizer's behavior; no transactional device-update guarantee is added.
    pub fn try_step_with_lrs(&mut self,rates:&[LearningRate],module:M,gradients:GradientsParams)
        -> Result<(M,GradientsParams),GroupedOptimizerError> {
        let (module,gradients)=self.step_with(rates,module,gradients.into())?;
        let GradAdaptor::Single(gradients)=gradients else {unreachable!("original single native gradient input")};
        Ok((module,gradients))
    }
    /// The same original multi-device gradient/device pairs pass through native
    /// group mappers; unused pairs remain in their original container unchanged.
    pub fn try_step_multi_with_lrs(&mut self,rates:&[LearningRate],module:M,gradients:MultiGradientsParams)
        -> Result<(M,MultiGradientsParams),GroupedOptimizerError> {
        let (module,gradients)=self.step_with(rates,module,gradients.into())?;
        let GradAdaptor::Multi(gradients)=gradients else {unreachable!("original multi-device native gradient input")};
        Ok((module,gradients))
    }
    /// Restore original group histories only for the identical saved ID routes.
    /// Caller restores original model IDs/configurations first, without inferred
    /// regrouping, state resets, master casts or changes to group clippers.
    pub fn try_load_record(mut self,record:GroupedOptimizerAdaptorRecord<O,B>) -> Result<Self,GroupedOptimizerError> {
        if record.version!=1 || record.routes!=self.route_record() {return Err(GroupedOptimizerError::State("saved original group routing differs"));}
        self.validate_histories(&record.records.iter().collect::<Vec<_>>())?;
        for (group,records) in self.groups.iter_mut().zip(record.records) {group.records=records;}
        Ok(self)
    }
}

impl<O,M,B> Optimizer<M,B> for GroupedOptimizerAdaptor<O,M,B>
where B:AutodiffBackend,M:AutodiffModule<B>,O:SimpleOptimizer<B::InnerBackend> {
    type Record=GroupedOptimizerAdaptorRecord<O,B>;
    /// Standard optimizer contract uses the supplied rate for all explicit groups.
    /// Use the explicit step methods when unused derivatives must be returned.
    fn step(&mut self,lr:LearningRate,module:M,gradients:GradientsParams) -> M {
        self.try_step_with_lrs(&alloc::vec![lr;self.groups.len()],module,gradients).unwrap_or_else(|error|panic!("{error}")).0
    }
    fn step_multi(&mut self,lr:LearningRate,module:M,gradients:MultiGradientsParams) -> M {
        self.try_step_multi_with_lrs(&alloc::vec![lr;self.groups.len()],module,gradients).unwrap_or_else(|error|panic!("{error}")).0
    }
    fn to_record(&self) -> Self::Record {
        Self::Record {version:1,routes:self.route_record(),records:self.groups.iter().map(|group|group.records.clone()).collect()}
    }
    fn load_record(self,record:Self::Record) -> Self {self.try_load_record(record).unwrap_or_else(|error|panic!("{error}"))}
}
