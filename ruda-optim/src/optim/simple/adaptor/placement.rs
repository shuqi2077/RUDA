use super::*;
use alloc::{collections::{BTreeMap,BTreeSet},vec::Vec};
use core::fmt;
use ruda_model::module::ModuleVisitor;

/// Native history identity/rank or actual tied-leaf placement mismatch.
#[derive(Clone,Debug,PartialEq,Eq)]
pub enum OptimizerStatePlacementError {
    UnknownParameter(u64),
    ParameterRank {parameter:u64,recorded:usize,actual:usize},
    AliasPlacement(u64),
    Group(GroupedOptimizerError),
}
impl fmt::Display for OptimizerStatePlacementError {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::UnknownParameter(id)=>write!(f,"unknown native optimizer history parameter {id}"),
            Self::ParameterRank {parameter,recorded,actual}=>write!(f,"native optimizer history rank for {parameter}: {recorded}, actual parameter rank: {actual}"),
            Self::AliasPlacement(id)=>write!(f,"tied native parameter {id} has different prepared shape/device"),
            Self::Group(error)=>fmt::Display::fmt(error,f)}
    }
}
impl core::error::Error for OptimizerStatePlacementError {}

pub(super) fn record_devices<B,M,O>(module:&M,records:&HashMap<ParamId,AdaptorRecord<O,B>>)
    -> Result<BTreeMap<ParamId,B::Device>,OptimizerStatePlacementError>
where B:AutodiffBackend,M:AutodiffModule<B>,O:SimpleOptimizer<B::InnerBackend> {
    parameter_devices::<B,M,_>(module,records.iter().map(|(id,record)|(*id,record.parameter_rank())))
}

pub(crate) fn parameter_devices<B,M,I>(module:&M,parameters:I) -> Result<BTreeMap<ParamId,B::Device>,OptimizerStatePlacementError>
where B:AutodiffBackend,M:AutodiffModule<B>,I:IntoIterator<Item=(ParamId,usize)> {
    struct Placement<B:AutodiffBackend> {
        requested:BTreeSet<ParamId>,found:BTreeMap<ParamId,(Vec<usize>,B::Device)>,error:Option<OptimizerStatePlacementError>,
    }
    impl<B:AutodiffBackend> ModuleVisitor<B> for Placement<B> {
        fn visit_float<const D:usize>(&mut self,param:&Param<Tensor<B,D>>) {
            if self.error.is_some() || !self.requested.contains(&param.id) {return;}
            let metadata=(param.lazy_shape().to_vec(),param.lazy_device());
            if self.found.insert(param.id,metadata.clone()).is_some_and(|previous|previous!=metadata) {
                self.error=Some(OptimizerStatePlacementError::AliasPlacement(param.id.val()));
            }
        }
    }
    let ranks=parameters.into_iter().collect::<BTreeMap<_,_>>();
    let mut placement=Placement::<B> {requested:ranks.keys().copied().collect(),found:BTreeMap::new(),error:None};
    module.visit(&mut placement);
    if let Some(error)=placement.error {return Err(error);}
    for (id,rank) in ranks {
        let (shape,_)=placement.found.get(&id).ok_or(OptimizerStatePlacementError::UnknownParameter(id.val()))?;
        if shape.len()!=rank {
            return Err(OptimizerStatePlacementError::ParameterRank {parameter:id.val(),recorded:rank,actual:shape.len()});
        }
    }
    Ok(placement.found.into_iter().map(|(id,(_,device))|(id,device)).collect())
}

impl<O,M,B> OptimizerAdaptor<O,M,B>
where B:AutodiffBackend,M:AutodiffModule<B>,O:SimpleOptimizer<B::InnerBackend> {
    /// Place original rank-tagged native history on the restored model's actual
    /// per-parameter devices before the next update. Validate all recorded IDs,
    /// rank tags and tied shape/device metadata before any history move.
    /// Frozen histories stay present; absent histories stay absent. Prepared lazy
    /// weights and unselected packed bases are not initialized or moved.
    /// Original optimizer/clipping options and state precision remain unchanged.
    pub fn try_load_record_for_model(mut self,module:&M,records:HashMap<ParamId,AdaptorRecord<O,B>>)
        -> Result<Self,OptimizerStatePlacementError> {
        let devices=record_devices::<B,M,O>(module,&records)?;
        self.records=records.into_iter().map(|(id,record)| {
            let device=devices.get(&id).expect("validated original optimizer history placement");(id,record.to_device(device))
        }).collect();
        Ok(self)
    }
}
