use super::*;

/// One explicit original native optimizer configuration and its separate logical per-parameter clipper.
/// Same optimizer type can have different decay, momentum, beta, epsilon, master scale or clipping per group.
#[derive(Clone)]
pub struct FullyShardedElementwiseGroup<O> {
    /// Caller-prepared original coordinate optimizer, without shape/name-based role selection.
    pub optimizer:O,
    /// Caller-selected original per-parameter clipping, before any native master-wrapper transform.
    pub clipping:Option<GradientClipping>,
}
impl<O> FullyShardedElementwiseGroup<O> {
    /// Declare actual group options; learning rates are supplied explicitly at each complete optimizer step.
    pub fn new(optimizer:O,clipping:Option<GradientClipping>) -> Self {Self {optimizer,clipping}}
}

/// Actual original group routing alongside one canonical local native history container.
pub struct FullyShardedGroupedElementwiseRecord<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> {
    version:u32,
    inner:FullyShardedElementwiseRecord<B,O>,
    routes:Vec<(u64,usize)>,
    group_count:usize,
}
impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> Clone for FullyShardedGroupedElementwiseRecord<B,O> {
    fn clone(&self) -> Self {Self {version:self.version,inner:self.inner.clone(),routes:self.routes.clone(),group_count:self.group_count}}
}
impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> Record<B> for FullyShardedGroupedElementwiseRecord<B,O>
    where O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    type Item<S:PrecisionSettings>=(u32,<FullyShardedElementwiseRecord<B,O> as Record<B>>::Item<S>,Vec<(u64,usize)>,usize);
    fn into_item<S:PrecisionSettings>(self) -> Self::Item<S> {(self.version,self.inner.into_item::<S>(),self.routes,self.group_count)}
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device) -> Self {
        Self {version:item.0,inner:FullyShardedElementwiseRecord::from_item::<S>(item.1,device),routes:item.2,group_count:item.3}
    }
}

/// Explicit parameter-group native FSDP optimization, with one global presence vote/update per canonical ID.
/// Bias/norm/base/adapter roles, groups and per-group rates are declared, not inferred from tensor dimensions.
#[derive(Clone)]
pub struct FullyShardedGroupedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend> {
    inner:FullyShardedElementwiseOptimizer<O,M,B,C>,
    groups:Vec<FullyShardedElementwiseGroup<O>>,
    routes:BTreeMap<ParamId,usize>,
}
impl<O,M,B,C> FullyShardedGroupedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
        O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    /// Assign every actual parameter identity once, including tied/frozen identities, to its original group.
    pub fn new(module:&M,parameters:&[FullyShardedOptimizerParameter<C>],groups:Vec<FullyShardedElementwiseGroup<O>>,routes:&[(ParamId,usize)])
        -> Result<Self,FullyShardedElementwiseError<C::Error>> {
        let first=groups.first().ok_or(FullyShardedElementwiseError::Configuration("at least one explicit native optimizer group required"))?;
        for group in &groups {group.optimizer.validate_fully_sharded_execution().map_err(FullyShardedElementwiseError::Configuration)?;}
        let inner=FullyShardedElementwiseOptimizer::new(first.optimizer.clone(),module,parameters,None)?;
        let mut routing=BTreeMap::new();
        for &(id,group) in routes {
            if group>=groups.len() || routing.insert(id,group).is_some() {return Err(FullyShardedElementwiseError::Configuration("duplicate parameter route or unknown native group"));}
        }
        if routing.len()!=inner.placement.len() || inner.placement.iter().any(|entry|!routing.contains_key(&ParamId::from(entry.0))) {
            return Err(FullyShardedElementwiseError::Configuration("every actual native local leaf requires an explicit optimizer group"));
        }
        Ok(Self {inner,groups,routes:routing})
    }
    /// Actual original group options, without synthesizing defaults for undeclared roles.
    pub fn groups(&self) -> &[FullyShardedElementwiseGroup<O>] {&self.groups}
    /// Actual number of canonical native histories, independent of how many shared roles reference a leaf.
    pub fn state_parameter_count(&self) -> usize {self.inner.states.len()}
    /// Original source-defined group for this actual canonical parameter ID.
    pub fn group_for(&self,id:ParamId) -> Option<usize> {self.routes.get(&id).copied()}
    /// Actual local native history, unchanged for globally unused or frozen leaves.
    pub fn state(&self,id:ParamId) -> Option<&O::State<1>> {self.inner.states.get(&id)}
    /// Step once with independently caller-declared learning rates for every original group.
    /// Full logical clipping/master transforms retain group options; no extra gradient reduction occurs.
    pub fn try_step_with_lrs(&mut self,rates:&[LearningRate],module:M,gradients:GradientsParams) -> Result<M,FullyShardedElementwiseError<C::Error>> {
        if rates.len()!=self.groups.len() || rates.iter().any(|rate|!rate.is_finite() || *rate<0.0) {
            return Err(FullyShardedElementwiseError::Configuration("one finite nonnegative learning rate per actual native group required"));
        }
        let groups=&self.groups;let routes=&self.routes;
        self.inner.step_configured(module,gradients,|id| {
            let index=*routes.get(&id).expect("validated canonical original optimizer route");
            let group=&groups[index];(group.optimizer.clone(),group.clipping.clone(),rates[index])
        })
    }
    /// Same actual group options/rates and original native update expression,
    /// returning rank-local model/gradient inputs if preparation returns an error.
    /// No partial group history commit or automatic transport/optimizer retry.
    pub fn try_step_recoverable_with_lrs(&mut self,rates:&[LearningRate],module:M,gradients:GradientsParams)
        -> Result<M,FullyShardedElementwiseStepFailure<M,C::Error>> {
        if rates.len()!=self.groups.len() || rates.iter().any(|rate|!rate.is_finite() || *rate<0.0) {
            return Err(FullyShardedElementwiseStepFailure {module,gradients,
                error:FullyShardedElementwiseError::Configuration("one finite nonnegative learning rate per actual native group required")});
        }
        let groups=&self.groups;let routes=&self.routes;
        self.inner.step_recoverable_configured(module,gradients,|id| {
            let index=*routes.get(&id).expect("validated canonical original optimizer route");
            let group=&groups[index];(group.optimizer.clone(),group.clipping.clone(),rates[index])
        })
    }
    fn route_record(&self) -> Vec<(u64,usize)> {self.routes.iter().map(|(id,group)|(id.val(),*group)).collect()}
    /// Restore original local histories only when exact saved group membership and ownership still match.
    /// Actual numerical group configurations remain the caller-prepared original configurations.
    pub fn try_load_record(mut self,record:FullyShardedGroupedElementwiseRecord<B,O>) -> Result<Self,FullyShardedElementwiseError<C::Error>> {
        if record.version!=1 || record.group_count!=self.groups.len() || record.routes!=self.route_record()
            || record.inner.version!=1 || record.inner.placement!=self.inner.placement {
            return Err(FullyShardedElementwiseError::State("original native optimizer grouping/ownership differs"));
        }
        for (id,state) in &record.inner.states {
            let index=*self.routes.get(id).ok_or(FullyShardedElementwiseError::State("unknown saved grouped optimizer history"))?;
            let spec=self.inner.placement.iter().find(|entry|entry.0==id.val()).ok_or(FullyShardedElementwiseError::State("unknown saved grouped parameter"))?;
            let count=spec.1.iter().product::<usize>();
            validate_state::<B::InnerBackend,O>(state,[count.div_ceil(spec.3 as usize)],self.groups[index].optimizer.shard_gradient_dtype(spec.4)).map_err(FullyShardedElementwiseError::State)?;
            self.groups[index].optimizer.validate_fully_sharded_history(state).map_err(FullyShardedElementwiseError::State)?;
        }
        self.inner.states=record.inner.states;Ok(self)
    }
    /// Eager original history placement after exact per-group route/options and
    /// logical ownership validation. Work/master buffers retain original dtype,
    /// counters and optional branches; unrelated parameter devices are preserved.
    pub fn try_load_record_for_model(self,module:&M,record:FullyShardedGroupedElementwiseRecord<B,O>)
        -> Result<Self,FullyShardedElementwiseError<C::Error>> {
        inspect::<B,M>(module,&self.inner.placement,true).map_err(FullyShardedElementwiseError::Arguments)?;
        let mut restored=self.try_load_record(record)?;restored.inner.place_histories_on_model(module)?;Ok(restored)
    }
}
impl<O,M,B,C> Optimizer<M,B> for FullyShardedGroupedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:ElementwiseShardOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
        O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    type Record=FullyShardedGroupedElementwiseRecord<B,O>;
    fn step(&mut self,lr:LearningRate,module:M,gradients:GradientsParams) -> M {
        let rates=alloc::vec![lr;self.groups.len()];self.try_step_with_lrs(&rates,module,gradients).unwrap_or_else(|error|panic!("{error}"))
    }
    fn step_multi(&mut self,_lr:LearningRate,_module:M,_gradients:MultiGradientsParams) -> M {panic!("actual grouped FSDP owners use step on each original rank")}
    fn to_record(&self) -> Self::Record {
        Self::Record {version:1,inner:self.inner.to_record(),routes:self.route_record(),group_count:self.groups.len()}
    }
    fn load_record(self,record:Self::Record) -> Self {self.try_load_record(record).unwrap_or_else(|error|panic!("{error}"))}
}

impl<O,M,B,C> FullyShardedGroupedElementwiseOptimizer<O,M,B,C>
    where B:AutodiffBackend,M:AutodiffModule<B>,O:crate::FlatShardElementwiseOptimizer<B::InnerBackend>,C:BroadcastTensorCollective<B::InnerBackend>,
        O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1> {
    /// Streaming original native histories into exact declared groups/owners, preserving actual frozen histories.
    pub fn try_import_native_records<I:IntoIterator<Item=(ParamId,crate::record::AdaptorRecord<O,B>)>>(&mut self,records:I) -> Result<(),crate::OptimizerShardError> {
        let mut states=HashMap::new();
        for (id,record) in records {
            if states.contains_key(&id) {return Err(crate::OptimizerShardError::Placement("duplicate original grouped history identity"));}
            let index=*self.routes.get(&id).ok_or(crate::OptimizerShardError::Placement("unknown original grouped optimizer identity"))?;
            let spec=self.inner.placement.iter().find(|entry|entry.0==id.val()).ok_or(crate::OptimizerShardError::Placement("unknown original native grouped parameter"))?;
            let shard=crate::FlatOptimizerTensorShard::new(spec.1.clone(),spec.2,spec.3)?;
            let state=match record {crate::record::AdaptorRecord::V1(record)=>O::partition_native_history(record,&shard)?};
            let (_,slots)=shard.geometry()?;
            validate_state::<B::InnerBackend,O>(&state,[slots],self.groups[index].optimizer.shard_gradient_dtype(spec.4)).map_err(crate::OptimizerShardError::DType)?;
            self.groups[index].optimizer.validate_fully_sharded_history(&state).map_err(crate::OptimizerShardError::Placement)?;
            states.insert(id,state);
        }
        self.inner.states=states;Ok(())
    }
}
impl<B:AutodiffBackend,O:ElementwiseShardOptimizer<B::InnerBackend>> FullyShardedGroupedElementwiseRecord<B,O>
    where O::State<1>:OptimizerCheckpointBuffers<B::InnerBackend,1>+crate::OptimizerCheckpointScalars {
    /// Per-parameter explicit ownership migration retaining each original native
    /// group index. Locally owned expert/TP parameter lists may differ; original
    /// group definitions/count and a parameter's selected source-group route must
    /// agree. No reclassification, group-default assignment or clock advance.
    pub fn reshard_explicit(records:&[Self],migrations:&[FullyShardedOptimizerMigration],device:&B::Device)
        -> Result<Self,crate::OptimizerShardError> {
        let invalid=crate::OptimizerShardError::Placement;
        let first=records.first().ok_or(invalid("actual original grouped optimizer archives required"))?;
        let mut routes=Vec::with_capacity(records.len());
        for record in records {
            let routing=record.routes.iter().copied().collect::<BTreeMap<_,_>>();
            if record.version!=1 || record.group_count!=first.group_count || record.group_count==0 || routing.len()!=record.routes.len()
                || routing.len()!=record.inner.placement.len() || routing.values().any(|index|*index>=record.group_count)
                || record.inner.placement.iter().any(|entry|!routing.contains_key(&entry.0)) {
                return Err(invalid("original native optimizer group definitions/routes differ"));
            }
            routes.push(routing);
        }
        let mut target_routes=Vec::with_capacity(migrations.len());
        for migration in migrations {
            let mut group=None;
            for index in &migration.source_records {
                let route=routes.get(*index).and_then(|routes|routes.get(&migration.target.parameter.val())).copied()
                    .ok_or(invalid("selected original group route missing"))?;
                if group.is_some_and(|previous|previous!=route) {return Err(invalid("original parameter group differs across selected source owners"));}
                group=Some(route);
            }
            target_routes.push((migration.target.parameter.val(),group.ok_or(invalid("explicit original group owners required"))?));
        }
        let sources=records.iter().map(|record|&record.inner).collect::<Vec<_>>();
        let inner=super::migration::migrate_optimizer_record::<B,O>(&sources,migrations,device)?;
        target_routes.sort_by_key(|entry|entry.0);
        Ok(Self {version:1,inner,routes:target_routes,group_count:first.group_count})
    }
    /// Exact original group routing and independent native histories across a complete rank-set ownership change.
    pub fn reshard(records:&[Self],target_rank:u32,target_world:u32,device:&B::Device) -> Result<Self,crate::OptimizerShardError> {
        let first=records.first().ok_or(crate::OptimizerShardError::Placement("complete original grouped optimizer rank set required"))?;
        for record in records {
            if record.version!=1 || record.group_count!=first.group_count || record.routes!=first.routes {
                return Err(crate::OptimizerShardError::Placement("actual native group routing differs across saved owners"));
            }
        }
        let inner=records.iter().map(|record|record.inner.clone()).collect::<Vec<_>>();
        let inner=FullyShardedElementwiseRecord::reshard(&inner,target_rank,target_world,device)?;
        Ok(Self {version:1,inner,routes:first.routes.clone(),group_count:first.group_count})
    }
}
