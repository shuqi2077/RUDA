//! ZeRO stage 1: one owner updates each complete parameter and stores its optimizer state.
//! Parameters and reduced gradients remain replicated. Transport is supplied by DataParallel.
use super::*;
use crate::{LearningRate, Optimizer, SimpleOptimizer, adaptor::OptimizerAdaptor};
use ruda_model::record::{PrecisionSettings, Record};

/// Rank-local optimizer-state sharding over an initialized replicated model.
///
/// Ownership is explicit, in first-occurrence order of distinct trainable parameter IDs.
/// Each parameter is updated as a complete tensor, including Muon's matrix operation.
/// Save every rank's record together with that rank's model and continuation state.
pub struct Zero1<B, M, O, C = RankCommunicator<TensorDevice<<B as AutodiffBackend>::InnerBackend>>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
    C: DataParallelCommunicator<B::InnerBackend>,
{
    session: DataParallel<B, C>,
    optimizer: OptimizerAdaptor<O, M, B>,
    owners: Vec<u32>,
    ids: Vec<ParamId>,
}

/// Versioned rank-local state, retaining world size, owners and model parameter IDs.
pub struct Zero1Record<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
{
    version: u32,
    rank: u32,
    world_size: u32,
    owners: Vec<u32>,
    ids: Vec<u64>,
    optimizer: <OptimizerAdaptor<O, M, B> as Optimizer<M, B>>::Record,
}

/// Result of a completed, globally weighted update and parameter synchronization.
pub struct Zero1Step<M> {
    /// Updated replica with the original local IDs and shared aliases.
    pub model: M,
    /// Total effective samples/tokens contributing to this update.
    pub global_weight: u64,
}

impl<B, M, O, C> Zero1<B, M, O, C>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
    C: DataParallelCommunicator<B::InnerBackend>,
{
    /// Attach an optimizer to an initialized session with explicit rank ownership.
    /// All ranks must supply the same owners and compatible optimizer options.
    pub fn new(
        session: DataParallel<B, C>,
        model: &M,
        optimizer: O,
        owners: Vec<u32>,
    ) -> Result<Self, DataParallelError> {
        let mut schema = Schema::new(session.synchronize_buffers);
        model.visit(&mut schema);
        let mut ids = Vec::new();
        for (parameter, id) in schema.contract.iter().zip(&schema.ids) {
            if parameter.trainable && !ids.contains(id) {
                ids.push(*id);
            }
        }
        let mut error = schema.device_error;
        if schema.contract != session.contract || schema.ids != session.ids {
            error = Some("model differs from the initialized replica".into());
        }
        if owners.len() != ids.len() || owners.iter().any(|&rank| rank >= session.world_size()) {
            error = Some("provide one valid owner for each distinct trainable parameter".into());
        }
        let requests = gather::<B::InnerBackend, C, _>(
            &session.communicator, &(owners.clone(), error),
        )?;
        for (other_owners, error) in requests {
            if let Some(error) = error { return Err(contract(error)); }
            if other_owners != owners { return Err(contract("ranks disagree on optimizer owners")); }
        }
        Ok(Self { session, optimizer: optimizer.into(), owners, ids })
    }

    /// Rank owning each distinct trainable parameter, in model visitation order.
    pub fn owners(&self) -> &[u32] { &self.owners }

    /// Number of parameters whose optimizer state belongs to this rank.
    pub fn owned_parameter_count(&self) -> usize {
        self.owners.iter().filter(|&&owner| owner == self.session.rank()).count()
    }

    /// Number of optimizer records actually allocated on this rank.
    /// Stateless optimizers and globally unused parameters need not have records.
    pub fn state_parameter_count(&self) -> usize { self.optimizer.to_record().len() }

    /// Reduce local loss-sum gradients, update owned parameters and broadcast their new values.
    /// The learning rate must agree across ranks. Does not advance a scheduler or clear a caller's accumulator.
    pub fn step(
        &mut self, lr: LearningRate, model: M, gradients: GradientsParams,
        local_weight: u64, policy: MissingGradientPolicy,
    ) -> Result<Zero1Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, local_weight, policy, false)
    }

    /// Keep reduced gradients in FP32 for a compatible optimizer such as Fp32MasterOptimizer.
    /// Parameter storage and ownership are unchanged; all ranks choose this variant together.
    pub fn step_fp32(
        &mut self, lr: LearningRate, model: M, gradients: GradientsParams,
        local_weight: u64, policy: MissingGradientPolicy,
    ) -> Result<Zero1Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, local_weight, policy, true)
    }

    fn step_inner(
        &mut self, lr: LearningRate, model: M, gradients: GradientsParams,
        local_weight: u64, policy: MissingGradientPolicy, fp32: bool,
    ) -> Result<Zero1Step<M>, DataParallelError> {
        let rates = gather::<B::InnerBackend, C, _>(&self.session.communicator, &lr.to_bits())?;
        if !lr.is_finite() || lr < 0. || rates.iter().any(|&rate| rate != lr.to_bits()) {
            return Err(contract("learning rates must be finite, nonnegative and identical"));
        }
        let reduced = if fp32 {
            self.session.reduce_fp32(&model, gradients, local_weight, policy)?
        } else {
            self.session.reduce(&model, gradients, local_weight, policy)?
        };
        let ownership: HashMap<_, _> = self.ids.iter().copied().zip(self.owners.iter().copied()).collect();
        let mut filter = OwnedGradients::<B> {
            input: reduced.gradients, output: GradientsParams::new(),
            ownership: &ownership, rank: self.session.rank(), visited: Vec::new(),
            backend: PhantomData,
        };
        model.visit(&mut filter);
        let model = self.optimizer.step(lr, model, filter.output);
        let mut broadcast = OwnedBroadcast::<B, C> {
            communicator: &self.session.communicator, ownership: &ownership,
            updated: TensorContainer::new(), error: None, backend: PhantomData,
        };
        let model = model.map(&mut broadcast);
        if let Some(error) = broadcast.error { return Err(error.into()); }
        Ok(Zero1Step { model, global_weight: reduced.global_weight })
    }

    /// Snapshot this rank's state at a completed update boundary.
    /// Recreate identical optimizer options and restore this rank's matching model first.
    pub fn to_record(&self) -> Zero1Record<B, M, O> {
        Zero1Record {
            version: 1, rank: self.session.rank(), world_size: self.session.world_size(),
            owners: self.owners.clone(), ids: self.ids.iter().map(ParamId::val).collect(),
            optimizer: self.optimizer.to_record(),
        }
    }

    /// Validate and load a matching rank-local snapshot collectively, without updating parameters.
    /// World-size/ownership changes require explicit repartitioning, not this restore method.
    pub fn load_record(&mut self, record: Zero1Record<B, M, O>) -> Result<(), DataParallelError> {
        let ids: Vec<_> = self.ids.iter().map(ParamId::val).collect();
        let error = if record.version != 1 || record.rank != self.session.rank()
            || record.world_size != self.session.world_size() || record.owners != self.owners
            || record.ids != ids {
            Some("ZeRO record version, rank, world size, owners or local model IDs differ".to_string())
        } else if record.optimizer.keys().any(|id| {
            self.ids.iter().position(|candidate| candidate == id)
                .is_none_or(|index| self.owners[index] != self.session.rank())
        }) {
            Some("optimizer record contains a non-owned parameter".to_string())
        } else { None };
        for error in gather::<B::InnerBackend, C, _>(&self.session.communicator, &error)? {
            if let Some(error) = error { return Err(contract(error)); }
        }
        // Moving the adaptor through its existing load_record API preserves its algorithm/options.
        let optimizer = OptimizerAdaptor::from(self.optimizer.optim().clone()).load_record(record.optimizer);
        self.optimizer = optimizer;
        Ok(())
    }
}

impl<B, M, O> Record<B> for Zero1Record<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
{
    type Item<S: PrecisionSettings> = <(
        u32, u32, u32, Vec<u32>, Vec<u64>,
        <OptimizerAdaptor<O, M, B> as Optimizer<M, B>>::Record,
    ) as Record<B>>::Item<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version, self.rank, self.world_size, self.owners, self.ids, self.optimizer).into_item::<S>()
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        let (version, rank, world_size, owners, ids, optimizer) = Record::<B>::from_item::<S>(item, device);
        Self { version, rank, world_size, owners, ids, optimizer }
    }
}

struct OwnedGradients<'a, B: AutodiffBackend> {
    input: GradientsParams,
    output: GradientsParams,
    ownership: &'a HashMap<ParamId, u32>,
    rank: u32,
    visited: Vec<ParamId>,
    backend: PhantomData<B>,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for OwnedGradients<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.ownership.get(&param.id) != Some(&self.rank) || self.visited.contains(&param.id) { return; }
        self.visited.push(param.id);
        if let Some(gradient) = self.input.remove::<B::InnerBackend, D>(param.id) {
            self.output.register(param.id, gradient);
        }
    }
}

struct OwnedBroadcast<'a, B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> {
    communicator: &'a C,
    ownership: &'a HashMap<ParamId, u32>,
    updated: TensorContainer<ParamId>,
    error: Option<TensorDeviceError>,
    backend: PhantomData<B>,
}
impl<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> ModuleMapper<B>
    for OwnedBroadcast<'_, B, C>
{
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if self.error.is_some() || !param.is_require_grad() { return param; }
        let Some(&owner) = self.ownership.get(&param.id) else { return param; };
        let (id, value, mapper) = param.consume();
        let tensor = if let Some(tensor) = self.updated.get::<B>(&id) {
            Tensor::from_primitive(tensor)
        } else {
            match self.communicator.broadcast_float(value.clone().inner().into_primitive().tensor(), owner) {
                Ok(tensor) => {
                    let tensor = Tensor::<B, D>::from_inner(Tensor::<B::InnerBackend, D>::from_primitive(TensorPrimitive::Float(tensor))).require_grad();
                    self.updated.register::<B>(id, tensor.clone().into_primitive());
                    tensor
                }
                Err(error) => { self.error = Some(error); value }
            }
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}
