//! Explicit parameter-group optimizer ownership over the original complete model.

use super::*;
use crate::data_parallel::selected::reduce_selected;

/// ZeRO-1 optimizer-state ownership for one selected replica parameter group.
///
/// Each selected trainable parameter has one explicit owner within this group.
/// Only that owner updates its complete tensor, allowing matrix optimizers such
/// as Muon as well as elementwise and FP32-master optimizers. Unselected model
/// parameters and derivatives are left intact for other optimizer/parallel groups.
/// Parameter and gradient storage of the selected subset remain replicated.
pub struct SelectedZero1<
    B,
    M,
    O,
    C = RankCommunicator<TensorDevice<<B as AutodiffBackend>::InnerBackend>>,
> where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
    C: DataParallelCommunicator<B::InnerBackend>,
{
    inner: Zero1<B, M, O, C>,
    parameters: Vec<ParamId>,
}

/// One selected-group update, returning the actual model and untouched outside gradients.
pub struct SelectedZero1Step<M> {
    /// Original full model with only the selected trainable parameters updated.
    pub model: M,
    /// Original derivatives outside this session's selected ID set.
    pub remaining_gradients: GradientsParams,
    /// Exact weight sum for means, or number of active participants for SUM-only updates.
    pub global_weight: u64,
}

/// Exact selected-group membership/schema and rank-local optimizer ownership state.
/// Save together with the original full model and pending accumulation state.
pub struct SelectedZero1Record<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
{
    version: u32,
    parameters: Vec<u64>,
    schema: String,
    optimizer: Zero1Record<B, M, O>,
}

fn parameter_set(parameters: &[ParamId]) -> Vec<u64> {
    let mut ids = parameters.iter().map(ParamId::val).collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

impl<B, M, O, C> SelectedZero1<B, M, O, C>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
    C: DataParallelCommunicator<B::InnerBackend>,
{
    /// Attach the caller's optimizer and explicit owners to a selected replica session.
    ///
    /// `owners` follows first full-model visitation order of distinct selected
    /// trainable IDs, not the order of the selection slice. Each rank must supply
    /// the same owners and compatible optimizer options. Selection paths and
    /// aliases are those already established by `SelectedDataParallel::initialize`.
    /// No optimizer state is allocated for unselected parameters.
    pub fn new(
        session: SelectedDataParallel<B, C>,
        model: &M,
        optimizer: O,
        owners: Vec<u32>,
    ) -> Result<Self, DataParallelError> {
        let (session, parameters) = session.into_parts();
        let inner = Zero1::from_session(session, model, optimizer, owners, Some(&parameters))?;
        Ok(Self { inner, parameters })
    }

    /// Original rank-local selected parameter IDs, including selected frozen weights.
    pub fn parameters(&self) -> &[ParamId] {
        &self.parameters
    }

    /// Actual distinct selected trainable IDs in the ownership assignment's order.
    pub fn trainable_parameters(&self) -> &[ParamId] {
        &self.inner.ids
    }

    /// Explicit owner for each entry in `trainable_parameters`.
    pub fn owners(&self) -> &[u32] {
        self.inner.owners()
    }

    /// Rank within the caller-selected replica group.
    pub fn rank(&self) -> u32 {
        self.inner.session.rank()
    }

    /// Number of ranks participating in this selected group.
    pub fn world_size(&self) -> u32 {
        self.inner.session.world_size()
    }

    /// Number of complete selected parameter tensors assigned to this rank.
    pub fn owned_parameter_count(&self) -> usize {
        self.inner.owned_parameter_count()
    }

    /// Number of optimizer records currently allocated for this rank's selected owners.
    pub fn state_parameter_count(&self) -> usize {
        self.inner.state_parameter_count()
    }

    /// Reduce selected local loss-sum derivatives into a weighted mean, update owners and broadcast.
    /// All remaining native derivatives are returned unchanged. No scheduler or reset is implicit.
    pub fn step(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
    ) -> Result<SelectedZero1Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, local_weight, policy, false, true)
    }

    /// Weighted-mean selected update retaining FP32 derivatives for compatible master optimizers.
    /// Unselected derivatives keep their original storage and dtype.
    pub fn step_fp32(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
    ) -> Result<SelectedZero1Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, local_weight, policy, true, true)
    }

    /// Update from a SUM of selected derivatives without a second objective normalization.
    ///
    /// Appropriate when each contribution already carries a global expert-world
    /// denominator. `local_active` is contribution eligibility, not a sample count.
    /// Inactive ranks contribute zero; `global_weight` counts active participants.
    /// Globally absent selected derivatives cause no optimizer step or weight decay.
    pub fn step_sum(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_active: bool,
        policy: MissingGradientPolicy,
    ) -> Result<SelectedZero1Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, u64::from(local_active), policy, false, false)
    }

    /// SUM-only selected update with FP32 gradients retained for the owner's master optimizer.
    pub fn step_sum_fp32(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_active: bool,
        policy: MissingGradientPolicy,
    ) -> Result<SelectedZero1Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, u64::from(local_active), policy, true, false)
    }

    fn step_inner(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
        fp32: bool,
        normalize: bool,
    ) -> Result<SelectedZero1Step<M>, DataParallelError> {
        let rates = gather::<B::InnerBackend, C, _>(
            &self.inner.session.communicator, &lr.to_bits(),
        )?;
        if !lr.is_finite() || lr < 0. || rates.iter().any(|&other| other != lr.to_bits()) {
            return Err(contract("selected ZeRO-1 learning rates must be finite, nonnegative and identical"));
        }
        let reduced = reduce_selected(
            &self.inner.session, &self.parameters, &model, gradients,
            local_weight, policy, fp32, normalize,
        )?;
        let (selected, remaining_gradients) =
            reduced.gradients.partition::<B::InnerBackend>(&self.parameters);
        let model = self.inner.update_owned(lr, model, selected, Some(&self.parameters))?;
        Ok(SelectedZero1Step { model, remaining_gradients, global_weight: reduced.global_weight })
    }

    /// Snapshot exact selected paths/aliases/flags and this rank's owned optimizer state.
    /// Use full-precision recorder settings for FP32 master weights and optimizer state.
    pub fn to_record(&self) -> SelectedZero1Record<B, M, O> {
        SelectedZero1Record {
            version: 1,
            parameters: parameter_set(&self.parameters),
            schema: serde_json::to_string(&self.inner.session.contract)
                .expect("selected parameter contract serialization failed"),
            optimizer: self.inner.to_record(),
        }
    }

    /// Restore matching group membership and rank-local ownership collectively.
    ///
    /// Restore the matching original model IDs first. Differences in selected
    /// frozen IDs, paths, shapes, dtypes, alias topology, owners, rank or world size
    /// are rejected before replacing state. World/group changes require explicit
    /// repartitioning; no optimizer algorithm/options or parameters are changed here.
    pub fn load_record(
        &mut self,
        record: SelectedZero1Record<B, M, O>,
    ) -> Result<(), DataParallelError> {
        let schema = serde_json::to_string(&self.inner.session.contract)
            .map_err(|error| contract(error.to_string()))?;
        let error = if record.version != 1
            || record.parameters != parameter_set(&self.parameters)
            || record.schema != schema
        {
            Some("selected ZeRO-1 membership or original replica schema differs".to_string())
        } else {
            None
        };
        for error in gather::<B::InnerBackend, C, _>(&self.inner.session.communicator, &error)? {
            if let Some(error) = error {
                return Err(contract(error));
            }
        }
        self.inner.load_record(record.optimizer)
    }
}

impl<B, M, O> Record<B> for SelectedZero1Record<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: SimpleOptimizer<B::InnerBackend>,
{
    type Item<S: PrecisionSettings> =
        <(u32, Vec<u64>, String, Zero1Record<B, M, O>) as Record<B>>::Item<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version, self.parameters, self.schema, self.optimizer).into_item::<S>()
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        let (version, parameters, schema, optimizer) = Record::<B>::from_item::<S>(item, device);
        Self { version, parameters, schema, optimizer }
    }
}
