//! Parameter-subset gradient/state sharding without rebuilding a proxy model.

use super::*;

/// ZeRO-2 for an explicit replica subset of the actual full rank-local model.
///
/// Selected parameters remain replicated within this caller-owned group, while
/// their gradients and elementwise optimizer state are split into equal padded
/// flat slices. Different unselected experts/shards need not share a model schema,
/// and unselected tensors are not materialized or mapped by this session.
/// Padding is only transport storage; original shapes and tied aliases are restored.
/// Matrix-wide optimizers requiring complete tensors must use selected ZeRO-1 instead.
pub struct SelectedZero2<
    B,
    M,
    O,
    C = RankCommunicator<TensorDevice<<B as AutodiffBackend>::InnerBackend>>,
> where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: ElementwiseShardOptimizer<B::InnerBackend>,
    C: ShardedCommunicator<B::InnerBackend>,
{
    inner: Zero2<B, M, O, C>,
    parameters: Vec<ParamId>,
}

/// Updated actual full model plus the original derivatives belonging to other groups.
pub struct SelectedZero2Step<M> {
    /// Complete rank-local model with only selected trainable parameters updated.
    pub model: M,
    /// Unselected input derivatives, without normalization, conversion or copying.
    pub remaining_gradients: GradientsParams,
    /// Exact mean denominator, or active participant count for a SUM-only update.
    pub global_weight: u64,
}

/// Exact selected membership/schema together with this group's rank-local flat states.
pub struct SelectedZero2Record<B, O>
where
    B: AutodiffBackend,
    O: ElementwiseShardOptimizer<B::InnerBackend>,
{
    version: u32,
    parameters: Vec<u64>,
    schema: String,
    optimizer: Zero2Record<B, O>,
}

fn parameter_set(parameters: &[ParamId]) -> Vec<u64> {
    let mut ids = parameters.iter().map(ParamId::val).collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

impl<B, M, O, C> SelectedZero2<B, M, O, C>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: ElementwiseShardOptimizer<B::InnerBackend>,
    C: ShardedCommunicator<B::InnerBackend>,
{
    /// Attach a compatible original elementwise optimizer to a selected replica group.
    ///
    /// Selected trainable tensors must be nonempty; an empty selection is valid.
    /// Group size/rank, selected paths, shapes, aliases and frozen flags come from
    /// the initialized session. Every rank supplies the same optimizer options.
    /// No optimizer state or gradient shard is created for unselected parameters.
    pub fn new(
        session: SelectedDataParallel<B, C>,
        model: &M,
        optimizer: O,
    ) -> Result<Self, DataParallelError> {
        let (session, parameters) = session.into_parts();
        let inner = Zero2::from_session(session, model, optimizer, Some(&parameters))?;
        Ok(Self { inner, parameters })
    }

    /// Original selected rank-local IDs, including selected frozen parameters.
    pub fn parameters(&self) -> &[ParamId] {
        &self.parameters
    }

    /// Distinct selected trainable IDs in original full-model visitation order.
    pub fn trainable_parameters(&self) -> &[ParamId] {
        &self.inner.ids
    }

    /// Rank owning this group's equal padded flat slices.
    pub fn rank(&self) -> u32 {
        self.inner.rank()
    }

    /// Number of slices/participants in this explicit replica group.
    pub fn world_size(&self) -> u32 {
        self.inner.session.world_size()
    }

    /// Number of selected parameter-slice states actually allocated on this rank.
    pub fn state_parameter_count(&self) -> usize {
        self.inner.state_parameter_count()
    }

    /// Reduce-scatter selected loss-sum gradients, normalize by exact global weight and update.
    ///
    /// Communication is FP32. Each local gradient slice uses the optimizer's
    /// declared gradient dtype, including FP32-master optimizers. Updated slices
    /// are gathered into each selected parameter's original native storage/shape.
    /// Unselected derivatives are returned for independent groups; no scheduler,
    /// clipping, accumulation reset or group topology is inferred.
    pub fn step(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
    ) -> Result<SelectedZero2Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, local_weight, policy, true)
    }

    /// Reduce-scatter selected SUM-only derivatives, with no second objective normalization.
    ///
    /// Use when contributions already contain a global expert-world denominator.
    /// `local_active` controls eligibility, not gradient scaling or token weights.
    /// Globally absent derivatives remain absent and cause no update/weight decay.
    /// An all-inactive window preserves selected weights/state and returns all
    /// unselected derivatives. `global_weight` counts active participants only.
    pub fn step_sum(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_active: bool,
        policy: MissingGradientPolicy,
    ) -> Result<SelectedZero2Step<M>, DataParallelError> {
        self.step_inner(lr, model, gradients, u64::from(local_active), policy, false)
    }

    fn step_inner(
        &mut self,
        lr: LearningRate,
        model: M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
        normalize: bool,
    ) -> Result<SelectedZero2Step<M>, DataParallelError> {
        let (updated, remaining_gradients) = self.inner.step_inner(
            lr, model, gradients, local_weight, policy, Some(&self.parameters), normalize,
        )?;
        Ok(SelectedZero2Step {
            model: updated.model,
            remaining_gradients,
            global_weight: updated.global_weight,
        })
    }

    /// Snapshot selected-group metadata and actual rank-local optimizer slice state.
    /// Save with the matching original full model and pending accumulation state;
    /// use full-precision settings for FP32 master/moment buffers.
    pub fn to_record(&self) -> SelectedZero2Record<B, O> {
        SelectedZero2Record {
            version: 1,
            parameters: parameter_set(&self.parameters),
            schema: serde_json::to_string(&self.inner.session.contract)
                .expect("selected parameter contract serialization failed"),
            optimizer: self.inner.to_record(),
        }
    }

    /// Restore matching membership/schema and this group's original rank-local slices.
    ///
    /// Restore matching full-model IDs first. All ranks reject mismatched selected
    /// frozen IDs, paths, shapes, dtypes, aliases, rank/world size or state keys
    /// before replacing optimizer state. World changes/resharding are not implicit.
    pub fn load_record(
        &mut self,
        record: SelectedZero2Record<B, O>,
    ) -> Result<(), DataParallelError> {
        let schema = serde_json::to_string(&self.inner.session.contract)
            .map_err(|error| contract(error.to_string()))?;
        let error = if record.version != 1
            || record.parameters != parameter_set(&self.parameters)
            || record.schema != schema
        {
            Some("selected ZeRO-2 membership or original replica schema differs".to_string())
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

impl<B, O> Record<B::InnerBackend> for SelectedZero2Record<B, O>
where
    B: AutodiffBackend,
    O: ElementwiseShardOptimizer<B::InnerBackend>,
{
    type Item<S: PrecisionSettings> =
        <(u32, Vec<u64>, String, Zero2Record<B, O>) as Record<B::InnerBackend>>::Item<S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version, self.parameters, self.schema, self.optimizer).into_item::<S>()
    }

    fn from_item<S: PrecisionSettings>(
        item: Self::Item<S>,
        device: &ruda_model::tensor::Device<B::InnerBackend>,
    ) -> Self {
        let (version, parameters, schema, optimizer) =
            Record::<B::InnerBackend>::from_item::<S>(item, device);
        Self { version, parameters, schema, optimizer }
    }
}
