use super::*;
use ruda_model::record::{Record, PrecisionSettings, RecorderError};
use crate::GradientsParamsRecord;

/// Explicit selected-gradient arithmetic, independent of model/owner names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectedGradientReductionMode {
    /// Add already normalized contributions without a second denominator.
    Sum,
    /// Same SUM retaining FP32 work gradients for master-parameter updates.
    SumFp32,
    /// Divide actual local loss-SUM derivatives by the original group weight.
    WeightedMean,
    /// Same original weighted mean retaining native FP32 work derivatives.
    WeightedMeanFp32,
}

/// Caller-provided original contribution for one ordered reduction stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectedGradientContribution {
    /// Eligibility only, not a supervised-token denominator or world-size scale.
    Sum { active: bool },
    /// Exact original local sample/token count for unnormalized local SUM gradients.
    WeightedMean { local_weight: u64 },
}
impl SelectedGradientReductionMode {
    fn accepts(self, contribution: SelectedGradientContribution) -> bool {
        matches!((self, contribution),
            (Self::Sum | Self::SumFp32, SelectedGradientContribution::Sum { .. }) |
            (Self::WeightedMean | Self::WeightedMeanFp32, SelectedGradientContribution::WeightedMean { .. }))
    }
}

/// Original local selector/schema/topology and arithmetic for one stage.
/// Actual peer membership and replica contents remain caller-loaded contracts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradientReductionStageContract {
    rank: u32,
    world: u32,
    parameters: Vec<u64>,
    schema: Vec<ParameterContract>,
    mode: SelectedGradientReductionMode,
    missing: MissingGradientPolicy,
    buffers: bool,
}
impl GradientReductionStageContract {
    pub fn rank(&self) -> u32 { self.rank }
    pub fn world_size(&self) -> u32 { self.world }
    pub fn parameters(&self) -> &[u64] { &self.parameters }
    pub fn mode(&self) -> SelectedGradientReductionMode { self.mode }
    pub fn missing_policy(&self) -> MissingGradientPolicy { self.missing }
}

/// One real native reduction stage. The fixed actual model type permits
/// heterogeneous original transport implementations in an ordered plan.
pub trait SelectedGradientReduction<B: AutodiffBackend, M: AutodiffModule<B>> {
    fn contract(&self) -> GradientReductionStageContract;
    fn validate_model(&self, model: &M) -> Result<(), DataParallelError>;
    fn reduce(&self, model: &M, gradients: GradientsParams, contribution: SelectedGradientContribution)
        -> Result<DataParallelGradients, DataParallelError>;
}

/// Adapter over the original selected replica session, without new kernels,
/// inferred expert ownership, clipping, scaler or optimizer/scheduler steps.
pub struct SelectedReplicaGradientStage<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> {
    session: SelectedDataParallel<B, C>,
    mode: SelectedGradientReductionMode,
    missing: MissingGradientPolicy,
}
impl<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> SelectedReplicaGradientStage<B, C> {
    pub fn new(session: SelectedDataParallel<B, C>, mode: SelectedGradientReductionMode, missing: MissingGradientPolicy) -> Self {
        Self { session, mode, missing }
    }
    pub fn session(&self) -> &SelectedDataParallel<B, C> { &self.session }
    pub fn into_session(self) -> SelectedDataParallel<B, C> { self.session }
}
impl<B: AutodiffBackend, M: AutodiffModule<B>, C: DataParallelCommunicator<B::InnerBackend>> SelectedGradientReduction<B, M>
    for SelectedReplicaGradientStage<B, C> {
    fn contract(&self) -> GradientReductionStageContract {
        GradientReductionStageContract { rank: self.session.rank(), world: self.session.world_size(),
            parameters: self.session.parameters.iter().map(|id| id.val()).collect(), schema: self.session.inner.contract.clone(),
            mode: self.mode, missing: self.missing, buffers: self.session.inner.synchronize_buffers }
    }
    fn validate_model(&self, model: &M) -> Result<(), DataParallelError> { self.session.validate_loaded(model) }
    fn reduce(&self, model: &M, gradients: GradientsParams, contribution: SelectedGradientContribution)
        -> Result<DataParallelGradients, DataParallelError> {
        match (self.mode, contribution) {
            (SelectedGradientReductionMode::Sum, SelectedGradientContribution::Sum { active }) => self.session.sum(model, gradients, active, self.missing),
            (SelectedGradientReductionMode::SumFp32, SelectedGradientContribution::Sum { active }) => self.session.sum_fp32(model, gradients, active, self.missing),
            (SelectedGradientReductionMode::WeightedMean, SelectedGradientContribution::WeightedMean { local_weight }) => self.session.reduce(model, gradients, local_weight, self.missing),
            (SelectedGradientReductionMode::WeightedMeanFp32, SelectedGradientContribution::WeightedMean { local_weight }) => self.session.reduce_fp32(model, gradients, local_weight, self.missing),
            _ => Err(contract("original reduction mode and local contribution kind differ")),
        }
    }
}

/// Ordered original stage bindings and local coordinator topology, without
/// gradient payloads, native communicator serialization or guessed group maps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradientReductionPlanContract {
    version: u32,
    coordinator_rank: u32,
    coordinator_world: u32,
    stages: Vec<GradientReductionStageContract>,
}
impl<B: Backend> Record<B> for GradientReductionPlanContract {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self { self }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self { item }
}
impl GradientReductionPlanContract {
    pub fn stages(&self) -> &[GradientReductionStageContract] { &self.stages }
    pub fn coordinator_rank(&self) -> u32 { self.coordinator_rank }
    pub fn coordinator_world_size(&self) -> u32 { self.coordinator_world }
    pub fn validate(&self) -> Result<(), DataParallelError> {
        if self.version != 1 || self.coordinator_world == 0 || self.coordinator_rank >= self.coordinator_world {
            return Err(contract("invalid original gradient-plan coordinator topology"));
        }
        for stage in &self.stages {
            if stage.world == 0 || stage.rank >= stage.world || stage.world > self.coordinator_world {
                return Err(contract("gradient-plan coordinator must cover every original stage"));
            }
            let mut ids = std::collections::HashSet::new();
            if stage.parameters.iter().any(|id| !ids.insert(*id)) { return Err(contract("duplicate original selected stage parameter")); }
        }
        Ok(())
    }
}

/// Native pending derivatives at an exact stage boundary. Successful previous
/// reductions are retained; the failed stage's original input is retained too.
#[derive(Debug)]
pub struct GradientReductionProgress {
    plan: GradientReductionPlanContract,
    gradients: GradientsParams,
    contributions: Vec<SelectedGradientContribution>,
    next_stage: usize,
    global_weights: Vec<u64>,
}
impl GradientReductionProgress {
    pub fn gradients(&self) -> &GradientsParams { &self.gradients }
    pub fn next_stage(&self) -> usize { self.next_stage }
    pub fn contributions(&self) -> &[SelectedGradientContribution] { &self.contributions }
    pub fn stage_global_weights(&self) -> &[u64] { &self.global_weights }
    pub fn plan_contract(&self) -> &GradientReductionPlanContract { &self.plan }
    pub fn into_gradients(self) -> GradientsParams { self.gradients }
    fn validate(&self) -> Result<(), DataParallelError> {
        validate_progress(&self.plan, &self.contributions, self.next_stage, &self.global_weights)
    }
}

/// Successful full ordered reduction, without an implicit parameter update.
#[derive(Debug)]
pub struct GradientReductionPlanOutput {
    pub gradients: GradientsParams,
    /// SUM stages count active group members; mean stages count original tokens/samples.
    pub stage_global_weights: Vec<u64>,
}

#[derive(Debug)]
pub enum GradientReductionPlanError {
    Stage { index: usize, error: DataParallelError },
    Peer { index: usize, rank: usize, message: String },
    Coordinator(DataParallelError),
}
impl fmt::Display for GradientReductionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { Self::Stage { index, error } => write!(f, "native gradient stage {index}: {error}"),
            Self::Peer { index, rank, message } => write!(f, "gradient stage {index}, coordinator rank {rank}: {message}"),
            Self::Coordinator(error) => write!(f, "gradient-plan coordinator: {error}") }
    }
}
impl Error for GradientReductionPlanError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self { Self::Stage { error, .. } | Self::Coordinator(error) => Some(error), Self::Peer { .. } => None }
    }
}

/// Original failure and retained exact pending stage, not a silent optimizer skip.
#[derive(Debug)]
pub struct GradientReductionPlanFailure {
    pub error: GradientReductionPlanError,
    pub progress: GradientReductionProgress,
}
impl fmt::Display for GradientReductionPlanFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(&self.error, f) }
}
impl Error for GradientReductionPlanFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> { Some(&self.error) }
}

#[derive(Serialize, Deserialize)]
struct PlanWindow { stages: usize, next: usize, error: Option<String> }
#[derive(Serialize, Deserialize)]
struct StageOutcome { error: Option<String> }

/// Explicit ordered native replica reductions. Repeated IDs are allowed across
/// different stages, e.g. separately declared replication axes. No stage, SUM,
/// mean or contribution is selected by shape, parameter name or expert count.
pub struct SelectedGradientReductionPlan<B: AutodiffBackend, M: AutodiffModule<B>, G: DataParallelCommunicator<B::InnerBackend>> {
    stages: Vec<Box<dyn SelectedGradientReduction<B, M>>>,
    coordinator: G,
}
impl<B: AutodiffBackend, M: AutodiffModule<B>, G: DataParallelCommunicator<B::InnerBackend>> SelectedGradientReductionPlan<B, M, G> {
    /// Coordinator membership must contain all actual stage participants; all
    /// members use matching stage count/order and enter one subgroup per slot.
    pub fn new(stages: Vec<Box<dyn SelectedGradientReduction<B, M>>>, coordinator: G) -> Self { Self { stages, coordinator } }
    pub fn len(&self) -> usize { self.stages.len() }
    pub fn is_empty(&self) -> bool { self.stages.is_empty() }
    pub fn contract(&self) -> GradientReductionPlanContract {
        GradientReductionPlanContract { version: 1, coordinator_rank: self.coordinator.rank(), coordinator_world: self.coordinator.world_size(),
            stages: self.stages.iter().map(|stage| stage.contract()).collect() }
    }
    pub fn validate_contract(&self, saved: &GradientReductionPlanContract) -> Result<(), DataParallelError> {
        saved.validate()?;
        if self.contract() != *saved { return Err(contract("original gradient plan topology, selectors or stage options differ")); }
        Ok(())
    }
    pub fn reduce(&self, model: &M, gradients: GradientsParams, contributions: &[SelectedGradientContribution])
        -> Result<GradientReductionPlanOutput, GradientReductionPlanFailure> {
        self.resume(model, GradientReductionProgress { plan: self.contract(), gradients,
            contributions: contributions.to_vec(), next_stage: 0, global_weights: Vec::new() })
    }

    /// Resume the same original inputs/options from an actual pending boundary.
    /// Native source primitives are retained before a stage so peer/transport
    /// failure cannot double-apply a SUM when the caller retries that stage.
    /// This invokes no backward replay, automatic retry, optimizer or scheduler.
    pub fn resume(&self, model: &M, mut progress: GradientReductionProgress)
        -> Result<GradientReductionPlanOutput, GradientReductionPlanFailure> {
        let validation = progress.validate().and_then(|_| self.validate_contract(&progress.plan))
            .and_then(|_| self.stages.iter().try_for_each(|stage| stage.validate_model(model)));
        let window = PlanWindow { stages: self.stages.len(), next: progress.next_stage, error: validation.err().map(|error| error.to_string()) };
        let windows = match gather::<B::InnerBackend, G, _>(&self.coordinator, &window) {
            Ok(windows) => windows, Err(error) => return Err(GradientReductionPlanFailure { error: GradientReductionPlanError::Coordinator(error), progress }),
        };
        for (rank, peer) in windows.iter().enumerate() {
            if let Some(message) = &peer.error {
                return Err(GradientReductionPlanFailure { error: GradientReductionPlanError::Peer { index: progress.next_stage, rank, message: message.clone() }, progress });
            }
            if peer.stages != window.stages || peer.next != window.next {
                return Err(GradientReductionPlanFailure { error: GradientReductionPlanError::Coordinator(contract("gradient-plan stage counts or pending positions differ")), progress });
            }
        }
        while progress.next_stage < self.stages.len() {
            let index = progress.next_stage;
            let original = progress.gradients.clone_native::<B::InnerBackend>();
            let result = self.stages[index].reduce(model, core::mem::take(&mut progress.gradients), progress.contributions[index]);
            let outcome = StageOutcome { error: result.as_ref().err().map(|error| error.to_string()) };
            let outcomes = match gather::<B::InnerBackend, G, _>(&self.coordinator, &outcome) {
                Ok(outcomes) => outcomes,
                Err(error) => { progress.gradients = original; return Err(GradientReductionPlanFailure { error: GradientReductionPlanError::Coordinator(error), progress }); }
            };
            if let Some((rank, peer)) = outcomes.iter().enumerate().find(|(_, peer)| peer.error.is_some()) {
                progress.gradients = original;
                let error = match result { Err(error) => GradientReductionPlanError::Stage { index, error },
                    Ok(_) => GradientReductionPlanError::Peer { index, rank, message: peer.error.clone().unwrap() } };
                return Err(GradientReductionPlanFailure { error, progress });
            }
            match result {
                Ok(reduced) => { progress.gradients = reduced.gradients; progress.global_weights.push(reduced.global_weight); progress.next_stage += 1; }
                Err(error) => { progress.gradients = original; return Err(GradientReductionPlanFailure { error: GradientReductionPlanError::Stage { index, error }, progress }); }
            }
        }
        Ok(GradientReductionPlanOutput { gradients: progress.gradients, stage_global_weights: progress.global_weights })
    }
}

fn validate_progress(plan: &GradientReductionPlanContract, contributions: &[SelectedGradientContribution], next: usize, weights: &[u64])
    -> Result<(), DataParallelError> {
    plan.validate()?;
    if contributions.len() != plan.stages.len() || next > plan.stages.len() || weights.len() != next {
        return Err(contract("original pending stage/weight/contribution counts differ"));
    }
    if plan.stages.iter().zip(contributions).any(|(stage, input)| !stage.mode.accepts(*input)) {
        return Err(contract("original local contribution kind differs from selected stage arithmetic"));
    }
    Ok(())
}

/// Durable original plan position, inputs and exact pending native derivatives.
/// Uses the existing gradient archive without narrowing its native payloads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientReductionProgressRecord {
    version: u32,
    plan: GradientReductionPlanContract,
    contributions: Vec<SelectedGradientContribution>,
    next_stage: usize,
    global_weights: Vec<u64>,
    gradients: GradientsParamsRecord,
}
impl<B: Backend> Record<B> for GradientReductionProgressRecord {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self { self }
    fn from_item<P: PrecisionSettings>(item: Self, _: &B::Device) -> Self { item }
}
impl GradientReductionProgress {
    pub fn to_record<B: AutodiffBackend>(&self) -> Result<GradientReductionProgressRecord, RecorderError> {
        self.validate().map_err(|error| RecorderError::Unknown(error.to_string()))?;
        Ok(GradientReductionProgressRecord { version: 1, plan: self.plan.clone(), contributions: self.contributions.clone(),
            next_stage: self.next_stage, global_weights: self.global_weights.clone(), gradients: self.gradients.try_to_record::<B::InnerBackend>()? })
    }
    pub async fn to_record_async<B: AutodiffBackend>(&self) -> Result<GradientReductionProgressRecord, RecorderError> {
        self.validate().map_err(|error| RecorderError::Unknown(error.to_string()))?;
        Ok(GradientReductionProgressRecord { version: 1, plan: self.plan.clone(), contributions: self.contributions.clone(),
            next_stage: self.next_stage, global_weights: self.global_weights.clone(), gradients: self.gradients.to_record_async::<B::InnerBackend>().await? })
    }
}
impl GradientReductionProgressRecord {
    pub fn plan_contract(&self) -> &GradientReductionPlanContract { &self.plan }
    pub fn next_stage(&self) -> usize { self.next_stage }
    /// Restore actual IDs/shape/work precision on each original model parameter's
    /// own device; actual matching loaded plan/model are validated before resume.
    pub fn restore_for_model<B: AutodiffBackend, M: AutodiffModule<B>>(self, model: &M) -> Result<GradientReductionProgress, RecorderError> {
        if self.version != 1 { return Err(RecorderError::Unknown("unsupported gradient-plan continuation version".into())); }
        validate_progress(&self.plan, &self.contributions, self.next_stage, &self.global_weights)
            .map_err(|error| RecorderError::Unknown(error.to_string()))?;
        let gradients = GradientsParams::from_record_for_model::<B, M>(self.gradients, model)?;
        Ok(GradientReductionProgress { plan: self.plan, contributions: self.contributions, next_stage: self.next_stage,
            global_weights: self.global_weights, gradients })
    }
}
