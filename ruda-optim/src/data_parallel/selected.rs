//! Explicit replica groups over parameters of the original rank-local model.

use super::*;
use std::collections::HashSet;

pub(super) fn visit_selection<
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    V: ModuleVisitor<B>,
>(model: &M, visitor: &mut V, selection: Option<&[ParamId]>) -> bool {
    let Some(selection) = selection else {
        model.visit(visitor);
        return true;
    };
    let mut selected = SelectionVisitor {
        inner: visitor,
        ids: selection.iter().copied().collect(),
        seen: HashSet::new(),
    };
    model.visit(&mut selected);
    selected.ids.len() == selection.len() && selected.ids == selected.seen
}

struct SelectionVisitor<'a, V> {
    inner: &'a mut V,
    ids: HashSet<ParamId>,
    seen: HashSet<ParamId>,
}

impl<B: AutodiffBackend, V: ModuleVisitor<B>> ModuleVisitor<B> for SelectionVisitor<'_, V> {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.inner.enter_module(name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        self.inner.exit_module(name, container_type);
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.ids.contains(&param.id) {
            self.seen.insert(param.id);
            self.inner.visit_float(param);
        }
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        if self.ids.contains(&param.id) {
            self.seen.insert(param.id);
            self.inner.visit_int(param);
        }
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        if self.ids.contains(&param.id) {
            self.seen.insert(param.id);
            self.inner.visit_bool(param);
        }
    }
}

pub(super) fn map_selection<
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    V: ModuleMapper<B>,
>(model: M, mapper: &mut V, selection: Option<&[ParamId]>) -> M {
    let Some(selection) = selection else {
        return model.map(mapper);
    };
    model.map(&mut SelectionMapper {
        inner: mapper,
        ids: selection.iter().copied().collect(),
    })
}

struct SelectionMapper<'a, V> {
    inner: &'a mut V,
    ids: HashSet<ParamId>,
}

impl<B: AutodiffBackend, V: ModuleMapper<B>> ModuleMapper<B> for SelectionMapper<'_, V> {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.inner.enter_module(name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        self.inner.exit_module(name, container_type);
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if self.ids.contains(&param.id) {
            self.inner.map_float(param)
        } else {
            param
        }
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<B, D, Int>>) -> Param<Tensor<B, D, Int>> {
        if self.ids.contains(&param.id) {
            self.inner.map_int(param)
        } else {
            param
        }
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        if self.ids.contains(&param.id) {
            self.inner.map_bool(param)
        } else {
            param
        }
    }
}

pub(super) fn selected_device_matches<B: AutodiffBackend, M: AutodiffModule<B>>(
    model: &M,
    device: &B::Device,
    selection: Option<&[ParamId]>,
) -> bool {
    let Some(selection) = selection else {
        return model.devices().iter().all(|actual| actual == device);
    };
    let mut visitor = SelectedDevice::<B> {
        device,
        matches: true,
    };
    let known = visit_selection::<B, _, _>(model, &mut visitor, Some(selection));
    known && visitor.matches
}

struct SelectedDevice<'a, B: AutodiffBackend> {
    device: &'a B::Device,
    matches: bool,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for SelectedDevice<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.matches &= param.val().device() == *self.device;
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        self.matches &= param.val().device() == *self.device;
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        self.matches &= param.val().device() == *self.device;
    }
}

/// Replicate an explicit floating-parameter subset of the actual rank-local model.
///
/// Select non-expert weights on an expert group, or one owned expert's replicas
/// on its separate data group, without imposing a common schema on other weights.
/// Unselected parameters may have different shapes, dtypes or devices across ranks.
/// They are neither initialized, broadcast, converted nor moved by this session.
/// Parameter IDs select every tied occurrence; original full paths establish
/// collective order, so IDs may differ between ranks. Group membership and roots
/// are caller-supplied. Transport defaults to ruCCL's host-staged implementation.
#[derive(Debug)]
pub struct SelectedDataParallel<
    B: AutodiffBackend,
    C: DataParallelCommunicator<B::InnerBackend> = RankCommunicator<
        TensorDevice<<B as AutodiffBackend>::InnerBackend>,
    >,
> {
    pub(super) inner: DataParallel<B, C>,
    pub(super) parameters: Vec<ParamId>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct ReductionMode {
    normalize: bool,
    fp32: bool,
}

impl<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> SelectedDataParallel<B, C> {
    /// Read actual loaded selected metadata without initialization, mutation or
    /// communication; reductions retain their existing collective agreement.
    pub fn validate_loaded<M: AutodiffModule<B>>(&self, model: &M) -> Result<(), DataParallelError> {
        self.inner.validate_loaded_inner(model, Some(&self.parameters))
    }
    /// Bind only explicit caller-loaded replica IDs without copying values or
    /// replacing existing autodiff leaves, aliases, IDs or record mappers.
    /// This can attach SUM-only groups to restored FSDP/EP models/optimizers.
    pub fn bind_loaded<M: AutodiffModule<B>>(communicator: C, model: &M, parameters: &[ParamId])
        -> Result<Self, DataParallelError> {
        let inner = DataParallel::bind_inner(communicator, model, 0, false, Some(parameters), true)?;
        Ok(Self { inner, parameters: parameters.to_vec() })
    }

    /// Validate selected native NF4/AWQ/integer/Bool state metadata too, retaining
    /// the actual payloads unchanged and outside floating gradient reduction.
    pub fn bind_loaded_with_buffers<M: AutodiffModule<B>>(communicator: C, model: &M, parameters: &[ParamId])
        -> Result<Self, DataParallelError> {
        let inner = DataParallel::bind_inner(communicator, model, 0, true, Some(parameters), true)?;
        Ok(Self { inner, parameters: parameters.to_vec() })
    }

    /// Collectively validate and broadcast only the selected F32/F16/BF16 parameters.
    ///
    /// The actual full model is returned with its original IDs, frozen flags,
    /// tied aliases and record mappers. Duplicate or absent selected IDs are
    /// rejected collectively. All ranks must select matching paths, tensor
    /// metadata and alias topology, but need not have matching unselected experts.
    /// An empty subset is valid. Integer/Bool buffers must not be selected here;
    /// unselected packed quantized bases remain untouched. Initialize before
    /// constructing optimizer state for these replicas.
    pub fn initialize<M: AutodiffModule<B>>(
        communicator: C,
        model: M,
        root: u32,
        parameters: &[ParamId],
    ) -> Result<(Self, M), DataParallelError> {
        let (inner, model) =
            DataParallel::initialize_inner(communicator, model, root, false, Some(parameters))?;
        Ok((
            Self {
                inner,
                parameters: parameters.to_vec(),
            },
            model,
        ))
    }

    /// Initialize selected floating weights and native integer/Bool parameter buffers together.
    ///
    /// Select actual NF4 U8 bytes and original scales/codebook, or AWQ I32/U32
    /// words and scales/bias, together with the desired adapter parameters.
    /// Integer payloads retain their native width and bit pattern; no dense base,
    /// requantization, scale conversion or optimizer state is created for them.
    /// Quantization/architecture settings remain those of the caller's prepared
    /// original model. This broadcasts once, not before every forward pass.
    /// Unselected experts/parameters/devices remain outside this group's schema.
    pub fn initialize_with_buffers<M: AutodiffModule<B>>(
        communicator: C,
        model: M,
        root: u32,
        parameters: &[ParamId],
    ) -> Result<(Self, M), DataParallelError> {
        let (inner, model) =
            DataParallel::initialize_inner(communicator, model, root, true, Some(parameters))?;
        Ok((Self { inner, parameters: parameters.to_vec() }, model))
    }

    /// Rank within this explicit replica group.
    pub fn rank(&self) -> u32 {
        self.inner.rank()
    }

    /// Number of participants in this explicit replica group.
    pub fn world_size(&self) -> u32 {
        self.inner.world_size()
    }

    /// Original rank-local parameter IDs selected at initialization.
    pub fn parameters(&self) -> &[ParamId] {
        &self.parameters
    }

    pub(super) fn into_parts(self) -> (DataParallel<B, C>, Vec<ParamId>) {
        (self.inner, self.parameters)
    }

    /// Reduce selected local loss-sum derivatives into a global weighted mean.
    ///
    /// `local_weight` is the effective sample/token count of the accumulation
    /// window. All unselected derivatives in the supplied full gradient container
    /// retain their native handles and dtype. Half gradients communicate in FP32
    /// and cast back to parameter dtype. This performs no optimizer update,
    /// clipping, scheduling or accumulator reset.
    pub fn reduce<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
    ) -> Result<DataParallelGradients, DataParallelError> {
        self.reduce_selected(model, gradients, local_weight, policy, false, true)
    }

    /// Weighted-mean reduction with selected derivatives retained in FP32.
    /// Accepts original parameter dtype or FP32 for FP32-master optimization.
    /// Unselected derivatives are not cast or normalized.
    pub fn reduce_fp32<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
    ) -> Result<DataParallelGradients, DataParallelError> {
        self.reduce_selected(model, gradients, local_weight, policy, true, true)
    }

    /// Sum selected derivatives without dividing by group size or token count.
    ///
    /// Use for contributions already normalized by a global expert-parallel
    /// objective. `local_active` controls contribution eligibility, not scaling:
    /// inactive ranks contribute zero. The result's `global_weight` counts active
    /// participants and is not a token denominator. Globally absent derivatives
    /// remain absent; an all-inactive window returns only unchanged unselected
    /// derivatives. All ranks must enter the same reduction method and policy.
    pub fn sum<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_active: bool,
        policy: MissingGradientPolicy,
    ) -> Result<DataParallelGradients, DataParallelError> {
        self.reduce_selected(model, gradients, u64::from(local_active), policy, false, false)
    }

    /// SUM-only selected reduction retaining FP32 derivatives for master weights.
    /// Contribution eligibility and unchanged unselected derivatives match `sum`.
    pub fn sum_fp32<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_active: bool,
        policy: MissingGradientPolicy,
    ) -> Result<DataParallelGradients, DataParallelError> {
        self.reduce_selected(model, gradients, u64::from(local_active), policy, true, false)
    }

    fn reduce_selected<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
        fp32: bool,
        normalize: bool,
    ) -> Result<DataParallelGradients, DataParallelError> {
        reduce_selected(&self.inner, &self.parameters, model, gradients,
            local_weight, policy, fp32, normalize)
    }
}

pub(super) fn reduce_selected<
    B: AutodiffBackend,
    C: DataParallelCommunicator<B::InnerBackend>,
    M: AutodiffModule<B>,
>(
    session: &DataParallel<B, C>,
    parameters: &[ParamId],
    model: &M,
    gradients: GradientsParams,
    local_weight: u64,
    policy: MissingGradientPolicy,
    fp32: bool,
    normalize: bool,
) -> Result<DataParallelGradients, DataParallelError> {
    agree_mode(session, fp32, normalize)?;
    let (selected, untouched) = gradients.partition::<B::InnerBackend>(parameters);
    let reduced = session.reduce_inner(
        model, selected, local_weight, policy, fp32, Some(parameters), normalize,
    )?;
    let gradients = reduced.gradients
        .merge_disjoint::<B::InnerBackend>(untouched)
        .map_err(|error| contract(error.to_string()))?;
    Ok(DataParallelGradients {
        gradients,
        global_weight: reduced.global_weight,
    })
}

pub(super) fn agree_mode<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>>(
    session: &DataParallel<B, C>,
    fp32: bool,
    normalize: bool,
) -> Result<(), DataParallelError> {
    let mode = ReductionMode { normalize, fp32 };
    let modes = gather::<B::InnerBackend, C, _>(&session.communicator, &mode)?;
    if modes.iter().any(|other| other != &mode) {
        return Err(contract("selected replicas disagree on SUM/mean or FP32 reduction"));
    }
    Ok(())
}
