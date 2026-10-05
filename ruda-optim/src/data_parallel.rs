//! Replicated data parallelism with explicit rank/device ownership and token weights.

use crate::GradientsParams;
use ruccl::{
    ReduceOperation,
    rank::{ElementType, communicator::RankCommunicator},
    tensor_device::{TensorDevice, TensorDeviceError},
};
use ruda_model::{
    module::{AutodiffModule, ModuleMapper, ModuleVisitor, Param, ParamId},
    tensor::{
        Bool, BoolDType, DType, Int, Tensor, TensorMetadata, TensorPrimitive,
        backend::{AutodiffBackend, Backend},
        container::TensorContainer,
        ops::{BoolTensorOps, FloatTensorOps, IntTensorOps},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, error::Error, fmt, marker::PhantomData};

/// An explicit policy for trainable parameters unused by a local backward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingGradientPolicy {
    /// Require a gradient on each rank with nonzero local weight.
    Error,
    /// Contribute zeros for missing gradients; globally unused parameters remain absent.
    Zero,
}

/// Collective transport failure or a training contract rejected by all ranks.
#[derive(Debug)]
pub enum DataParallelError {
    /// Error from the explicitly selected communicator.
    Collective(TensorDeviceError),
    /// Inconsistent model structure, gradients, root or reduction settings.
    Contract(String),
}

impl fmt::Display for DataParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collective(e) => e.fmt(f),
            Self::Contract(e) => f.write_str(e),
        }
    }
}
impl Error for DataParallelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Collective(e) => Some(e),
            Self::Contract(_) => None,
        }
    }
}
impl From<TensorDeviceError> for DataParallelError {
    fn from(e: TensorDeviceError) -> Self {
        Self::Collective(e)
    }
}

/// Transport for replicated parameters and gradients on an explicitly owned device.
/// Metadata must be returned in rank order. Tensor operations must preserve shape,
/// dtype and the input value, and finish before returning their output.
pub trait DataParallelCommunicator<B: Backend> {
    /// This communicator's rank.
    fn rank(&self) -> u32;
    /// Number of participating ranks.
    fn world_size(&self) -> u32;
    /// Device owned by this rank.
    fn device(&self) -> &B::Device;
    /// Gather variable-length training metadata, in rank order.
    fn all_gather_bytes(&self, payload: Vec<u8>) -> Result<Vec<Vec<u8>>, DataParallelError>;
    /// Broadcast a floating tensor without changing its dtype or shape.
    fn broadcast_float(
        &self,
        value: B::FloatTensorPrimitive,
        root: u32,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError>;
    /// Broadcast an I32/I64 parameter buffer without floating conversion.
    fn broadcast_int(
        &self,
        value: B::IntTensorPrimitive,
        root: u32,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError>;
    /// Reduce floating gradients using the caller's operation.
    fn all_reduce_float(
        &self,
        value: B::FloatTensorPrimitive,
        operation: ReduceOperation,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError>;
}

impl<B: Backend> DataParallelCommunicator<B> for RankCommunicator<TensorDevice<B>> {
    fn rank(&self) -> u32 {
        RankCommunicator::rank(self)
    }
    fn world_size(&self) -> u32 {
        RankCommunicator::world_size(self)
    }
    fn device(&self) -> &B::Device {
        self.execution().device()
    }
    fn all_gather_bytes(&self, mut payload: Vec<u8>) -> Result<Vec<Vec<u8>>, DataParallelError> {
        let host = self.host();
        let (lengths, _) = host
            .all_gather_host_staged(
                ElementType::U64,
                1,
                (payload.len() as u64).to_le_bytes().to_vec(),
            )
            .map_err(TensorDeviceError::from)?;
        let lengths = lengths
            .chunks_exact(8)
            .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
            .map(|size| usize::try_from(size).map_err(|_| contract("metadata size overflow")))
            .collect::<Result<Vec<_>, _>>()?;
        let stride = lengths.iter().copied().max().unwrap_or(1).max(1);
        payload.resize(stride, 0);
        let (payload, _) = host
            .all_gather_host_staged(ElementType::U8, stride, payload)
            .map_err(TensorDeviceError::from)?;
        Ok(payload
            .chunks_exact(stride)
            .zip(lengths)
            .map(|(bytes, size)| bytes[..size].to_vec())
            .collect())
    }
    fn broadcast_float(
        &self,
        value: B::FloatTensorPrimitive,
        root: u32,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        RankCommunicator::broadcast_float(self, value, root)
    }
    fn broadcast_int(
        &self,
        value: B::IntTensorPrimitive,
        root: u32,
    ) -> Result<B::IntTensorPrimitive, TensorDeviceError> {
        RankCommunicator::broadcast_int(self, value, root)
    }
    fn all_reduce_float(
        &self,
        value: B::FloatTensorPrimitive,
        operation: ReduceOperation,
    ) -> Result<B::FloatTensorPrimitive, TensorDeviceError> {
        RankCommunicator::all_reduce_float(self, value, operation)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ParameterContract {
    path: Vec<String>,
    shape: Vec<usize>,
    dtype: String,
    trainable: bool,
    alias: usize,
}

#[derive(Debug)]
struct Schema {
    contract: Vec<ParameterContract>,
    ids: Vec<ParamId>,
    path: Vec<String>,
    aliases: HashMap<ParamId, usize>,
    storage_aliases: HashMap<(ParamId, bool), usize>,
    synchronize_buffers: bool,
    device_error: Option<String>,
}

impl Schema {
    fn new(synchronize_buffers: bool) -> Self {
        Self {
            contract: Vec::new(),
            ids: Vec::new(),
            path: Vec::new(),
            aliases: HashMap::new(),
            storage_aliases: HashMap::new(),
            synchronize_buffers,
            device_error: None,
        }
    }

    fn register(&mut self, id: ParamId, shape: Vec<usize>, dtype: DType, trainable: bool) {
        let position = self.contract.len();
        let alias = *self.aliases.entry(id).or_insert(position);
        let storage_alias = *self
            .storage_aliases
            .entry((id, trainable))
            .or_insert(position);
        let dtype = format!("{dtype:?}");
        if storage_alias != position {
            let previous = &self.contract[storage_alias];
            if previous.shape != shape || previous.dtype != dtype {
                self.device_error =
                    Some("one parameter ID has inconsistent tied tensor metadata".into());
            }
        }
        self.ids.push(id);
        self.contract.push(ParameterContract {
            path: self.path.clone(),
            shape,
            dtype,
            trainable,
            alias,
        });
    }
}
impl<B: AutodiffBackend> ModuleVisitor<B> for Schema {
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        if !matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            self.device_error =
                Some("data parallelism supports F32, F16 and BF16 parameters".into());
        }
        self.register(
            param.id,
            tensor.dims().to_vec(),
            tensor.dtype(),
            tensor.is_require_grad(),
        );
    }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        if !self.synchronize_buffers {
            self.device_error =
                Some("integer parameter buffers require explicit synchronization".into());
            return;
        }
        let tensor = param.val();
        if !matches!(tensor.dtype(), DType::I32 | DType::I64) {
            self.device_error = Some("data parallel buffers support I32 and I64 integers".into());
        }
        self.register(param.id, tensor.dims().to_vec(), tensor.dtype(), false);
    }
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        if !self.synchronize_buffers {
            self.device_error =
                Some("Bool parameter buffers require explicit synchronization".into());
            return;
        }
        let tensor = param.val();
        self.register(param.id, tensor.dims().to_vec(), tensor.dtype(), false);
    }
}

/// A rank-local replicated training session, independent of model family.
///
/// Defaults to ruCCL's host-staged tensor transport. A device-native communicator
/// can reuse the same replica, gradient-weighting and optimizer contracts.
/// Local parameter IDs may differ between ranks; module paths and alias topology
/// determine collective order. Keep each rank's optimizer and continuation state
/// with its own model record when checkpointing.
#[derive(Debug)]
pub struct DataParallel<
    B: AutodiffBackend,
    C: DataParallelCommunicator<B::InnerBackend> = RankCommunicator<TensorDevice<B::InnerBackend>>,
> {
    communicator: C,
    contract: Vec<ParameterContract>,
    ids: Vec<ParamId>,
    synchronize_buffers: bool,
    backend: PhantomData<B>,
}

/// Globally token-weighted gradients for one accumulation window.
#[derive(Debug)]
pub struct DataParallelGradients {
    /// Sum of local unnormalized gradients divided by `global_weight`.
    pub gradients: GradientsParams,
    /// Exact sum of caller-supplied sample/token weights across ranks.
    pub global_weight: u64,
}

#[derive(Serialize, Deserialize)]
struct Initialization {
    contract: Vec<ParameterContract>,
    root: u32,
    synchronize_buffers: bool,
    error: Option<String>,
}
#[derive(Serialize, Deserialize)]
struct Window {
    weight: u64,
    policy: MissingGradientPolicy,
    fp32_gradients: bool,
    present: Vec<bool>,
    error: Option<String>,
}

impl<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> DataParallel<B, C> {
    /// Validate replica structure, then broadcast floating parameters from `root`.
    ///
    /// Call before constructing an optimizer or after restoring matching rank
    /// checkpoints. The returned model retains local IDs, tied parameter aliases,
    /// frozen flags and record mappers. All ranks must enter the same operation.
    pub fn initialize<M: AutodiffModule<B>>(
        communicator: C,
        model: M,
        root: u32,
    ) -> Result<(Self, M), DataParallelError> {
        Self::initialize_inner(communicator, model, root, false)
    }

    /// Initialize a replica and explicitly broadcast integer and Bool parameter buffers too.
    /// I32/I64 buffers retain their width; Bool buffers are transported as integer 0/1.
    /// Buffers retain local IDs and aliases and never enter gradient reduction or optimization.
    /// This synchronizes buffers once, not automatically before each forward pass.
    pub fn initialize_with_buffers<M: AutodiffModule<B>>(
        communicator: C,
        model: M,
        root: u32,
    ) -> Result<(Self, M), DataParallelError> {
        Self::initialize_inner(communicator, model, root, true)
    }

    fn initialize_inner<M: AutodiffModule<B>>(
        communicator: C,
        model: M,
        root: u32,
        synchronize_buffers: bool,
    ) -> Result<(Self, M), DataParallelError> {
        let mut schema = Schema::new(synchronize_buffers);
        model.visit(&mut schema);
        let mut error = schema.device_error;
        if root >= communicator.world_size() {
            error = Some("broadcast root is outside the world".into());
        }
        if model
            .devices()
            .iter()
            .any(|device| device != communicator.device())
        {
            error = Some("each replica must reside on its communicator's device".into());
        }
        let requests = gather::<B::InnerBackend, C, _>(
            &communicator,
            &Initialization {
                contract: schema.contract.clone(),
                root,
                synchronize_buffers,
                error,
            },
        )?;
        for (rank, request) in requests.iter().enumerate() {
            if let Some(error) = &request.error {
                return Err(contract(format!("rank {rank}: {error}")));
            }
            if request.root != root
                || request.contract != schema.contract
                || request.synchronize_buffers != synchronize_buffers
            {
                return Err(contract(
                    "replica paths, shapes, dtypes, frozen flags or tied aliases differ",
                ));
            }
        }
        let mut mapper = Broadcast::<B, C> {
            communicator: &communicator,
            root,
            tensors: TensorContainer::new(),
            integers: HashMap::new(),
            booleans: HashMap::new(),
            error: None,
        };
        let model = model.map(&mut mapper);
        if let Some(error) = mapper.error {
            return Err(error.into());
        }
        Ok((
            Self {
                communicator,
                contract: schema.contract,
                ids: schema.ids,
                synchronize_buffers,
                backend: PhantomData,
            },
            model,
        ))
    }

    /// This replica's rank.
    pub fn rank(&self) -> u32 {
        self.communicator.rank()
    }
    /// Number of replicas.
    pub fn world_size(&self) -> u32 {
        self.communicator.world_size()
    }

    /// Synchronize accumulated gradients of **local loss sums**, not local means.
    ///
    /// `local_weight` is their effective sample/token count over the entire
    /// window. This gives a global token mean even with unequal batch sizes,
    /// ignored labels or a rank with no supervised tokens. Synchronize only at
    /// an accumulation boundary; perform the optimizer update afterwards.
    /// No scheduler, optimizer, clipping or accumulator reset is implicit.
    /// Half-precision gradients are summed and normalized in FP32, then cast
    /// back to their parameter dtype for the existing optimizer contract.
    pub fn reduce<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
    ) -> Result<DataParallelGradients, DataParallelError> {
        self.reduce_inner(model, gradients, local_weight, policy, false)
    }

    /// Reduce into FP32 gradients for explicit FP32-master optimizer updates.
    ///
    /// Accepts gradients in their parameter dtype or FP32, including FP32
    /// accumulation for half-precision parameters. Loss-sum weighting and missing
    /// gradient semantics match `reduce`. All ranks must select this same method.
    pub fn reduce_fp32<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
    ) -> Result<DataParallelGradients, DataParallelError> {
        self.reduce_inner(model, gradients, local_weight, policy, true)
    }

    fn reduce_inner<M: AutodiffModule<B>>(
        &self,
        model: &M,
        gradients: GradientsParams,
        local_weight: u64,
        policy: MissingGradientPolicy,
        fp32_gradients: bool,
    ) -> Result<DataParallelGradients, DataParallelError> {
        let mut schema = Schema::new(self.synchronize_buffers);
        model.visit(&mut schema);
        let mut check = Check::<B> {
            gradients: &gradients,
            ids: Vec::new(),
            present: Vec::new(),
            error: schema.device_error,
            device: self.communicator.device(),
            fp32_gradients,
        };
        model.visit(&mut check);
        if schema.contract != self.contract || schema.ids != self.ids {
            check.error =
                Some("model structure or parameter IDs changed after initialization".into());
        }
        if check.present.iter().filter(|present| **present).count() != gradients.len() {
            check.error = Some("gradient container includes an unknown or frozen parameter".into());
        }
        if model
            .devices()
            .iter()
            .any(|device| device != self.communicator.device())
        {
            check.error = Some("replica moved off its communicator's device".into());
        }
        if policy == MissingGradientPolicy::Error
            && local_weight > 0
            && check.present.contains(&false)
        {
            check.error = Some("a trainable parameter is missing its local gradient".into());
        }
        let windows = gather::<B::InnerBackend, C, _>(
            &self.communicator,
            &Window {
                weight: local_weight,
                policy,
                fp32_gradients,
                present: check.present.clone(),
                error: check.error,
            },
        )?;
        let mut global_weight = 0_u64;
        let mut globally_present = vec![false; check.present.len()];
        for (rank, window) in windows.iter().enumerate() {
            if let Some(error) = &window.error {
                return Err(contract(format!("rank {rank}: {error}")));
            }
            if window.policy != policy
                || window.fp32_gradients != fp32_gradients
                || window.present.len() != globally_present.len()
            {
                return Err(contract("ranks disagree on gradient reduction policy"));
            }
            global_weight = global_weight
                .checked_add(window.weight)
                .ok_or_else(|| contract("global weight overflow"))?;
            if window.weight > 0 {
                for (any, present) in globally_present.iter_mut().zip(&window.present) {
                    *any |= present;
                }
            }
        }
        if global_weight == 0 {
            return Err(contract("cannot normalize a zero-weight global window"));
        }
        let mut reducer = Reduce::<B, C> {
            communicator: &self.communicator,
            input: gradients,
            output: GradientsParams::new(),
            visited: Vec::new(),
            index: 0,
            globally_present: &globally_present,
            local_weight,
            global_weight,
            fp32_gradients,
            error: None,
            backend: PhantomData,
        };
        model.visit(&mut reducer);
        if let Some(error) = reducer.error {
            return Err(error.into());
        }
        Ok(DataParallelGradients {
            gradients: reducer.output,
            global_weight,
        })
    }
}

fn contract(message: impl Into<String>) -> DataParallelError {
    DataParallelError::Contract(message.into())
}

fn gather<B: Backend, C: DataParallelCommunicator<B>, T: Serialize + DeserializeOwned>(
    communicator: &C,
    value: &T,
) -> Result<Vec<T>, DataParallelError> {
    let payload = serde_json::to_vec(value).map_err(|e| contract(e.to_string()))?;
    communicator
        .all_gather_bytes(payload)?
        .into_iter()
        .map(|bytes| serde_json::from_slice(&bytes).map_err(|e| contract(e.to_string())))
        .collect()
}

struct Broadcast<'a, B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> {
    communicator: &'a C,
    root: u32,
    tensors: TensorContainer<(ParamId, bool)>,
    integers: HashMap<ParamId, B::IntTensorPrimitive>,
    booleans: HashMap<ParamId, B::BoolTensorPrimitive>,
    error: Option<TensorDeviceError>,
}
impl<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> ModuleMapper<B>
    for Broadcast<'_, B, C>
{
    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        if self.error.is_some() {
            return param;
        }
        let tensor = if let Some(value) = self.integers.get(&param.id) {
            Tensor::<B, D, Int>::from_primitive(value.clone())
        } else {
            match self
                .communicator
                .broadcast_int(param.val().inner().into_primitive(), self.root)
            {
                Ok(value) => {
                    let tensor = Tensor::<B, D, Int>::from_inner(
                        Tensor::<B::InnerBackend, D, Int>::from_primitive(value),
                    );
                    self.integers
                        .insert(param.id, tensor.clone().into_primitive());
                    tensor
                }
                Err(error) => {
                    self.error = Some(error);
                    return param;
                }
            }
        };
        let (id, _, mapper) = param.consume();
        Param::from_mapped_value(id, tensor, mapper)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        if self.error.is_some() {
            return param;
        }
        let tensor = if let Some(value) = self.booleans.get(&param.id) {
            Tensor::<B, D, Bool>::from_primitive(value.clone())
        } else {
            let value = param.val().inner();
            let dtype: BoolDType = value.dtype().into();
            let integers = value.int().cast(ruda_model::tensor::IntDType::I32);
            match self
                .communicator
                .broadcast_int(integers.into_primitive(), self.root)
            {
                Ok(value) => {
                    let zeros = B::InnerBackend::int_equal_elem(value, 0.into(), dtype);
                    let tensor = Tensor::<B, D, Bool>::from_inner(
                        Tensor::<B::InnerBackend, D, Bool>::from_primitive(
                            B::InnerBackend::bool_not(zeros),
                        ),
                    );
                    self.booleans
                        .insert(param.id, tensor.clone().into_primitive());
                    tensor
                }
                Err(error) => {
                    self.error = Some(error);
                    return param;
                }
            }
        };
        let (id, _, mapper) = param.consume();
        Param::from_mapped_value(id, tensor, mapper)
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if self.error.is_some() {
            return param;
        }
        let value = param.val();
        let trainable = value.is_require_grad();
        let key = (param.id, trainable);
        let tensor = if let Some(tensor) = self.tensors.get::<B>(&key) {
            Tensor::from_primitive(tensor)
        } else {
            match self
                .communicator
                .broadcast_float(value.inner().into_primitive().tensor(), self.root)
            {
                Ok(value) => {
                    let tensor = Tensor::<B, D>::from_inner(Tensor::from_primitive(
                        TensorPrimitive::Float(value),
                    ))
                    .set_require_grad(trainable);
                    self.tensors
                        .register::<B>(key, tensor.clone().into_primitive());
                    tensor
                }
                Err(error) => {
                    self.error = Some(error);
                    return param;
                }
            }
        };
        param.map(|_| tensor)
    }
}

struct Check<'a, B: AutodiffBackend> {
    gradients: &'a GradientsParams,
    ids: Vec<ParamId>,
    present: Vec<bool>,
    error: Option<String>,
    device: &'a B::Device,
    fp32_gradients: bool,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for Check<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let value = param.val();
        if !value.is_require_grad() || self.ids.contains(&param.id) {
            return;
        }
        let gradient = self.gradients.primitive::<B::InnerBackend>(param.id);
        if let Some(gradient) = &gradient {
            let valid = match gradient {
                TensorPrimitive::Float(gradient) => {
                    gradient.shape() == value.shape()
                        && (gradient.dtype() == value.dtype()
                            || (self.fp32_gradients && gradient.dtype() == DType::F32))
                        && &B::InnerBackend::float_device(gradient) == self.device
                }
                TensorPrimitive::QFloat(_) => false,
            };
            if !valid {
                self.error =
                    Some("gradient shape, dtype or device differs from its parameter".into());
            }
        }
        self.ids.push(param.id);
        self.present.push(gradient.is_some());
    }
}

struct Reduce<'a, B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> {
    communicator: &'a C,
    input: GradientsParams,
    output: GradientsParams,
    visited: Vec<ParamId>,
    index: usize,
    globally_present: &'a [bool],
    local_weight: u64,
    global_weight: u64,
    fp32_gradients: bool,
    error: Option<TensorDeviceError>,
    backend: PhantomData<B>,
}
impl<B: AutodiffBackend, C: DataParallelCommunicator<B::InnerBackend>> ModuleVisitor<B>
    for Reduce<'_, B, C>
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let value = param.val();
        if !value.is_require_grad() || self.visited.contains(&param.id) || self.error.is_some() {
            return;
        }
        self.visited.push(param.id);
        let present = self.globally_present[self.index];
        self.index += 1;
        if !present {
            return;
        }
        let zero = || {
            Tensor::<B::InnerBackend, D>::zeros(value.dims(), &value.device()).cast(value.dtype())
        };
        let gradient = if self.local_weight == 0 {
            zero()
        } else {
            self.input
                .remove::<B::InnerBackend, D>(param.id)
                .unwrap_or_else(zero)
        };
        let gradient = gradient.cast(DType::F32);
        match self
            .communicator
            .all_reduce_float(gradient.into_primitive().tensor(), ReduceOperation::Sum)
        {
            Ok(gradient) => {
                let gradient =
                    Tensor::<B::InnerBackend, D>::from_primitive(TensorPrimitive::Float(gradient))
                        .div_scalar(self.global_weight as f64);
                let gradient = if self.fp32_gradients {
                    gradient
                } else {
                    gradient.cast(value.dtype())
                };
                self.output.register(param.id, gradient);
            }
            Err(error) => self.error = Some(error),
        }
    }
}

#[cfg(test)]
mod tests;
