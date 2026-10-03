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
        Bool, DType, Int, Tensor, TensorPrimitive, backend::AutodiffBackend,
        container::TensorContainer,
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, error::Error, fmt};

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
    device_error: Option<String>,
}

impl Schema {
    fn new() -> Self {
        Self {
            contract: Vec::new(),
            ids: Vec::new(),
            path: Vec::new(),
            aliases: HashMap::new(),
            device_error: None,
        }
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
        let position = self.contract.len();
        let alias = *self.aliases.entry(param.id).or_insert(position);
        if alias != position {
            let previous = &self.contract[alias];
            if previous.shape != tensor.dims().to_vec()
                || previous.dtype != format!("{:?}", tensor.dtype())
                || previous.trainable != tensor.is_require_grad()
            {
                self.device_error =
                    Some("one parameter ID has inconsistent tied tensor metadata".into());
            }
        }
        self.ids.push(param.id);
        self.contract.push(ParameterContract {
            path: self.path.clone(),
            shape: tensor.dims().to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
            trainable: tensor.is_require_grad(),
            alias,
        });
    }
    fn visit_int<const D: usize>(&mut self, _: &Param<Tensor<B, D, Int>>) {
        self.device_error =
            Some("integer parameter buffers require explicit synchronization".into());
    }
    fn visit_bool<const D: usize>(&mut self, _: &Param<Tensor<B, D, Bool>>) {
        self.device_error = Some("Bool parameter buffers require explicit synchronization".into());
    }
}

/// A rank-local replicated training session, independent of model family.
///
/// Uses ruCCL's existing host-staged tensor transport, not NCCL or model sharding.
/// Local parameter IDs may differ between ranks; module paths and alias topology
/// determine collective order. Keep each rank's optimizer and continuation state
/// with its own model record when checkpointing.
#[derive(Debug)]
pub struct DataParallel<B: AutodiffBackend> {
    communicator: RankCommunicator<TensorDevice<B::InnerBackend>>,
    contract: Vec<ParameterContract>,
    ids: Vec<ParamId>,
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
    error: Option<String>,
}
#[derive(Serialize, Deserialize)]
struct Window {
    weight: u64,
    policy: MissingGradientPolicy,
    present: Vec<bool>,
    error: Option<String>,
}

impl<B: AutodiffBackend> DataParallel<B> {
    /// Validate replica structure, then broadcast floating parameters from `root`.
    ///
    /// Call before constructing an optimizer or after restoring matching rank
    /// checkpoints. The returned model retains local IDs, tied parameter aliases,
    /// frozen flags and record mappers. All ranks must enter the same operation.
    pub fn initialize<M: AutodiffModule<B>>(
        communicator: RankCommunicator<TensorDevice<B::InnerBackend>>,
        model: M,
        root: u32,
    ) -> Result<(Self, M), DataParallelError> {
        let mut schema = Schema::new();
        model.visit(&mut schema);
        let mut error = schema.device_error;
        if root >= communicator.world_size() {
            error = Some("broadcast root is outside the world".into());
        }
        if model
            .devices()
            .iter()
            .any(|device| device != communicator.execution().device())
        {
            error = Some("each replica must reside on its communicator's device".into());
        }
        let requests = gather(
            &communicator,
            &Initialization {
                contract: schema.contract.clone(),
                root,
                error,
            },
        )?;
        for (rank, request) in requests.iter().enumerate() {
            if let Some(error) = &request.error {
                return Err(contract(format!("rank {rank}: {error}")));
            }
            if request.root != root || request.contract != schema.contract {
                return Err(contract(
                    "replica paths, shapes, dtypes, frozen flags or tied aliases differ",
                ));
            }
        }
        let mut mapper = Broadcast::<B> {
            communicator: &communicator,
            root,
            tensors: TensorContainer::new(),
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
        let mut schema = Schema::new();
        model.visit(&mut schema);
        let mut check = Check::<B> {
            gradients: &gradients,
            ids: Vec::new(),
            present: Vec::new(),
            error: schema.device_error,
            device: self.communicator.execution().device(),
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
            .any(|device| device != self.communicator.execution().device())
        {
            check.error = Some("replica moved off its communicator's device".into());
        }
        if policy == MissingGradientPolicy::Error
            && local_weight > 0
            && check.present.contains(&false)
        {
            check.error = Some("a trainable parameter is missing its local gradient".into());
        }
        let windows = gather(
            &self.communicator,
            &Window {
                weight: local_weight,
                policy,
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
            if window.policy != policy || window.present.len() != globally_present.len() {
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
        let mut reducer = Reduce::<B> {
            communicator: &self.communicator,
            input: gradients,
            output: GradientsParams::new(),
            visited: Vec::new(),
            index: 0,
            globally_present: &globally_present,
            local_weight,
            global_weight,
            error: None,
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

fn gather<B: ruda_model::tensor::backend::Backend, T: Serialize + DeserializeOwned>(
    communicator: &RankCommunicator<TensorDevice<B>>,
    value: &T,
) -> Result<Vec<T>, DataParallelError> {
    let mut payload = serde_json::to_vec(value).map_err(|e| contract(e.to_string()))?;
    let host = communicator.host();
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
    payload
        .chunks_exact(stride)
        .zip(lengths)
        .map(|(bytes, size)| {
            serde_json::from_slice(&bytes[..size]).map_err(|e| contract(e.to_string()))
        })
        .collect()
}

struct Broadcast<'a, B: AutodiffBackend> {
    communicator: &'a RankCommunicator<TensorDevice<B::InnerBackend>>,
    root: u32,
    tensors: TensorContainer<ParamId>,
    error: Option<TensorDeviceError>,
}
impl<B: AutodiffBackend> ModuleMapper<B> for Broadcast<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if self.error.is_some() {
            return param;
        }
        let tensor = if let Some(tensor) = self.tensors.get::<B>(&param.id) {
            Tensor::from_primitive(tensor)
        } else {
            let value = param.val();
            let trainable = value.is_require_grad();
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
                        .register::<B>(param.id, tensor.clone().into_primitive());
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
}
impl<B: AutodiffBackend> ModuleVisitor<B> for Check<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let value = param.val();
        if !value.is_require_grad() || self.ids.contains(&param.id) {
            return;
        }
        let gradient = self.gradients.get::<B::InnerBackend, D>(param.id);
        if let Some(gradient) = &gradient {
            if gradient.dims() != value.dims()
                || gradient.dtype() != value.dtype()
                || &gradient.device() != self.device
            {
                self.error =
                    Some("gradient shape, dtype or device differs from its parameter".into());
            }
        }
        self.ids.push(param.id);
        self.present.push(gradient.is_some());
    }
}

struct Reduce<'a, B: AutodiffBackend> {
    communicator: &'a RankCommunicator<TensorDevice<B::InnerBackend>>,
    input: GradientsParams,
    output: GradientsParams,
    visited: Vec<ParamId>,
    index: usize,
    globally_present: &'a [bool],
    local_weight: u64,
    global_weight: u64,
    error: Option<TensorDeviceError>,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for Reduce<'_, B> {
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
                        .div_scalar(self.global_weight as f64)
                        .cast(value.dtype());
                self.output.register(param.id, gradient);
            }
            Err(error) => self.error = Some(error),
        }
    }
}

#[cfg(test)]
mod tests;
