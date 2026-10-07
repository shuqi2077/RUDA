use super::{
    AutodiffBackend, AutodiffModule, Backend, GradientsParams, LBFGS, LBFGSState,
    LearningRate, Tensor, ToElement, flatten_params_inner,
    reductions::VectorReductions,
};
use alloc::vec::Vec;
use core::{fmt, ops::Range};
use ruda_model::{
    record::{PrecisionSettings, Record},
    tensor::{BroadcastTensorCollective, TensorPrimitive},
};
use serde::{Deserialize, Serialize};

/// Disjoint contiguous pieces of the original ordered, flattened trainable parameter vector.
/// Every rank supplies its actual unique-parameter visitation order and a nonempty local piece.
/// Replicated parameters and data-parallel reductions must be handled separately.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LBFGSShardLayout {
    /// Real unpadded flattened lengths in communicator rank order.
    pub lengths: Vec<usize>,
}

impl LBFGSShardLayout {
    /// Declare the original rank-ordered vector partitions without guessing a model layout.
    pub fn new(lengths: Vec<usize>) -> Self {
        Self { lengths }
    }

    /// Validate actual rank/world placement and checked complete-vector geometry.
    pub fn validate(&self, rank: u32, world: u32) -> Result<(), LBFGSShardError> {
        if world == 0 || rank >= world || self.lengths.len() != world as usize {
            return Err(LBFGSShardError::Layout("rank/world does not match vector layout"));
        }
        if self.lengths.contains(&0) {
            return Err(LBFGSShardError::Layout("each vector shard must be nonempty"));
        }
        self.global_len()?;
        Ok(())
    }

    /// Complete real vector length, excluding transport padding or replicated storage.
    pub fn global_len(&self) -> Result<usize, LBFGSShardError> {
        self.lengths.iter().try_fold(0usize, |total, length| {
            total.checked_add(*length)
                .ok_or(LBFGSShardError::Layout("complete vector length overflows"))
        })
    }

    /// This rank's original complete-vector interval.
    pub fn range(&self, rank: u32) -> Result<Range<usize>, LBFGSShardError> {
        let rank = rank as usize;
        if rank >= self.lengths.len() {
            return Err(LBFGSShardError::Layout("rank is outside vector layout"));
        }
        let start = self.lengths[..rank].iter().try_fold(0usize, |total, length| {
            total.checked_add(*length)
                .ok_or(LBFGSShardError::Layout("vector interval overflows"))
        })?;
        let end = start.checked_add(self.lengths[rank])
            .ok_or(LBFGSShardError::Layout("vector interval end overflows"))?;
        Ok(start..end)
    }
}

/// Native sharded L-BFGS placement, history or transport-result contract failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LBFGSShardError {
    /// Invalid explicitly declared vector partition.
    Layout(&'static str),
    /// A native history/gradient/direction vector has incompatible geometry.
    Shape(&'static str),
    /// Actual vector precision differs from the original parameter/history precision.
    DType,
    /// Actual vector resides on a different local device.
    Device,
    /// Serialized schema, owning rank or original complete layout differs.
    Record,
}

impl fmt::Display for LBFGSShardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(message) => write!(formatter, "L-BFGS shard layout: {message}"),
            Self::Shape(message) => write!(formatter, "L-BFGS vector shape: {message}"),
            Self::DType => write!(formatter, "L-BFGS vector precision does not match"),
            Self::Device => write!(formatter, "L-BFGS vector device does not match"),
            Self::Record => write!(formatter, "L-BFGS shard checkpoint placement does not match"),
        }
    }
}

impl core::error::Error for LBFGSShardError {}

/// Returned native shard contract or actual communicator error; no local update fallback.
#[derive(Debug)]
pub enum LBFGSShardedError<E: fmt::Debug> {
    /// Native state/parameter/transport-result contract failure.
    State(LBFGSShardError),
    /// Actual collective failure.
    Collective(E),
}

impl<E: fmt::Debug> From<LBFGSShardError> for LBFGSShardedError<E> {
    fn from(error: LBFGSShardError) -> Self {
        Self::State(error)
    }
}

impl<E: fmt::Debug> fmt::Display for LBFGSShardedError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(error) => write!(formatter, "{error}"),
            Self::Collective(error) => write!(formatter, "L-BFGS collective failed: {error:?}"),
        }
    }
}

impl<E: fmt::Debug> core::error::Error for LBFGSShardedError<E> {}

impl<B: Backend> LBFGSState<B> {
    fn vectors(&self) -> impl Iterator<Item = &Tensor<B, 1>> {
        self.history_s.iter().chain(self.history_y.iter())
            .chain(self.d.iter()).chain(self.prev_flat_grad.iter())
    }

    fn validate_vectors(
        &self,
        length: usize,
        reference: Option<&Tensor<B, 1>>,
    ) -> Result<(), LBFGSShardError> {
        if self.history_s.len() != self.history_y.len() {
            return Err(LBFGSShardError::Shape("history displacement/gradient counts differ"));
        }
        let reference = reference.or_else(|| self.vectors().next());
        for value in self.vectors() {
            if value.dims() != [length] {
                return Err(LBFGSShardError::Shape("native history/direction/gradient length"));
            }
            if let Some(reference) = reference {
                if value.dtype() != reference.dtype() {
                    return Err(LBFGSShardError::DType);
                }
                if value.device() != reference.device() {
                    return Err(LBFGSShardError::Device);
                }
            }
        }
        Ok(())
    }

    /// Slice actual complete flattened histories/direction/gradient into a declared rank interval.
    /// Preserve the original iteration, step, previous loss and genuinely absent optional vectors.
    /// The original full vector must use rank-concatenated unique-parameter visitation order.
    pub fn try_into_shard(
        self,
        layout: &LBFGSShardLayout,
        rank: u32,
        world: u32,
    ) -> Result<LBFGSShardedState<B>, LBFGSShardError> {
        layout.validate(rank, world)?;
        self.validate_vectors(layout.global_len()?, None)?;
        let interval = layout.range(rank)?;
        let slice = |value: Tensor<B, 1>| {
            if world == 1 { value } else { value.slice(interval.clone()) }
        };
        let state = Self {
            history_s: self.history_s.into_iter().map(slice).collect(),
            history_y: self.history_y.into_iter().map(slice).collect(),
            d: self.d.map(slice),
            t: self.t,
            prev_flat_grad: self.prev_flat_grad.map(slice),
            prev_loss: self.prev_loss,
            g_iter: self.g_iter,
        };
        LBFGSShardedState::from_local_state(state, layout, rank, world)
    }
}

/// Actual local L-BFGS state with the original complete-vector placement metadata.
#[derive(Clone)]
pub struct LBFGSShardedState<B: Backend> {
    version: u32,
    rank: u32,
    layout: LBFGSShardLayout,
    state: LBFGSState<B>,
}

impl<B: Backend> LBFGSShardedState<B> {
    /// Bind actual loaded local state to its explicitly declared original vector placement.
    pub fn from_local_state(
        state: LBFGSState<B>,
        layout: &LBFGSShardLayout,
        rank: u32,
        world: u32,
    ) -> Result<Self, LBFGSShardError> {
        layout.validate(rank, world)?;
        state.validate_vectors(layout.lengths[rank as usize], None)?;
        Ok(Self { version: 1, rank, layout: layout.clone(), state })
    }

    /// Original rank-local native state; no complete history is gathered.
    pub fn state(&self) -> &LBFGSState<B> { &self.state }

    /// Actual owning communicator rank.
    pub fn rank(&self) -> u32 { self.rank }

    /// Original rank-concatenated complete-vector layout.
    pub fn layout(&self) -> &LBFGSShardLayout { &self.layout }

    /// Validate original schema/placement and every actual native local vector.
    pub fn validate_placement(
        &self,
        layout: &LBFGSShardLayout,
        rank: u32,
        world: u32,
    ) -> Result<(), LBFGSShardError> {
        layout.validate(rank, world)?;
        if self.version != 1 || self.rank != rank || &self.layout != layout {
            return Err(LBFGSShardError::Record);
        }
        self.state.validate_vectors(layout.lengths[rank as usize], None)
    }

    /// Move this rank's original history/gradient/direction, retaining placement and counters.
    pub fn to_device(mut self, device: &B::Device) -> Self {
        self.state = self.state.to_device(device);
        self
    }
}

impl<B: Backend> Record<B> for LBFGSShardedState<B> {
    type Item<S: PrecisionSettings> = (
        u32, u32, Vec<usize>, <LBFGSState<B> as Record<B>>::Item<S>,
    );

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version, self.rank, self.layout.lengths, self.state.into_item::<S>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            version: item.0,
            rank: item.1,
            layout: LBFGSShardLayout::new(item.2),
            state: LBFGSState::<B>::from_item::<S>(item.3, device),
        }
    }
}

struct ShardedReductions<'a, C> {
    communicator: &'a C,
}

impl<C> ShardedReductions<'_, C> {
    fn sum<B: Backend>(
        &self,
        value: Tensor<B, 1>,
    ) -> Result<Tensor<B, 1>, LBFGSShardedError<C::Error>>
    where C: BroadcastTensorCollective<B> {
        let dtype = value.dtype();
        let device = value.device();
        let output = self.communicator.all_reduce_sum(value.into_primitive().tensor())
            .map_err(LBFGSShardedError::Collective)?;
        let output = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(output));
        if output.dims() != [1] {
            return Err(LBFGSShardError::Shape("scalar all-reduce result").into());
        }
        if output.dtype() != dtype { return Err(LBFGSShardError::DType.into()); }
        if output.device() != device { return Err(LBFGSShardError::Device.into()); }
        Ok(output)
    }
}

impl<B: Backend, C: BroadcastTensorCollective<B>> VectorReductions<B> for ShardedReductions<'_, C> {
    type Error = LBFGSShardedError<C::Error>;

    fn dot(
        &mut self,
        lhs: &Tensor<B, 1>,
        rhs: &Tensor<B, 1>,
    ) -> Result<Tensor<B, 1>, Self::Error> {
        self.sum(lhs.clone().dot(rhs.clone()))
    }

    fn sum_abs(&mut self, value: &Tensor<B, 1>) -> Result<f64, Self::Error> {
        Ok(self.sum(value.clone().abs().sum())?.into_scalar().to_f64())
    }

    fn max_abs(&mut self, value: &Tensor<B, 1>) -> Result<f64, Self::Error> {
        let local = value.clone().abs().max();
        let dtype = local.dtype();
        let device = local.device();
        let output = self.communicator.all_gather_float(local.into_primitive().tensor())
            .map_err(LBFGSShardedError::Collective)?;
        let output = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(output));
        if output.dims() != [self.communicator.world_size() as usize] {
            return Err(LBFGSShardError::Shape("scalar maxima gather result").into());
        }
        if output.dtype() != dtype { return Err(LBFGSShardError::DType.into()); }
        if output.device() != device { return Err(LBFGSShardError::Device.into()); }
        Ok(output.max().into_scalar().to_f64())
    }
}

impl<B: AutodiffBackend> LBFGS<B> {
    /// Export native local histories with exact owning rank and original vector placement.
    pub fn to_sharded_record<C: BroadcastTensorCollective<B::InnerBackend>>(
        &self,
        layout: &LBFGSShardLayout,
        communicator: &C,
    ) -> Result<LBFGSShardedState<B::InnerBackend>, LBFGSShardError> {
        LBFGSShardedState::from_local_state(
            self.state.clone(), layout, communicator.rank(), communicator.world_size(),
        )
    }

    /// Restore original local histories/counters after restoring the matching parameter shard.
    /// The caller retains the original optimizer configuration and unique-parameter ordering.
    pub fn load_sharded_record<C: BroadcastTensorCollective<B::InnerBackend>>(
        mut self,
        record: LBFGSShardedState<B::InnerBackend>,
        layout: &LBFGSShardLayout,
        communicator: &C,
    ) -> Result<Self, LBFGSShardError> {
        record.validate_placement(layout, communicator.rank(), communicator.world_size())?;
        self.state = record.state;
        Ok(self)
    }

    /// Optimize disjoint actual parameter shards with global L-BFGS curvature and line search.
    /// The closure must return the same complete objective value on every rank and its exact
    /// local gradient of that objective, entering all model collectives in matching order.
    /// All ranks retain identical configuration/history counts/scalar state. Replicas must not
    /// be counted as disjoint parameters. This does not choose a model partition or loss scaling.
    pub fn step_sharded<M, F, C>(
        &mut self,
        lr: LearningRate,
        module: M,
        mut closure: F,
        layout: &LBFGSShardLayout,
        communicator: &C,
    ) -> Result<(M, f64), LBFGSShardedError<C::Error>>
    where
        M: AutodiffModule<B> + Clone,
        F: FnMut(M) -> (f64, GradientsParams),
        C: BroadcastTensorCollective<B::InnerBackend>,
    {
        self.try_step_sharded(lr, module, |model| Ok(closure(model)), layout, communicator)
    }

    /// Fallible objective variant of native sharded L-BFGS, using the same complete two-loop
    /// recursion and strong-Wolfe interpolation/evaluation budget as the original optimizer.
    /// Only scalar reductions cross ranks; parameter and history vectors remain local.
    /// Returned failure leaves the original optimizer state installed. Retain matching model
    /// checkpoints; this does not roll back external closure effects or repair a failed transport.
    pub fn try_step_sharded<M, F, C>(
        &mut self,
        lr: LearningRate,
        module: M,
        closure: F,
        layout: &LBFGSShardLayout,
        communicator: &C,
    ) -> Result<(M, f64), LBFGSShardedError<C::Error>>
    where
        M: AutodiffModule<B> + Clone,
        F: FnMut(M) -> Result<(f64, GradientsParams), LBFGSShardedError<C::Error>>,
        C: BroadcastTensorCollective<B::InnerBackend>,
    {
        layout.validate(communicator.rank(), communicator.world_size())?;
        let parameters = flatten_params_inner::<B, M>(&module)
            .ok_or(LBFGSShardError::Shape("rank has no trainable parameter vector"))?;
        let length = layout.lengths[communicator.rank() as usize];
        if parameters.dims() != [length] {
            return Err(LBFGSShardError::Shape("rank-local unique parameter vector length").into());
        }
        self.state.validate_vectors(length, Some(&parameters))?;
        self.try_step_with_reductions(
            lr, module, closure, &mut ShardedReductions { communicator }, Some(parameters),
        )
    }
}
