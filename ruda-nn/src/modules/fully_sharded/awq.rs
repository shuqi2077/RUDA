use super::*;
use ruda_model::tensor::{FrozenAwqOps, IntegerTensorCollective};
use core::fmt;

/// Immutable exact integer words stored only in this rank's equal padded slice.
/// Word boundaries are preserved; packed codes never travel through float tensors.
#[derive(Module, Debug)]
pub struct ShardedPackedParameter<B: Backend> {
    /// Actual U8/I32/I64 native local words; integer parameters have no optimizer gradient.
    pub local: Param<Tensor<B, 1, Int>>,
    /// Original complete integer tensor axes, excluding local padding.
    pub logical_shape: Vec<usize>,
    /// Actual data-group owner rank.
    pub rank: usize,
    /// Original number of equal padded integer slices.
    pub world_size: usize,
}

impl<B: Backend> ShardedPackedParameter<B> {
    /// Load an actual local slice with its original ID and logical axes.
    pub fn from_local(local: Param<Tensor<B, 1, Int>>, logical_shape: Vec<usize>, rank: usize, world_size: usize) -> Self {
        assert!(world_size > 0 && rank < world_size, "invalid packed shard geometry");
        let elements = logical_shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("packed logical size overflows");
        let size = elements.div_ceil(world_size);
        size.checked_mul(world_size).expect("packed padded size overflows");
        assert_eq!(local.val().dims(), [size], "local packed word count differs");
        assert!(matches!(local.val().dtype(), DType::U8 | DType::I32 | DType::I64), "packed shard storage must retain U8/I32/I64");
        Self { local, logical_shape, rank, world_size }
    }

    /// Slice actual source words on the original backend/device, not on the host.
    /// Padding words are zero; no quantization, bit reinterpretation or float cast.
    pub fn from_full<const D: usize>(parameter: Param<Tensor<B, D, Int>>, rank: usize, world_size: usize) -> Self {
        assert!(world_size > 0 && rank < world_size, "invalid packed shard topology");
        let value = parameter.val(); let shape = value.dims().to_vec();
        let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("packed logical size overflows");
        let size = elements.div_ceil(world_size);
        size.checked_mul(world_size).expect("packed padded size overflows");
        let start = rank * size; let end = (start + size).min(elements);
        let mut local = Tensor::<B, 1, Int>::zeros([size], (&value.device(), value.dtype()));
        if start < end { local = local.slice_assign([0..end-start], value.reshape([elements]).slice([start..end])); }
        Self::from_local(Param::initialized(parameter.id, local), shape, rank, world_size)
    }

    /// Exact native integer gather with padding removed before original-axis restore.
    pub fn gather_inference<C: IntegerTensorCollective<B>, const D: usize>(&self, communicator: C)
        -> Result<Tensor<B, D, Int>, C::Error> {
        assert_eq!(communicator.rank() as usize, self.rank, "packed gather rank differs");
        assert_eq!(communicator.world_size() as usize, self.world_size, "packed gather world differs");
        let shape: [usize; D] = self.logical_shape.clone().try_into().expect("packed logical tensor rank differs");
        let local = self.local.val(); let dtype = local.dtype(); let device = local.device();
        let elements = self.logical_shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).expect("packed size overflows");
        let size = elements.div_ceil(self.world_size);
        assert_eq!(local.dims(), [size], "loaded packed local word count differs");
        let padded = size.checked_mul(self.world_size).expect("packed gather size overflows");
        let full = Tensor::<B, 1, Int>::from_primitive(communicator.all_gather_int(local.into_primitive())?);
        assert_eq!(full.dims(), [padded], "packed transport shape differs");
        assert_eq!(full.dtype(), dtype, "packed transport changed word dtype");
        assert_eq!(full.device(), device, "packed transport changed device");
        Ok(full.slice([0..elements]).reshape(shape))
    }
}

impl<B: Backend, S: CheckpointStrategy> ShardedPackedParameter<Autodiff<B, S>> {
    /// Gather immutable words using the actual inner transport, with no integer AD surrogate.
    pub fn gather<C: IntegerTensorCollective<B>, const D: usize>(&self, communicator: C)
        -> Result<Tensor<Autodiff<B, S>, D, Int>, C::Error> {
        let native = ShardedPackedParameter::from_local(
            Param::initialized(self.local.id, self.local.val().inner()), self.logical_shape.clone(), self.rank, self.world_size,
        );
        native.gather_inference::<C, D>(communicator).map(Tensor::from_inner)
    }
}

/// Fully data-sharded frozen AWQ base: words, zero points, scales and bias are
/// persistent local slices, not permanently replicated base tensors. Forward
/// gathers packed payload transiently; backward retains packed buffers, not a dense shadow.
#[derive(Module, Debug)]
pub struct FullyShardedAwqLinear<B: Backend> {
    /// Actual original packed weight-word shard.
    pub qweight: ShardedPackedParameter<B>,
    /// Actual original packed zero-point shard.
    pub qzeros: ShardedPackedParameter<B>,
    /// Actual frozen original scale shard.
    pub scales: ShardedParameter<B>,
    /// Optional actual frozen bias shard.
    pub bias: Option<ShardedParameter<B>>,
    /// Original AWQ group size.
    pub group_size: usize,
}

impl<B: Backend> FullyShardedAwqLinear<B> {
    /// Convert original loaded packed base storage into this rank's actual slices.
    pub fn from_full(layer:crate::FrozenAwqLinear<B>,rank:usize,world_size:usize) -> Self {
        ShardingContext::new(rank,world_size).awq(layer)
    }
    /// Assemble actual local base slices, retaining their values and IDs.
    pub fn from_shards(qweight: ShardedPackedParameter<B>, qzeros: ShardedPackedParameter<B>,
        scales: ShardedParameter<B>, bias: Option<ShardedParameter<B>>, group_size: usize) -> Self {
        let layer = Self { qweight, qzeros, scales, bias, group_size };layer.validate();layer
    }

    /// Validate original logical layout and local precision/trainability/topology.
    pub fn validate(&self) {
        let [input, packed_output]: [usize; 2] = self.qweight.logical_shape.clone().try_into().expect("AWQ words must be a matrix");
        let [groups, output]: [usize; 2] = self.scales.logical_shape.clone().try_into().expect("AWQ scales must be a matrix");
        assert!(input > 0 && output > 0 && self.group_size > 0, "AWQ dimensions/group must be positive");
        assert_eq!(input % self.group_size, 0, "AWQ input groups must be complete");
        assert_eq!(output % 8, 0, "AWQ output must be divisible by eight");
        assert_eq!([groups, packed_output], [input / self.group_size, output / 8], "AWQ grouped geometry differs");
        assert_eq!(self.qzeros.logical_shape, [groups, packed_output], "AWQ zero-point shape differs");
        assert!(input.checked_mul(output).is_some_and(|n|n<=u32::MAX as usize), "AWQ matrix indexing overflows");
        let topology = (self.qweight.rank, self.qweight.world_size);
        let device = self.qweight.local.val().device();
        for words in [&self.qweight, &self.qzeros] {
            assert_eq!((words.rank, words.world_size), topology, "AWQ packed topologies differ");
            assert_eq!(words.local.val().dtype(), DType::I32, "AWQ word dtype must remain I32");
            assert_eq!(words.local.val().device(), device, "AWQ packed devices differ");
            let _ = ShardedPackedParameter::from_local(words.local.clone(), words.logical_shape.clone(), words.rank, words.world_size);
        }
        for value in core::iter::once(&self.scales).chain(self.bias.iter()) {
            assert_eq!((value.rank, value.world_size), topology, "AWQ floating topologies differ");
            assert_eq!(value.local.val().device(), device, "AWQ floating devices differ");
            assert!(!value.local.val().is_require_grad(), "AWQ base scales/bias must be frozen");
            assert!(matches!(value.local.val().dtype(), DType::F16 | DType::BF16 | DType::F32), "AWQ floating storage is unsupported");
            let _ = ShardedParameter::from_local(value.local.clone(), value.logical_shape.clone(), value.rank, value.world_size);
        }
        if let Some(bias) = &self.bias {
            assert_eq!(bias.logical_shape, [output], "AWQ bias width differs");
            assert_eq!(bias.local.val().dtype(), self.scales.local.val().dtype(), "AWQ bias/scale dtypes differ");
        }
    }

    /// Native transient packed projection with its actual frozen base IDs.
    pub fn gather_inference<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<crate::FrozenAwqLinear<B>, C::Error> {
        self.validate();
        let weight = self.qweight.gather_inference::<C, 2>(communicator.clone())?;
        let zeros = self.qzeros.gather_inference::<C, 2>(communicator.clone())?;
        let scales = self.scales.gather_inference::<C, 2>(communicator.clone())?;
        let bias = self.bias.as_ref().map(|bias|bias.gather_inference::<C, 1>(communicator).map(|value|Param::initialized(bias.local.id, value))).transpose()?;
        Ok(crate::FrozenAwqLinear::from_parameters(Param::initialized(self.qweight.local.id, weight),
            Param::initialized(self.qzeros.local.id, zeros), Param::initialized(self.scales.local.id, scales), bias, self.group_size))
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedAwqLinear<Autodiff<B, S>> {
    /// Gather the real immutable packed base on its original data transport.
    pub fn gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<crate::FrozenAwqLinear<Autodiff<B, S>>, C::Error> {
        self.validate();
        let weight = self.qweight.gather::<C, 2>(communicator.clone())?;
        let zeros = self.qzeros.gather::<C, 2>(communicator.clone())?;
        let scales = self.scales.gather::<C, 2>(communicator.clone())?;
        let bias = self.bias.as_ref().map(|bias|bias.gather::<C, 1>(communicator).map(|value|Param::initialized(bias.local.id, value))).transpose()?;
        Ok(crate::FrozenAwqLinear::from_parameters(Param::initialized(self.qweight.local.id, weight),
            Param::initialized(self.qzeros.local.id, zeros), Param::initialized(self.scales.local.id, scales), bias, self.group_size))
    }
}

/// Packed base and floating adapters are all persistently data-sharded. Adapter
/// derivatives use existing SUM/reduce-scatter, with no second gradient reduction.
#[derive(Module, Debug)]
pub struct FullyShardedAwqLoRALinear<B: Backend> {
    /// Actual immutable word/zero/scale/bias slices.
    pub base: FullyShardedAwqLinear<B>,
    /// Original trainable rank-local adapter A storage.
    pub adapter_a: FullyShardedLinear<B>,
    /// Original trainable rank-local adapter B storage.
    pub adapter_b: FullyShardedLinear<B>,
    /// Original adapter-only dropout.
    pub dropout: crate::Dropout,
    /// Original LoRA/rsLoRA scale.
    pub scale: f64,
}

impl<B: Backend> FullyShardedAwqLoRALinear<B> {
    /// Validate loaded adapter logical shapes, storage, trainability and topology
    /// against the actual packed base before entering any collective.
    pub fn validate(&self) {
        self.base.validate();
        let input=self.base.qweight.logical_shape[0];let output=self.base.scales.logical_shape[1];
        let [a_input,rank]:[usize;2]=self.adapter_a.weight.logical_shape.clone().try_into().expect("AWQ adapter A must be a matrix");
        assert!(rank>0 && self.scale.is_finite(),"invalid AWQ adapter rank/scale");
        assert_eq!(a_input,input,"AWQ adapter A input width differs");
        assert_eq!(self.adapter_b.weight.logical_shape,[rank,output],"AWQ adapter B shape differs");
        assert!(self.adapter_a.bias.is_none() && self.adapter_b.bias.is_none(),"AWQ adapters must be bias-free");
        let device=self.base.qweight.local.val().device();
        let topology=(self.base.qweight.rank,self.base.qweight.world_size);
        for parameter in [&self.adapter_a.weight,&self.adapter_b.weight] {
            assert_eq!((parameter.rank,parameter.world_size),topology,"AWQ adapter/base topologies differ");
            let value=parameter.local.val();
            assert_eq!(value.device(),device,"AWQ adapter/base devices differ");
            assert!(matches!(value.dtype(),DType::F16|DType::BF16|DType::F32),"unsupported AWQ adapter storage");
            assert!(!B::ad_enabled(&device) || value.is_require_grad(),"AWQ adapters must be trainable on the AD backend");
            let _=ShardedParameter::from_local(parameter.local.clone(),parameter.logical_shape.clone(),parameter.rank,parameter.world_size);
        }
    }
    /// Convert loaded native packed base/adapters into actual local slices.
    pub fn from_full(layer: crate::AwqLoRALinear<B>, rank: usize, world_size: usize) -> Self {
        ShardingContext::new(rank, world_size).awq_lora(layer)
    }
    /// Transient native module for inference; no adapter merge or requantization.
    pub fn gather_inference<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<crate::AwqLoRALinear<B>, C::Error> {
        self.validate();
        Ok(crate::AwqLoRALinear {base:self.base.gather_inference(communicator.clone())?,
            adapter_a:self.adapter_a.gather_inference(communicator.clone())?,adapter_b:self.adapter_b.gather_inference(communicator)?,
            dropout:self.dropout.clone(),scale:self.scale})
    }
}

impl<B: Backend, S: CheckpointStrategy> FullyShardedAwqLoRALinear<Autodiff<B, S>> {
    /// Original differentiable A/B gathers with an unchanged packed frozen base.
    pub fn gather<C: IntegerTensorCollective<B>>(&self, communicator: C) -> Result<crate::AwqLoRALinear<Autodiff<B, S>>, C::Error> {
        self.validate();
        Ok(crate::AwqLoRALinear {base:self.base.gather(communicator.clone())?,
            adapter_a:self.adapter_a.gather(communicator.clone())?,adapter_b:self.adapter_b.gather(communicator)?,
            dropout:self.dropout.clone(),scale:self.scale})
    }
}

/// Underlying collective or native packed projection failure, with no float fallback.
#[derive(Debug)]
pub enum FullyShardedAwqError<C: fmt::Debug, Q: fmt::Debug> {
    /// Original selected data transport error.
    Collective(C),
    /// Native AWQ validation/launch error.
    Projection(Q),
}
impl<C: fmt::Debug, Q: fmt::Debug> fmt::Display for FullyShardedAwqError<C, Q> {
    fn fmt(&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Collective(error)=>write!(f,"AWQ shard transport: {error:?}"),Self::Projection(error)=>write!(f,"AWQ projection: {error:?}")}
    }
}
impl<C: fmt::Debug, Q: fmt::Debug> core::error::Error for FullyShardedAwqError<C, Q> {}

macro_rules! awq_forward {
    ($module:ident) => {
        impl<B: FrozenAwqOps> $module<B> {
            /// Original native packed projection after gathering actual data shards.
            pub fn forward_inference<C: IntegerTensorCollective<B>, const D: usize>(&self, input:Tensor<B,D>, communicator:C)
                -> Result<Tensor<B,D>,FullyShardedAwqError<C::Error,B::AwqError>> {
                self.gather_inference(communicator).map_err(FullyShardedAwqError::Collective)?.forward(input).map_err(FullyShardedAwqError::Projection)
            }
        }
        impl<B: FrozenAwqOps, S: CheckpointStrategy> $module<Autodiff<B,S>> {
            /// Native packed forward with input and adapter derivatives on the original AD backend.
            pub fn forward<C: IntegerTensorCollective<B>, const D: usize>(&self, input:Tensor<Autodiff<B,S>,D>, communicator:C)
                -> Result<Tensor<Autodiff<B,S>,D>,FullyShardedAwqError<C::Error,<Autodiff<B,S> as FrozenAwqOps>::AwqError>> {
                self.gather(communicator).map_err(FullyShardedAwqError::Collective)?.forward(input).map_err(FullyShardedAwqError::Projection)
            }
        }
    };
}
awq_forward!(FullyShardedAwqLinear);
awq_forward!(FullyShardedAwqLoRALinear);
