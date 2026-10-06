//! Rust model layers whose persistent parameters/gradients are equal flat shards.
use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,collective};
use ruda_model::{
    module::{Module,Param,ParamId},
    tensor::{Tensor,Int,DType,FloatDType,TensorPrimitive,backend::Backend,ops::ModuleOps,module::{linear,embedding}},
};
use ruda_autodiff::tensor_parallel::BroadcastTensorCollective;

/// Explicit construction context preserving one local autograd leaf per source ID.
/// Reuse a context for all tied layers, then drop it after model construction. It
/// retains only local shards, never the complete source tensors or optimizer states.
pub struct ShardingContext<B:Backend> {
    rank:usize,
    world_size:usize,
    parameters:BTreeMap<ParamId,ShardedParameter<B>>,
}

impl<B:Backend> ShardingContext<B> {
    /// Select the actual data-axis topology; no communicator or device is inferred.
    pub fn new(rank:usize,world_size:usize)->Self {
        assert!(world_size>0 && rank<world_size,"invalid sharding context topology");
        Self{rank,world_size,parameters:BTreeMap::new()}
    }

    /// Return the same local Param/autograd leaf for each alias of a source ID.
    /// Reusing an ID with different logical metadata or trainability is an error.
    pub fn parameter<const D:usize>(&mut self,parameter:Param<Tensor<B,D>>)->ShardedParameter<B> {
        let value=parameter.val();
        if let Some(shard)=self.parameters.get(&parameter.id) {
            let local=shard.local.val();
            assert_eq!(shard.logical_shape,value.dims().to_vec(),"tied parameter dimensions differ");
            assert_eq!(local.dtype(),value.dtype(),"tied parameter storage dtypes differ");
            assert!(local.device()==value.device(),"tied parameter devices differ");
            assert_eq!(local.is_require_grad(),value.is_require_grad(),"tied parameter trainability differs");
            return shard.clone();
        }
        let id=parameter.id;
        let shard=ShardedParameter::from_full(parameter,self.rank,self.world_size);
        self.parameters.insert(id,shard.clone());
        shard
    }

    /// Shard an existing dense projection, including any tied weight or bias.
    pub fn linear(&mut self,layer:crate::Linear<B>)->FullyShardedLinear<B> {
        FullyShardedLinear{weight:self.parameter(layer.weight),bias:layer.bias.map(|bias|self.parameter(bias))}
    }

    /// Share embedding storage with another explicitly tied projection/table.
    pub fn embedding(&mut self,layer:crate::Embedding<B>)->FullyShardedEmbedding<B> {
        FullyShardedEmbedding{weight:self.parameter(layer.weight)}
    }

    /// Shard a caller-injected adapter without duplicating shared local leaves.
    pub fn lora(&mut self,layer:crate::LoRALinear<B>)->FullyShardedLoRALinear<B> {
        FullyShardedLoRALinear{base:self.linear(layer.base),adapter_a:self.linear(layer.adapter_a),
            adapter_b:self.linear(layer.adapter_b),dropout:layer.dropout,scale:layer.scale}
    }

    /// Shard RMSNorm's existing affine parameter with its actual epsilon.
    pub fn rms_norm(&mut self,layer:crate::RmsNorm<B>)->FullyShardedRmsNorm<B> {
        FullyShardedRmsNorm{gamma:self.parameter(layer.gamma),epsilon:layer.epsilon}
    }

    /// Shard LayerNorm affine values without guessing or resetting its epsilon.
    pub fn layer_norm(&mut self,layer:crate::LayerNorm<B>)->FullyShardedLayerNorm<B> {
        let epsilon=layer.epsilon();
        FullyShardedLayerNorm{gamma:self.parameter(layer.gamma),beta:layer.beta.map(|beta|self.parameter(beta)),epsilon}
    }

    /// Build a gated feed-forward block with one shared leaf for each tied matrix.
    pub fn gated_mlp(&mut self,gate:crate::Linear<B>,up:crate::Linear<B>,down:crate::Linear<B>)->FullyShardedGatedMLP<B> {
        let gate=self.linear(gate);
        let up=self.linear(up);
        let down=self.linear(down);
        FullyShardedGatedMLP::from_shards(gate,up,down)
    }

    /// Tie a logical vocabulary projection to the already sharded embedding.
    /// Transposition occurs only on the gathered value, never on local storage.
    pub fn tied_projection(&mut self,embedding:&FullyShardedEmbedding<B>,bias:Option<Param<Tensor<B,1>>>)->FullyShardedProjection<B> {
        assert_eq!((embedding.weight.rank,embedding.weight.world_size),(self.rank,self.world_size),"embedding data topology differs from the construction context");
        FullyShardedProjection::from_embedding(embedding,bias.map(|bias|self.parameter(bias)))
    }
}

/// A logical parameter backed only by this rank's padded element slice.
///
/// Tied parameters may share the same local Param ID. Gathered values are
/// transient tensors, not registered full parameters. Backward sums independent
/// rank-local losses into a reduce-scattered gradient. Normalize the loss by the
/// global sample/token weight BEFORE backward; do not all-reduce these gradients
/// a second time. Select an elementwise optimizer for these flattened records.
#[derive(Module,Debug)]
pub struct ShardedParameter<B:Backend> {
    /// Rank-local storage; padding lies only after the logical element count.
    pub local:Param<Tensor<B,1>>,
    /// Full logical dimensions, excluding padding.
    pub logical_shape:Vec<usize>,
    /// Owner of this equal contiguous slice.
    pub rank:usize,
    /// Number of equal padded slices.
    pub world_size:usize,
}

impl<B:Backend> ShardedParameter<B> {
    /// Construct from an already loaded local checkpoint slice; no full tensor is needed.
    pub fn from_local(local:Param<Tensor<B,1>>,logical_shape:Vec<usize>,rank:usize,world_size:usize)->Self {
        assert!(world_size>0 && rank<world_size,"invalid parameter shard topology");
        assert!(!logical_shape.is_empty() && logical_shape.iter().all(|&n|n>0),"logical parameter dimensions must be positive");
        let elements=logical_shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("logical parameter size overflow");
        assert_eq!(local.val().dims(),[elements.div_ceil(world_size)],"local parameter slice length differs");
        assert!(matches!(local.val().dtype(),DType::F32|DType::F16|DType::BF16),"floating parameter storage required");
        Self{local,logical_shape,rank,world_size}
    }

    /// Slice a caller-loaded logical value on its current backend/device.
    /// Existing parameter IDs are retained for explicitly shared/tied weights.
    pub fn from_full<const D:usize>(parameter:Param<Tensor<B,D>>,rank:usize,world_size:usize)->Self {
        assert!(world_size>0 && rank<world_size,"invalid parameter shard topology");
        let value=parameter.val();
        let shape=value.dims().to_vec();
        let elements=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("logical parameter size overflow");
        assert!(elements>0,"logical parameter cannot be empty");
        let size=elements.div_ceil(world_size);
        let start=rank*size;
        let end=(start+size).min(elements);
        let mut local=Tensor::<B,1>::zeros([size],&value.device()).cast(value.dtype());
        if end>start {
            local=local.slice_assign([0..end-start],value.clone().reshape([elements]).slice([start..end]));
        }
        // A shard is a new leaf: its optimizer must not keep the full source graph.
        let trainable=value.is_require_grad();
        let local=local.detach().set_require_grad(trainable);
        Self::from_local(Param::initialized(parameter.id,local),shape,rank,world_size)
    }
}

impl<B:Backend,S:CheckpointStrategy> ShardedParameter<Autodiff<B,S>> {
    /// Gather a logical value; backward uses the data-parallel SUM derivative.
    /// All ranks must execute the same collective-bearing layer order.
    pub fn gather<C:BroadcastTensorCollective<B>,const D:usize>(&self,communicator:C)
        ->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.gather_inner(communicator,None)
    }

    /// Explicit gather/arithmetic precision without changing parameter storage.
    /// Casting before gather also keeps its collective backward in this dtype.
    pub fn gather_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,communicator:C,dtype:FloatDType)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.gather_inner(communicator,Some(dtype.into()))
    }

    fn gather_inner<C:BroadcastTensorCollective<B>,const D:usize>(&self,communicator:C,dtype:Option<DType>)
        ->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        assert_eq!(communicator.rank() as usize,self.rank,"parameter shard rank differs");
        assert_eq!(communicator.world_size() as usize,self.world_size,"parameter shard world size differs");
        let shape:[usize;D]=self.logical_shape.clone().try_into().expect("logical parameter rank differs");
        let elements=self.logical_shape.iter().product();
        let local=self.local.val();
        let local=if let Some(dtype)=dtype {local.cast(dtype)} else {local};
        collective::all_gather(local,communicator).map(|full|full.slice([0..elements]).reshape(shape))
    }
}

/// Linear projection with local flattened weight/bias and ordinary global output.
/// Weight logical layout is RUDA's `[input,output]`, not `[output,input]`.
#[derive(Module,Debug)]
pub struct FullyShardedLinear<B:Backend> {
    /// Full logical matrix metadata and its local storage.
    pub weight:ShardedParameter<B>,
    /// Optional full logical output-bias metadata and local storage.
    pub bias:Option<ShardedParameter<B>>,
}

impl<B:Backend> FullyShardedLinear<B> {
    /// Convert caller-loaded dense weights into local slices before optimizer creation.
    pub fn from_full(layer:crate::Linear<B>,rank:usize,world_size:usize)->Self {
        Self{weight:ShardedParameter::from_full(layer.weight,rank,world_size),
             bias:layer.bias.map(|bias|ShardedParameter::from_full(bias,rank,world_size))}
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedLinear<Autodiff<B,S>> {
    /// Materialize this layer's parameters, project local data, and reduce-scatter in backward.
    /// Full gathered weights may be retained by the selected autodiff checkpoint strategy.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        let weight=self.weight.gather::<C,2>(communicator.clone())?;
        let bias=match &self.bias {Some(bias)=>Some(bias.gather::<C,1>(communicator)?),None=>None};
        Ok(linear(input,weight,bias))
    }
}

/// Embedding with local flattened table slices and globally indexed token lookup.
#[derive(Module,Debug)]
pub struct FullyShardedEmbedding<B:Backend> {
    /// Full `[vocabulary,features]` metadata and local table storage.
    pub weight:ShardedParameter<B>,
}

impl<B:Backend> FullyShardedEmbedding<B> {
    /// Split a loaded ordinary embedding on its current backend.
    pub fn from_full(layer:crate::Embedding<B>,rank:usize,world_size:usize)->Self {
        Self{weight:ShardedParameter::from_full(layer.weight,rank,world_size)}
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedEmbedding<Autodiff<B,S>> {
    /// Gather this table, preserving the existing integer lookup and padding semantics.
    pub fn forward<C:BroadcastTensorCollective<B>>(&self,input:Tensor<Autodiff<B,S>,2,Int>,communicator:C)
        ->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        Ok(embedding(self.weight.gather::<C,2>(communicator)?,input))
    }
}

/// Data-sharded projection sharing exact `[vocabulary,width]` embedding storage.
#[derive(Module,Debug)]
pub struct FullyShardedProjection<B:Backend> {
    /// Same local Param ID/autograd leaf as the source embedding.
    pub weight:ShardedParameter<B>,
    /// Optional logical output bias and its local slice.
    pub bias:Option<ShardedParameter<B>>,
}

impl<B:Backend> FullyShardedProjection<B> {
    /// Retain a real embedding/head tie without copying or transposing its leaf.
    pub fn from_embedding(embedding:&FullyShardedEmbedding<B>,bias:Option<ShardedParameter<B>>)->Self {
        assert_eq!(embedding.weight.logical_shape.len(),2,"projection table must be two-dimensional");
        if let Some(bias)=&bias {
            assert_eq!(bias.logical_shape,[embedding.weight.logical_shape[0]],"projection bias width differs");
            assert_eq!((bias.rank,bias.world_size),(embedding.weight.rank,embedding.weight.world_size),"projection bias data topology differs");
        }
        Self{weight:embedding.weight.clone(),bias}
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedProjection<Autodiff<B,S>> {
    /// Full logical output with data SUM derivatives on the shared local table.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        let weight=self.weight.gather::<C,2>(communicator.clone())?.transpose();
        let bias=match &self.bias {Some(bias)=>Some(bias.gather::<C,1>(communicator)?),None=>None};
        Ok(linear(input,weight,bias))
    }

    /// Choose arithmetic/gather precision; output retains the incoming activation
    /// dtype, while the persistent embedding/head storage remains unchanged.
    pub fn forward_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C,dtype:FloatDType)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        let output_dtype=input.dtype();
        let weight=self.weight.gather_with_compute_dtype::<C,2>(communicator.clone(),dtype)?.transpose();
        let bias=match &self.bias {
            Some(bias)=>Some(bias.gather_with_compute_dtype::<C,1>(communicator,dtype)?),None=>None,
        };
        Ok(linear(input.cast(dtype),weight,bias).cast(output_dtype))
    }
}

/// Native gated feed-forward block with only data-sharded persistent parameters.
#[derive(Module,Debug)]
pub struct FullyShardedGatedMLP<B:Backend> {
    /// Input-to-hidden gating projection.
    pub gate:FullyShardedLinear<B>,
    /// Input-to-hidden value projection.
    pub up:FullyShardedLinear<B>,
    /// Hidden-to-output projection; its bias is applied only by this projection.
    pub down:FullyShardedLinear<B>,
}

impl<B:Backend> FullyShardedGatedMLP<B> {
    /// Use actual loaded projections and preserve ties within this block.
    pub fn from_full(gate:crate::Linear<B>,up:crate::Linear<B>,down:crate::Linear<B>,rank:usize,world_size:usize)->Self {
        ShardingContext::new(rank,world_size).gated_mlp(gate,up,down)
    }

    /// Compose explicitly loaded slices on the same data topology.
    pub fn from_shards(gate:FullyShardedLinear<B>,up:FullyShardedLinear<B>,down:FullyShardedLinear<B>)->Self {
        let shape=&gate.weight.logical_shape;
        assert!(shape.len()==2 && up.weight.logical_shape==*shape && down.weight.logical_shape.len()==2,"gated MLP projection ranks or gate/up widths differ");
        assert_eq!(down.weight.logical_shape[0],shape[1],"gated MLP intermediate width differs");
        let topology=(gate.weight.rank,gate.weight.world_size);
        for layer in [&gate,&up,&down] {
            assert_eq!((layer.weight.rank,layer.weight.world_size),topology,"gated MLP data topologies differ");
            if let Some(bias)=&layer.bias {
                assert_eq!(bias.logical_shape,[layer.weight.logical_shape[1]],"gated MLP bias width differs");
                assert_eq!((bias.rank,bias.world_size),topology,"gated MLP bias data topology differs");
            }
        }
        Self{gate,up,down}
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedGatedMLP<Autodiff<B,S>> {
    /// SwiGLU on actual local data; no tensor-parallel or model-family inference.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with(input,communicator,ruda_model::tensor::activation::silu)
    }

    /// Explicit caller-selected gate activation with the same DP derivatives.
    pub fn forward_with<C:BroadcastTensorCollective<B>,F,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C,activation:F)->Result<Tensor<Autodiff<B,S>,D>,C::Error>
    where F:FnOnce(Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        let gate=self.gate.forward(input.clone(),communicator.clone())?;
        let up=self.up.forward(input,communicator.clone())?;
        self.down.forward(activation(gate)*up,communicator)
    }
}

/// Native LoRA whose frozen base and trainable adapters all retain local DP slices.
/// There is no persistent duplicate full base or adapter matrix. Ordinary layer
/// records contain only slices and logical geometry, retaining tied parameter IDs.
#[derive(Module,Debug)]
pub struct FullyShardedLoRALinear<B:Backend> {
    /// Frozen element-sharded base projection.
    pub base:FullyShardedLinear<B>,
    /// Data slices of `[input,rank]`, with its own floating storage dtype.
    pub adapter_a:FullyShardedLinear<B>,
    /// Data slices of `[rank,output]`, with its own floating storage dtype.
    pub adapter_b:FullyShardedLinear<B>,
    /// Adapter-only input dropout, using this data replica's backend RNG.
    pub dropout:crate::Dropout,
    /// Explicit alpha/rank or alpha/sqrt(rank) coefficient from the source adapter.
    pub scale:f64,
}

impl<B:Backend> FullyShardedLoRALinear<B> {
    /// Consume caller-loaded/injected dense LoRA before creating local optimizers.
    /// No parameters are reinitialized and no base checkpoint is inferred.
    pub fn from_full(layer:crate::LoRALinear<B>,rank:usize,world_size:usize)->Self {
        ShardingContext::new(rank,world_size).lora(layer)
    }
}

/// Last-axis RMS normalization with persistent data-sharded affine storage.
#[derive(Module,Debug)]
pub struct FullyShardedRmsNorm<B:Backend> {
    /// Logical affine vector and its local data slice.
    pub gamma:ShardedParameter<B>,
    /// Actual source normalization epsilon.
    pub epsilon:f64,
}

impl<B:Backend> FullyShardedRmsNorm<B> {
    /// Convert an existing normalization before creating the shard optimizer.
    pub fn from_full(layer:crate::RmsNorm<B>,rank:usize,world_size:usize)->Self {
        ShardingContext::new(rank,world_size).rms_norm(layer)
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedRmsNorm<Autodiff<B,S>> {
    /// FP32 statistics/affine arithmetic, retaining the input activation dtype.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_compute_dtype(input,communicator,FloatDType::F32)
    }

    /// Explicit arithmetic/collective precision without changing affine storage.
    pub fn forward_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C,dtype:FloatDType)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        assert!(D>0 && self.gamma.logical_shape==[input.dims()[D-1]],"RMSNorm affine width differs");
        let output_dtype=input.dtype();
        let gamma=self.gamma.gather_with_compute_dtype::<C,1>(communicator,dtype)?;
        let input=input.cast(dtype);
        let rms=(input.clone().square().mean_dim(D-1)+self.epsilon).sqrt();
        Ok(((input/rms)*gamma.unsqueeze::<D>()).cast(output_dtype))
    }
}

/// Backend-native last-axis LayerNorm with data-sharded scale/optional bias.
#[derive(Module,Debug)]
pub struct FullyShardedLayerNorm<B:Backend> {
    /// Local slices of the logical affine scale.
    pub gamma:ShardedParameter<B>,
    /// Optional local slices of the logical affine bias.
    pub beta:Option<ShardedParameter<B>>,
    /// Actual source normalization epsilon.
    pub epsilon:f64,
}

impl<B:Backend> FullyShardedLayerNorm<B> {
    /// Convert loaded scale/bias without initializing another normalization.
    pub fn from_full(layer:crate::LayerNorm<B>,rank:usize,world_size:usize)->Self {
        ShardingContext::new(rank,world_size).layer_norm(layer)
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedLayerNorm<Autodiff<B,S>> {
    /// FP32 normalization/affine computation, preserving the input storage dtype.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_compute_dtype(input,communicator,FloatDType::F32)
    }

    /// Use the same native LayerNorm/autodiff operation with explicit precision.
    pub fn forward_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C,dtype:FloatDType)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        assert!(D>0 && self.gamma.logical_shape==[input.dims()[D-1]],"LayerNorm affine width differs");
        let output_dtype=input.dtype();
        let gamma=self.gamma.gather_with_compute_dtype::<C,1>(communicator.clone(),dtype)?.into_primitive().tensor();
        let beta=match &self.beta {
            Some(beta)=>{
                assert_eq!(beta.logical_shape,self.gamma.logical_shape,"LayerNorm bias width differs");
                Some(beta.gather_with_compute_dtype::<C,1>(communicator,dtype)?.into_primitive().tensor())
            }
            None=>None,
        };
        let output=<Autodiff<B,S> as ModuleOps<Autodiff<B,S>>>::layer_norm(input.cast(dtype).into_primitive().tensor(),gamma,beta,self.epsilon);
        Ok(Tensor::<Autodiff<B,S>,D>::from_primitive(TensorPrimitive::Float(output)).cast(output_dtype))
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedLoRALinear<Autodiff<B,S>> {
    /// Gather this projection's values; backward reduce-scatters DP gradient sums.
    /// Normalize local loss sums by global token weight before backward. All ranks
    /// must use the same collective-bearing order; do not reduce these slices again.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(
        &self,input:Tensor<Autodiff<B,S>,D>,communicator:C)->Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        let adapted=input.clone().cast(self.adapter_a.weight.local.val().dtype());
        let adapted=self.dropout.forward(adapted);
        let base=self.base.forward(input,communicator.clone())?;
        let hidden=self.adapter_a.forward(adapted,communicator.clone())?
            .cast(self.adapter_b.weight.local.val().dtype());
        let update=self.adapter_b.forward(hidden,communicator)?.mul_scalar(self.scale);
        let dtype=base.dtype();
        Ok(base+update.cast(dtype))
    }
}
