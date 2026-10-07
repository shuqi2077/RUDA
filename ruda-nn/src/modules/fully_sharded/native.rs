use super::*;

impl<B:Backend> ShardedParameter<B> {
    /// Gather actual local storage on the native inference backend, not through an autodiff wrapper.
    /// Padding is removed before restoring the original logical axes. This never registers a
    /// persistent complete parameter or selects a different communicator/backend.
    pub fn gather_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        self.gather_inference_inner(communicator,None)
    }

    /// Explicit native gather precision, retaining the actual local parameter storage unchanged.
    pub fn gather_inference_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(&self,communicator:C,dtype:FloatDType)
        -> Result<Tensor<B,D>,C::Error> {
        self.gather_inference_inner(communicator,Some(dtype.into()))
    }

    fn gather_inference_inner<C:BroadcastTensorCollective<B>,const D:usize>(&self,communicator:C,dtype:Option<DType>)
        -> Result<Tensor<B,D>,C::Error> {
        assert_eq!(communicator.rank() as usize,self.rank,"native parameter shard rank differs");
        assert_eq!(communicator.world_size() as usize,self.world_size,"native parameter shard world size differs");
        let shape:[usize;D] = self.logical_shape.clone().try_into().expect("native logical parameter rank differs");
        let elements = self.logical_shape.iter().try_fold(1usize,|total,axis|total.checked_mul(*axis)).expect("native logical parameter size overflow");
        let local = self.local.val();
        let local = if let Some(dtype) = dtype {local.cast(dtype)} else {local};
        let dtype = local.dtype();let device = local.device();
        let padded = local.dims()[0].checked_mul(self.world_size).expect("native parameter gather length overflow");
        let output = communicator.all_gather_float(local.into_primitive().tensor())?;
        let output = Tensor::<B,1>::from_primitive(TensorPrimitive::Float(output));
        assert_eq!(output.dims(),[padded],"native parameter transport gather shape differs");
        assert_eq!(output.dtype(),dtype,"native parameter transport changed precision");
        assert_eq!(output.device(),device,"native parameter transport changed device");
        Ok(output.slice([0..elements]).reshape(shape))
    }
}

impl<B:Backend> FullyShardedLinear<B> {
    /// Original native projection with transiently gathered real weight and optional bias.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        let weight = self.weight.gather_inference::<C,2>(communicator.clone())?;
        let bias = self.bias.as_ref().map(|bias|bias.gather_inference::<C,1>(communicator)).transpose()?;
        Ok(linear(input,weight,bias))
    }
    /// Explicit projection/gather arithmetic precision; output returns to incoming activation storage.
    pub fn forward_inference_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,dtype:FloatDType)
        -> Result<Tensor<B,D>,C::Error> {
        let storage = input.dtype();
        let weight = self.weight.gather_inference_with_compute_dtype::<C,2>(communicator.clone(),dtype)?;
        let bias = self.bias.as_ref().map(|bias|bias.gather_inference_with_compute_dtype::<C,1>(communicator,dtype)).transpose()?;
        Ok(linear(input.cast(dtype),weight,bias).cast(storage))
    }
}

impl<B:Backend> FullyShardedEmbedding<B> {
    /// Native original integer lookup from the actual sharded table, without changing token IDs.
    pub fn forward_inference<C:BroadcastTensorCollective<B>>(&self,input:Tensor<B,2,Int>,communicator:C)
        -> Result<Tensor<B,3>,C::Error> {
        Ok(embedding(self.weight.gather_inference::<C,2>(communicator)?,input))
    }
    /// Explicit transient embedding output/gather precision without altering persistent table storage.
    pub fn forward_inference_with_compute_dtype<C:BroadcastTensorCollective<B>>(&self,input:Tensor<B,2,Int>,communicator:C,dtype:FloatDType)
        -> Result<Tensor<B,3>,C::Error> {
        Ok(embedding(self.weight.gather_inference_with_compute_dtype::<C,2>(communicator,dtype)?,input))
    }
}

impl<B:Backend> FullyShardedProjection<B> {
    /// Native vocabulary projection using the actual shared row-major embedding/head storage.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        let weight = self.weight.gather_inference::<C,2>(communicator.clone())?.transpose();
        let bias = self.bias.as_ref().map(|bias|bias.gather_inference::<C,1>(communicator)).transpose()?;
        Ok(linear(input,weight,bias))
    }
    /// Explicit native row-major head arithmetic; only gathered values are transposed.
    pub fn forward_inference_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,dtype:FloatDType)
        -> Result<Tensor<B,D>,C::Error> {
        let storage = input.dtype();
        let weight = self.weight.gather_inference_with_compute_dtype::<C,2>(communicator.clone(),dtype)?.transpose();
        let bias = self.bias.as_ref().map(|bias|bias.gather_inference_with_compute_dtype::<C,1>(communicator,dtype)).transpose()?;
        Ok(linear(input.cast(dtype),weight,bias).cast(storage))
    }
}

impl<B:Backend> FullyShardedLoRALinear<B> {
    /// Original native dense base plus actual separate A/B adapters; no weight merge is performed.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        let adapted = self.dropout.forward(input.clone().cast(self.adapter_a.weight.local.val().dtype()));
        let base = self.base.forward_inference(input,communicator.clone())?;
        let hidden = self.adapter_a.forward_inference(adapted,communicator.clone())?.cast(self.adapter_b.weight.local.val().dtype());
        let update = self.adapter_b.forward_inference(hidden,communicator)?.mul_scalar(self.scale);
        let storage = base.dtype();Ok(base+update.cast(storage))
    }
}

impl<B:Backend> FullyShardedGatedMLP<B> {
    /// Original gated native projection order with the explicitly supplied real activation.
    pub fn forward_inference_with<C:BroadcastTensorCollective<B>,F,const D:usize>(&self,input:Tensor<B,D>,communicator:C,activation:F)
        -> Result<Tensor<B,D>,C::Error> where F:FnOnce(Tensor<B,D>)->Tensor<B,D> {
        let gate = activation(self.gate.forward_inference(input.clone(),communicator.clone())?);
        let up = self.up.forward_inference(input,communicator.clone())?;
        assert_eq!(gate.dims(),up.dims(),"native sharded gate activation changed geometry");
        self.down.forward_inference(gate*up,communicator)
    }
    /// Native SwiGLU using the same original SiLU tensor activation, not a new parameterized gate.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        self.forward_inference_with(input,communicator,ruda_model::tensor::activation::silu)
    }
}

impl<B:Backend> FullyShardedRmsNorm<B> {
    /// Native FP32 statistics and affine application, retaining incoming activation storage.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        self.forward_inference_with_compute_dtype(input,communicator,FloatDType::F32)
    }
    /// Explicit statistics/collective precision without changing the original affine storage.
    pub fn forward_inference_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,dtype:FloatDType)
        -> Result<Tensor<B,D>,C::Error> {
        assert!(D>0 && self.gamma.logical_shape==[input.dims()[D-1]],"native RMSNorm affine width differs");
        let storage = input.dtype();
        let gamma = self.gamma.gather_inference_with_compute_dtype::<C,1>(communicator,dtype)?;
        let input = input.cast(dtype);let rms = (input.clone().square().mean_dim(D-1)+self.epsilon).sqrt();
        Ok(((input/rms)*gamma.unsqueeze::<D>()).cast(storage))
    }
}

impl<B:Backend> FullyShardedLayerNorm<B> {
    /// Native last-axis FP32 normalization with original epsilon and optional affine bias.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        self.forward_inference_with_compute_dtype(input,communicator,FloatDType::F32)
    }
    /// Original native LayerNorm/autodiff-independent operation with explicit arithmetic precision.
    pub fn forward_inference_with_compute_dtype<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,dtype:FloatDType)
        -> Result<Tensor<B,D>,C::Error> {
        assert!(D>0 && self.gamma.logical_shape==[input.dims()[D-1]],"native LayerNorm affine width differs");
        let storage = input.dtype();
        let gamma = self.gamma.gather_inference_with_compute_dtype::<C,1>(communicator.clone(),dtype)?.into_primitive().tensor();
        let beta = self.beta.as_ref().map(|beta| {
            assert_eq!(beta.logical_shape,self.gamma.logical_shape,"native LayerNorm affine bias width differs");
            beta.gather_inference_with_compute_dtype::<C,1>(communicator,dtype).map(|beta|beta.into_primitive().tensor())
        }).transpose()?;
        let output = <B as ModuleOps<B>>::layer_norm(input.cast(dtype).into_primitive().tensor(),gamma,beta,self.epsilon);
        Ok(Tensor::<B,D>::from_primitive(TensorPrimitive::Float(output)).cast(storage))
    }
}
