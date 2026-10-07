use super::*;
use tensor_parallel::{inference_gather,inference_scatter,inference_sum,VocabParallelLossLayout};

impl<B:Backend> FullyShardedColumnParallelLinear<B> {
    /// Native inference from actual DP slices of a TP-local output-column projection.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,
        data:C,tensor:T,gather_output:bool) -> Result<Tensor<B,D>,HybridParallelError<C::Error,T::Error>> {
        assert!(D>0,"native hybrid column projection needs a feature axis");
        let output = self.local.forward_inference(input,data).map_err(HybridParallelError::Data)?;
        if gather_output {inference_gather(output,tensor,D-1).map_err(HybridParallelError::Tensor)} else {Ok(output)}
    }
}

impl<B:Backend> FullyShardedRowParallelLinear<B> {
    /// Native DP gather, TP partial sum, then exactly one actual replicated output-bias addition.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,
        data:C,tensor:T,input_is_parallel:bool) -> Result<Tensor<B,D>,HybridParallelError<C::Error,T::Error>> {
        assert!(D>0,"native hybrid row projection needs a feature axis");
        let input = if input_is_parallel {input} else {inference_scatter(input,&tensor,D-1)};
        let weight = self.local.weight.gather_inference::<C,2>(data.clone()).map_err(HybridParallelError::Data)?;
        let output = inference_sum(linear(input,weight,None),tensor).map_err(HybridParallelError::Tensor)?;
        Ok(if let Some(bias) = &self.local.bias {
            let bias = bias.gather_inference::<C,1>(data).map_err(HybridParallelError::Data)?;
            let mut shape = [1;D];shape[D-1] = bias.dims()[0];output+bias.reshape(shape)
        } else {output})
    }
}

impl<B:Backend> FullyShardedColumnParallelLoRA<B> {
    /// Native actual base/A/B values and mixed storage, without merging or creating adapter dropout.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,
        adapter_input:Option<Tensor<B,D>>,data:C,tensor:T,gather_output:bool) -> Result<Tensor<B,D>,HybridParallelError<C::Error,T::Error>> {
        assert!(D>0,"native hybrid adapter needs a feature axis");
        let adapted = adapter_input.unwrap_or_else(||input.clone());assert_eq!(adapted.dims(),input.dims(),"native hybrid adapter input geometry differs");
        let base = self.base.forward_inference(input,data.clone(),tensor.clone(),false)?;
        let storage = base.dtype();
        let a = self.adapter_a.weight.gather_inference::<C,2>(data.clone()).map_err(HybridParallelError::Data)?;
        let hidden = linear(adapted.cast(a.dtype()),a,None).cast(self.adapter_b.weight.local.val().dtype());
        let update = self.adapter_b.forward_inference(hidden,data).map_err(HybridParallelError::Data)?;
        let output = base+update.mul_scalar(self.scale).cast(storage);
        if gather_output {inference_gather(output,tensor,D-1).map_err(HybridParallelError::Tensor)} else {Ok(output)}
    }
}

impl<B:Backend> FullyShardedRowParallelLoRA<B> {
    /// Original native row-local A, TP-summed hidden update and DP-gathered replicated B.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,
        adapter_input:Option<Tensor<B,D>>,data:C,tensor:T,input_is_parallel:bool) -> Result<Tensor<B,D>,HybridParallelError<C::Error,T::Error>> {
        assert!(D>0,"native hybrid adapter needs a feature axis");
        let adapted = adapter_input.unwrap_or_else(||input.clone());assert_eq!(adapted.dims(),input.dims(),"native hybrid adapter input geometry differs");
        let (input,adapted) = if input_is_parallel {(input,adapted)} else {
            (inference_scatter(input,&tensor,D-1),inference_scatter(adapted,&tensor,D-1))
        };
        let base = self.base.forward_inference(input,data.clone(),tensor.clone(),true)?;
        let storage = base.dtype();
        let hidden = self.adapter_a.forward_inference(adapted.cast(self.adapter_a.weight.local.val().dtype()),data.clone()).map_err(HybridParallelError::Data)?;
        let hidden = inference_sum(hidden,tensor).map_err(HybridParallelError::Tensor)?.cast(self.adapter_b.weight.local.val().dtype());
        let update = self.adapter_b.forward_inference(hidden,data).map_err(HybridParallelError::Data)?;
        Ok(base+update.mul_scalar(self.scale).cast(storage))
    }
}

impl<B:Backend> FullyShardedTensorParallelGatedMlp<B> {
    /// Native actual TP-local intermediate and DP-sharded parameters with caller-selected gate activation.
    pub fn forward_inference_with<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>,F,const D:usize>(&self,input:Tensor<B,D>,data:C,tensor:T,activation:F)
        -> Result<Tensor<B,D>,HybridParallelError<C::Error,T::Error>> where F:FnOnce(Tensor<B,D>)->Tensor<B,D> {
        let gate = self.gate.forward_inference(input.clone(),data.clone(),tensor.clone(),false)?;
        let up = self.up.forward_inference(input,data.clone(),tensor.clone(),false)?;
        let gate = activation(gate);assert_eq!(gate.dims(),up.dims(),"native hybrid gate activation changed geometry");
        self.down.forward_inference(gate*up,data,tensor,true)
    }
    /// Native original SiLU-gated FFN, retaining real local intermediates and output-bias placement.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,data:C,tensor:T)
        -> Result<Tensor<B,D>,HybridParallelError<C::Error,T::Error>> {
        self.forward_inference_with(input,data,tensor,silu)
    }
}

impl<B:Backend> FullyShardedVocabParallelEmbedding<B> {
    /// Native actual TP-local table from DP slices with explicit real/padded vocabulary placement.
    /// Configured padding-row values and exact I64 global IDs use the existing original native lookup.
    pub fn forward_inference_with_layout<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>>(&self,tokens:Tensor<B,2,Int>,
        data:C,tensor:T,layout:&VocabParallelLossLayout) -> Result<Tensor<B,3>,HybridParallelError<C::Error,T::Error>> {
        let weight = self.weight.gather_inference::<C,2>(data).map_err(HybridParallelError::Data)?;
        let layer = tensor_parallel::VocabParallelEmbedding {local:crate::Embedding {weight:Param::initialized(self.weight.local.id,weight)},
            vocabulary_start:self.vocabulary_start,vocabulary_size:self.vocabulary_size,padding_index:self.padding_index};
        layer.forward_inference_with_layout(tokens,tensor,layout).map_err(HybridParallelError::Tensor)
    }
}

impl<B:Backend> FullyShardedVocabParallelProjection<B> {
    /// Native actual shared row-major embedding/head storage, without a persistent transposed copy.
    pub fn forward_inference_with_layout<C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,
        data:C,tensor:T,layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,HybridParallelError<C::Error,T::Error>> {
        let weight = self.weight.gather_inference::<C,2>(data.clone()).map_err(HybridParallelError::Data)?;
        let bias = self.bias.as_ref().map(|bias|bias.gather_inference::<C,1>(data).map(|value|Param::initialized(bias.local.id,value))).transpose()
            .map_err(HybridParallelError::Data)?;
        let layer = tensor_parallel::VocabParallelProjection {weight:Param::initialized(self.weight.local.id,weight),bias};
        layer.forward_inference_with_layout(input,tensor,layout,gather_output).map_err(HybridParallelError::Tensor)
    }
}
