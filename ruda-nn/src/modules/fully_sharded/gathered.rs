use super::*;

impl<B:Backend> FullyShardedLinear<B> {
    /// Transient original native module values, retaining IDs without creating persistent full weights.
    pub fn gather_inference<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::Linear<B>,C::Error> {
        let weight = Param::initialized(self.weight.local.id,self.weight.gather_inference::<C,2>(communicator.clone())?);
        let bias = self.bias.as_ref().map(|bias|bias.gather_inference::<C,1>(communicator).map(|value|Param::initialized(bias.local.id,value))).transpose()?;
        Ok(crate::Linear {weight,bias})
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedLinear<Autodiff<B,S>> {
    /// Transient original projection over differentiably gathered values; local leaves remain authoritative.
    pub fn gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::Linear<Autodiff<B,S>>,C::Error> {
        let weight = Param::initialized(self.weight.local.id,self.weight.gather::<C,2>(communicator.clone())?);
        let bias = self.bias.as_ref().map(|bias|bias.gather::<C,1>(communicator).map(|value|Param::initialized(bias.local.id,value))).transpose()?;
        Ok(crate::Linear {weight,bias})
    }
}

impl<B:Backend> FullyShardedLoRALinear<B> {
    /// Gather actual native base/A/B values without merging adapters or changing dropout/scaling.
    pub fn gather_inference<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::LoRALinear<B>,C::Error> {
        Ok(crate::LoRALinear {base:self.base.gather_inference(communicator.clone())?,adapter_a:self.adapter_a.gather_inference(communicator.clone())?,
            adapter_b:self.adapter_b.gather_inference(communicator)?,dropout:self.dropout.clone(),scale:self.scale})
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedLoRALinear<Autodiff<B,S>> {
    /// Differentiably gather the original independent base/A/B leaves without detachment or merge.
    pub fn gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::LoRALinear<Autodiff<B,S>>,C::Error> {
        Ok(crate::LoRALinear {base:self.base.gather(communicator.clone())?,adapter_a:self.adapter_a.gather(communicator.clone())?,
            adapter_b:self.adapter_b.gather(communicator)?,dropout:self.dropout.clone(),scale:self.scale})
    }
}

impl<B:Backend> FullyShardedLayerNorm<B> {
    /// Transient original native affine leaves and epsilon, without random or placeholder parameters.
    pub fn gather_inference<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::LayerNorm<B>,C::Error> {
        let gamma = Param::initialized(self.gamma.local.id,self.gamma.gather_inference::<C,1>(communicator.clone())?);
        let beta = self.beta.as_ref().map(|beta|beta.gather_inference::<C,1>(communicator).map(|value|Param::initialized(beta.local.id,value))).transpose()?;
        Ok(crate::LayerNorm::from_parameters(gamma,beta,self.epsilon))
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedLayerNorm<Autodiff<B,S>> {
    /// Original differentiable affine values; full-value retention follows the selected checkpoint strategy.
    pub fn gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::LayerNorm<Autodiff<B,S>>,C::Error> {
        let gamma = Param::initialized(self.gamma.local.id,self.gamma.gather::<C,1>(communicator.clone())?);
        let beta = self.beta.as_ref().map(|beta|beta.gather::<C,1>(communicator).map(|value|Param::initialized(beta.local.id,value))).transpose()?;
        Ok(crate::LayerNorm::from_parameters(gamma,beta,self.epsilon))
    }
}

impl<B:Backend> FullyShardedRmsNorm<B> {
    /// Transient native RMSNorm preserving the actual affine ID and original epsilon.
    pub fn gather_inference<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::RmsNorm<B>,C::Error> {
        Ok(crate::RmsNorm {gamma:Param::initialized(self.gamma.local.id,self.gamma.gather_inference::<C,1>(communicator)?),epsilon:self.epsilon})
    }
}

impl<B:Backend,S:CheckpointStrategy> FullyShardedRmsNorm<Autodiff<B,S>> {
    /// Transient native RMSNorm whose affine derivative reaches the actual local parameter leaf.
    pub fn gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<crate::RmsNorm<Autodiff<B,S>>,C::Error> {
        Ok(crate::RmsNorm {gamma:Param::initialized(self.gamma.local.id,self.gamma.gather::<C,1>(communicator)?),epsilon:self.epsilon})
    }
}
