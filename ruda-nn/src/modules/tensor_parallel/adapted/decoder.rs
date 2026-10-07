use super::{Autodiff,Backend,BroadcastTensorCollective,CheckpointStrategy,AttentionParallelGroups,Dropout,Module,Tensor,geometry};
use super::{TensorParallelAdaptedGroupedQueryAttention,TensorParallelAdaptedStackLayer};
use super::super::TensorParallelCrossAttentionBlock;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions},cache::ProjectedKvCache,
    transformer::{AdaptedCrossAttentionBlock,DecoderCrossAttention,DenseTransformerNorm,AttentionAdapterTarget}};
use ruda_model::tensor::Bool;

mod layer;
mod stack;
pub use layer::*;
pub use stack::*;

/// Actual selected cross-attention adapters on explicit local query/KV/output shards.
#[derive(Module,Debug)]
pub struct TensorParallelAdaptedCrossAttentionBlock<B: Backend> {
    /// Original selected projections, with their actual A/B storage, IDs and scales.
    pub attention: TensorParallelAdaptedGroupedQueryAttention<B>,
    /// Original replicated query/residual normalization.
    pub query_norm: DenseTransformerNorm<B>,
    /// Original optional replicated encoder-memory normalization.
    pub memory_norm: Option<DenseTransformerNorm<B>>,
    /// Original full-residual branch dropout; corresponding replicas must share its values.
    pub residual_dropout: Dropout,
    /// Original pre/post query normalization rule.
    pub norm_first: bool,
}

impl<B: Backend> TensorParallelAdaptedCrossAttentionBlock<B> {
    /// Connect actual loaded local adapters without reselecting targets or freezing other parameters.
    pub fn from_sharded_block(block: AdaptedCrossAttentionBlock<B>) -> Self {
        assert_eq!(block.query_norm.width(),geometry(&block.attention.query)[0],"adapted parallel cross query norm width differs");
        if let Some(norm) = &block.memory_norm {assert_eq!(norm.width(),geometry(&block.attention.key)[0],"adapted parallel cross memory norm width differs");}
        Self {attention:TensorParallelAdaptedGroupedQueryAttention::from_shard(block.attention),query_norm:block.query_norm,
            memory_norm:block.memory_norm,residual_dropout:block.residual_dropout,norm_first:block.norm_first}
    }

    /// Return original native local containers for existing model/adapter-record integrations.
    pub fn into_local_block(self) -> AdaptedCrossAttentionBlock<B> {
        AdaptedCrossAttentionBlock {attention:self.attention.local,query_norm:self.query_norm,memory_norm:self.memory_norm,
            residual_dropout:self.residual_dropout,norm_first:self.norm_first}
    }

    fn source<const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {if self.norm_first {self.query_norm.forward(input)} else {input}}
    fn memory<const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {if let Some(norm) = &self.memory_norm {norm.forward(input)} else {input}}
    fn finish_with<R,const D: usize>(&self,input: Tensor<B,D>,branch: Tensor<B,D>,dropout: R) -> Tensor<B,D>
        where R: FnOnce(Tensor<B,D>)->Tensor<B,D> {
        let output = input+dropout(branch);
        if self.norm_first {output} else {self.query_norm.forward(output)}
    }
    fn finish<const D: usize>(&self,input: Tensor<B,D>,branch: Tensor<B,D>) -> Tensor<B,D> {
        self.finish_with(input,branch,|branch|self.residual_dropout.forward(branch))
    }

    /// Native actual unmerged adapters on independent query/source rows and positions.
    pub fn forward_inference<C,F>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,
        communicator: C,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let query = self.attention.project_query_inference(self.source(input.clone()));
        let (key,value) = self.attention.project_memory_inference(self.memory(memory));
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"native adapted cross positions changed local query/source heads");
        Ok(self.finish(input,self.attention.forward_projected_inference(query,key,value,masks,options,communicator)?))
    }

    /// Native immutable source preparation retains actual memory norm, selected adapters and source positions.
    pub fn prepare_memory_inference<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,positions: F) -> ProjectedKvCache<B>
        where F: FnOnce(Tensor<B,4>)->Tensor<B,4> {
        let (key,value) = self.attention.project_memory_inference(self.memory(memory));
        let geometry = key.dims();let key = positions(key);
        assert_eq!(key.dims(),geometry,"native adapted prepared source positions changed local keys");
        ProjectedKvCache::from_projected(key,value,visible,0)
    }

    /// Native query-only continuation over positioned source K/V, with the actual decoder offset.
    pub fn forward_cached_inference<C,F>(&self,input: Tensor<B,3>,memory: &ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,
        communicator: C,position: usize,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let query = self.attention.project_query_inference(self.source(input.clone()));
        let geometry = query.dims();let query = positions(query,position);
        assert_eq!(query.dims(),geometry,"native adapted cached cross positions changed local queries");
        Ok(self.finish(input,self.attention.forward_cached_memory_inference(query,memory,masks,options,communicator)?))
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedCrossAttentionBlock<Autodiff<B,S>> {
    /// Actual selected cross-attention adapters with explicit head/KV replica gradient semantics.
    pub fn forward<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,memory: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        self.forward_with_transforms(input,memory,masks,options,groups,positions,|_,module,input|module.forward(input),|branch|self.residual_dropout.forward(branch))
    }

    /// Explicit per-target A-input dropout and actual shared residual transform.
    /// Matching replica draws and source/query position policies are not inferred from model names.
    pub fn forward_with_transforms<C,K,F,A,R>(&self,input: Tensor<Autodiff<B,S>,3>,memory: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F,mut dropout: A,branch_output: R) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>),
            A: FnMut(AttentionAdapterTarget,&Dropout,Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3>,R: FnOnce(Tensor<Autodiff<B,S>,3>)->Tensor<Autodiff<B,S>,3> {
        let query = self.attention.project_query_with_adapter_dropout(self.source(input.clone()),groups.heads.clone(),|module,input|dropout(AttentionAdapterTarget::Query,module,input))?;
        let (key,value) = self.attention.project_memory_with_adapter_dropout(self.memory(memory),groups,&mut dropout)?;
        let geometry = (query.dims(),key.dims());let (query,key) = positions(query,key);
        assert_eq!((query.dims(),key.dims()),geometry,"adapted parallel cross positions changed actual local head geometry");
        let branch = self.attention.forward_projected_with_adapter_dropout(query,key,value,masks,options,groups.heads.clone(),
            |module,input|dropout(AttentionAdapterTarget::Output,module,input))?;
        Ok(self.finish_with(input,branch,branch_output))
    }

    /// Prepare actual adapted source K/V once; retained source history is intentionally detached.
    pub fn prepare_memory<C,K,F>(&self,memory: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<ProjectedKvCache<Autodiff<B,S>>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,F: FnOnce(Tensor<Autodiff<B,S>,4>)->Tensor<Autodiff<B,S>,4> {
        let (key,value) = self.attention.project_memory(self.memory(memory),groups)?;
        let geometry = key.dims();let key = positions(key);
        assert_eq!(key.dims(),geometry,"adapted parallel prepared source positions changed actual local keys");
        Ok(ProjectedKvCache::from_projected(key,value,visible,0))
    }

    /// Query-only cached continuation; actual decoder offsets never derive from encoder source length.
    pub fn forward_cached<C,F>(&self,input: Tensor<Autodiff<B,S>,3>,memory: &ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        communicator: C,position: usize,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<Autodiff<B,S>,4>,usize)->Tensor<Autodiff<B,S>,4> {
        let query = self.attention.project_query(self.source(input.clone()),communicator.clone())?;
        let geometry = query.dims();let query = positions(query,position);
        assert_eq!(query.dims(),geometry,"adapted parallel cached cross positions changed local queries");
        Ok(self.finish(input,self.attention.forward_cached_memory(query,memory,masks,options,communicator)?))
    }
}

/// Original dense/adapted encoder-memory stage, with no implicit target conversion.
#[derive(Module,Debug)]
pub enum TensorParallelAdaptedDecoderCrossAttention<B: Backend> {
    /// Original dense memory attention and original trainable flags.
    Dense(TensorParallelCrossAttentionBlock<B>),
    /// Explicit selected native LoRA/rsLoRA projection adapters.
    Adapted(TensorParallelAdaptedCrossAttentionBlock<B>),
}

impl<B: Backend> TensorParallelAdaptedDecoderCrossAttention<B> {
    /// Preserve the exact original memory-stage selection and all parameter identities.
    pub fn from_sharded_stage(stage: DecoderCrossAttention<B>) -> Self {
        match stage {DecoderCrossAttention::Dense(block)=>Self::Dense(TensorParallelCrossAttentionBlock::from_sharded_block(block)),
            DecoderCrossAttention::Adapted(block)=>Self::Adapted(TensorParallelAdaptedCrossAttentionBlock::from_sharded_block(block))}
    }
    /// Return the native actual dense/adapted stage for existing save/load integrations.
    pub fn into_local_stage(self) -> DecoderCrossAttention<B> {
        match self {Self::Dense(block)=>DecoderCrossAttention::Dense(block.into_local_block()),Self::Adapted(block)=>DecoderCrossAttention::Adapted(block.into_local_block())}
    }
    fn query_width(&self) -> usize {
        match self {Self::Dense(block)=>block.attention.local.query.weight.val().dims()[0],Self::Adapted(block)=>geometry(&block.attention.local.query)[0]}
    }

    /// Native inference with the actual independent query/source positions.
    pub fn forward_inference<C,F>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,communicator: C,positions: F)
        -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        match self {Self::Dense(block)=>block.forward_inference(input,memory,masks,options,communicator,positions),
            Self::Adapted(block)=>block.forward_inference(input,memory,masks,options,communicator,positions)}
    }
    /// Native immutable source state on the actual selected memory stage.
    pub fn prepare_memory_inference<F>(&self,memory: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>,positions: F) -> ProjectedKvCache<B>
        where F: FnOnce(Tensor<B,4>)->Tensor<B,4> {
        match self {Self::Dense(block)=>block.prepare_memory_inference(memory,visible,positions),Self::Adapted(block)=>block.prepare_memory_inference(memory,visible,positions)}
    }
    /// Native cached query stage without reevaluating retained encoder K/V.
    pub fn forward_cached_inference<C,F>(&self,input: Tensor<B,3>,memory: &ProjectedKvCache<B>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions,
        communicator: C,position: usize,positions: F) -> Result<Tensor<B,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        match self {Self::Dense(block)=>block.forward_cached_inference(input,memory,masks,options,communicator,position,positions),
            Self::Adapted(block)=>block.forward_cached_inference(input,memory,masks,options,communicator,position,positions)}
    }
}

impl<B: Backend,S: CheckpointStrategy> TensorParallelAdaptedDecoderCrossAttention<Autodiff<B,S>> {
    /// Actual native dense/adapted cross graph with explicit head/KV replica groups.
    pub fn forward<C,K,F>(&self,input: Tensor<Autodiff<B,S>,3>,memory: Tensor<Autodiff<B,S>,3>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,
            F: FnOnce(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>)->(Tensor<Autodiff<B,S>,4>,Tensor<Autodiff<B,S>,4>) {
        match self {Self::Dense(block)=>block.forward(input,memory,masks,options,groups,positions),Self::Adapted(block)=>block.forward(input,memory,masks,options,groups,positions)}
    }
    /// Actual per-stage source preparation, preserving its selected adapter parameters.
    pub fn prepare_memory<C,K,F>(&self,memory: Tensor<Autodiff<B,S>,3>,visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,
        groups: &AttentionParallelGroups<C,K>,positions: F) -> Result<ProjectedKvCache<Autodiff<B,S>>,C::Error>
        where C: BroadcastTensorCollective<B>,K: BroadcastTensorCollective<B,Error=C::Error>,F: FnOnce(Tensor<Autodiff<B,S>,4>)->Tensor<Autodiff<B,S>,4> {
        match self {Self::Dense(block)=>block.prepare_memory(memory,visible,groups,positions),Self::Adapted(block)=>block.prepare_memory(memory,visible,groups,positions)}
    }
    /// Actual selected query/output projections on detached immutable source history.
    pub fn forward_cached<C,F>(&self,input: Tensor<Autodiff<B,S>,3>,memory: &ProjectedKvCache<Autodiff<B,S>>,masks: DenseAttentionMask<Autodiff<B,S>>,options: DenseAttentionOptions,
        communicator: C,position: usize,positions: F) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C: BroadcastTensorCollective<B>,F: FnOnce(Tensor<Autodiff<B,S>,4>,usize)->Tensor<Autodiff<B,S>,4> {
        match self {Self::Dense(block)=>block.forward_cached(input,memory,masks,options,communicator,position,positions),
            Self::Adapted(block)=>block.forward_cached(input,memory,masks,options,communicator,position,positions)}
    }
}
