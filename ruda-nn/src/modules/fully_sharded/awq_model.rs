use super::*;
use ruda_model::tensor::{Bool,FrozenAwqOps,IntegerTensorCollective};
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::TransformerKvCache,transformer::{AwqTransformerHead,AwqTransformerModel}};

/// Exact native output projection/norm/dropout with integer and floating storage sharded.
#[derive(Module,Debug)]
pub struct FullyShardedAwqTransformerHead<B:Backend> {
    /// Actual original dense/LoRA/AWQ/AWQ-LoRA output projection choice.
    pub projection:FullyShardedAwqProjection<B>,
    /// Original optional head affine norm.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Original independent head dropout.
    pub dropout:crate::Dropout,
}

/// Complete actual embeddings/backbone/final-norm/head graph with only local persistent leaves.
#[derive(Module,Debug)]
pub struct FullyShardedAwqTransformerModel<B:Backend> {
    /// Original input tables, norms and dropout, retaining all explicit ties.
    pub embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Actual original ordered mixed-projection blocks.
    pub backbone:FullyShardedAwqTransformerStack<B>,
    /// Original optional independent model-final norm.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Actual original independently selected output projection/norm/dropout.
    pub head:FullyShardedAwqTransformerHead<B>,
}
impl<B:Backend> ShardingContext<B> {
    /// Partition every actual original head leaf through the shared-ID context.
    pub fn awq_transformer_head(&mut self,head:AwqTransformerHead<B>) -> FullyShardedAwqTransformerHead<B> {
        FullyShardedAwqTransformerHead {projection:self.awq_projection(head.projection),
            normalization:head.normalization.map(|norm|self.normalization(norm)),dropout:head.dropout}
    }
    /// Partition the complete actual model, retaining one canonical local leaf for each tie.
    pub fn awq_transformer_model(&mut self,model:AwqTransformerModel<B>) -> FullyShardedAwqTransformerModel<B> {
        FullyShardedAwqTransformerModel {embeddings:self.transformer_embeddings(model.embeddings),
            backbone:self.awq_transformer_stack(model.backbone),normalization:model.normalization.map(|norm|self.normalization(norm)),
            head:self.awq_transformer_head(model.head)}
    }
}
impl<B:Backend> FullyShardedAwqTransformerModel<B> {
    /// Convert actual caller-loaded native components without random replacement values.
    pub fn from_full(model:AwqTransformerModel<B>,rank:usize,world:usize) -> Self {ShardingContext::new(rank,world).awq_transformer_model(model)}
    /// Prepare the actual native per-layer cache metadata.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {self.backbone.new_kv_cache(initial_capacity)}
}

macro_rules! awq_model_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$embed:ident,$forward:ident,$packed:ident) => {
        impl<$($generics)*> FullyShardedAwqTransformerHead<$backend> {
            /// Transient exact original head over actual packed/floating local values.
            pub fn $gather<C:IntegerTensorCollective<B>>(&self,communicator:C) -> Result<AwqTransformerHead<$backend>,C::Error> {
                Ok(AwqTransformerHead::from_projection(self.projection.$gather(communicator.clone())?,
                    self.normalization.as_ref().map(|norm|norm.$gather(communicator)).transpose()?,self.dropout.clone()))
            }
            /// Native logits on actual hidden states after exact original data gathers.
            pub fn $forward<C:IntegerTensorCollective<B>,const D:usize>(&self,hidden:Tensor<$backend,D>,communicator:C)
                -> Result<Tensor<$backend,D>,FullyShardedAwqError<C::Error,<$backend as FrozenAwqOps>::AwqError>> {
                self.$gather(communicator).map_err(FullyShardedAwqError::Collective)?.forward(hidden).map_err(FullyShardedAwqError::Projection)
            }
        }
        impl<$($generics)*> FullyShardedAwqTransformerModel<$backend> {
            /// Complete actual native ID-to-logits graph. All architecture-owned positions,
            /// optional table IDs and loss supervision remain explicit original caller inputs.
            pub fn $forward<C,F>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,
                options:DenseAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,3>,FullyShardedAwqError<C::Error,<$backend as FrozenAwqOps>::AwqError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                let hidden=self.embeddings.$embed(input,communicator.clone()).map_err(FullyShardedAwqError::Collective)?;
                let hidden=self.backbone.$forward(hidden,masks,options,communicator.clone(),positions)?;
                let hidden=if let Some(norm)=&self.normalization {norm.$gather(communicator.clone()).map_err(FullyShardedAwqError::Collective)?.forward(hidden)} else {hidden};
                self.head.$forward(hidden,communicator)
            }
            /// Actual flat-document complete model. The native input helper preserves
            /// original token/position/type metadata and explicit mixed lookup-row dtypes.
            pub fn $packed<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,
                masks:Option<&[PackedDocumentAttentionMask<$backend>]>,options:PackedAttentionOptions,communicator:C,positions:F)
                -> Result<Tensor<$backend,2>,FullyShardedAwqError<C::Error,<$backend as FrozenAwqOps>::AwqError>>
                where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                let input=super::model::packed_input(input,layout);
                let hidden=self.embeddings.$embed(input,communicator.clone()).map_err(FullyShardedAwqError::Collective)?
                    .reshape([layout.tokens(),self.embeddings.hidden_width()]);
                let hidden=self.backbone.$packed(hidden,layout,masks,options,communicator.clone(),positions)?;
                let hidden=if let Some(norm)=&self.normalization {norm.$gather(communicator.clone()).map_err(FullyShardedAwqError::Collective)?.forward(hidden)} else {hidden};
                self.head.$forward(hidden,communicator)
            }
        }
    };
}
awq_model_execution!(B,[B:FrozenAwqOps],gather_inference,forward_inference,forward_inference,forward_packed_inference);
awq_model_execution!(Autodiff<B,S>,[B:FrozenAwqOps,S:CheckpointStrategy],gather,forward,forward,forward_packed);

impl<B:FrozenAwqOps> FullyShardedAwqTransformerModel<B> {
    /// Original complete new-token inference with positioned native KV cache and actual logits.
    /// Partial cache failures retain the original cache recovery requirement.
    pub fn forward_cached_inference<C,F>(&self,input:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,mut positions:F)
        -> Result<Tensor<B,3>,FullyShardedAwqError<C::Error,B::AwqError>>
        where C:IntegerTensorCollective<B>,F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let mut hidden=self.embeddings.forward_inference(input,communicator.clone()).map_err(FullyShardedAwqError::Collective)?;
        cache.validate_layers(self.backbone.blocks.len());let rows=(hidden.dims()[0],hidden.dims()[1]);
        let next=cache.position().checked_add(rows.1).expect("cached sharded model position overflows");
        for (index,block) in self.backbone.blocks.iter().enumerate() {
            hidden=block.forward_cached_inference(hidden,visible.clone(),&mut cache.layers_mut()[index],masks.clone(),options,communicator.clone(),
                |query,key,position|positions(index,query,key,position))?;
            assert_eq!((hidden.dims()[0],hidden.dims()[1]),rows,"cached model layer changed actual rows");
        }
        cache.finish_chunk(next);
        let hidden=if let Some(norm)=&self.normalization {norm.gather_inference(communicator.clone()).map_err(FullyShardedAwqError::Collective)?.forward(hidden)} else {hidden};
        self.head.forward_inference(hidden,communicator)
    }
}
