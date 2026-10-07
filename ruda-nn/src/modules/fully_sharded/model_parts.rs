use super::*;
use crate::{Dropout,pool::{pool_sequence,pool_packed_sequences,SequencePooling,SequencePoolOutput},
    attention::PackedSequenceLayout,transformer::{TransformerEmbeddings,TransformerHead,AdaptedTransformerHead,
        AdaptedProjection,DenseTransformerNorm,SequenceHeadOutput}};
use ruda_model::tensor::Bool;

/// Backend-neutral token/position/type payload shared with the existing native model APIs.
pub type FullyShardedTransformerInput<B,const D:usize=2> = crate::tensor_parallel::TensorParallelTransformerInput<B,D>;

/// Original input tables, optional affine normalization and dropout, all persistent leaves sharded.
#[derive(Module,Debug)]
pub struct FullyShardedTransformerEmbeddings<B:Backend> {
    /// Actual token table; its row-major leaf can also be the actual output head.
    pub token:FullyShardedEmbedding<B>,
    /// Original optional learned positions, never inferred from document offsets.
    pub position:Option<FullyShardedEmbedding<B>>,
    /// Original optional type/segment table.
    pub token_type:Option<FullyShardedEmbedding<B>>,
    /// Original combined-input normalization.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Original combined-input dropout.
    pub dropout:Dropout,
}

/// Actual original output storage choice, without transposing a registered persistent leaf.
#[derive(Module,Debug)]
pub enum FullyShardedHeadProjection<B:Backend> {
    /// Original [hidden,classes] dense or native base/A/B projection.
    Column(FullyShardedAdaptedProjection<B>),
    /// Explicit shared [classes,hidden] table; no tie is inferred from compatible shapes.
    RowMajor(FullyShardedProjection<B>),
}

/// Complete native output head with sharded projection and optional affine leaves.
#[derive(Module,Debug)]
pub struct FullyShardedTransformerHead<B:Backend> {
    /// Caller-declared actual output storage and adapter choice.
    pub projection:FullyShardedHeadProjection<B>,
    /// Original pre-projection normalization, independent of a model-final norm.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Original head-input dropout, independent of any adapter-input dropout.
    pub dropout:Dropout,
}

/// Transient actual native projection values, not a new registered parameter or checkpoint.
enum GatheredHeadProjection<B:Backend> {
    Column(AdaptedProjection<B>),
    RowMajor {weight:Tensor<B,2>,bias:Option<Tensor<B,1>>},
}

/// Transient complete head, reusable across real projection chunks after one fallible gather.
/// Training values retain the actual local-leaf graph and selected checkpoint retention behavior.
pub struct GatheredFullyShardedTransformerHead<B:Backend> {
    projection:GatheredHeadProjection<B>,
    normalization:Option<DenseTransformerNorm<B>>,
    dropout:Dropout,
}

impl<B:Backend> ShardingContext<B> {
    /// Partition actual loaded tables and preserve ties to every other component using this context.
    pub fn transformer_embeddings(&mut self,embeddings:TransformerEmbeddings<B>) -> FullyShardedTransformerEmbeddings<B> {
        FullyShardedTransformerEmbeddings {token:self.embedding(embeddings.token),
            position:embeddings.position.map(|table|self.embedding(table)),
            token_type:embeddings.token_type.map(|table|self.embedding(table)),
            normalization:embeddings.normalization.map(|norm|self.normalization(norm)),dropout:embeddings.dropout}
    }

    /// Partition an actual dense native head without creating adapters or changing its dropout order.
    pub fn transformer_head(&mut self,head:TransformerHead<B>) -> FullyShardedTransformerHead<B> {
        FullyShardedTransformerHead::from_parts(FullyShardedHeadProjection::Column(
            FullyShardedAdaptedProjection::Dense(self.linear(head.projection))),
            head.normalization.map(|norm|self.normalization(norm)),head.dropout)
    }

    /// Partition original independent base/A/B storage, retaining all frozen/trainable flags.
    pub fn adapted_transformer_head(&mut self,head:AdaptedTransformerHead<B>) -> FullyShardedTransformerHead<B> {
        FullyShardedTransformerHead::from_parts(FullyShardedHeadProjection::Column(
            FullyShardedAdaptedProjection::LoRA(self.lora(head.projection))),
            head.normalization.map(|norm|self.normalization(norm)),head.dropout)
    }

    /// Connect an explicitly tied actual input table as the output, without copying its local leaf.
    pub fn tied_transformer_head(&mut self,table:&FullyShardedEmbedding<B>,bias:Option<Param<Tensor<B,1>>>,
        normalization:Option<DenseTransformerNorm<B>>,dropout:Dropout) -> FullyShardedTransformerHead<B> {
        let projection=self.tied_projection(table,bias);
        FullyShardedTransformerHead::from_parts(FullyShardedHeadProjection::RowMajor(projection),
            normalization.map(|norm|self.normalization(norm)),dropout)
    }
}

pub(super) fn projection_geometry<B:Backend>(projection:&FullyShardedAdaptedProjection<B>) -> (usize,usize) {
    let weight=match projection {FullyShardedAdaptedProjection::Dense(layer)=>&layer.weight,
        FullyShardedAdaptedProjection::LoRA(layer)=>&layer.base.weight};
    assert_eq!(weight.logical_shape.len(),2,"sharded projection must be a matrix");
    (weight.logical_shape[0],weight.logical_shape[1])
}

impl<B:Backend> FullyShardedTransformerNorm<B> {
    /// Original logical affine feature count, excluding local shard padding.
    pub fn width(&self) -> usize {
        let gamma=match self {Self::Layer(norm)=>&norm.gamma,Self::Rms(norm)=>&norm.gamma};
        assert_eq!(gamma.logical_shape.len(),1,"sharded transformer affine must be a vector");gamma.logical_shape[0]
    }
}

impl<B:Backend> FullyShardedTransformerEmbeddings<B> {
    /// Original logical hidden width, not the flat local parameter size.
    pub fn hidden_width(&self) -> usize {
        assert_eq!(self.token.weight.logical_shape.len(),2,"sharded token table must be a matrix");
        self.token.weight.logical_shape[1]
    }
}

impl<B:Backend> FullyShardedTransformerHead<B> {
    /// Assemble already prepared actual local modules; no weights are initialized or implicitly tied.
    pub fn from_parts(projection:FullyShardedHeadProjection<B>,normalization:Option<FullyShardedTransformerNorm<B>>,dropout:Dropout) -> Self {
        let head=Self {projection,normalization,dropout};
        if let Some(norm)=&head.normalization {assert_eq!(norm.width(),head.hidden_width(),"sharded head norm/input widths differ");}
        assert!(head.dropout.prob.is_finite() && (0.0..=1.0).contains(&head.dropout.prob),"invalid sharded head dropout");head
    }
    /// Actual incoming hidden width under the explicitly declared source layout.
    pub fn hidden_width(&self) -> usize {
        match &self.projection {FullyShardedHeadProjection::Column(layer)=>projection_geometry(layer).0,
            FullyShardedHeadProjection::RowMajor(layer)=>{
                assert_eq!(layer.weight.logical_shape.len(),2,"sharded output table must be a matrix");layer.weight.logical_shape[1]
            }}
    }
    /// Real output class/vocabulary count, with no padded data-shard elements.
    pub fn classes(&self) -> usize {
        match &self.projection {FullyShardedHeadProjection::Column(layer)=>projection_geometry(layer).1,
            FullyShardedHeadProjection::RowMajor(layer)=>layer.weight.logical_shape[0]}
    }
}

impl<B:Backend> GatheredFullyShardedTransformerHead<B> {
    /// Original norm -> head dropout -> actual dense/LoRA/shared-table projection expression.
    pub fn forward<const D:usize>(&self,hidden:Tensor<B,D>) -> Tensor<B,D> {
        let hidden=if let Some(norm)=&self.normalization {norm.forward(hidden)} else {hidden};
        let hidden=self.dropout.forward(hidden);
        match &self.projection {GatheredHeadProjection::Column(layer)=>layer.forward(hidden),
            GatheredHeadProjection::RowMajor {weight,bias}=>linear(hidden,weight.clone().transpose(),bias.clone())}
    }
    /// Preserve actual visible-token/count metadata from the original native pooler.
    pub fn forward_pooled(&self,pooled:SequencePoolOutput<B>) -> SequenceHeadOutput<B> {
        SequenceHeadOutput {logits:self.forward(pooled.values),valid_rows:pooled.valid_rows,token_counts:pooled.token_counts}
    }
    /// Explicit sequence pooling before the original complete output head.
    pub fn forward_sequence(&self,hidden:Tensor<B,3>,visible:Tensor<B,2,Bool>,pooling:SequencePooling) -> SequenceHeadOutput<B> {
        self.forward_pooled(pool_sequence(hidden,visible,pooling))
    }
    /// Independent packed-document pooling, including actual empty document metadata.
    pub fn forward_packed_sequences(&self,hidden:Tensor<B,2>,layout:&PackedSequenceLayout,
        visible:Option<Tensor<B,1,Bool>>,pooling:SequencePooling) -> SequenceHeadOutput<B> {
        self.forward_pooled(pool_packed_sequences(hidden,layout,visible,pooling))
    }
}

macro_rules! gathered_model_parts {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$forward:ident) => {
        impl<$($generics)*> FullyShardedTransformerEmbeddings<$backend> {
            /// Transient original loaded native tables/norm; gathers preserve actual local graph identities.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<TransformerEmbeddings<$backend>,C::Error> {
                let table=|table:&FullyShardedEmbedding<$backend>| {
                    table.weight.$gather::<C,2>(communicator.clone()).map(|value|crate::Embedding {
                        weight:Param::initialized(table.weight.local.id,value)})
                };
                Ok(TransformerEmbeddings::from_tables(table(&self.token)?,
                    self.position.as_ref().map(&table).transpose()?,self.token_type.as_ref().map(&table).transpose()?,
                    self.normalization.as_ref().map(|norm|norm.$gather(communicator.clone())).transpose()?,self.dropout.clone()))
            }
            /// Original table metadata, mixed lookup-row precision and combined-input expression.
            pub fn $forward<C:BroadcastTensorCollective<B>>(&self,input:FullyShardedTransformerInput<$backend>,communicator:C)
                -> Result<Tensor<$backend,3>,C::Error> {
                let tables=self.$gather(communicator)?;
                Ok(match input.embedding_dtypes {
                    Some((compute,output))=>tables.forward_with_compute_dtype(input.tokens,input.positions,input.token_types,compute,output),
                    None=>tables.forward(input.tokens,input.positions,input.token_types),
                })
            }
        }
        impl<$($generics)*> FullyShardedTransformerHead<$backend> {
            /// Fallible transport completes before infallible native chunk-projection callbacks.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<GatheredFullyShardedTransformerHead<$backend>,C::Error> {
                let projection=match &self.projection {
                    FullyShardedHeadProjection::Column(layer)=>GatheredHeadProjection::Column(layer.$gather(communicator.clone())?),
                    FullyShardedHeadProjection::RowMajor(layer)=>GatheredHeadProjection::RowMajor {
                        weight:layer.weight.$gather::<C,2>(communicator.clone())?,
                        bias:layer.bias.as_ref().map(|bias|bias.$gather::<C,1>(communicator.clone())).transpose()?,
                    },
                };
                Ok(GatheredFullyShardedTransformerHead {projection,
                    normalization:self.normalization.as_ref().map(|norm|norm.$gather(communicator.clone())).transpose()?,dropout:self.dropout.clone()})
            }
            /// Actual complete token/sequence head, retaining all original leading dimensions.
            pub fn $forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<$backend,D>,communicator:C)
                -> Result<Tensor<$backend,D>,C::Error> {Ok(self.$gather(communicator)?.forward(hidden))}
        }
    };
}
gathered_model_parts!(B,[B:Backend],gather_inference,forward_inference);
gathered_model_parts!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward);
