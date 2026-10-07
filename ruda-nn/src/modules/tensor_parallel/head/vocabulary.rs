use super::*;
use crate::{loss::LossTerms,transformer::DenseTransformerNorm};
use super::super::{VocabParallelProjection,VocabParallelEmbedding};
use ruda_model::module::Param;

/// Native normalized/dropout vocabulary head sharing actual row-major embedding storage.
/// Its weight stays `[local_vocabulary,hidden]` with the original embedding parameter ID.
/// No transposed parameter copy, automatic vocabulary partition or output gather is introduced.
#[derive(Module,Debug)]
pub struct VocabParallelTransformerHead<B:Backend> {
    /// Actual native vocabulary rows and optional local output bias.
    pub projection:VocabParallelProjection<B>,
    /// Caller-selected replicated preprojection normalization, retaining its native parameters.
    pub normalization:Option<DenseTransformerNorm<B>>,
    /// Original head-input dropout before the tied projection.
    pub dropout:Dropout,
}

impl<B:Backend> VocabParallelTransformerHead<B> {
    /// Connect actual loaded vocabulary storage and norm/dropout without replacing weight IDs.
    pub fn from_projection(projection:VocabParallelProjection<B>,normalization:Option<DenseTransformerNorm<B>>,dropout:Dropout,
        layout:&VocabParallelLossLayout,rank:usize) -> Self {
        let [width,hidden] = projection.weight.val().dims();
        assert!(hidden > 0,"native vocabulary head hidden width must be positive");
        assert_eq!(width,layout.interval(rank).len(),"native head vocabulary rows differ from actual rank interval");
        if let Some(norm) = &normalization {assert_eq!(norm.width(),hidden,"native vocabulary head norm/hidden width differs");}
        if let Some(bias) = &projection.bias {
            assert_eq!(bias.val().dims(),[width],"native vocabulary head local bias width differs");
            assert_eq!(bias.val().device(),projection.weight.val().device(),"native vocabulary head bias/weight devices differ");
        }
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid native vocabulary head dropout");
        Self {projection,normalization,dropout}
    }

    /// Retain this exact embedding/head tie with explicit corresponding vocabulary layout.
    /// Original embedding padding-gradient policy remains in its lookup branch, not imposed on
    /// the head: an actual padding token may still be an explicitly selected output class.
    pub fn from_embedding(embedding:&VocabParallelEmbedding<B>,bias:Option<Param<Tensor<B,1>>>,normalization:Option<DenseTransformerNorm<B>>,
        dropout:Dropout,layout:&VocabParallelLossLayout,rank:usize) -> Self {
        assert_eq!(embedding.vocabulary_start,layout.interval(rank).start,"embedding/head vocabulary starts differ");
        assert_eq!(embedding.vocabulary_size,layout.vocabulary_size(),"embedding/head logical vocabularies differ");
        Self::from_projection(VocabParallelProjection::from_embedding(embedding,bias),normalization,dropout,layout,rank)
    }

    /// Native inference retaining original normalization/dropout order and tied embedding storage.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<B,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,C::Error> {
        let hidden = if let Some(norm) = &self.normalization {norm.forward(hidden)} else {hidden};
        self.projection.forward_inference_with_layout(self.dropout.forward(hidden),communicator,layout,gather_output)
    }
}

impl<B:Backend,S:CheckpointStrategy> VocabParallelTransformerHead<Autodiff<B,S>> {
    /// Local tied-head logits with native embedding/head leaf gradients and full hidden derivatives.
    /// Corresponding replicated dropout is caller-owned; native parameter ties are never detached.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        self.forward_with_dropout(hidden,communicator,layout,gather_output,|dropout,input|dropout.forward(input))
    }

    /// Apply an explicit shared head-input dropout transform while keeping the native tie.
    pub fn forward_with_dropout<C,F,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool,dropout:F) -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        let hidden = if let Some(norm) = &self.normalization {norm.forward(hidden)} else {hidden};
        self.projection.forward_with_layout(transformed(&self.dropout,hidden,dropout),communicator,layout,gather_output)
    }
}

super::training::head_objectives!(VocabParallelTransformerHead);
