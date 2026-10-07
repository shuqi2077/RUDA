use super::*;
use crate::loss::LossTerms;

/// Explicit output projection kind for one actual locally partitioned model.
/// Row-major vocabulary variants retain embedding ties; column-major variants
/// retain ordinary native class-head storage. No conversion or merge is implicit.
#[derive(Module,Debug)]
pub enum TensorParallelOutputHead<B:Backend> {
    /// Native `[hidden,local_classes]` output projection.
    Linear(TensorParallelTransformerHead<B>),
    /// Original local linear base/B with replicated adapter A.
    AdaptedLinear(TensorParallelAdaptedTransformerHead<B>),
    /// Native `[local_vocabulary,hidden]` rows, including explicit embedding ties.
    Vocabulary(VocabParallelTransformerHead<B>),
    /// Frozen row-major vocabulary base and original independent A/B adapters.
    AdaptedVocabulary(VocabParallelAdaptedTransformerHead<B>),
}

impl<B:Backend> TensorParallelOutputHead<B> {
    /// Actual hidden input width, independent of the output class placement.
    pub fn hidden_width(&self) -> usize {
        match self {
            Self::Linear(head)=>head.local.projection.weight.val().dims()[0],
            Self::AdaptedLinear(head)=>head.local.projection.base.weight.val().dims()[0],
            Self::Vocabulary(head)=>head.projection.weight.val().dims()[1],
            Self::AdaptedVocabulary(head)=>head.projection.base.weight.val().dims()[1],
        }
    }

    /// Actual local storage classes, including caller-declared trailing padding.
    pub fn local_classes(&self) -> usize {
        match self {
            Self::Linear(head)=>head.local.projection.weight.val().dims()[1],
            Self::AdaptedLinear(head)=>head.local.projection.base.weight.val().dims()[1],
            Self::Vocabulary(head)=>head.projection.weight.val().dims()[0],
            Self::AdaptedVocabulary(head)=>head.projection.base.weight.val().dims()[0],
        }
    }

    /// Dispatch native inference without transposing parameter storage or merging adapters.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<B,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<B,D>,C::Error> {
        match self {
            Self::Linear(head)=>head.forward_inference(hidden,communicator,layout,gather_output),
            Self::AdaptedLinear(head)=>head.forward_inference(hidden,communicator,layout,gather_output),
            Self::Vocabulary(head)=>head.forward_inference(hidden,communicator,layout,gather_output),
            Self::AdaptedVocabulary(head)=>head.forward_inference(hidden,communicator,layout,gather_output),
        }
    }
}

impl<B:Backend,S:CheckpointStrategy> TensorParallelOutputHead<Autodiff<B,S>> {
    /// Original native projection graph and exact local class layout.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        match self {
            Self::Linear(head)=>head.forward(hidden,communicator,layout,gather_output),
            Self::AdaptedLinear(head)=>head.forward(hidden,communicator,layout,gather_output),
            Self::Vocabulary(head)=>head.forward(hidden,communicator,layout,gather_output),
            Self::AdaptedVocabulary(head)=>head.forward(hidden,communicator,layout,gather_output),
        }
    }

    /// Explicit shared input/adapter dropout, retaining each native branch's dtype/order.
    /// The adapter callback is unused for actual dense heads.
    pub fn forward_with_dropouts<C,F,A,const D:usize>(&self,hidden:Tensor<Autodiff<B,S>,D>,communicator:C,
        layout:&VocabParallelLossLayout,gather_output:bool,head_dropout:F,adapter_dropout:A)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error>
        where C:BroadcastTensorCollective<B>,F:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D>,
            A:FnOnce(&Dropout,Tensor<Autodiff<B,S>,D>)->Tensor<Autodiff<B,S>,D> {
        match self {
            Self::Linear(head)=>head.forward_with_dropout(hidden,communicator,layout,gather_output,head_dropout),
            Self::Vocabulary(head)=>head.forward_with_dropout(hidden,communicator,layout,gather_output,head_dropout),
            Self::AdaptedLinear(head)=>head.forward_with_dropouts(hidden,communicator,layout,gather_output,head_dropout,adapter_dropout),
            Self::AdaptedVocabulary(head)=>head.forward_with_dropouts(hidden,communicator,layout,gather_output,head_dropout,adapter_dropout),
        }
    }
}

super::training::head_objectives!(TensorParallelOutputHead);
