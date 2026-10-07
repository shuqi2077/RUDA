use super::*;
use crate::loss::{CausalCrossEntropyConfig,CausalLoss};

impl<B:Backend,S:CheckpointStrategy> TensorParallelEncoderDecoderModel<Autodiff<B,S>> {
    /// Complete paired model graph, with independently declared source/target/output vocabularies.
    pub fn forward_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool) -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        let memory = self.encode_with(source,source_group,source_layout,encoder)?;
        let hidden = self.decode_hidden_with(target,memory,target_group,target_layout,decoder)?;
        self.head.forward(hidden,output_group,output_layout,gather_output)
    }

    /// Complete teacher-forced paired objective over explicitly same-position decoder targets.
    /// Native criterion chunks/sentinel/smoothing are retained; already aligned labels are not shifted again.
    pub fn forward_aligned_loss_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>>,target:TensorParallelTransformerInput<Autodiff<B,S>>,
        labels:Tensor<Autodiff<B,S>,2,Int>,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        assert!(!criterion.shift,"explicitly aligned paired model labels must not be shifted again");
        assert!(labels.dims() == target.tokens.dims() && labels.device() == target.tokens.device(),"paired model target/label rows or device differ");
        let memory = self.encode_with(source,source_group,source_layout,encoder)?;
        let hidden = self.decode_hidden_with(target,memory,target_group,target_layout,decoder)?;
        self.head.forward_causal_loss(hidden,labels,criterion,output_group,output_layout,label_smoothing)
    }

    /// Actual paired packed source/target model graph with separate independent-document boundaries.
    pub fn forward_packed_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,
        target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,output_group:O,output_layout:&VocabParallelLossLayout,gather_output:bool)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(source_packed.documents(),target_packed.documents(),"packed paired model document counts differ");
        let memory = self.encode_packed_with(source,source_packed,source_group,source_layout,encoder)?;
        let hidden = self.decode_packed_hidden_with(target,memory,source_packed,target_packed,target_group,target_layout,decoder)?;
        self.head.forward(hidden,output_group,output_layout,gather_output)
    }

    /// Native packed paired teacher forcing, without shifting aligned targets or mixing source documents.
    pub fn forward_packed_aligned_loss_with<C,T,O,E,F>(&self,source:TensorParallelTransformerInput<Autodiff<B,S>,1>,target:TensorParallelTransformerInput<Autodiff<B,S>,1>,
        labels:Tensor<Autodiff<B,S>,1,Int>,source_packed:&PackedSequenceLayout,target_packed:&PackedSequenceLayout,
        source_group:C,source_layout:&VocabParallelLossLayout,encoder:E,target_group:T,target_layout:&VocabParallelLossLayout,decoder:F,
        output_group:O,output_layout:&VocabParallelLossLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert!(!criterion.shift,"explicitly aligned packed paired model labels must not be shifted again");
        assert!(labels.dims() == target.tokens.dims() && labels.device() == target.tokens.device(),"packed paired model target/label rows or device differ");
        assert_eq!(source_packed.documents(),target_packed.documents(),"packed paired model document counts differ");
        let memory = self.encode_packed_with(source,source_packed,source_group,source_layout,encoder)?;
        let hidden = self.decode_packed_hidden_with(target,memory,source_packed,target_packed,target_group,target_layout,decoder)?;
        self.head.forward_packed_causal_loss(hidden,labels,target_packed,criterion,output_group,output_layout,label_smoothing)
    }
}
