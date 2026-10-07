use super::*;
use crate::loss::{CausalCrossEntropyConfig,CausalLoss};
use ruda_model::{tensor::Bool,data::causal::{PaddedCausalBatch,PackedCausalBatch}};

impl<B:Backend,S:CheckpointStrategy> TensorParallelTransformerModel<Autodiff<B,S>> {
    /// Actual native collator batch through this complete model and its native sharded causal criterion.
    /// The input callback assigns real learned-table/type/dtype operands; the layer callback receives
    /// original padding visibility and reset position IDs for architecture-owned attention/rotary handling.
    pub fn forward_padded_causal_batch_with<C,O,I,F>(&self,batch:PaddedCausalBatch<Autodiff<B,S>>,input_group:C,
        input_layout:&VocabParallelLossLayout,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,prepare_input:I,mut layer:F)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            I:FnOnce(Tensor<Autodiff<B,S>,2,Int>,&Tensor<Autodiff<B,S>,2,Int>)->TensorParallelTransformerInput<Autodiff<B,S>>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,&Tensor<Autodiff<B,S>,2,Bool>,
                &Tensor<Autodiff<B,S>,2,Int>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        criterion.forward_sharded_padded_batch(batch,output_layout,output_group,|tokens,visible,positions,_| {
            let shape = tokens.dims();let device = tokens.device();let input = prepare_input(tokens,&positions);
            assert!(input.tokens.dims() == shape && input.tokens.device() == device,"prepared parallel batch input changed actual rows/device");
            self.forward_hidden_with(input,input_group,input_layout,|index,block,hidden|layer(index,block,hidden,&visible,&positions))
        },|rows,group|self.head.forward(rows,group.clone(),output_layout,false),label_smoothing)
    }

    /// Actual independent packed collator documents through the complete model and native causal loss.
    /// Original boundaries/target alignment/sentinel are reused, not reconstructed from padding heuristics.
    pub fn forward_packed_causal_batch_with<C,O,I,F>(&self,batch:PackedCausalBatch<Autodiff<B,S>>,input_group:C,
        input_layout:&VocabParallelLossLayout,output_group:O,output_layout:&VocabParallelLossLayout,
        criterion:&CausalCrossEntropyConfig,label_smoothing:f64,prepare_input:I,mut layer:F)
        -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,O:BroadcastTensorCollective<B,Error=C::Error>,
            I:FnOnce(Tensor<Autodiff<B,S>,1,Int>,&PackedSequenceLayout)->TensorParallelTransformerInput<Autodiff<B,S>,1>,
            F:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,&PackedSequenceLayout)
                ->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        criterion.forward_sharded_packed_batch(batch,output_layout,output_group,|tokens,packed,_| {
            let shape = tokens.dims();let device = tokens.device();let input = prepare_input(tokens,packed);
            assert!(input.tokens.dims() == shape && input.tokens.device() == device,"prepared parallel packed input changed actual token length/device");
            self.forward_packed_hidden_with(input,packed,input_group,input_layout,|index,block,hidden|layer(index,block,hidden,packed))
        },|rows,group|self.head.forward(rows,group.clone(),output_layout,false),label_smoothing)
    }
}
