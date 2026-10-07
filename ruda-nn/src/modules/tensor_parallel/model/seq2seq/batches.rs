use super::*;
use crate::loss::{CausalCrossEntropyConfig,CausalLoss};
use ruda_model::data::seq2seq::{PaddedTokenBatch,PackedTokenBatch,PaddedSeq2SeqBatch,PackedSeq2SeqBatch};

impl<B:Backend,S:CheckpointStrategy> TensorParallelEncoderDecoderModel<Autodiff<B,S>> {
    /// Actual native paired padded collator data through the complete model and aligned sharded loss.
    /// Original source/target padding/reset-position metadata is visible to each actual architecture callback.
    pub fn forward_padded_batch_with<C,T,O,I,J,E,F>(&self,batch:PaddedSeq2SeqBatch<Autodiff<B,S>>,source_group:C,
        source_layout:&VocabParallelLossLayout,target_group:T,target_layout:&VocabParallelLossLayout,output_group:O,
        output_layout:&VocabParallelLossLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,
        source_input:I,target_input:J,mut encoder:E,mut decoder:F) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            I:FnOnce(&PaddedTokenBatch<Autodiff<B,S>>)->TensorParallelTransformerInput<Autodiff<B,S>>,
            J:FnOnce(&PaddedTokenBatch<Autodiff<B,S>>)->TensorParallelTransformerInput<Autodiff<B,S>>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,&PaddedTokenBatch<Autodiff<B,S>>)
                ->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,3>,Tensor<Autodiff<B,S>,3>,
                &PaddedTokenBatch<Autodiff<B,S>>,&PaddedTokenBatch<Autodiff<B,S>>)->Result<Tensor<Autodiff<B,S>,3>,C::Error> {
        criterion.forward_sharded_seq2seq_padded_batch(batch,output_layout,output_group,|source,target,_| {
            let source_ids = source_input(&source);let target_ids = target_input(&target);
            assert!(source_ids.tokens.dims() == source.input_ids.dims() && source_ids.tokens.device() == source.input_ids.device(),"prepared paired source input changed actual rows/device");
            assert!(target_ids.tokens.dims() == target.input_ids.dims() && target_ids.tokens.device() == target.input_ids.device(),"prepared paired target input changed actual rows/device");
            let memory = self.encode_with(source_ids,source_group,source_layout,|index,block,hidden|encoder(index,block,hidden,&source))?;
            self.decode_hidden_with(target_ids,memory,target_group,target_layout,|index,block,hidden,memory|decoder(index,block,hidden,memory,&source,&target))
        },|rows,group|self.head.forward(rows,group.clone(),output_layout,false),label_smoothing)
    }

    /// Native paired packed collator data, reusing exact independent source/target boundaries and alignment.
    /// Input/attention callbacks receive actual metadata; no model-family position or mask rule is inferred.
    pub fn forward_packed_batch_with<C,T,O,I,J,E,F>(&self,batch:PackedSeq2SeqBatch<Autodiff<B,S>>,source_group:C,
        source_layout:&VocabParallelLossLayout,target_group:T,target_layout:&VocabParallelLossLayout,output_group:O,
        output_layout:&VocabParallelLossLayout,criterion:&CausalCrossEntropyConfig,label_smoothing:f64,
        source_input:I,target_input:J,mut encoder:E,mut decoder:F) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where C:BroadcastTensorCollective<B>,T:BroadcastTensorCollective<B,Error=C::Error>,O:BroadcastTensorCollective<B,Error=C::Error>,
            I:FnOnce(&PackedTokenBatch<Autodiff<B,S>>,&PackedSequenceLayout)->TensorParallelTransformerInput<Autodiff<B,S>,1>,
            J:FnOnce(&PackedTokenBatch<Autodiff<B,S>>,&PackedSequenceLayout)->TensorParallelTransformerInput<Autodiff<B,S>,1>,
            E:FnMut(usize,&TensorParallelAdaptedStackLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,&PackedTokenBatch<Autodiff<B,S>>,&PackedSequenceLayout)
                ->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            F:FnMut(usize,&TensorParallelAdaptedEncoderDecoderLayer<Autodiff<B,S>>,Tensor<Autodiff<B,S>,2>,Tensor<Autodiff<B,S>,2>,
                &PackedTokenBatch<Autodiff<B,S>>,&PackedTokenBatch<Autodiff<B,S>>,&PackedSequenceLayout,&PackedSequenceLayout)
                ->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        criterion.forward_sharded_seq2seq_packed_batch(batch,output_layout,output_group,|source,target,source_packed,target_packed,_| {
            let source_ids = source_input(&source,source_packed);let target_ids = target_input(&target,target_packed);
            assert!(source_ids.tokens.dims() == source.input_ids.dims() && source_ids.tokens.device() == source.input_ids.device(),"prepared paired packed source changed actual length/device");
            assert!(target_ids.tokens.dims() == target.input_ids.dims() && target_ids.tokens.device() == target.input_ids.device(),"prepared paired packed target changed actual length/device");
            let memory = self.encode_packed_with(source_ids,source_packed,source_group,source_layout,|index,block,hidden|encoder(index,block,hidden,&source,source_packed))?;
            self.decode_packed_hidden_with(target_ids,memory,source_packed,target_packed,target_group,target_layout,
                |index,block,hidden,memory|decoder(index,block,hidden,memory,&source,&target,source_packed,target_packed))
        },|rows,group|self.head.forward(rows,group.clone(),output_layout,false),label_smoothing)
    }
}
