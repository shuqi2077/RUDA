use super::*;
use ruda_model::data::{causal::{CausalTargetAlignment,PaddedCausalBatch,PackedCausalBatch},
    seq2seq::{PaddedTokenBatch,PackedTokenBatch,PaddedSeq2SeqBatch,PackedSeq2SeqBatch}};

impl CausalCrossEntropyConfig {
    /// Train directly from native padded causal batches, retaining real visibility/reset positions.
    /// The caller's distributed backbone must honor those operands; labels are not inferred from masks.
    pub fn forward_sharded_padded_batch<B,S,C,H,P>(&self,batch:PaddedCausalBatch<Autodiff<B,S>>,layout:&VocabParallelLossLayout,
        communicator:C,forward_hidden:H,project:P,label_smoothing:f64) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>,
            H:FnOnce(Tensor<Autodiff<B,S>,2,Int>,Tensor<Autodiff<B,S>,2,Bool>,Tensor<Autodiff<B,S>,2,Int>,&C)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            P:FnMut(Tensor<Autodiff<B,S>,2>,&C)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(self.ignore_index,batch.ignore_index,"causal collator/loss sentinels differ");
        assert_eq!(self.shift,batch.target_alignment == CausalTargetAlignment::NextToken,"causal collator/loss alignment differs");
        let shape = batch.input_ids.dims();let device = batch.input_ids.device();
        assert!(batch.labels.dims() == shape && batch.attention_mask.dims() == shape && batch.position_ids.dims() == shape,
            "padded causal operands differ from actual input rows");
        assert!(batch.labels.device() == device && batch.attention_mask.device() == device && batch.position_ids.device() == device,
            "padded causal batch operands must share a device");
        let hidden = forward_hidden(batch.input_ids,batch.attention_mask,batch.position_ids,&communicator)?;
        self.forward_sharded_hidden(hidden,batch.labels,layout,communicator,project,label_smoothing)
    }

    /// Train actual independent packed causal documents through an explicit distributed backbone.
    pub fn forward_sharded_packed_batch<B,S,C,H,P>(&self,batch:PackedCausalBatch<Autodiff<B,S>>,layout:&VocabParallelLossLayout,
        communicator:C,forward_hidden:H,project:P,label_smoothing:f64) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>,
            H:FnOnce(Tensor<Autodiff<B,S>,1,Int>,&PackedSequenceLayout,&C)->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            P:FnMut(Tensor<Autodiff<B,S>,2>,&C)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert_eq!(self.ignore_index,batch.ignore_index,"packed causal collator/loss sentinels differ");
        assert_eq!(self.shift,batch.target_alignment == CausalTargetAlignment::NextToken,"packed causal collator/loss alignment differs");
        assert_eq!(batch.input_ids.dims(),batch.labels.dims(),"packed causal input/label geometry differs");
        assert_eq!(batch.input_ids.device(),batch.labels.device(),"packed causal input/labels must share a device");
        let packed = PackedSequenceLayout::new(batch.boundaries,batch.input_ids.dims()[0]);
        let hidden = forward_hidden(batch.input_ids,&packed,&communicator)?;
        self.forward_sharded_packed_hidden(hidden,batch.labels,&packed,layout,communicator,project,label_smoothing)
    }

    /// Native paired padding-aware encoder/decoder batches with sharded-vocabulary objectives.
    /// Same-position decoder labels are used exactly once, with no second causal shift.
    pub fn forward_sharded_seq2seq_padded_batch<B,S,C,H,P>(&self,batch:PaddedSeq2SeqBatch<Autodiff<B,S>>,layout:&VocabParallelLossLayout,
        communicator:C,forward_hidden:H,project:P,label_smoothing:f64) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>,
            H:FnOnce(PaddedTokenBatch<Autodiff<B,S>>,PaddedTokenBatch<Autodiff<B,S>>,&C)->Result<Tensor<Autodiff<B,S>,3>,C::Error>,
            P:FnMut(Tensor<Autodiff<B,S>,2>,&C)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert!(!self.shift,"already-aligned paired decoder targets must not be shifted again");
        assert_eq!(self.ignore_index,batch.decoder.ignore_index,"paired collator/loss sentinels differ");
        assert_eq!(batch.decoder.target_alignment,CausalTargetAlignment::SamePosition,"paired decoder targets must be same-position aligned");
        batch.encoder.validate().expect("invalid actual paired encoder input metadata");
        let decoder = batch.decoder.token_inputs();decoder.validate().expect("invalid actual paired decoder input metadata");
        assert_eq!(batch.encoder.examples(),decoder.examples(),"paired encoder/decoder example counts differ");
        let shape = decoder.input_ids.dims();let device = decoder.input_ids.device();
        assert_eq!(batch.encoder.input_ids.device(),device,"paired encoder/decoder must share a device");
        assert_eq!(batch.decoder.labels.device(),device,"paired decoder inputs/labels must share a device");
        assert_eq!(batch.decoder.labels.dims(),shape,"paired decoder input/label geometry differs");
        let hidden = forward_hidden(batch.encoder,decoder,&communicator)?;
        assert_eq!((hidden.dims()[0],hidden.dims()[1]),(shape[0],shape[1]),"paired forward changed actual decoder rows");
        assert_eq!(hidden.device(),device,"paired forward changed actual decoder device");
        self.forward_sharded_hidden(hidden,batch.decoder.labels,layout,communicator,project,label_smoothing)
    }

    /// Independent native source/target packed boundaries, with already-aligned decoder labels.
    /// Source/target attention and reset positions remain the actual backbone's explicit contract.
    pub fn forward_sharded_seq2seq_packed_batch<B,S,C,H,P>(&self,batch:PackedSeq2SeqBatch<Autodiff<B,S>>,layout:&VocabParallelLossLayout,
        communicator:C,forward_hidden:H,project:P,label_smoothing:f64) -> Result<CausalLoss<Autodiff<B,S>>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B>,
            H:FnOnce(PackedTokenBatch<Autodiff<B,S>>,PackedTokenBatch<Autodiff<B,S>>,&PackedSequenceLayout,&PackedSequenceLayout,&C)
                ->Result<Tensor<Autodiff<B,S>,2>,C::Error>,
            P:FnMut(Tensor<Autodiff<B,S>,2>,&C)->Result<Tensor<Autodiff<B,S>,2>,C::Error> {
        assert!(!self.shift,"already-aligned packed decoder targets must not be shifted again");
        assert_eq!(self.ignore_index,batch.decoder.ignore_index,"packed paired collator/loss sentinels differ");
        assert_eq!(batch.decoder.target_alignment,CausalTargetAlignment::SamePosition,"packed paired decoder targets must be same-position aligned");
        batch.encoder.validate().expect("invalid actual packed encoder input metadata");
        let decoder = batch.decoder.token_inputs();decoder.validate().expect("invalid actual packed decoder input metadata");
        assert_eq!(batch.encoder.examples(),decoder.examples(),"packed paired encoder/decoder example counts differ");
        let tokens = decoder.input_ids.dims()[0];let device = decoder.input_ids.device();
        assert_eq!(batch.encoder.input_ids.device(),device,"packed paired encoder/decoder must share a device");
        assert_eq!(batch.decoder.labels.device(),device,"packed paired decoder inputs/labels must share a device");
        assert_eq!(batch.decoder.labels.dims(),[tokens],"packed paired decoder input/label geometry differs");
        let source_layout = PackedSequenceLayout::new(batch.encoder.boundaries.clone(),batch.encoder.input_ids.dims()[0]);
        let target_layout = PackedSequenceLayout::new(decoder.boundaries.clone(),tokens);
        let hidden = forward_hidden(batch.encoder,decoder,&source_layout,&target_layout,&communicator)?;
        assert_eq!(hidden.dims()[0],tokens,"packed paired forward changed actual decoder rows");
        assert_eq!(hidden.device(),device,"packed paired forward changed actual decoder device");
        self.forward_sharded_packed_hidden(hidden,batch.decoder.labels,&target_layout,layout,communicator,project,label_smoothing)
    }
}
