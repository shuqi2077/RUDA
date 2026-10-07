use ruda_model::{data::{causal::CausalTargetAlignment,seq2seq::{PaddedTokenBatch,PackedTokenBatch,
    PaddedSeq2SeqBatch,PackedSeq2SeqBatch}},tensor::{Tensor,backend::Backend}};
use crate::{attention::PackedSequenceLayout,loss::{CausalCrossEntropyConfig,CausalLoss}};

impl CausalCrossEntropyConfig {
    /// Train a paired padding-aware encoder/decoder on explicit already-aligned labels.
    /// The caller owns both actual forward stages, position rules and attention masks;
    /// projection uses the existing exact full-vocabulary token chunks and smoothing.
    pub fn forward_seq2seq_padded_batch<B: Backend>(&self,batch: PaddedSeq2SeqBatch<B>,
        forward_hidden: impl FnOnce(PaddedTokenBatch<B>,PaddedTokenBatch<B>)->Tensor<B,3>,
        project: impl Fn(Tensor<B,2>)->Tensor<B,2>,label_smoothing: f64) -> CausalLoss<B> {
        assert!(self.token_chunk_size > 0,"token_chunk_size must be positive");
        assert!(label_smoothing.is_finite() && (0.0..=1.0).contains(&label_smoothing),"label smoothing must be in [0,1]");
        assert!(!self.shift,"already-aligned encoder-decoder supervision must not be shifted again");
        assert_eq!(self.ignore_index,batch.decoder.ignore_index,"paired collator and loss ignore indices differ");
        assert_eq!(batch.decoder.target_alignment,CausalTargetAlignment::SamePosition,"decoder targets must be same-position aligned");
        batch.encoder.validate().expect("invalid actual encoder input metadata");
        let decoder = batch.decoder.token_inputs();
        decoder.validate().expect("invalid actual decoder input metadata");
        assert_eq!(batch.encoder.examples(),decoder.examples(),"actual encoder/decoder example counts differ");
        let geometry = decoder.input_ids.dims();
        let device = decoder.input_ids.device();
        assert_eq!(batch.encoder.input_ids.device(),device,"encoder and decoder must share a device");
        assert_eq!(batch.decoder.labels.device(),device,"decoder inputs and labels must share a device");
        assert_eq!(batch.decoder.labels.dims(),geometry,"decoder input/label geometry differs");
        let hidden = forward_hidden(batch.encoder,decoder);
        assert_eq!((hidden.dims()[0],hidden.dims()[1]),(geometry[0],geometry[1]),"encoder-decoder forward changed target geometry");
        assert_eq!(hidden.device(),device,"encoder-decoder forward changed target device");
        self.forward_hidden_with_smoothing(hidden,batch.decoder.labels,project,label_smoothing)
    }

    /// Train actual paired packed documents with independent source/decoder boundaries.
    /// Labels are already aligned, so all real document-start targets remain intact.
    pub fn forward_seq2seq_packed_batch<B: Backend>(&self,batch: PackedSeq2SeqBatch<B>,
        forward_hidden: impl FnOnce(PackedTokenBatch<B>,PackedTokenBatch<B>,&PackedSequenceLayout,&PackedSequenceLayout)->Tensor<B,2>,
        project: impl Fn(Tensor<B,2>)->Tensor<B,2>,label_smoothing: f64) -> CausalLoss<B> {
        assert!(self.token_chunk_size > 0,"token_chunk_size must be positive");
        assert!(label_smoothing.is_finite() && (0.0..=1.0).contains(&label_smoothing),"label smoothing must be in [0,1]");
        assert!(!self.shift,"already-aligned encoder-decoder supervision must not be shifted again");
        assert_eq!(self.ignore_index,batch.decoder.ignore_index,"paired collator and loss ignore indices differ");
        assert_eq!(batch.decoder.target_alignment,CausalTargetAlignment::SamePosition,"decoder targets must be same-position aligned");
        batch.encoder.validate().expect("invalid actual packed encoder metadata");
        let decoder = batch.decoder.token_inputs();
        decoder.validate().expect("invalid actual packed decoder metadata");
        assert_eq!(batch.encoder.examples(),decoder.examples(),"actual packed encoder/decoder example counts differ");
        let tokens = decoder.input_ids.dims()[0];
        let device = decoder.input_ids.device();
        assert_eq!(batch.encoder.input_ids.device(),device,"packed encoder and decoder must share a device");
        assert_eq!(batch.decoder.labels.device(),device,"packed decoder inputs and labels must share a device");
        assert_eq!(batch.decoder.labels.dims(),[tokens],"packed decoder input/label geometry differs");
        let encoder_layout = PackedSequenceLayout::new(batch.encoder.boundaries.clone(),batch.encoder.input_ids.dims()[0]);
        let decoder_layout = PackedSequenceLayout::new(decoder.boundaries.clone(),tokens);
        let hidden = forward_hidden(batch.encoder,decoder,&encoder_layout,&decoder_layout);
        assert_eq!(hidden.dims()[0],tokens,"packed encoder-decoder forward changed target rows");
        assert_eq!(hidden.device(),device,"packed encoder-decoder forward changed target device");
        self.forward_packed_hidden_with_smoothing(hidden,batch.decoder.labels,&decoder_layout,project,label_smoothing)
    }
}
