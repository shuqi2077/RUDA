//! Paired encoder/decoder collation with explicit already-aligned teacher-forcing targets.
use crate::tensor::{Bool,DType,Int,Tensor,TensorData,backend::Backend};
use super::causal::{CausalBatch,CausalBatchLayout,CausalBatcher,CausalExample,CausalPadding,CausalTargetAlignment,
    PaddedCausalBatch,PackedCausalBatch};
use serde::{Deserialize,Serialize};
use std::{error::Error,fmt};

/// Invalid actual paired input/label geometry or explicit layout configuration.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Seq2SeqDataError(pub String);
impl fmt::Display for Seq2SeqDataError {
    fn fmt(&self,formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(&self.0) }
}
impl Error for Seq2SeqDataError {}
fn invalid(message: impl Into<String>) -> Seq2SeqDataError { Seq2SeqDataError(message.into()) }

/// One actual encoded source and its explicit decoder inputs and same-position labels.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct Seq2SeqExample {
    /// Actual source IDs; no implicit tokenization, BOS, EOS or separator insertion.
    pub encoder_input_ids: Vec<i64>,
    /// Actual nonempty teacher-forcing inputs, prepared by the caller's tokenizer/model contract.
    pub decoder_input_ids: Vec<i64>,
    /// Label t is supervised by decoder hidden position t; no second shift is performed.
    pub labels: Vec<i64>,
}

/// Paired document layout, with independently supplied source/decoder padding policies.
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub enum Seq2SeqBatchLayout {
    /// Independent source/decoder concatenations; corresponding document indices remain paired.
    Packed,
    /// Independent rectangular source/decoder lengths and explicit tokenizer padding IDs.
    Padded {
        /// Actual encoder padding ID, not inferred from decoder or EOS.
        encoder_pad_token_id: i64,
        /// Actual decoder padding ID, not inferred from encoder or EOS.
        decoder_pad_token_id: i64,
        /// Source padding direction.
        encoder_padding: CausalPadding,
        /// Decoder padding direction.
        decoder_padding: CausalPadding,
    },
}

/// Native paired collation without model-specific target construction or truncation.
#[derive(Clone,Debug,PartialEq,Eq,Serialize,Deserialize)]
pub struct Seq2SeqBatcher {
    /// Actual source/decoder tensor layout.
    pub layout: Seq2SeqBatchLayout,
    /// Explicit decoder target sentinel; independent of either attention mask.
    pub ignore_index: i64,
    /// Optional source limit; excess documents are rejected, never truncated.
    pub maximum_encoder_length: Option<usize>,
    /// Optional decoder limit, independently enforced before uploading tensors.
    pub maximum_decoder_length: Option<usize>,
}

/// Actual padded token inputs without labels, for a model's encoder or decoder.
#[derive(Clone,Debug)]
pub struct PaddedTokenBatch<B: Backend> {
    /// [examples,maximum actual length], including explicitly masked padding cells.
    pub input_ids: Tensor<B,2,Int>,
    /// True exactly at actual input tokens, not a supervision/prompt mask.
    pub attention_mask: Tensor<B,2,Bool>,
    /// Actual per-document reset positions, with zero in masked padding cells.
    pub position_ids: Tensor<B,2,Int>,
    /// Original lengths in actual paired example order.
    pub lengths: Vec<usize>,
}

impl<B: Backend> PaddedTokenBatch<B> {
    /// Actual document count, independent of rectangular token storage size.
    pub fn examples(&self) -> usize { self.lengths.len() }

    /// Check metadata geometry/device without downloading token or mask contents.
    pub fn validate(&self) -> Result<(),Seq2SeqDataError> {
        let shape = self.input_ids.dims();
        let device = self.input_ids.device();
        if shape[0] != self.lengths.len() || self.lengths.iter().any(|&length|length > shape[1])
            || self.attention_mask.dims() != shape || self.position_ids.dims() != shape
            || self.attention_mask.device() != device || self.position_ids.device() != device {
            return Err(invalid("padded input lengths, visibility, positions or device differ"));
        }
        Ok(())
    }

    /// Move actual input tensors without changing their geometry or host length metadata.
    pub fn to_device(mut self,device: &B::Device) -> Self {
        self.input_ids = self.input_ids.to_device(device);
        self.attention_mask = self.attention_mask.to_device(device);
        self.position_ids = self.position_ids.to_device(device);
        self
    }
}

/// Actual flat document inputs without labels or synthetic separator tokens.
#[derive(Clone,Debug)]
pub struct PackedTokenBatch<B: Backend> {
    /// Flat actual tokenizer IDs.
    pub input_ids: Tensor<B,1,Int>,
    /// Actual input visibility; ordinary packed collation marks every real token visible.
    pub attention_mask: Tensor<B,1,Bool>,
    /// Actual per-document positions reset to zero.
    pub position_ids: Tensor<B,1,Int>,
    /// Cumulative source/decoder boundaries, independently beginning at zero.
    pub boundaries: Vec<usize>,
}

impl<B: Backend> PackedTokenBatch<B> {
    /// Actual retained document count, including any explicitly empty sources.
    pub fn examples(&self) -> usize { self.boundaries.len().saturating_sub(1) }

    /// Check exact physical boundaries/geometry/device without tensor readback.
    pub fn validate(&self) -> Result<(),Seq2SeqDataError> {
        let shape = self.input_ids.dims();
        let device = self.input_ids.device();
        if self.boundaries.is_empty() || self.boundaries[0] != 0 || self.boundaries.last() != Some(&shape[0])
            || self.boundaries.windows(2).any(|pair|pair[0] > pair[1])
            || self.attention_mask.dims() != shape || self.position_ids.dims() != shape
            || self.attention_mask.device() != device || self.position_ids.device() != device {
            return Err(invalid("packed input boundaries, visibility, positions or device differ"));
        }
        Ok(())
    }

    /// Move actual flat inputs while retaining their exact document boundaries.
    pub fn to_device(mut self,device: &B::Device) -> Self {
        self.input_ids = self.input_ids.to_device(device);
        self.attention_mask = self.attention_mask.to_device(device);
        self.position_ids = self.position_ids.to_device(device);
        self
    }
}

impl<B: Backend> PaddedCausalBatch<B> {
    /// Label-free input view; cloning tensor handles does not duplicate token values.
    pub fn token_inputs(&self) -> PaddedTokenBatch<B> {
        PaddedTokenBatch {input_ids:self.input_ids.clone(),attention_mask:self.attention_mask.clone(),
            position_ids:self.position_ids.clone(),lengths:self.lengths.clone()}
    }
}

impl<B: Backend> PackedCausalBatch<B> {
    /// Label-free flat input view with the exact original document boundaries.
    pub fn token_inputs(&self) -> PackedTokenBatch<B> {
        PackedTokenBatch {input_ids:self.input_ids.clone(),attention_mask:self.attention_mask.clone(),
            position_ids:self.position_ids.clone(),boundaries:self.boundaries.clone()}
    }
}

/// Native paired padded inputs and unchanged aligned decoder supervision.
#[derive(Clone,Debug)]
pub struct PaddedSeq2SeqBatch<B: Backend> {
    /// Actual encoder inputs; no unused encoder-label tensor is allocated.
    pub encoder: PaddedTokenBatch<B>,
    /// Actual decoder inputs/labels; target_alignment is SamePosition.
    pub decoder: PaddedCausalBatch<B>,
}

/// Native paired packed inputs with independent actual source/decoder boundaries.
#[derive(Clone,Debug)]
pub struct PackedSeq2SeqBatch<B: Backend> {
    /// Actual encoder documents, including explicitly empty sources if supplied.
    pub encoder: PackedTokenBatch<B>,
    /// Actual aligned decoder supervision; no label shift crosses documents.
    pub decoder: PackedCausalBatch<B>,
}

/// Explicit native paired device batch for padded or packed encoder-decoder models.
#[derive(Clone,Debug)]
pub enum Seq2SeqBatch<B: Backend> {
    /// Independently padded source and decoder input axes.
    Padded(PaddedSeq2SeqBatch<B>),
    /// Independently packed paired document axes.
    Packed(PackedSeq2SeqBatch<B>),
}

impl<B: Backend> Seq2SeqBatch<B> {
    /// Actual decoder supervised labels for weighted gradient accumulation.
    pub fn supervised_tokens(&self) -> usize {
        match self {Self::Padded(batch)=>batch.decoder.supervised_tokens,Self::Packed(batch)=>batch.decoder.supervised_tokens}
    }

    /// Actual paired example count for committing the source sampler cursor.
    pub fn examples(&self) -> usize {
        match self {Self::Padded(batch)=>batch.decoder.examples(),Self::Packed(batch)=>batch.decoder.examples()}
    }

    /// Move both source and decoder tensors to the same caller-selected device.
    pub fn to_device(self,device: &B::Device) -> Self {
        match self {
            Self::Padded(batch)=>Self::Padded(PaddedSeq2SeqBatch {encoder:batch.encoder.to_device(device),decoder:batch.decoder.to_device(device)}),
            Self::Packed(batch)=>Self::Packed(PackedSeq2SeqBatch {encoder:batch.encoder.to_device(device),decoder:batch.decoder.to_device(device)}),
        }
    }
}

impl Seq2SeqBatcher {
    /// Validate both actual sequence axes and all explicit policies before tensor upload.
    pub fn validate(&self,samples: &[Seq2SeqExample]) -> Result<(),Seq2SeqDataError> {
        if samples.is_empty() || self.maximum_encoder_length == Some(0) || self.maximum_decoder_length == Some(0) {
            return Err(invalid("paired batch needs actual examples and positive optional sequence limits"));
        }
        if matches!(self.layout,Seq2SeqBatchLayout::Padded {encoder_pad_token_id,decoder_pad_token_id,..}
            if encoder_pad_token_id < 0 || decoder_pad_token_id < 0) {
            return Err(invalid("supply nonnegative actual source/decoder padding IDs"));
        }
        let mut encoder_total = 0usize;
        let mut decoder_total = 0usize;
        for (index,sample) in samples.iter().enumerate() {
            let encoder_length = sample.encoder_input_ids.len();
            let decoder_length = sample.decoder_input_ids.len();
            if decoder_length == 0 || decoder_length != sample.labels.len()
                || sample.encoder_input_ids.iter().chain(&sample.decoder_input_ids).any(|&token|token < 0)
                || sample.labels.iter().any(|&label|label < 0 && label != self.ignore_index) {
                return Err(invalid(format!("paired example {index} has invalid token IDs or aligned decoder labels")));
            }
            if self.maximum_encoder_length.is_some_and(|limit|encoder_length > limit)
                || self.maximum_decoder_length.is_some_and(|limit|decoder_length > limit)
                || i64::try_from(encoder_length).is_err() || i64::try_from(decoder_length).is_err() {
                return Err(invalid(format!("paired example {index} exceeds its explicit sequence/integer limit")));
            }
            encoder_total = encoder_total.checked_add(encoder_length).ok_or_else(||invalid("packed source token count overflow"))?;
            decoder_total = decoder_total.checked_add(decoder_length).ok_or_else(||invalid("packed decoder token count overflow"))?;
        }
        if matches!(self.layout,Seq2SeqBatchLayout::Padded {..}) {
            let encoder_max = samples.iter().map(|sample|sample.encoder_input_ids.len()).max().unwrap();
            let decoder_max = samples.iter().map(|sample|sample.decoder_input_ids.len()).max().unwrap();
            samples.len().checked_mul(encoder_max).ok_or_else(||invalid("padded source token count overflow"))?;
            samples.len().checked_mul(decoder_max).ok_or_else(||invalid("padded decoder token count overflow"))?;
        }
        Ok(())
    }

    /// Collate exact supplied pairs; reuse native aligned decoder label/count handling.
    /// No implicit teacher-forcing shift, BOS/EOS insertion, truncation or prompt masking.
    pub fn collate<B: Backend>(&self,samples: Vec<Seq2SeqExample>,device: &B::Device) -> Result<Seq2SeqBatch<B>,Seq2SeqDataError> {
        self.validate(&samples)?;
        let (encoder,decoder): (Vec<_>,Vec<_>) = samples.into_iter().map(|sample|
            (sample.encoder_input_ids,CausalExample {input_ids:sample.decoder_input_ids,labels:sample.labels})).unzip();
        let decoder_layout = match self.layout {
            Seq2SeqBatchLayout::Packed=>CausalBatchLayout::Packed,
            Seq2SeqBatchLayout::Padded {decoder_pad_token_id,decoder_padding,..}=>
                CausalBatchLayout::Padded {pad_token_id:decoder_pad_token_id,padding:decoder_padding},
        };
        let decoder = CausalBatcher {layout:decoder_layout,ignore_index:self.ignore_index,
            target_alignment:CausalTargetAlignment::SamePosition,maximum_sequence_length:self.maximum_decoder_length}
            .collate(decoder,device).map_err(|error|invalid(error.0))?;
        match decoder {
            CausalBatch::Packed(decoder)=>{
                let total: usize = encoder.iter().map(Vec::len).sum();
                let mut ids = Vec::with_capacity(total);
                let mut positions = Vec::with_capacity(total);
                let mut boundaries = Vec::with_capacity(encoder.len()+1);
                boundaries.push(0);
                for sequence in encoder {
                    positions.extend((0..sequence.len()).map(|position|position as i64));
                    ids.extend(sequence);
                    boundaries.push(ids.len());
                }
                Ok(Seq2SeqBatch::Packed(PackedSeq2SeqBatch {decoder,encoder:PackedTokenBatch {
                    input_ids:Tensor::from_data(TensorData::new(ids,[total]),(device,DType::I64)),
                    position_ids:Tensor::from_data(TensorData::new(positions,[total]),(device,DType::I64)),
                    attention_mask:Tensor::from_data(TensorData::new(vec![true;total],[total]),device),boundaries,
                }}))
            },
            CausalBatch::Padded(decoder)=>{
                let Seq2SeqBatchLayout::Padded {encoder_pad_token_id,encoder_padding,..} = self.layout
                    else { return Err(invalid("actual decoder layout differs from paired collation policy")); };
                let lengths: Vec<_> = encoder.iter().map(Vec::len).collect();
                let maximum = *lengths.iter().max().unwrap();
                let slots = encoder.len().checked_mul(maximum).ok_or_else(||invalid("source padded token count overflow"))?;
                let shape = [encoder.len(),maximum];
                let mut ids = vec![encoder_pad_token_id;slots];
                let mut positions = vec![0i64;slots];
                let mut visible = vec![false;slots];
                for (row,sequence) in encoder.into_iter().enumerate() {
                    let start = match encoder_padding {CausalPadding::Right=>0,CausalPadding::Left=>maximum-sequence.len()};
                    for (position,token) in sequence.into_iter().enumerate() {
                        let cell = row*maximum+start+position;
                        ids[cell] = token;
                        positions[cell] = position as i64;
                        visible[cell] = true;
                    }
                }
                Ok(Seq2SeqBatch::Padded(PaddedSeq2SeqBatch {decoder,encoder:PaddedTokenBatch {
                    input_ids:Tensor::from_data(TensorData::new(ids,shape),(device,DType::I64)),
                    position_ids:Tensor::from_data(TensorData::new(positions,shape),(device,DType::I64)),
                    attention_mask:Tensor::from_data(TensorData::new(visible,shape),device),lengths,
                }}))
            },
        }
    }
}

#[cfg(feature = "dataset")]
impl<B: Backend> super::dataloader::batcher::Batcher<B,Seq2SeqExample,Result<Seq2SeqBatch<B>,Seq2SeqDataError>> for Seq2SeqBatcher {
    fn batch(&self,items: Vec<Seq2SeqExample>,device: &B::Device) -> Result<Seq2SeqBatch<B>,Seq2SeqDataError> {
        self.collate(items,device)
    }
}
