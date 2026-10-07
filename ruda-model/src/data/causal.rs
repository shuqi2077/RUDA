//! Causal data collation independent of architecture and tokenizer selection.
use crate::tensor::{Bool, DType, Int, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt};

/// Invalid actual input/label geometry or explicit batch configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalDataError(pub String);

impl fmt::Display for CausalDataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(&self.0) }
}
impl Error for CausalDataError {}
fn invalid(message: impl Into<String>) -> CausalDataError { CausalDataError(message.into()) }

/// One already-encoded document with caller-declared supervision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalExample {
    /// Nonnegative actual tokenizer IDs, without batch-padding tokens.
    pub input_ids: Vec<i64>,
    /// Explicit token targets or the collator's selected ignore_index.
    pub labels: Vec<i64>,
}

impl CausalExample {
    /// Validate a nonempty encoded document without guessing prompt spans.
    pub fn validate(&self, ignore_index: i64) -> Result<(), CausalDataError> {
        if self.input_ids.is_empty() || self.input_ids.len() != self.labels.len() {
            return Err(invalid("causal input_ids and labels must be equally sized nonempty sequences"));
        }
        if self.input_ids.iter().any(|&token| token < 0)
            || self.labels.iter().any(|&label| label < 0 && label != ignore_index) {
            return Err(invalid("causal token IDs must be nonnegative and negative labels must equal ignore_index"));
        }
        Ok(())
    }

    /// Actual encoded sequence length, excluding any future batch padding.
    pub fn length(&self) -> usize { self.input_ids.len() }
}

/// Direction of rectangular batch padding, not a truncation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalPadding {
    /// Place all real tokens before the padded suffix.
    Right,
    /// Place all real tokens after the padded prefix.
    Left,
}

/// Device batch layout explicitly selected by the caller's model contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalBatchLayout {
    /// Concatenate documents, retaining cumulative boundaries and reset positions.
    Packed,
    /// Rectangular token rows with an explicit tokenizer padding ID/direction.
    Padded {
        /// Actual tokenizer's pad ID; never inferred from EOS or vocabulary size.
        pad_token_id: i64,
        /// Which side of each real document receives batch padding.
        padding: CausalPadding,
    },
}

/// Alignment used by the downstream causal loss.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalTargetAlignment {
    /// Hidden position t predicts label t+1; document-start targets are ignored.
    NextToken,
    /// Hidden position t predicts label t; supplied real labels remain unchanged.
    SamePosition,
}

/// Explicit supervision/layout policy shared by every call of a data batcher.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalBatcher {
    /// Actual device token-axis geometry.
    pub layout: CausalBatchLayout,
    /// Label sentinel, distinct from attention visibility and padding token ID.
    pub ignore_index: i64,
    /// Must match the downstream loss's causal-shift setting.
    pub target_alignment: CausalTargetAlignment,
    /// Optional maximum actual sequence length; excess data is rejected, not cut.
    pub maximum_sequence_length: Option<usize>,
}

/// Flat actual documents, ready for an explicit packed-attention model.
#[derive(Clone, Debug)]
pub struct PackedCausalBatch<B: Backend> {
    /// Actual flat token IDs, without separator or filler tokens.
    pub input_ids: Tensor<B, 1, Int>,
    /// Supervised labels with ignored document starts for next-token training.
    pub labels: Tensor<B, 1, Int>,
    /// True for every real token; no physical padding in this representation.
    pub attention_mask: Tensor<B, 1, Bool>,
    /// Per-document positions reset to zero.
    pub position_ids: Tensor<B, 1, Int>,
    /// Cumulative physical token boundaries, beginning with zero.
    pub boundaries: Vec<usize>,
    /// Maximum actual document length.
    pub maximum_sequence_length: usize,
    /// Actual nonignored supervised targets under the selected alignment.
    pub supervised_tokens: usize,
    /// Actual label sentinel used during collation.
    pub ignore_index: i64,
    /// Target alignment recorded for the downstream loss contract.
    pub target_alignment: CausalTargetAlignment,
}

impl<B: Backend> PackedCausalBatch<B> {
    /// Actual document count, not token count or a padded batch dimension.
    pub fn examples(&self) -> usize { self.boundaries.len() - 1 }

    /// Move the actual tensors to the caller-selected backend device.
    pub fn to_device(mut self, device: &B::Device) -> Self {
        self.input_ids = self.input_ids.to_device(device);
        self.labels = self.labels.to_device(device);
        self.attention_mask = self.attention_mask.to_device(device);
        self.position_ids = self.position_ids.to_device(device);
        self
    }
}

/// Rectangular actual documents with explicitly masked batch-padding cells.
#[derive(Clone, Debug)]
pub struct PaddedCausalBatch<B: Backend> {
    /// Token IDs shaped [actual examples, maximum actual sequence length].
    pub input_ids: Tensor<B, 2, Int>,
    /// Same shape as input_ids; all padding cells contain ignore_index.
    pub labels: Tensor<B, 2, Int>,
    /// True exactly at real source tokens, never at padding cells.
    pub attention_mask: Tensor<B, 2, Bool>,
    /// Real positions reset per document; masked padding positions are zero.
    pub position_ids: Tensor<B, 2, Int>,
    /// Original encoded lengths in actual batch order.
    pub lengths: Vec<usize>,
    /// Actual nonignored supervised targets under the selected alignment.
    pub supervised_tokens: usize,
    /// Actual label sentinel used during collation.
    pub ignore_index: i64,
    /// Target alignment recorded for the downstream loss contract.
    pub target_alignment: CausalTargetAlignment,
}

impl<B: Backend> PaddedCausalBatch<B> {
    /// Actual source example count; no synthetic rows are created.
    pub fn examples(&self) -> usize { self.lengths.len() }

    /// Move the actual tensors while retaining the original length metadata.
    pub fn to_device(mut self, device: &B::Device) -> Self {
        self.input_ids = self.input_ids.to_device(device);
        self.labels = self.labels.to_device(device);
        self.attention_mask = self.attention_mask.to_device(device);
        self.position_ids = self.position_ids.to_device(device);
        self
    }
}

/// A device batch carrying its actual geometry, not an inferred model family.
#[derive(Clone, Debug)]
pub enum CausalBatch<B: Backend> {
    /// Flat documents for a boundary-aware packed model.
    Packed(PackedCausalBatch<B>),
    /// Masked rectangular rows for a padding-aware model.
    Padded(PaddedCausalBatch<B>),
}

impl<B: Backend> CausalBatch<B> {
    /// Actual supervised targets used for weighted microbatch accumulation.
    pub fn supervised_tokens(&self) -> usize {
        match self { Self::Packed(batch) => batch.supervised_tokens, Self::Padded(batch) => batch.supervised_tokens }
    }

    /// Actual encoded example count for committing the sample cursor.
    pub fn examples(&self) -> usize {
        match self { Self::Packed(batch) => batch.examples(), Self::Padded(batch) => batch.examples() }
    }

    /// Move a packed or rectangular batch without altering its layout.
    pub fn to_device(self, device: &B::Device) -> Self {
        match self { Self::Packed(batch) => Self::Packed(batch.to_device(device)), Self::Padded(batch) => Self::Padded(batch.to_device(device)) }
    }
}

impl CausalBatcher {
    /// Validate explicit policy and all actual source examples before uploading.
    pub fn validate(&self, samples: &[CausalExample]) -> Result<(), CausalDataError> {
        if samples.is_empty() || self.maximum_sequence_length == Some(0) {
            return Err(invalid("causal batch must contain actual examples and a positive optional sequence limit"));
        }
        if matches!(self.layout, CausalBatchLayout::Padded { pad_token_id, .. } if pad_token_id < 0) {
            return Err(invalid("supply a nonnegative actual padding token ID"));
        }
        for (index, sample) in samples.iter().enumerate() {
            sample.validate(self.ignore_index).map_err(|error| invalid(format!("causal example {index}: {error}")))?;
            if self.maximum_sequence_length.is_some_and(|limit| sample.length() > limit)
                || i64::try_from(sample.length()).is_err() {
                return Err(invalid(format!("causal example {index} exceeds the explicit sequence/integer limit")));
            }
        }
        Ok(())
    }

    /// Collate explicit labels/IDs on the selected device, without tokenization.
    ///
    /// Next-token mode ignores each document's first target even for left
    /// padding, so a padded hidden state cannot supervise the first real token.
    /// Packed mode never inserts EOS, changes labels within a document, or
    /// authorizes a model to attend across the retained boundaries.
    pub fn collate<B: Backend>(&self, samples: Vec<CausalExample>, device: &B::Device) -> Result<CausalBatch<B>, CausalDataError> {
        self.validate(&samples)?;
        let lengths: Vec<_> = samples.iter().map(CausalExample::length).collect();
        let maximum = *lengths.iter().max().unwrap();
        let mut supervised_tokens = 0usize;
        let label_at = |sample: &CausalExample, position: usize| {
            if position == 0 && self.target_alignment == CausalTargetAlignment::NextToken { self.ignore_index }
            else { sample.labels[position] }
        };
        match self.layout {
            CausalBatchLayout::Packed => {
                let total = lengths.iter().try_fold(0usize, |total, &length| total.checked_add(length))
                    .ok_or_else(|| invalid("packed causal token count overflow"))?;
                let mut ids = Vec::with_capacity(total);
                let mut labels = Vec::with_capacity(total);
                let mut positions = Vec::with_capacity(total);
                let mut boundaries = Vec::with_capacity(samples.len() + 1);
                boundaries.push(0);
                for sample in samples {
                    for (position, &token) in sample.input_ids.iter().enumerate() {
                        let label = label_at(&sample, position);
                        supervised_tokens += usize::from(label != self.ignore_index);
                        ids.push(token);
                        labels.push(label);
                        positions.push(position as i64);
                    }
                    boundaries.push(ids.len());
                }
                Ok(CausalBatch::Packed(PackedCausalBatch {
                    input_ids: Tensor::from_data(TensorData::new(ids, [total]), (device, DType::I64)),
                    labels: Tensor::from_data(TensorData::new(labels, [total]), (device, DType::I64)),
                    position_ids: Tensor::from_data(TensorData::new(positions, [total]), (device, DType::I64)),
                    attention_mask: Tensor::from_data(TensorData::new(vec![true; total], [total]), device),
                    boundaries, maximum_sequence_length: maximum, supervised_tokens,
                    ignore_index: self.ignore_index, target_alignment: self.target_alignment,
                }))
            }
            CausalBatchLayout::Padded { pad_token_id, padding } => {
                let slots = samples.len().checked_mul(maximum).ok_or_else(|| invalid("padded causal token count overflow"))?;
                let shape = [samples.len(), maximum];
                let mut ids = vec![pad_token_id; slots];
                let mut labels = vec![self.ignore_index; slots];
                let mut positions = vec![0i64; slots];
                let mut mask = vec![false; slots];
                for (row, sample) in samples.iter().enumerate() {
                    let start = match padding { CausalPadding::Right => 0, CausalPadding::Left => maximum - sample.length() };
                    for (position, &token) in sample.input_ids.iter().enumerate() {
                        let cell = row * maximum + start + position;
                        let label = label_at(sample, position);
                        supervised_tokens += usize::from(label != self.ignore_index);
                        ids[cell] = token;
                        labels[cell] = label;
                        positions[cell] = position as i64;
                        mask[cell] = true;
                    }
                }
                Ok(CausalBatch::Padded(PaddedCausalBatch {
                    input_ids: Tensor::from_data(TensorData::new(ids, shape), (device, DType::I64)),
                    labels: Tensor::from_data(TensorData::new(labels, shape), (device, DType::I64)),
                    position_ids: Tensor::from_data(TensorData::new(positions, shape), (device, DType::I64)),
                    attention_mask: Tensor::from_data(TensorData::new(mask, shape), device),
                    lengths, supervised_tokens, ignore_index: self.ignore_index, target_alignment: self.target_alignment,
                }))
            }
        }
    }
}

#[cfg(feature = "dataset")]
impl<B: Backend> super::dataloader::batcher::Batcher<B, CausalExample, Result<CausalBatch<B>, CausalDataError>> for CausalBatcher {
    fn batch(&self, items: Vec<CausalExample>, device: &B::Device) -> Result<CausalBatch<B>, CausalDataError> {
        self.collate(items, device)
    }
}
