use alloc::vec::Vec;
use ruda_model::tensor::{Bool, DType, FloatDType, Tensor, backend::Backend};
use crate::attention::{PackedSequenceLayout, PackedCompressedAttentionOutput, CompressedAttentionProjection};
use super::{MhcResidualBlock, MhcResidualStack, MhcResidualBranch, MhcResidualBranchShape};
use super::compressed::{normalized, visible};

/// Packed residual streams and independent per-document indexer objectives.
#[derive(Debug)]
pub struct PackedMhcResidualBlockOutput<B: Backend> {
    pub state: Tensor<B, 4>,
    pub document_indexer_losses: Tensor<B, 1>,
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualBlock<B, P, F> {
    pub fn try_forward_packed_with<R, G>(&self, state: Tensor<B, 4>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, branch: G) -> Result<Tensor<B, 4>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.packed_with(state, layout, valid, false, false, branch).map(|result| result.state)
    }

    pub fn try_forward_packed_with_aux<R, G>(&self, state: Tensor<B, 4>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool, branch: G) -> Result<PackedMhcResidualBlockOutput<B>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.packed_with(state, layout, valid, true, indexer_warmup, branch)
    }

    fn packed_with<R, G>(&self, state: Tensor<B, 4>, layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>,
        auxiliary: bool, warmup: bool, branch: G) -> Result<PackedMhcResidualBlockOutput<B>, R>
    where G: FnOnce(&F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let tokens = layout.tokens();
        let width = self.attention.width;
        assert_eq!(state.dims(), [1, tokens, self.attention_connection.streams, width], "mHC packed stream geometry differs");
        if let Some(valid) = &valid {
            assert_eq!(valid.dims(), [tokens], "mHC packed visibility count differs");
            assert_eq!(valid.device(), state.device(), "mHC packed visibility device differs");
        }
        let compute = if state.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
        if tokens == 0 {
            // No fabricated source rows: a zero-token rank still receives and executes actual remote expert assignments.
            let empty = state.clone().sum_dim(2).squeeze_dim(2);
            let update = branch(&self.feed_forward, normalized(empty, &self.ffn_norm, self.epsilon))?;
            assert_eq!(update.dims(), [1, 0, width], "mHC empty expert branch produced source rows");
            let losses = Tensor::zeros([layout.documents()], (&state.device(), compute));
            let state = state + update.unsqueeze_dim(2);
            return Ok(PackedMhcResidualBlockOutput { state, document_indexer_losses: losses });
        }
        let (query, mappings) = self.attention_connection.pre(state.clone());
        let query = normalized(query, &self.attention_norm, self.epsilon);
        let valid_rows = visible(&query, valid.clone().map(|value| value.reshape([1, tokens])));
        let query = query.reshape([tokens, width]);
        let result = if auxiliary { self.attention.forward_packed_with_aux(query, layout, valid, warmup) }
        else { PackedCompressedAttentionOutput { output: self.attention.forward_packed(query, layout, valid),
            document_indexer_losses: Tensor::zeros([layout.documents()], (&state.device(), compute)) } };
        // One FFN invocation for all local documents; expert worlds may have different document counts.
        let state = self.finish_with(state, result.output.reshape([1, tokens, width]), mappings, valid_rows, branch)?;
        Ok(PackedMhcResidualBlockOutput { state, document_indexer_losses: result.document_indexer_losses })
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranchShape<B>> MhcResidualStack<B, P, F> {
    pub fn try_forward_packed_with<R, G>(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, branch: G) -> Result<Tensor<B, 2>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.packed_with(input, layout, valid, false, false, branch).map(|result| result.output)
    }

    pub fn try_forward_packed_with_aux<R, G>(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool, branch: G) -> Result<PackedCompressedAttentionOutput<B>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        self.packed_with(input, layout, valid, true, indexer_warmup, branch)
    }

    fn packed_with<R, G>(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>,
        auxiliary: bool, warmup: bool, mut branch: G) -> Result<PackedCompressedAttentionOutput<B>, R>
    where G: FnMut(usize, &F, Tensor<B, 3>) -> Result<Tensor<B, 3>, R> {
        let [tokens, width] = input.dims();
        assert_eq!((layout.tokens(), width), (tokens, self.layers[0].attention.width), "mHC packed payload/layout geometry differs");
        let compute = if input.dtype() == DType::F64 { DType::F64 } else { DType::F32 };
        let mut losses = Tensor::zeros([layout.documents()], (&input.device(), compute));
        let input = input.reshape([1, tokens, width]);
        let visibility = visible(&input, valid.map(|value| value.reshape([1, tokens])));
        let dtype = input.dtype();
        let input = input * visibility.clone().cast::<FloatDType>(dtype.into()).reshape([1, tokens, 1]);
        let mut state = self.layers[0].attention_connection.expand(input);
        for (index, layer) in self.layers.iter().enumerate() {
            let mask = Some(visibility.clone().reshape([tokens]));
            if auxiliary {
                let result = layer.try_forward_packed_with_aux(state, layout, mask, warmup, |feed, input| branch(index, feed, input))?;
                state = result.state;
                losses = losses + result.document_indexer_losses;
            } else {
                state = layer.try_forward_packed_with(state, layout, mask, |feed, input| branch(index, feed, input))?;
            }
        }
        let output = if tokens == 0 { state.sum_dim(2).squeeze_dim(2) }
        else { self.layers.last().unwrap().ffn_connection.reduce(state) };
        let output = normalized(output, &self.final_norm, self.epsilon).reshape([tokens, width]);
        Ok(PackedCompressedAttentionOutput { output, document_indexer_losses: losses })
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>, F: MhcResidualBranch<B>> MhcResidualStack<B, P, F> {
    pub fn forward_packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>)
        -> Result<Tensor<B, 2>, F::Error> {
        self.packed(input, layout, valid, false, false).map(|result| result.output)
    }

    pub fn forward_packed_with_aux(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool) -> Result<PackedCompressedAttentionOutput<B>, F::Error> {
        self.packed(input, layout, valid, true, indexer_warmup)
    }

    fn packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>,
        auxiliary: bool, warmup: bool) -> Result<PackedCompressedAttentionOutput<B>, F::Error> {
        let [tokens, width] = input.dims();
        assert_eq!((layout.tokens(), width), (tokens, self.layers[0].attention.width), "mHC packed payload/layout geometry differs");
        let device = input.device();
        let storage = input.dtype();
        assert!(matches!(storage, DType::F16 | DType::BF16 | DType::F32 | DType::F64), "mHC packed activations must be floating");
        let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
        if let Some(valid) = &valid {
            assert_eq!(valid.dims(), [tokens], "mHC packed visibility count differs");
            assert_eq!(valid.device(), device, "mHC packed visibility device differs");
        }
        let mut outputs = Vec::new();
        let mut losses = Vec::with_capacity(layout.documents());
        for bounds in layout.boundaries().windows(2) {
            let (start, end) = (bounds[0], bounds[1]);
            let length = end - start;
            if length == 0 { losses.push(Tensor::<B, 1>::zeros([1], (&device, compute))); continue; }
            let document = input.clone().slice_dim(0, start..end).reshape([1, length, width]);
            let valid = valid.as_ref().map(|valid| valid.clone().slice_dim(0, start..end).reshape([1, length]));
            if auxiliary {
                let result = self.forward_with_aux(document, valid, warmup)?;
                outputs.push(result.output.reshape([length, width]));
                losses.push(result.indexer_loss);
            } else {
                outputs.push(self.forward(document, valid)?.reshape([length, width]));
                losses.push(Tensor::<B, 1>::zeros([1], (&device, compute)));
            }
        }
        let output = if outputs.is_empty() {
            Tensor::<B, 2>::zeros([tokens, width], (&device, storage)) + input.sum().mul_scalar(0).reshape([1, 1])
        } else { Tensor::cat(outputs, 0) };
        let document_indexer_losses = if losses.is_empty() { Tensor::zeros([0], (&device, compute)) } else { Tensor::cat(losses, 0) };
        Ok(PackedCompressedAttentionOutput { output, document_indexer_losses })
    }
}
