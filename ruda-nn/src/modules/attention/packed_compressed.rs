use alloc::vec::Vec;
use ruda_model::tensor::{Bool, Tensor, backend::Backend};
use super::{CompressedAttention, PackedSequenceLayout};
use super::sparse_ops::work_dtype;

/// Actual packed token output and one independent indexer KL value per document.
/// Loss reduction across documents is explicitly left to the training objective.
#[derive(Clone, Debug)]
pub struct PackedCompressedAttentionOutput<B: Backend> {
    pub output: Tensor<B, 2>,
    pub document_indexer_losses: Tensor<B, 1>,
}

impl<B: Backend> CompressedAttention<B> {
    /// Independent document CSA/HCA: RoPE positions, local windows, incomplete
    /// blocks and overlap reset at every actual document boundary.
    pub fn forward_packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>) -> Tensor<B, 2> {
        self.packed(input, layout, valid, false, false).output
    }

    /// Preserve each document's active-query-normalized KL independently,
    /// including disconnected zeros for empty documents and HCA.
    pub fn forward_packed_with_aux(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool) -> PackedCompressedAttentionOutput<B> {
        self.packed(input, layout, valid, true, indexer_warmup)
    }

    fn packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>,
        auxiliary: bool, warmup: bool) -> PackedCompressedAttentionOutput<B> {
        let [tokens, width] = input.dims();
        assert_eq!((layout.tokens(), width), (tokens, self.width), "packed compression layout/input geometry differs");
        let device = input.device();
        let storage = input.dtype();
        let compute = work_dtype(storage);
        assert_eq!(device, self.parts.query_down.weight.val().device(), "packed compression device differs");
        if let Some(valid) = &valid {
            assert_eq!(valid.dims(), [tokens], "packed compression validity count differs");
            assert_eq!(valid.device(), device, "packed compression validity device differs");
        }
        let mut outputs = Vec::new();
        let mut losses = Vec::with_capacity(layout.documents());
        for bounds in layout.boundaries().windows(2) {
            let (start, end) = (bounds[0], bounds[1]);
            if start == end {
                losses.push(Tensor::<B, 1>::zeros([1], (&device, compute)));
                continue;
            }
            let length = end - start;
            let document = input.clone().slice_dim(0, start..end).reshape([1, length, width]);
            let valid = valid.as_ref().map(|valid| valid.clone().slice_dim(0, start..end).reshape([1, length]));
            if auxiliary {
                let result = self.forward_with_aux(document, valid, warmup);
                outputs.push(result.output.reshape([length, width]));
                losses.push(result.indexer_loss);
            } else {
                outputs.push(self.forward(document, valid).reshape([length, width]));
                losses.push(Tensor::<B, 1>::zeros([1], (&device, compute)));
            }
        }
        let output = if outputs.is_empty() {
            Tensor::<B, 2>::zeros([tokens, width], (&device, storage)) + input.sum().mul_scalar(0).reshape([1, 1])
        } else { Tensor::cat(outputs, 0) };
        let document_indexer_losses = if losses.is_empty() { Tensor::zeros([0], (&device, compute)) }
            else { Tensor::cat(losses, 0) };
        PackedCompressedAttentionOutput { output, document_indexer_losses }
    }
}
