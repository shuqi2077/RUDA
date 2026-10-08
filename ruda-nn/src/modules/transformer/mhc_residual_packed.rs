use alloc::vec::Vec;
use ruda_model::tensor::{Bool, DType, Tensor, backend::Backend};
use crate::attention::{PackedSequenceLayout, PackedCompressedAttentionOutput, CompressedAttentionProjection};
use super::{MhcResidualStack, MhcResidualBranch};

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
