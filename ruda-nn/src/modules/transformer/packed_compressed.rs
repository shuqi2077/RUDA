use alloc::vec::Vec;
use ruda_model::tensor::{Bool, DType, Int, Tensor, backend::Backend};
use crate::attention::{PackedSequenceLayout, PackedCompressedAttentionOutput, CompressedAttentionProjection};
use super::{HybridAttentionBackbone, HybridAttentionLanguageModel};

impl<B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionBackbone<B, P> {
    /// Native full backbone over actual packed documents with no synthetic separator,
    /// no inter-document compression/window visibility and no global token-square mask.
    pub fn forward_packed(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>) -> Tensor<B, 2> {
        self.packed(tokens, layout, valid, false, false).output
    }

    pub fn forward_packed_with_aux(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool) -> PackedCompressedAttentionOutput<B> {
        self.packed(tokens, layout, valid, true, indexer_warmup)
    }

    fn packed(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout, valid: Option<Tensor<B, 1, Bool>>,
        auxiliary: bool, warmup: bool) -> PackedCompressedAttentionOutput<B> {
        let count = tokens.dims()[0];
        assert_eq!(layout.tokens(), count, "packed hybrid token count differs from document layout");
        assert!(matches!(tokens.dtype(), DType::I32 | DType::I64), "packed hybrid tokens must use I32/I64");
        let weight = self.embedding.weight.val();
        let device = weight.device();
        let storage = weight.dtype();
        let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
        let width = weight.dims()[1];
        assert_eq!(tokens.device(), device, "packed hybrid token device differs");
        if let Some(valid) = &valid {
            assert_eq!(valid.dims(), [count], "packed hybrid validity count differs");
            assert_eq!(valid.device(), device, "packed hybrid validity device differs");
        }
        let mut outputs = Vec::new();
        let mut losses = Vec::with_capacity(layout.documents());
        for bounds in layout.boundaries().windows(2) {
            let (start, end) = (bounds[0], bounds[1]);
            let length = end - start;
            if length == 0 {
                losses.push(Tensor::<B, 1>::zeros([1], (&device, compute)));
                continue;
            }
            let document = tokens.clone().slice_dim(0, start..end).reshape([1, length]);
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
            Tensor::<B, 2>::zeros([count, width], (&device, storage))
        } else { Tensor::cat(outputs, 0) };
        let document_indexer_losses = if losses.is_empty() { Tensor::zeros([0], (&device, compute)) }
            else { Tensor::cat(losses, 0) };
        PackedCompressedAttentionOutput { output, document_indexer_losses }
    }
}

impl<B: Backend, P: CompressedAttentionProjection<B>> HybridAttentionLanguageModel<B, P> {
    pub fn forward_packed(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>) -> Tensor<B, 2> {
        self.project_tokens(self.backbone.forward_packed(tokens, layout, valid))
    }

    pub fn forward_packed_with_aux(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout,
        valid: Option<Tensor<B, 1, Bool>>, indexer_warmup: bool) -> PackedCompressedAttentionOutput<B> {
        let result = self.backbone.forward_packed_with_aux(tokens, layout, valid, indexer_warmup);
        PackedCompressedAttentionOutput { output: self.project_tokens(result.output), document_indexer_losses: result.document_indexer_losses }
    }

    /// Project only actual selected hidden rows against the complete original vocabulary.
    /// Chunking is controlled by the caller or the existing causal-loss implementation.
    pub fn project_tokens(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        let [tokens, width] = hidden.dims();
        let vocab = self.backbone.embedding.weight.val().dims()[0];
        if tokens == 0 {
            return Tensor::<B, 2>::zeros([0, vocab], (&hidden.device(), hidden.dtype())) + hidden.sum().mul_scalar(0).reshape([1, 1]);
        }
        self.head.forward(hidden.reshape([1, tokens, width]), &self.backbone.embedding).reshape([tokens, vocab])
    }
}
