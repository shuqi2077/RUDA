use ruda_model::tensor::{Tensor,backend::Backend};
use crate::attention::{PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask};
use super::{DenseCrossAttentionBlock,DenseEncoderDecoderLayer,DenseEncoderDecoderStack,
    AdaptedCrossAttentionBlock,DecoderCrossAttention,AdaptedEncoderDecoderLayer,AdaptedEncoderDecoderStack};
use super::dense::residual_branch;

fn check_documents<B: Backend>(input: &Tensor<B,2>,memory: &Tensor<B,2>,query_layout: &PackedSequenceLayout,
    memory_layout: &PackedSequenceLayout) {
    assert_eq!(input.dims()[0],query_layout.tokens(),"decoder target boundaries differ from actual token rows");
    assert_eq!(memory.dims()[0],memory_layout.tokens(),"decoder memory boundaries differ from actual token rows");
    assert_eq!(query_layout.documents(),memory_layout.documents(),"decoder target/memory document counts differ");
}

fn check_masks<B: Backend>(query_layout: &PackedSequenceLayout,self_masks: &[PackedDocumentAttentionMask<B>],
    cross_masks: &[PackedDocumentAttentionMask<B>]) {
    assert_eq!(self_masks.len(),query_layout.documents(),"decoder self masks/document count differs");
    assert_eq!(cross_masks.len(),query_layout.documents(),"decoder cross masks/document count differs");
}

impl<B: Backend> DenseCrossAttentionBlock<B> {
    /// Cross-attend actual paired documents with independent query/key validity and score bias.
    pub fn forward_packed_masked_with_positions<F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,masks: &[PackedDocumentAttentionMask<B>],
        options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_documents(&input,&memory,query_layout,memory_layout);
        assert_eq!(masks.len(),query_layout.documents(),"packed cross masks/document count differs");
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed(source,memory.clone(),memory);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"masked packed cross positions changed geometry");
            self.attention.forward_packed_masked_projected(query,key,value,query_layout,memory_layout,masks,options)
        })
    }
}

impl<B: Backend> AdaptedCrossAttentionBlock<B> {
    /// Native A/B training on corresponding actual target/memory documents, without padding.
    pub fn forward_packed(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_with_positions(input,memory,query_layout,memory_layout,options,|query,key|(query,key))
    }

    /// Actual packed query/memory positional transforms with original normalization order.
    pub fn forward_packed_with_positions<F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,options: PackedAttentionOptions,
        positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_documents(&input,&memory,query_layout,memory_layout);
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed(source,memory.clone(),memory);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted packed cross positions changed geometry");
            self.attention.forward_packed_projected(query,key,value,query_layout,memory_layout,options)
        })
    }

    /// Per-document query/key validity, pairwise visibility and trainable cross-attention bias.
    pub fn forward_packed_masked_with_positions<F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,masks: &[PackedDocumentAttentionMask<B>],
        options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_documents(&input,&memory,query_layout,memory_layout);
        assert_eq!(masks.len(),query_layout.documents(),"adapted packed cross masks/document count differs");
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed(source,memory.clone(),memory);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"masked adapted packed cross positions changed geometry");
            self.attention.forward_packed_masked_projected(query,key,value,query_layout,memory_layout,masks,options)
        })
    }
}

impl<B: Backend> DecoderCrossAttention<B> {
    /// Native flat-document attention using the actual original/adapted memory stage.
    pub fn forward_packed_with_positions<F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,options: PackedAttentionOptions,
        positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        match self {
            Self::Dense(block)=>block.forward_packed_with_positions(input,memory,query_layout,memory_layout,options,positions),
            Self::Adapted(block)=>block.forward_packed_with_positions(input,memory,query_layout,memory_layout,options,positions),
        }
    }

    /// Explicit per-document masks/bias on either actual dense/adapted memory stage.
    pub fn forward_packed_masked_with_positions<F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,masks: &[PackedDocumentAttentionMask<B>],
        options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        match self {
            Self::Dense(block)=>block.forward_packed_masked_with_positions(input,memory,query_layout,memory_layout,masks,options,positions),
            Self::Adapted(block)=>block.forward_packed_masked_with_positions(input,memory,query_layout,memory_layout,masks,options,positions),
        }
    }
}

impl<B: Backend> DenseEncoderDecoderLayer<B> {
    /// Actual paired-document self/cross masks and score biases, without positional inference.
    pub fn forward_packed_masked(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_masks: &[PackedDocumentAttentionMask<B>],self_options: PackedAttentionOptions,
        cross_masks: &[PackedDocumentAttentionMask<B>],cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_masked_with_positions(input,memory,query_layout,memory_layout,self_masks,self_options,cross_masks,cross_options,
            |query,key|(query,key),|query,key|(query,key))
    }

    /// Distinct packed self/cross positional transforms and exact original three-stage order.
    pub fn forward_packed_masked_with_positions<F,G>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,self_masks: &[PackedDocumentAttentionMask<B>],
        self_options: PackedAttentionOptions,cross_masks: &[PackedDocumentAttentionMask<B>],cross_options: PackedAttentionOptions,
        self_positions: F,cross_positions: G) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),
        G: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_documents(&input,&memory,query_layout,memory_layout);
        check_masks(query_layout,self_masks,cross_masks);
        let hidden = self.backbone.forward_packed_attention_masked(input,query_layout,self_masks,self_options,self_positions);
        let hidden = self.cross_attention.forward_packed_masked_with_positions(hidden,memory,query_layout,memory_layout,
            cross_masks,cross_options,cross_positions);
        self.backbone.forward_packed_feed_forward(hidden)
    }
}

impl<B: Backend> AdaptedEncoderDecoderLayer<B> {
    /// Native adapter training with paired packed targets and encoder-memory documents.
    pub fn forward_packed(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_options: PackedAttentionOptions,cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_with_positions(input,memory,query_layout,memory_layout,self_options,cross_options,
            |query,key|(query,key),|query,key|(query,key))
    }

    /// Explicit native packed self-attention, memory attention, then final adapted/dense FFN.
    pub fn forward_packed_with_positions<F,G>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,self_options: PackedAttentionOptions,
        cross_options: PackedAttentionOptions,self_positions: F,cross_positions: G) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),
        G: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_documents(&input,&memory,query_layout,memory_layout);
        let hidden = self.backbone.forward_packed_attention(input,query_layout,self_options,self_positions);
        let hidden = self.cross_attention.forward_packed_with_positions(hidden,memory,query_layout,memory_layout,cross_options,cross_positions);
        self.backbone.forward_packed_feed_forward(hidden)
    }

    /// Per-document native self/cross visibility and bias with selected A/B gradients.
    pub fn forward_packed_masked(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_masks: &[PackedDocumentAttentionMask<B>],self_options: PackedAttentionOptions,
        cross_masks: &[PackedDocumentAttentionMask<B>],cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_masked_with_positions(input,memory,query_layout,memory_layout,self_masks,self_options,cross_masks,cross_options,
            |query,key|(query,key),|query,key|(query,key))
    }

    /// Caller-owned packed self/cross positional transforms, masks and independent score biases.
    pub fn forward_packed_masked_with_positions<F,G>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,
        query_layout: &PackedSequenceLayout,memory_layout: &PackedSequenceLayout,self_masks: &[PackedDocumentAttentionMask<B>],
        self_options: PackedAttentionOptions,cross_masks: &[PackedDocumentAttentionMask<B>],cross_options: PackedAttentionOptions,
        self_positions: F,cross_positions: G) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),
        G: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_documents(&input,&memory,query_layout,memory_layout);
        check_masks(query_layout,self_masks,cross_masks);
        let hidden = self.backbone.forward_packed_attention_masked(input,query_layout,self_masks,self_options,self_positions);
        let hidden = self.cross_attention.forward_packed_masked_with_positions(hidden,memory,query_layout,memory_layout,
            cross_masks,cross_options,cross_positions);
        self.backbone.forward_packed_feed_forward(hidden)
    }
}

impl<B: Backend> DenseEncoderDecoderStack<B> {
    /// Shared explicit per-document self/cross masks/bias across every actual decoder layer.
    pub fn forward_packed_masked(&self,mut input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_masks: &[PackedDocumentAttentionMask<B>],self_options: PackedAttentionOptions,
        cross_masks: &[PackedDocumentAttentionMask<B>],cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        check_documents(&input,&memory,query_layout,memory_layout);
        check_masks(query_layout,self_masks,cross_masks);
        for layer in &self.layers {
            input = layer.forward_packed_masked(input,memory.clone(),query_layout,memory_layout,self_masks,self_options,cross_masks,cross_options);
        }
        input
    }

    /// Per-layer packed positions/masks/bias supplied against actual target/memory boundaries.
    pub fn forward_packed_with<F>(&self,mut input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,mut layer: F) -> Tensor<B,2>
    where F: FnMut(usize,&DenseEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>,&PackedSequenceLayout,&PackedSequenceLayout)->Tensor<B,2> {
        check_documents(&input,&memory,query_layout,memory_layout);
        for (index,block) in self.layers.iter().enumerate() { input = layer(index,block,input,memory.clone(),query_layout,memory_layout); }
        input
    }
}

impl<B: Backend> AdaptedEncoderDecoderStack<B> {
    /// Native fine-tuning on independent paired documents without a global packed attention mask.
    pub fn forward_packed(&self,mut input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_options: PackedAttentionOptions,cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        check_documents(&input,&memory,query_layout,memory_layout);
        for layer in &self.layers {
            input = layer.forward_packed(input,memory.clone(),query_layout,memory_layout,self_options,cross_options);
        }
        input
    }

    /// Shared actual packed self/cross visibility and bias for dense/adapted layers.
    pub fn forward_packed_masked(&self,mut input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_masks: &[PackedDocumentAttentionMask<B>],self_options: PackedAttentionOptions,
        cross_masks: &[PackedDocumentAttentionMask<B>],cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        check_documents(&input,&memory,query_layout,memory_layout);
        check_masks(query_layout,self_masks,cross_masks);
        for layer in &self.layers {
            input = layer.forward_packed_masked(input,memory.clone(),query_layout,memory_layout,self_masks,self_options,cross_masks,cross_options);
        }
        input
    }

    /// Actual per-layer packed positions, score bias and options with original adapter IDs.
    pub fn forward_packed_with<F>(&self,mut input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,mut layer: F) -> Tensor<B,2>
    where F: FnMut(usize,&AdaptedEncoderDecoderLayer<B>,Tensor<B,2>,Tensor<B,2>,&PackedSequenceLayout,&PackedSequenceLayout)->Tensor<B,2> {
        check_documents(&input,&memory,query_layout,memory_layout);
        for (index,block) in self.layers.iter().enumerate() { input = layer(index,block,input,memory.clone(),query_layout,memory_layout); }
        input
    }
}
