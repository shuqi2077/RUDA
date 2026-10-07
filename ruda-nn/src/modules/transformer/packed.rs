use ruda_model::tensor::{Tensor,Int,FloatDType,backend::Backend};
use crate::attention::{PackedSequenceLayout,PackedAttentionOptions};
use super::{DenseTransformerBlock,DenseTransformerStack,AdaptedTransformerBlock,AdaptedTransformerStack,AdaptedStackLayer,
    DenseCrossAttentionBlock,DenseEncoderDecoderLayer,DenseEncoderDecoderStack,TransformerEmbeddings};
use super::dense::residual_branch;

fn check_input<B: Backend>(input: &Tensor<B,2>,layout: &PackedSequenceLayout) {
    assert_eq!(input.dims()[0],layout.tokens(),"packed transformer boundaries differ from actual token rows");
}

impl<B: Backend> DenseTransformerBlock<B> {
    /// Actual packed documents with explicit causal/window rules and native autodiff.
    pub fn forward_packed(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_with_positions(input,layout,options,|query,key|(query,key))
    }

    /// Caller-owned packed Q/K positional transforms, retaining document boundaries.
    pub fn forward_packed_with_positions<F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,
        options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let hidden = self.forward_packed_attention(input,layout,options,positions);
        residual_branch(hidden,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,|source|self.feed_forward.forward(source))
    }

    /// Packed attention stage alone for actual encoder-decoder stage composition.
    pub fn forward_packed_attention<F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,
        options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_input(&input,layout);
        residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"packed query/key positions changed geometry");
            self.attention.forward_packed_projected(query,key,value,layout,layout,options)
        })
    }
}

impl<B: Backend> AdaptedTransformerBlock<B> {
    /// Packed native adapter training with actual independent-document attention.
    pub fn forward_packed(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_with_positions(input,layout,options,|query,key|(query,key))
    }

    /// Apply the caller's actual packed Q/K positions, without a global score mask.
    pub fn forward_packed_with_positions<F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,
        options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_input(&input,layout);
        let hidden = residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed(source.clone(),source.clone(),source);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"adapted packed positions changed geometry");
            self.attention.forward_packed_projected(query,key,value,layout,layout,options)
        });
        residual_branch(hidden,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,|source|self.feed_forward.forward(source))
    }
}

impl<B: Backend> AdaptedStackLayer<B> {
    /// Run either actual dense or actual adapted layer on flat native documents.
    pub fn forward_packed(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        match self {Self::Dense(block)=>block.forward_packed(input,layout,options),Self::Adapted(block)=>block.forward_packed(input,layout,options)}
    }

    /// Apply an explicit packed positional transform to either actual layer variant.
    pub fn forward_packed_with_positions<F>(&self,input: Tensor<B,2>,layout: &PackedSequenceLayout,
        options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        match self {
            Self::Dense(block)=>block.forward_packed_with_positions(input,layout,options,positions),
            Self::Adapted(block)=>block.forward_packed_with_positions(input,layout,options,positions),
        }
    }
}

impl<B: Backend> DenseTransformerStack<B> {
    /// Shared explicit packed attention rules for every actual layer.
    pub fn forward_packed(&self,mut input: Tensor<B,2>,layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        check_input(&input,layout);
        for block in &self.blocks { input = block.forward_packed(input,layout,options); }
        input
    }

    /// Per-layer packed position/window policies without model-family assumptions.
    pub fn forward_packed_with<F>(&self,mut input: Tensor<B,2>,layout: &PackedSequenceLayout,mut layer: F) -> Tensor<B,2>
    where F: FnMut(usize,&DenseTransformerBlock<B>,Tensor<B,2>,&PackedSequenceLayout)->Tensor<B,2> {
        check_input(&input,layout);
        for (index,block) in self.blocks.iter().enumerate() { input = layer(index,block,input,layout); }
        input
    }
}

impl<B: Backend> AdaptedTransformerStack<B> {
    /// Shared packed document boundaries for all selected/unselected native layers.
    pub fn forward_packed(&self,mut input: Tensor<B,2>,layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        check_input(&input,layout);
        for block in &self.layers { input = block.forward_packed(input,layout,options); }
        input
    }

    /// Explicit per-layer packed position/window rules while retaining actual adapter IDs.
    pub fn forward_packed_with<F>(&self,mut input: Tensor<B,2>,layout: &PackedSequenceLayout,mut layer: F) -> Tensor<B,2>
    where F: FnMut(usize,&AdaptedStackLayer<B>,Tensor<B,2>,&PackedSequenceLayout)->Tensor<B,2> {
        check_input(&input,layout);
        for (index,block) in self.layers.iter().enumerate() { input = layer(index,block,input,layout); }
        input
    }
}

impl<B: Backend> DenseCrossAttentionBlock<B> {
    /// Cross-attend paired actual packed target/memory documents, without padding.
    pub fn forward_packed(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_with_positions(input,memory,query_layout,memory_layout,options,|query,key|(query,key))
    }

    /// Actual query/memory positions and independent document lengths remain explicit.
    pub fn forward_packed_with_positions<F>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,options: PackedAttentionOptions,positions: F) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        check_input(&input,query_layout);
        check_input(&memory,memory_layout);
        assert_eq!(query_layout.documents(),memory_layout.documents(),"packed target/memory document counts differ");
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project_packed(source,memory.clone(),memory);
            let geometry = (query.dims(),key.dims());
            let (query,key) = positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"packed cross positions changed geometry");
            self.attention.forward_packed_projected(query,key,value,query_layout,memory_layout,options)
        })
    }
}

impl<B: Backend> DenseEncoderDecoderLayer<B> {
    /// Packed self-attention, paired packed memory attention, then the actual FFN.
    pub fn forward_packed(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_options: PackedAttentionOptions,cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        self.forward_packed_with_positions(input,memory,query_layout,memory_layout,self_options,cross_options,
            |query,key|(query,key),|query,key|(query,key))
    }

    /// Explicit packed self/cross positional transforms with unchanged residual ordering.
    pub fn forward_packed_with_positions<F,G>(&self,input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_options: PackedAttentionOptions,cross_options: PackedAttentionOptions,
        self_positions: F,cross_positions: G) -> Tensor<B,2>
    where F: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),
        G: FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let hidden = self.backbone.forward_packed_attention(input,query_layout,self_options,self_positions);
        let hidden = self.cross_attention.forward_packed_with_positions(hidden,memory,query_layout,memory_layout,cross_options,cross_positions);
        residual_branch(hidden,&self.backbone.feed_forward_norm,&self.backbone.residual_dropout,self.backbone.norm_first,
            |source|self.backbone.feed_forward.forward(source))
    }
}

impl<B: Backend> DenseEncoderDecoderStack<B> {
    /// Shared actual target/memory boundaries; no fabricated separator or alignment rule.
    pub fn forward_packed(&self,mut input: Tensor<B,2>,memory: Tensor<B,2>,query_layout: &PackedSequenceLayout,
        memory_layout: &PackedSequenceLayout,self_options: PackedAttentionOptions,cross_options: PackedAttentionOptions) -> Tensor<B,2> {
        check_input(&input,query_layout);
        check_input(&memory,memory_layout);
        assert_eq!(query_layout.documents(),memory_layout.documents(),"packed target/memory document counts differ");
        for layer in &self.layers { input = layer.forward_packed(input,memory.clone(),query_layout,memory_layout,self_options,cross_options); }
        input
    }
}

impl<B: Backend> TransformerEmbeddings<B> {
    /// Embed actual flat token/position/type IDs with no physical padding or automatic offsets.
    pub fn forward_packed(&self,input_ids: Tensor<B,1,Int>,position_ids: Option<Tensor<B,1,Int>>,
        token_type_ids: Option<Tensor<B,1,Int>>) -> Tensor<B,2> {
        let tokens = input_ids.dims()[0];
        let result = self.forward(input_ids.reshape([1,tokens]),position_ids.map(|ids| {
            assert_eq!(ids.dims(),[tokens],"packed learned positions differ from token geometry"); ids.reshape([1,tokens])
        }),token_type_ids.map(|ids| {
            assert_eq!(ids.dims(),[tokens],"packed token types differ from token geometry"); ids.reshape([1,tokens])
        }));
        let width = result.dims()[2];
        result.reshape([tokens,width])
    }

    /// Explicit mixed table arithmetic/output storage, still casting only looked-up rows.
    pub fn forward_packed_with_compute_dtype(&self,input_ids: Tensor<B,1,Int>,position_ids: Option<Tensor<B,1,Int>>,
        token_type_ids: Option<Tensor<B,1,Int>>,compute: FloatDType,output: FloatDType) -> Tensor<B,2> {
        let tokens = input_ids.dims()[0];
        let result = self.forward_with_compute_dtype(input_ids.reshape([1,tokens]),position_ids.map(|ids| {
            assert_eq!(ids.dims(),[tokens],"packed learned positions differ from token geometry"); ids.reshape([1,tokens])
        }),token_type_ids.map(|ids| {
            assert_eq!(ids.dims(),[tokens],"packed token types differ from token geometry"); ids.reshape([1,tokens])
        }),compute,output);
        let width = result.dims()[2];
        result.reshape([tokens,width])
    }
}
