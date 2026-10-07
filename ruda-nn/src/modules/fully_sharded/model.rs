use super::*;
use super::model_parts::projection_geometry;
use crate::{attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions},
    cache::TransformerKvCache,pool::SequencePooling,transformer::SequenceHeadOutput};
use ruda_model::tensor::Bool;

/// Complete actual input -> original ordered backbone -> optional final norm -> original head.
/// Only local parameter shards persist; full gathered values follow the selected AD checkpoint strategy.
#[derive(Module,Debug)]
pub struct FullyShardedTransformerModel<B:Backend> {
    /// Actual token/learned-position/type inputs.
    pub embeddings:FullyShardedTransformerEmbeddings<B>,
    /// Original complete dense/adapted backbone and real layer order.
    pub backbone:FullyShardedTransformerStack<B>,
    /// Explicit architecture-final norm, independent of input and head normalization.
    pub normalization:Option<FullyShardedTransformerNorm<B>>,
    /// Actual original dense/LoRA/explicitly tied output choice.
    pub head:FullyShardedTransformerHead<B>,
}

pub(super) fn validate_block_width<B:Backend>(block:&FullyShardedTransformerBlock<B>,width:usize) {
    assert_eq!((block.attention_norm.width(),block.feed_forward_norm.width()),(width,width),"sharded model block norm widths differ");
    let attention=&block.attention;
    assert_eq!(projection_geometry(&attention.query).0,width,"sharded model query input width differs");
    assert_eq!(projection_geometry(&attention.key).0,width,"sharded model key input width differs");
    assert_eq!(projection_geometry(&attention.value).0,width,"sharded model value input width differs");
    assert_eq!(projection_geometry(&attention.output).1,width,"sharded model attention output width differs");
    assert_eq!(projection_geometry(&block.feed_forward.up).0,width,"sharded model FFN input width differs");
    if let Some(gate)=&block.feed_forward.gate {assert_eq!(projection_geometry(gate).0,width,"sharded model gate input width differs");}
    assert_eq!(projection_geometry(&block.feed_forward.down).1,width,"sharded model FFN output width differs");
}

pub(super) fn packed_input<B:Backend>(input:FullyShardedTransformerInput<B,1>,layout:&PackedSequenceLayout)
    -> FullyShardedTransformerInput<B> {
    let count=input.tokens.dims()[0];assert_eq!(count,layout.tokens(),"sharded packed input/document lengths differ");
    let convert=|ids:Tensor<B,1,Int>| {assert_eq!(ids.dims(),[count],"sharded packed table metadata length differs");ids.reshape([1,count])};
    FullyShardedTransformerInput {tokens:input.tokens.reshape([1,count]),positions:input.positions.map(convert),
        token_types:input.token_types.map(convert),embedding_dtypes:input.embedding_dtypes}
}

impl<B:Backend> FullyShardedTransformerModel<B> {
    /// Assemble actual already-sharded components; reuse one ShardingContext while constructing shared roles.
    /// Masks, positional policies, vocabulary, trainability and precision are not selected by this constructor.
    pub fn from_parts(embeddings:FullyShardedTransformerEmbeddings<B>,backbone:FullyShardedTransformerStack<B>,
        normalization:Option<FullyShardedTransformerNorm<B>>,head:FullyShardedTransformerHead<B>) -> Self {
        let width=embeddings.hidden_width();assert_eq!(width,head.hidden_width(),"sharded model input/head widths differ");
        if let Some(norm)=&normalization {assert_eq!(norm.width(),width,"sharded model final norm width differs");}
        for block in &backbone.blocks {validate_block_width(block,width);}
        Self {embeddings,backbone,normalization,head}
    }
    /// Original cache for precisely the actual backbone layer count.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {self.backbone.new_kv_cache(initial_capacity)}
}

macro_rules! model_execution {
    ($backend:ty,[$($generics:tt)*],$gather:ident,$embed:ident,$hidden:ident,$packed_hidden:ident,$forward:ident,
        $packed_forward:ident,$positions:ident,$packed_positions:ident,$sequence:ident,$packed_sequence:ident,$hidden_with:ident,$packed_with:ident) => {
        impl<$($generics)*> FullyShardedTransformerModel<$backend> {
            /// Execute actual input tables and every original backbone layer with explicit fallible architecture hooks.
            pub fn $hidden_with<C,F>(&self,input:FullyShardedTransformerInput<$backend>,communicator:C,layer:F)
                -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error> {
                let hidden=self.embeddings.$embed(input,communicator.clone())?;
                let hidden=self.backbone.forward_with(hidden,layer)?;
                assert_eq!(hidden.dims()[2],self.head.hidden_width(),"sharded backbone changed hidden width");
                Ok(if let Some(norm)=&self.normalization {norm.$gather(communicator)?.forward(hidden)} else {hidden})
            }
            /// Flat actual document rows, with no padding or invented learned position IDs.
            pub fn $packed_with<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,communicator:C,layer:F)
                -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error> {
                let hidden=self.embeddings.$embed(packed_input(input,layout),communicator.clone())?.reshape([layout.tokens(),self.embeddings.hidden_width()]);
                let hidden=self.backbone.forward_packed_with(hidden,layer)?;
                assert_eq!(hidden.dims(),[layout.tokens(),self.head.hidden_width()],"sharded packed backbone changed actual row geometry");
                Ok(if let Some(norm)=&self.normalization {norm.$gather(communicator)?.forward(hidden)} else {hidden})
            }
            /// Complete model with explicit original per-layer Q/K position transformations and attention policy.
            pub fn $positions<C,P>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,
                options:DenseAttentionOptions,communicator:C,mut positions:P) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                let hidden=self.$hidden_with(input,communicator.clone(),|index,block,hidden|block.$forward(hidden,masks.clone(),options,communicator.clone(),
                    |query,key|positions(index,query,key)))?;
                self.head.$forward(hidden,communicator)
            }
            /// Complete actual model without an additional Q/K transform; supplied masks/options still define visibility.
            pub fn $forward<C:BroadcastTensorCollective<B>>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,
                options:DenseAttentionOptions,communicator:C) -> Result<Tensor<$backend,3>,C::Error> {
                self.$positions(input,masks,options,communicator,|_,query,key|(query,key))
            }
            /// Complete flat packed-document model, preserving independent boundaries and actual positional hooks.
            pub fn $packed_positions<C,P>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,
                options:PackedAttentionOptions,communicator:C,mut positions:P) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                let hidden=self.$packed_with(input,layout,communicator.clone(),|index,block,hidden|block.$packed_forward(hidden,layout,options,communicator.clone(),
                    |query,key|positions(index,query,key)))?;
                self.head.$forward(hidden,communicator)
            }
            /// Actual packed model with no extra Q/K transform; no dense cross-document visibility is introduced.
            pub fn $packed_forward<C:BroadcastTensorCollective<B>>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,
                options:PackedAttentionOptions,communicator:C) -> Result<Tensor<$backend,2>,C::Error> {
                self.$packed_positions(input,layout,options,communicator,|_,query,key|(query,key))
            }
            /// Actual hidden states before head projection, retaining original input/backbone/final-norm derivatives.
            pub fn $hidden<C,P>(&self,input:FullyShardedTransformerInput<$backend>,masks:DenseAttentionMask<$backend>,
                options:DenseAttentionOptions,communicator:C,mut positions:P) -> Result<Tensor<$backend,3>,C::Error>
                where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<$backend,4>,Tensor<$backend,4>)->(Tensor<$backend,4>,Tensor<$backend,4>) {
                self.$hidden_with(input,communicator.clone(),|index,block,hidden|block.$forward(hidden,masks.clone(),options,communicator.clone(),
                    |query,key|positions(index,query,key)))
            }
            /// Packed hidden states before output projection, with real independent document geometry.
            pub fn $packed_hidden<C,P>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,
                options:PackedAttentionOptions,communicator:C,mut positions:P) -> Result<Tensor<$backend,2>,C::Error>
                where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<$backend,3>,Tensor<$backend,3>)->(Tensor<$backend,3>,Tensor<$backend,3>) {
                self.$packed_with(input,layout,communicator.clone(),|index,block,hidden|block.$packed_forward(hidden,layout,options,communicator.clone(),
                    |query,key|positions(index,query,key)))
            }
            /// Actual pooled sequence output and exact visible-row/token metadata, not inferred loss labels.
            pub fn $sequence<C,F>(&self,input:FullyShardedTransformerInput<$backend>,visible:Tensor<$backend,2,Bool>,pooling:SequencePooling,
                communicator:C,layer:F) -> Result<SequenceHeadOutput<$backend>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,3>)->Result<Tensor<$backend,3>,C::Error> {
                let hidden=self.$hidden_with(input,communicator.clone(),layer)?;
                Ok(self.head.$gather(communicator)?.forward_sequence(hidden,visible,pooling))
            }
            /// Independent packed classification with original empty-document visibility and I64 counts.
            pub fn $packed_sequence<C,F>(&self,input:FullyShardedTransformerInput<$backend,1>,layout:&PackedSequenceLayout,visible:Option<Tensor<$backend,1,Bool>>,
                pooling:SequencePooling,communicator:C,layer:F) -> Result<SequenceHeadOutput<$backend>,C::Error>
                where C:BroadcastTensorCollective<B>,F:FnMut(usize,&FullyShardedTransformerBlock<$backend>,Tensor<$backend,2>)->Result<Tensor<$backend,2>,C::Error> {
                let hidden=self.$packed_with(input,layout,communicator.clone(),layer)?;
                Ok(self.head.$gather(communicator)?.forward_packed_sequences(hidden,layout,visible,pooling))
            }
        }
    };
}
model_execution!(B,[B:Backend],gather_inference,forward_inference,forward_hidden_inference,forward_packed_hidden_inference,
    forward_inference,forward_packed_inference,forward_inference_with_positions,forward_packed_inference_with_positions,
    forward_sequence_inference_with,forward_packed_sequences_inference_with,forward_hidden_inference_with,forward_packed_hidden_inference_with);
model_execution!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather,forward,forward_hidden,forward_packed_hidden,
    forward,forward_packed,forward_with_positions,forward_packed_with_positions,
    forward_sequence_with,forward_packed_sequences_with,forward_hidden_with,forward_packed_hidden_with);

impl<B:Backend> FullyShardedTransformerModel<B> {
    /// Native actual new-token chunk only; original layer caches and absolute query/key offsets are retained.
    /// A transport failure after cache mutation requires restoring the caller's last complete cache record.
    pub fn forward_cached_hidden_inference<C,P>(&self,input:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,positions:P)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.embeddings.forward_inference(input,communicator.clone())?;
        let hidden=self.backbone.forward_cached_inference(hidden,visible,cache,masks,options,communicator.clone(),positions)?;
        Ok(if let Some(norm)=&self.normalization {norm.gather_inference(communicator)?.forward(hidden)} else {hidden})
    }
    /// Complete native cached model logits for real incoming rows, without replaying old token embeddings.
    pub fn forward_cached_inference<C,P>(&self,input:FullyShardedTransformerInput<B>,visible:Option<Tensor<B,2,Bool>>,
        cache:&mut TransformerKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,communicator:C,positions:P)
        -> Result<Tensor<B,3>,C::Error>
        where C:BroadcastTensorCollective<B>,P:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.forward_cached_hidden_inference(input,visible,cache,masks,options,communicator.clone(),positions)?;
        self.head.forward_inference(hidden,communicator)
    }
}
