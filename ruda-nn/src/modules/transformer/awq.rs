use alloc::vec::Vec;
use ruda_model::{module::Module,tensor::{Bool,FrozenAwqOps,Tensor,backend::Backend}};
use crate::{Linear,LoRALinear,FrozenAwqLinear,AwqLoRALinear,Dropout,activation::Activation,
    attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask,
        dense_scaled_dot_product_attention,packed_scaled_dot_product_attention,packed_scaled_dot_product_attention_masked},
    cache::{ProjectedKvCache,TransformerKvCache}};
use super::DenseTransformerNorm;
use super::{TransformerProjectionShape,TransformerProjection,BackendProjection};
use super::dense::try_residual_branch;

/// Explicit per-projection dense, dense-LoRA, original packed AWQ or AWQ-LoRA
/// selection. No source weights are quantized or adapter roles inferred here.
#[derive(Module,Debug)]
pub enum AwqTransformerProjection<B:Backend> {
    /// Original dense module and its actual trainability.
    Dense(Linear<B>),
    /// Original floating-base native adapter.
    LoRA(LoRALinear<B>),
    /// Actual loaded immutable AWQ words/scales/bias.
    Awq(FrozenAwqLinear<B>),
    /// Actual loaded packed base and independently trainable floating A/B.
    AwqLoRA(AwqLoRALinear<B>),
}
impl<B:Backend> AwqTransformerProjection<B> {
    /// Actual logical input/output width, without decoding packed words.
    pub fn dimensions(&self) -> [usize;2] {
        match self {Self::Dense(layer)=>layer.weight.val().dims(),Self::LoRA(layer)=>layer.base.weight.val().dims(),
            Self::Awq(layer)=>layer.dimensions(),Self::AwqLoRA(layer)=>layer.base.dimensions()}
    }
}
impl<B:FrozenAwqOps> AwqTransformerProjection<B> {
    /// Apply only the selected original implementation, never a dense AWQ substitute.
    pub fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,B::AwqError> {
        match self {Self::Dense(layer)=>Ok(layer.forward(input)),Self::LoRA(layer)=>Ok(layer.forward(input)),
            Self::Awq(layer)=>layer.forward(input),Self::AwqLoRA(layer)=>layer.forward(input)}
    }
}

/// Model-independent GQA/MQA/self/cross attention with explicitly mixed projection storage.
#[derive(Module,Debug)]
pub struct AwqGroupedQueryAttention<B:Backend,P:Module<B>=AwqTransformerProjection<B>> {
    /// Actual caller-selected query projection.
    pub query:BackendProjection<B,P>,
    /// Actual caller-selected memory key projection.
    pub key:BackendProjection<B,P>,
    /// Actual caller-selected memory value projection.
    pub value:BackendProjection<B,P>,
    /// Actual caller-selected context/output projection.
    pub output:BackendProjection<B,P>,
    /// Original attention-probability dropout.
    pub dropout:Dropout,
    /// Original query head count.
    pub query_heads:usize,
    /// Original shared key/value head count.
    pub kv_heads:usize,
    /// Original feature width per head.
    pub head_dimension:usize,
}
impl<B:Backend,P:TransformerProjectionShape<B>> AwqGroupedQueryAttention<B,P> {
    /// Connect loaded projections using explicit head geometry and dropout.
    pub fn from_projections(query:P,key:P,value:P,
        output:P,query_heads:usize,kv_heads:usize,head_dimension:usize,dropout:Dropout) -> Self {
        assert!(query_heads>0 && kv_heads>0 && head_dimension>0 && query_heads.is_multiple_of(kv_heads),"invalid AWQ attention head geometry");
        let query_width=query_heads.checked_mul(head_dimension).expect("query head width overflows");
        let kv_width=kv_heads.checked_mul(head_dimension).expect("KV head width overflows");
        let [input,width]=query.dimensions();assert_eq!(width,query_width,"query projection/head width differs");
        assert_eq!(key.dimensions()[1],kv_width,"key projection/head width differs");
        assert_eq!(value.dimensions(),key.dimensions(),"key/value projection geometry differs");
        assert_eq!(output.dimensions(),[query_width,input],"attention output/residual width differs");
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid attention dropout");
        Self {query,key,value,output,dropout,query_heads,kv_heads,head_dimension}
    }
}
impl<B:Backend,P:TransformerProjection<B>> AwqGroupedQueryAttention<B,P> {
    fn check_packed_heads(&self,query:&Tensor<B,3>,key:&Tensor<B,3>,value:&Tensor<B,3>) {
        assert_eq!((query.dims()[1],query.dims()[2]),(self.query_heads,self.head_dimension),"packed query head geometry differs");
        assert_eq!((key.dims()[1],key.dims()[2]),(self.kv_heads,self.head_dimension),"packed key head geometry differs");
        assert_eq!((value.dims()[1],value.dims()[2]),(self.kv_heads,self.head_dimension),"packed value head geometry differs");
    }
    /// Original loaded Q/K/V projections with exposed dense head axes for RoPE.
    pub fn project(&self,query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>)
        -> Result<(Tensor<B,4>,Tensor<B,4>,Tensor<B,4>),P::Error> {
        let [batch,queries,_]=query.dims();let [key_batch,keys,_]=key.dims();let [value_batch,values,_]=value.dims();
        assert_eq!((batch,keys),(key_batch,values),"attention batches/key lengths differ");assert_eq!(batch,value_batch,"value batch differs");
        Ok((self.query.forward(query)?.reshape([batch,queries,self.query_heads,self.head_dimension]).swap_dims(1,2),
            self.key.forward(key)?.reshape([batch,keys,self.kv_heads,self.head_dimension]).swap_dims(1,2),
            self.value.forward(value)?.reshape([batch,keys,self.kv_heads,self.head_dimension]).swap_dims(1,2)))
    }
    /// Reuse native grouped attention, masks, FP32 work statistics and dropout.
    pub fn forward_projected(&self,query:Tensor<B,4>,key:Tensor<B,4>,value:Tensor<B,4>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions)
        -> Result<Tensor<B,3>,P::Error> {
        let [batch,heads,tokens,width]=query.dims();
        assert_eq!((heads,width),(self.query_heads,self.head_dimension),"query head geometry differs");
        assert_eq!((key.dims()[1],key.dims()[3]),(self.kv_heads,self.head_dimension),"key head geometry differs");
        assert_eq!((value.dims()[1],value.dims()[3]),(self.kv_heads,self.head_dimension),"value head geometry differs");
        let context=dense_scaled_dot_product_attention(query,key,value,masks,options,Some(&self.dropout));
        self.output.forward(context.swap_dims(1,2).reshape([batch,tokens,heads*width]))
    }
    /// Actual dense-axis self/cross attention with explicit visibility/window rules.
    pub fn forward(&self,query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions)
        -> Result<Tensor<B,3>,P::Error> {
        let (query,key,value)=self.project(query,key,value)?;self.forward_projected(query,key,value,masks,options)
    }
    /// Packed Q/K/V projection preserves exact actual token counts and head geometry.
    pub fn project_packed(&self,query:Tensor<B,2>,key:Tensor<B,2>,value:Tensor<B,2>)
        -> Result<(Tensor<B,3>,Tensor<B,3>,Tensor<B,3>),P::Error> {
        let queries=query.dims()[0];let keys=key.dims()[0];assert_eq!(value.dims()[0],keys,"packed key/value rows differ");
        Ok((self.query.forward(query)?.reshape([queries,self.query_heads,self.head_dimension]),
            self.key.forward(key)?.reshape([keys,self.kv_heads,self.head_dimension]),
            self.value.forward(value)?.reshape([keys,self.kv_heads,self.head_dimension])))
    }
    /// Reuse independent-document attention without padding or a global score matrix.
    pub fn forward_packed_projected(&self,query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>,
        query_layout:&PackedSequenceLayout,key_layout:&PackedSequenceLayout,options:PackedAttentionOptions)
        -> Result<Tensor<B,2>,P::Error> {
        self.check_packed_heads(&query,&key,&value);let tokens=query.dims()[0];
        let context=packed_scaled_dot_product_attention(query,key,value,query_layout,key_layout,options,Some(&self.dropout));
        self.output.forward(context.reshape([tokens,self.query_heads*self.head_dimension]))
    }
    /// Reuse original per-document score bias and query/key visibility exactly.
    pub fn forward_packed_masked_projected(&self,query:Tensor<B,3>,key:Tensor<B,3>,value:Tensor<B,3>,
        query_layout:&PackedSequenceLayout,key_layout:&PackedSequenceLayout,masks:&[PackedDocumentAttentionMask<B>],options:PackedAttentionOptions)
        -> Result<Tensor<B,2>,P::Error> {
        self.check_packed_heads(&query,&key,&value);let tokens=query.dims()[0];
        let context=packed_scaled_dot_product_attention_masked(query,key,value,query_layout,key_layout,masks,options,Some(&self.dropout));
        self.output.forward(context.reshape([tokens,self.query_heads*self.head_dimension]))
    }
    /// Append actual newly positioned K/V using the existing native cache contract.
    /// Partial projection failures after append require restoring the caller's cache record.
    pub fn forward_cached_with_positions<F>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        let (query,key,value)=self.project(input.clone(),input.clone(),input)?;
        let geometry=(query.dims(),key.dims());let (query,key)=positions(query,key,cache.position());
        assert_eq!((query.dims(),key.dims()),geometry,"cached positions changed projection geometry");
        let (key,value,masks)=crate::attention::append_cached_projected(&query,key,value,new_visible,cache,masks,
            (self.query_heads,self.kv_heads,self.head_dimension));
        self.forward_projected(query,key,value,masks,options)
    }
}

/// Ordinary/gated FFN with exact per-projection packed/dense/adapter choices.
#[derive(Module,Debug)]
pub struct AwqFeedForward<B:Backend,P:Module<B>=AwqTransformerProjection<B>> {
    /// Actual loaded value/up projection.
    pub up:P,
    /// Actual optional independent gate, retaining absence when ungated.
    pub gate:Option<P>,
    /// Actual loaded down projection.
    pub down:P,
    /// Existing native activation, including its actual trainable parameters.
    pub activation:Activation<B>,
    /// Existing intermediate dropout.
    pub dropout:Dropout,
}
impl<B:Backend,P:TransformerProjectionShape<B>> AwqFeedForward<B,P> {
    /// Connect actual loaded values with no quantization or random replacements.
    pub fn from_projections(up:P,gate:Option<P>,down:P,
        activation:Activation<B>,dropout:Dropout) -> Self {
        let [width,inner]=up.dimensions();assert_eq!(down.dimensions(),[inner,width],"FFN up/down widths differ");
        if let Some(gate)=&gate {assert_eq!(gate.dimensions(),[width,inner],"FFN gate/up geometry differs");}
        Self {up,gate,down,activation,dropout}
    }
}
impl<B:Backend,P:TransformerProjection<B>> AwqFeedForward<B,P> {
    /// Native original activation(gate)*up or activation(up), then dropout/down.
    pub fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,P::Error> {
        let up=self.up.forward(input.clone())?;
        let value=if let Some(gate)=&self.gate {
            let activated=self.activation.forward(gate.forward(input)?);assert_eq!(activated.dims(),up.dims(),"gate activation changed geometry");activated*up
        } else {self.activation.forward(up)};
        self.down.forward(self.dropout.forward(value))
    }
}

/// Complete native self-attention/FFN block, retaining original normalization order.
#[derive(Module,Debug)]
pub struct AwqTransformerBlock<B:Backend,P:Module<B>=AwqTransformerProjection<B>> {
    /// Actual independently selected query/key/value/output projections.
    pub attention:AwqGroupedQueryAttention<B,P>,
    /// Actual ordinary/gated native FFN.
    pub feed_forward:AwqFeedForward<B,P>,
    /// Original attention affine norm.
    pub attention_norm:DenseTransformerNorm<B>,
    /// Original independent FFN affine norm.
    pub feed_forward_norm:DenseTransformerNorm<B>,
    /// Original residual-branch dropout.
    pub residual_dropout:Dropout,
    /// Original pre/post normalization choice.
    pub norm_first:bool,
}
impl<B:Backend,P:TransformerProjection<B>> AwqTransformerBlock<B,P> {
    /// Original dense-axis complete block with caller-owned projected positions.
    pub fn forward_with_positions<F>(&self,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,P::Error> where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_feed_forward(self.forward_attention_with_positions(input,masks,options,positions)?)
    }
    /// Actual original attention/residual/norm stage, before inserting encoder-memory attention.
    pub fn forward_attention_with_positions<F>(&self,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,P::Error> where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        try_residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value)=self.attention.project(source.clone(),source.clone(),source)?;
            let shape=(query.dims(),key.dims());let (query,key)=positions(query,key);
            assert_eq!((query.dims(),key.dims()),shape,"positions changed head geometry");
            self.attention.forward_projected(query,key,value,masks,options)
        })
    }
    /// Actual original FFN/residual/norm stage, retaining dense or flat-document axes.
    pub fn forward_feed_forward<const D:usize>(&self,hidden:Tensor<B,D>) -> Result<Tensor<B,D>,P::Error> {
        try_residual_branch(hidden,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,|source|self.feed_forward.forward(source))
    }
    /// Complete block without an implicit model-specific positional transformation.
    pub fn forward(&self,input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions) -> Result<Tensor<B,3>,P::Error> {
        self.forward_with_positions(input,masks,options,|query,key|(query,key))
    }
    /// Packed document-local block with explicit masks/options and projected positions.
    pub fn forward_packed_with_positions<F>(&self,input:Tensor<B,2>,layout:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F)
        -> Result<Tensor<B,2>,P::Error> where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        self.forward_feed_forward(self.forward_packed_attention_with_positions(input,layout,masks,options,positions)?)
    }
    /// Original independent-document attention stage, before encoder-memory attention.
    pub fn forward_packed_attention_with_positions<F>(&self,input:Tensor<B,2>,layout:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F)
        -> Result<Tensor<B,2>,P::Error> where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        assert_eq!(input.dims()[0],layout.tokens(),"packed document boundaries differ from actual rows");
        try_residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value)=self.attention.project_packed(source.clone(),source.clone(),source)?;
            let geometry=(query.dims(),key.dims());let (query,key)=positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"packed positions changed geometry");
            if let Some(masks)=masks {self.attention.forward_packed_masked_projected(query,key,value,layout,layout,masks,options)}
            else {self.attention.forward_packed_projected(query,key,value,layout,layout,options)}
        })
    }
    /// Actual new-token cached attention and FFN, reusing the existing cache semantics.
    pub fn forward_cached_with_positions<F>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_feed_forward(self.forward_cached_attention_with_positions(input,new_visible,cache,masks,options,positions)?)
    }
    /// Original actual new-token attention stage without prematurely running the decoder FFN.
    pub fn forward_cached_attention_with_positions<F>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        try_residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source|
            self.attention.forward_cached_with_positions(source,new_visible,cache,masks,options,positions))
    }
}

/// Actual original block order with independently selected packed/adapter projections.
#[derive(Module,Debug)]
pub struct AwqTransformerStack<B:Backend,P:Module<B>=AwqTransformerProjection<B>> {
    /// Every loaded actual block, retaining cross-layer parameter identities.
    pub blocks:Vec<AwqTransformerBlock<B,P>>,
}
impl<B:Backend,P:Module<B>> AwqTransformerStack<B,P> {
    /// Original native cache metadata for the actual loaded layer count.
    pub fn new_kv_cache(&self,initial_capacity:usize) -> TransformerKvCache<B> {TransformerKvCache::new(self.blocks.len(),initial_capacity)}
}
impl<B:Backend,P:TransformerProjection<B>> AwqTransformerStack<B,P> {
    /// Original block order with caller-owned per-layer policy, preserving arbitrary hidden axes.
    pub fn forward_with<const D:usize,F>(&self,mut input:Tensor<B,D>,mut layer:F) -> Result<Tensor<B,D>,P::Error>
        where F:FnMut(usize,&AwqTransformerBlock<B,P>,Tensor<B,D>)->Result<Tensor<B,D>,P::Error> {
        for (index,block) in self.blocks.iter().enumerate() {input=layer(index,block,input)?;}Ok(input)
    }
    /// Whole native stack with explicit per-layer projected positions.
    pub fn forward_with_positions<F>(&self,mut input:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,mut positions:F)
        -> Result<Tensor<B,3>,P::Error> where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        for (index,block) in self.blocks.iter().enumerate() {input=block.forward_with_positions(input,masks.clone(),options,|query,key|positions(index,query,key))?;}
        Ok(input)
    }
    /// Whole packed stack with actual document-local visibility and positions.
    pub fn forward_packed_with_positions<F>(&self,mut input:Tensor<B,2>,layout:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,mut positions:F)
        -> Result<Tensor<B,2>,P::Error> where F:FnMut(usize,Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        for (index,block) in self.blocks.iter().enumerate() {input=block.forward_packed_with_positions(input,layout,masks,options,|query,key|positions(index,query,key))?;}
        Ok(input)
    }
    /// Original cached per-layer execution. Only a complete chunk advances the
    /// stack boundary; partial errors may require restoring the actual cache record.
    pub fn forward_cached_with_positions<F>(&self,mut input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut TransformerKvCache<B>,
        masks:DenseAttentionMask<B>,options:DenseAttentionOptions,mut positions:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnMut(usize,Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>) {
        cache.validate_layers(self.blocks.len());let rows=(input.dims()[0],input.dims()[1]);
        let next=cache.position().checked_add(rows.1).expect("cached stack position overflows");
        for (index,block) in self.blocks.iter().enumerate() {
            input=block.forward_cached_with_positions(input,new_visible.clone(),&mut cache.layers_mut()[index],masks.clone(),options,
                |query,key,position|positions(index,query,key,position))?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"cached block changed chunk rows");
        }
        cache.finish_chunk(next);Ok(input)
    }
}
