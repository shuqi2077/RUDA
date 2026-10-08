use alloc::vec::Vec;
use ruda_model::{module::Module,tensor::{Bool,Tensor,backend::Backend}};
use crate::{Dropout,attention::{DenseAttentionMask,DenseAttentionOptions,PackedSequenceLayout,PackedAttentionOptions,PackedDocumentAttentionMask},
    cache::{ProjectedKvCache,TransformerKvCache,EncoderDecoderKvCache}};
use super::{TransformerProjectionShape,TransformerProjection,ProjectedGroupedQueryAttention,ProjectedTransformerBlock,DenseTransformerNorm};
use super::dense::try_residual_branch;

/// Actual native encoder-memory stage with independent dense/AWQ/NF4 projection roles.
#[derive(Module,Debug)]
pub struct ProjectedCrossAttentionBlock<B:Backend,P:Module<B>> {
    /// Original independent query, memory key/value and output projections.
    pub attention:ProjectedGroupedQueryAttention<B,P>,
    /// Original query/residual norm.
    pub query_norm:DenseTransformerNorm<B>,
    /// Original optional independent encoder-memory norm.
    pub memory_norm:Option<DenseTransformerNorm<B>>,
    /// Original residual-branch dropout.
    pub residual_dropout:Dropout,
    /// Original pre/post-normalization convention.
    pub norm_first:bool,
}
impl<B:Backend,P:TransformerProjectionShape<B>> ProjectedCrossAttentionBlock<B,P> {
    /// Connect actual loaded components without allocating new weights or inferring memory width.
    pub fn from_parts(attention:ProjectedGroupedQueryAttention<B,P>,query_norm:DenseTransformerNorm<B>,memory_norm:Option<DenseTransformerNorm<B>>,
        residual_dropout:Dropout,norm_first:bool) -> Self {
        assert_eq!(query_norm.width(),attention.query.dimensions()[0],"cross query norm width differs");
        if let Some(norm)=&memory_norm {assert_eq!(norm.width(),attention.key.dimensions()[0],"cross memory norm width differs");}
        Self {attention,query_norm,memory_norm,residual_dropout,norm_first}
    }
}
impl<B:Backend,P:TransformerProjection<B>> ProjectedCrossAttentionBlock<B,P> {
    /// Actual dense-axis cross attention, retaining source memory gradients and explicit positions.
    pub fn forward_with_positions<F>(&self,input:Tensor<B,3>,memory:Tensor<B,3>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,positions:F)
        -> Result<Tensor<B,3>,P::Error> where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let memory=if let Some(norm)=&self.memory_norm {norm.forward(memory)} else {memory};
        try_residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value)=self.attention.project(source,memory.clone(),memory)?;
            let geometry=(query.dims(),key.dims());let (query,key)=positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"cross positions changed actual head geometry");
            self.attention.forward_projected(query,key,value,masks,options)
        })
    }
    /// Actual paired flat documents with independent source/target lengths and optional score masks.
    pub fn forward_packed_with_positions<F>(&self,input:Tensor<B,2>,memory:Tensor<B,2>,query_layout:&PackedSequenceLayout,memory_layout:&PackedSequenceLayout,
        masks:Option<&[PackedDocumentAttentionMask<B>]>,options:PackedAttentionOptions,positions:F) -> Result<Tensor<B,2>,P::Error>
        where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        assert_eq!(query_layout.documents(),memory_layout.documents(),"paired source/target document counts differ");
        assert_eq!(input.dims()[0],query_layout.tokens(),"paired target boundaries differ from actual rows");
        assert_eq!(memory.dims()[0],memory_layout.tokens(),"paired memory boundaries differ from actual rows");
        let memory=if let Some(norm)=&self.memory_norm {norm.forward(memory)} else {memory};
        try_residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value)=self.attention.project_packed(source,memory.clone(),memory)?;
            let geometry=(query.dims(),key.dims());let (query,key)=positions(query,key);
            assert_eq!((query.dims(),key.dims()),geometry,"packed cross positions changed actual geometry");
            if let Some(masks)=masks {self.attention.forward_packed_masked_projected(query,key,value,query_layout,memory_layout,masks,options)}
            else {self.attention.forward_packed_projected(query,key,value,query_layout,memory_layout,options)}
        })
    }
    /// Prepare actual original memory K/V once, without running a query projection.
    pub fn prepare_cached_memory<F>(&self,memory:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,start_position:usize,positions:F)
        -> Result<ProjectedKvCache<B>,P::Error> where F:FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let memory=if let Some(norm)=&self.memory_norm {norm.forward(memory)} else {memory};let [batch,tokens,_]=memory.dims();
        let key=self.attention.key.forward(memory.clone())?.reshape([batch,tokens,self.attention.kv_heads,self.attention.head_dimension]).swap_dims(1,2);
        let value=self.attention.value.forward(memory)?.reshape([batch,tokens,self.attention.kv_heads,self.attention.head_dimension]).swap_dims(1,2);
        let geometry=key.dims();let key=positions(key,start_position);assert_eq!(key.dims(),geometry,"prepared memory positions changed geometry");
        Ok(ProjectedKvCache::from_projected(key,value,visible,start_position))
    }
    /// Run only actual query/output weights against immutable already-positioned encoder memory.
    pub fn forward_cached_memory<F>(&self,input:Tensor<B,3>,memory:&ProjectedKvCache<B>,masks:DenseAttentionMask<B>,options:DenseAttentionOptions,
        query_position:usize,positions:F) -> Result<Tensor<B,3>,P::Error> where F:FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        try_residual_branch(input,&self.query_norm,&self.residual_dropout,self.norm_first,|source| {
            let [batch,tokens,_]=source.dims();
            let query=self.attention.query.forward(source)?.reshape([batch,tokens,self.attention.query_heads,self.attention.head_dimension]).swap_dims(1,2);
            let geometry=query.dims();let query=positions(query,query_position);assert_eq!(query.dims(),geometry,"cached cross query positions changed geometry");
            let (key,value,visible)=memory.prefix().expect("prepare actual encoder memory before cached cross attention");
            self.attention.forward_projected(query,key,value,crate::attention::cached_projected_masks(masks,visible),options)
        })
    }
}

/// Original self-attention, encoder-memory attention, then FFN with independent native packed roles.
#[derive(Module,Debug)]
pub struct ProjectedEncoderDecoderLayer<B:Backend,P:Module<B>> {
    /// Actual original native self-attention and final FFN stages.
    pub backbone:ProjectedTransformerBlock<B,P>,
    /// Actual independent encoder-memory stage, inserted before the FFN.
    pub cross_attention:ProjectedCrossAttentionBlock<B,P>,
}
impl<B:Backend,P:TransformerProjectionShape<B>> ProjectedEncoderDecoderLayer<B,P> {
    /// Connect actual original components with their original identities and residual widths.
    pub fn from_parts(backbone:ProjectedTransformerBlock<B,P>,cross_attention:ProjectedCrossAttentionBlock<B,P>) -> Self {
        assert_eq!(backbone.attention.query.dimensions()[0],cross_attention.attention.query.dimensions()[0],"decoder residual widths differ");
        Self {backbone,cross_attention}
    }
}
impl<B:Backend,P:TransformerProjection<B>> ProjectedEncoderDecoderLayer<B,P> {
    /// Exact original three-stage dense-axis graph with separately supplied self/cross positions.
    pub fn forward_with_positions<F,G>(&self,input:Tensor<B,3>,memory:Tensor<B,3>,self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,
        cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,self_positions:F,cross_positions:G) -> Result<Tensor<B,3>,P::Error>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),G:FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden=self.backbone.forward_attention_with_positions(input,self_masks,self_options,self_positions)?;
        let hidden=self.cross_attention.forward_with_positions(hidden,memory,cross_masks,cross_options,cross_positions)?;
        self.backbone.forward_feed_forward(hidden)
    }
    /// Exact original three-stage independent-document graph without padded source/target rows.
    pub fn forward_packed_with_positions<F,G>(&self,input:Tensor<B,2>,memory:Tensor<B,2>,query_layout:&PackedSequenceLayout,memory_layout:&PackedSequenceLayout,
        self_masks:Option<&[PackedDocumentAttentionMask<B>]>,self_options:PackedAttentionOptions,cross_masks:Option<&[PackedDocumentAttentionMask<B>]>,
        cross_options:PackedAttentionOptions,self_positions:F,cross_positions:G) -> Result<Tensor<B,2>,P::Error>
        where F:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>),G:FnOnce(Tensor<B,3>,Tensor<B,3>)->(Tensor<B,3>,Tensor<B,3>) {
        let hidden=self.backbone.forward_packed_attention_with_positions(input,query_layout,self_masks,self_options,self_positions)?;
        let hidden=self.cross_attention.forward_packed_with_positions(hidden,memory,query_layout,memory_layout,cross_masks,cross_options,cross_positions)?;
        self.backbone.forward_feed_forward(hidden)
    }
    /// Only new self-attention keys append; encoder memory remains positioned once.
    pub fn forward_cached_with_positions<F,G>(&self,input:Tensor<B,3>,new_visible:Option<Tensor<B,2,Bool>>,cache:&mut ProjectedKvCache<B>,memory:&ProjectedKvCache<B>,
        self_masks:DenseAttentionMask<B>,self_options:DenseAttentionOptions,cross_masks:DenseAttentionMask<B>,cross_options:DenseAttentionOptions,
        self_positions:F,cross_positions:G) -> Result<Tensor<B,3>,P::Error>
        where F:FnOnce(Tensor<B,4>,Tensor<B,4>,usize)->(Tensor<B,4>,Tensor<B,4>),G:FnOnce(Tensor<B,4>,usize)->Tensor<B,4> {
        let position=cache.position();
        let hidden=self.backbone.forward_cached_attention_with_positions(input,new_visible,cache,self_masks,self_options,self_positions)?;
        let hidden=self.cross_attention.forward_cached_memory(hidden,memory,cross_masks,cross_options,position,cross_positions)?;
        self.backbone.forward_feed_forward(hidden)
    }
}

/// Original native paired-layer order, with independently selected actual packed projection storage.
#[derive(Module,Debug)]
pub struct ProjectedEncoderDecoderStack<B:Backend,P:Module<B>> {
    /// Every actual loaded layer in its original order.
    pub layers:Vec<ProjectedEncoderDecoderLayer<B,P>>,
}
impl<B:Backend,P:TransformerProjection<B>> ProjectedEncoderDecoderStack<B,P> {
    /// Execute original dense paired rows with architecture-owned per-layer visibility and positions.
    pub fn forward_with<F>(&self,mut input:Tensor<B,3>,memory:Tensor<B,3>,mut layer:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,3>,Tensor<B,3>)->Result<Tensor<B,3>,P::Error> {
        for (index,block) in self.layers.iter().enumerate() {input=layer(index,block,input,memory.clone())?;}Ok(input)
    }
    /// Execute independently bounded packed source/target rows with architecture-owned layer policies.
    pub fn forward_packed_with<F>(&self,mut input:Tensor<B,2>,memory:Tensor<B,2>,query_layout:&PackedSequenceLayout,memory_layout:&PackedSequenceLayout,mut layer:F)
        -> Result<Tensor<B,2>,P::Error>
        where F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,2>,Tensor<B,2>)->Result<Tensor<B,2>,P::Error> {
        assert_eq!(query_layout.documents(),memory_layout.documents(),"paired document counts differ");
        assert_eq!(input.dims()[0],query_layout.tokens(),"target boundaries differ");assert_eq!(memory.dims()[0],memory_layout.tokens(),"source boundaries differ");
        for (index,block) in self.layers.iter().enumerate() {input=layer(index,block,input,memory.clone())?;}Ok(input)
    }
    /// Prepare exact paired native cache state, reusable by existing beam reorder/record continuation.
    pub fn prepare_kv_cache<F>(&self,memory:Tensor<B,3>,visible:Option<Tensor<B,2,Bool>>,start_position:usize,initial_capacity:usize,mut positions:F)
        -> Result<EncoderDecoderKvCache<B>,P::Error> where F:FnMut(usize,Tensor<B,4>,usize)->Tensor<B,4> {
        let memory=self.layers.iter().enumerate().map(|(index,layer)|layer.cross_attention.prepare_cached_memory(memory.clone(),visible.clone(),start_position,
            |key,position|positions(index,key,position))).collect::<Result<Vec<_>,_>>()?;
        Ok(EncoderDecoderKvCache::new(TransformerKvCache::new(self.layers.len(),initial_capacity),memory))
    }
    /// Execute only actual new decoder rows. A partial failure retains the original cache restore contract.
    pub fn forward_cached_with<F>(&self,mut input:Tensor<B,3>,cache:&mut EncoderDecoderKvCache<B>,mut layer:F) -> Result<Tensor<B,3>,P::Error>
        where F:FnMut(usize,&ProjectedEncoderDecoderLayer<B,P>,Tensor<B,3>,&mut ProjectedKvCache<B>,&ProjectedKvCache<B>)->Result<Tensor<B,3>,P::Error> {
        let (decoder,memory)=cache.parts_mut();decoder.validate_layers(self.layers.len());assert_eq!(memory.len(),self.layers.len(),"cached memory/layer counts differ");
        let rows=(input.dims()[0],input.dims()[1]);let next=decoder.position().checked_add(rows.1).expect("cached decoder position overflows");
        for (index,block) in self.layers.iter().enumerate() {
            input=layer(index,block,input,&mut decoder.layers_mut()[index],&memory[index])?;
            assert_eq!((input.dims()[0],input.dims()[1]),rows,"cached decoder changed actual new rows");
        }
        decoder.finish_chunk(next);Ok(input)
    }
}
