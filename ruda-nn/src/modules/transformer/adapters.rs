use ruda_model::{config::Config,module::Module,tensor::{DType,Tensor,backend::Backend}};
use crate::{Linear,LoRALinear,LoRALinearConfig,Dropout,activation::Activation,
    attention::{GroupedQueryAttention,DenseAttentionMask,DenseAttentionOptions,dense_scaled_dot_product_attention}};
use super::{DenseFeedForward,DenseTransformerBlock,DenseTransformerNorm};
use super::dense::residual_branch;

/// Adapter options applied only to explicitly selected native projections.
#[derive(Config,Debug)]
pub struct TransformerAdapterConfig {
    /// Rank, alpha and adapter-input dropout of RUDA's existing LoRA module.
    pub lora: LoRALinearConfig,
    /// Explicit A/B storage; None retains each selected base weight's dtype.
    pub adapter_dtype: Option<DType>,
    /// Explicit alpha/sqrt(rank) scaling rather than alpha/rank.
    #[config(default = false)]
    pub use_rslora: bool,
}

/// Actual original projection or frozen base plus recorded A/B adapter parameters.
#[derive(Module,Debug)]
pub enum AdaptedProjection<B: Backend> {
    /// Original projection, retaining its existing frozen/trainable setting.
    Dense(Linear<B>),
    /// Existing RUDA LoRA implementation, including native mixed A/B storage.
    LoRA(LoRALinear<B>),
}

impl<B: Backend> AdaptedProjection<B> {
    /// Apply the actual selected projection with no hidden dense weight merge.
    pub fn forward<const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {
        match self {Self::Dense(layer)=>layer.forward(input),Self::LoRA(layer)=>layer.forward(input)}
    }

    /// Consume adapters for dropout-free inference, not training-state conversion.
    pub fn merge(self) -> Linear<B> {
        match self {Self::Dense(layer)=>layer,Self::LoRA(layer)=>layer.merge()}
    }
}

impl TransformerAdapterConfig {
    fn wrap<B: Backend>(&self,base: Linear<B>,selected: bool) -> AdaptedProjection<B> {
        if selected {
            let dtype = self.adapter_dtype.unwrap_or_else(||base.weight.val().dtype());
            AdaptedProjection::LoRA(self.lora.init_with_options(base,dtype,self.use_rslora))
        } else { AdaptedProjection::Dense(base) }
    }
}

/// Exact native attention projection selection; no suffix/name pattern guessing.
#[derive(Config,Debug,Copy,PartialEq,Eq)]
pub enum AttentionAdapterTarget {
    /// Query projection.
    Query,
    /// Shared key projection.
    Key,
    /// Shared value projection.
    Value,
    /// Context/output projection.
    Output,
}

/// Native GQA/MQA with adapters on caller-selected actual projections.
#[derive(Module,Debug)]
pub struct AdaptedGroupedQueryAttention<B: Backend> {
    /// Actual dense/adapted query projection.
    pub query: AdaptedProjection<B>,
    /// Actual dense/adapted shared key projection.
    pub key: AdaptedProjection<B>,
    /// Actual dense/adapted shared value projection.
    pub value: AdaptedProjection<B>,
    /// Actual dense/adapted output projection.
    pub output: AdaptedProjection<B>,
    /// Attention-probability dropout inherited from the supplied layer.
    pub dropout: Dropout,
    /// Original actual query head count.
    pub query_heads: usize,
    /// Original actual KV head count.
    pub kv_heads: usize,
    /// Original per-head feature width.
    pub head_dimension: usize,
}

impl<B: Backend> AdaptedGroupedQueryAttention<B> {
    /// Consume existing projections and add only selected A/B matrices.
    /// Selected bases become frozen by LoRA; unselected flags remain unchanged.
    pub fn from_dense(base: GroupedQueryAttention<B>,config: &TransformerAdapterConfig,
        targets: &[AttentionAdapterTarget]) -> Self {
        for (i,target) in targets.iter().enumerate() {
            assert!(!targets[..i].contains(target),"duplicate attention adapter target");
        }
        Self {query:config.wrap(base.query,targets.contains(&AttentionAdapterTarget::Query)),
            key:config.wrap(base.key,targets.contains(&AttentionAdapterTarget::Key)),
            value:config.wrap(base.value,targets.contains(&AttentionAdapterTarget::Value)),
            output:config.wrap(base.output,targets.contains(&AttentionAdapterTarget::Output)),
            dropout:base.dropout,query_heads:base.query_heads,kv_heads:base.kv_heads,head_dimension:base.head_dimension}
    }

    /// Project real Q/K/V inputs, exposing heads for explicit positional transforms.
    pub fn project(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>)
        -> (Tensor<B,4>,Tensor<B,4>,Tensor<B,4>) {
        let [batch,queries,_] = query.dims();
        let [key_batch,keys,_] = key.dims();
        let [value_batch,values,_] = value.dims();
        assert_eq!((batch,keys),(key_batch,values),"adapted attention batches/key lengths differ");
        assert_eq!(batch,value_batch,"adapted value batch differs");
        (self.query.forward(query).reshape([batch,queries,self.query_heads,self.head_dimension]).swap_dims(1,2),
            self.key.forward(key).reshape([batch,keys,self.kv_heads,self.head_dimension]).swap_dims(1,2),
            self.value.forward(value).reshape([batch,keys,self.kv_heads,self.head_dimension]).swap_dims(1,2))
    }

    /// Native grouped attention and the actual dense/adapted output projection.
    pub fn forward_projected(&self,query: Tensor<B,4>,key: Tensor<B,4>,value: Tensor<B,4>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        let [batch,heads,queries,width] = query.dims();
        assert_eq!((heads,width),(self.query_heads,self.head_dimension),"adapted query head geometry differs");
        assert_eq!((key.dims()[1],key.dims()[3]),(self.kv_heads,self.head_dimension),"adapted key head geometry differs");
        assert_eq!((value.dims()[1],value.dims()[3]),(self.kv_heads,self.head_dimension),"adapted value head geometry differs");
        let context = dense_scaled_dot_product_attention(query,key,value,masks,options,Some(&self.dropout));
        self.output.forward(context.swap_dims(1,2).reshape([batch,queries,heads*width]))
    }

    /// Explicit-mask adapted self/cross attention, with no inferred position rule.
    pub fn forward(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        let (query,key,value) = self.project(query,key,value);
        self.forward_projected(query,key,value,masks,options)
    }

    /// Consume A/B updates into the original dense projection identities for inference.
    pub fn merge(self) -> GroupedQueryAttention<B> {
        GroupedQueryAttention {query:self.query.merge(),key:self.key.merge(),value:self.value.merge(),output:self.output.merge(),
            dropout:self.dropout,query_heads:self.query_heads,kv_heads:self.kv_heads,head_dimension:self.head_dimension}
    }
}

/// Explicit actual FFN projection targets.
#[derive(Config,Debug,Copy,PartialEq,Eq)]
pub enum FeedForwardAdapterTarget {
    /// Value/up projection.
    Up,
    /// Optional separate gate projection; selecting an absent gate is an error.
    Gate,
    /// Final/down projection.
    Down,
}

/// Ordinary/gated native FFN with recorded per-projection adapters.
#[derive(Module,Debug)]
pub struct AdaptedFeedForward<B: Backend> {
    /// Actual value projection.
    pub up: AdaptedProjection<B>,
    /// Optional actual gate projection.
    pub gate: Option<AdaptedProjection<B>>,
    /// Actual final output projection.
    pub down: AdaptedProjection<B>,
    /// Original activation, including every existing trainable activation parameter.
    pub activation: Activation<B>,
    /// Original intermediate dropout.
    pub dropout: Dropout,
}

impl<B: Backend> AdaptedFeedForward<B> {
    /// Add only explicitly selected adapters; other parameter flags remain intact.
    pub fn from_dense(base: DenseFeedForward<B>,config: &TransformerAdapterConfig,
        targets: &[FeedForwardAdapterTarget]) -> Self {
        for (i,target) in targets.iter().enumerate() {
            assert!(!targets[..i].contains(target),"duplicate feed-forward adapter target");
        }
        assert!(base.gate.is_some() || !targets.contains(&FeedForwardAdapterTarget::Gate),"cannot adapt an absent gate projection");
        Self {up:config.wrap(base.up,targets.contains(&FeedForwardAdapterTarget::Up)),
            gate:base.gate.map(|gate|config.wrap(gate,targets.contains(&FeedForwardAdapterTarget::Gate))),
            down:config.wrap(base.down,targets.contains(&FeedForwardAdapterTarget::Down)),activation:base.activation,dropout:base.dropout}
    }

    /// Same native FFN expression, with A/B gradients on selected projections.
    pub fn forward<const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {
        let up = self.up.forward(input.clone());
        let value = if let Some(gate) = &self.gate {
            let activated = self.activation.forward(gate.forward(input));
            assert_eq!(activated.dims(),up.dims(),"adapted gate activation must preserve intermediate geometry");
            activated*up
        }
            else { self.activation.forward(up) };
        self.down.forward(self.dropout.forward(value))
    }

    /// Dropout-free adapter merge for dense inference, not optimizer checkpoint resume.
    pub fn merge(self) -> DenseFeedForward<B> {
        DenseFeedForward {up:self.up.merge(),gate:self.gate.map(|gate|gate.merge()),down:self.down.merge(),
            activation:self.activation,dropout:self.dropout}
    }
}

/// Native block with explicit attention/FFN adapters and unchanged residual/norm order.
#[derive(Module,Debug)]
pub struct AdaptedTransformerBlock<B: Backend> {
    /// Actual adapted grouped attention.
    pub attention: AdaptedGroupedQueryAttention<B>,
    /// Actual adapted ordinary/gated FFN.
    pub feed_forward: AdaptedFeedForward<B>,
    /// Original attention norm parameters/flags.
    pub attention_norm: DenseTransformerNorm<B>,
    /// Original FFN norm parameters/flags.
    pub feed_forward_norm: DenseTransformerNorm<B>,
    /// Original residual dropout.
    pub residual_dropout: Dropout,
    /// Original pre/post norm order.
    pub norm_first: bool,
}

impl<B: Backend> AdaptedTransformerBlock<B> {
    /// Consume a real block; apply no blanket freezing of unselected parameters.
    /// For adapter-only training, explicitly freeze the base before this conversion.
    pub fn from_dense(base: DenseTransformerBlock<B>,config: &TransformerAdapterConfig,
        attention_targets: &[AttentionAdapterTarget],feed_forward_targets: &[FeedForwardAdapterTarget]) -> Self {
        assert!(!attention_targets.is_empty() || !feed_forward_targets.is_empty(),"at least one adapter target is required");
        Self {attention:AdaptedGroupedQueryAttention::from_dense(base.attention,config,attention_targets),
            feed_forward:AdaptedFeedForward::from_dense(base.feed_forward,config,feed_forward_targets),
            attention_norm:base.attention_norm,feed_forward_norm:base.feed_forward_norm,
            residual_dropout:base.residual_dropout,norm_first:base.norm_first}
    }

    /// Forward using the actual unchanged visibility/window contract.
    pub fn forward(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_with_positions(input,masks,options,|query,key|(query,key))
    }

    /// Preserve caller-owned Q/K positions while training the selected adapters.
    pub fn forward_with_positions<F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source| {
            let (query,key,value) = self.attention.project(source.clone(),source.clone(),source);
            let query_shape = query.dims();
            let key_shape = key.dims();
            let (query,key) = positions(query,key);
            assert_eq!(query.dims(),query_shape,"adapted query position transform changed geometry");
            assert_eq!(key.dims(),key_shape,"adapted key position transform changed geometry");
            self.attention.forward_projected(query,key,value,masks,options)
        });
        residual_branch(hidden,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,|source|self.feed_forward.forward(source))
    }

    /// Consume all selected adapters into the original dense block for inference.
    /// The old adapter optimizer state is not converted or reused.
    pub fn merge(self) -> DenseTransformerBlock<B> {
        DenseTransformerBlock {attention:self.attention.merge(),feed_forward:self.feed_forward.merge(),
            attention_norm:self.attention_norm,feed_forward_norm:self.feed_forward_norm,
            residual_dropout:self.residual_dropout,norm_first:self.norm_first}
    }
}
