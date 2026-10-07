use alloc::vec::Vec;
use ruda_model::{config::Config,module::Module,tensor::{DType,FloatDType,Tensor,backend::Backend}};
use crate::{Dropout,DropoutConfig,LayerNorm,LayerNormConfig,RmsNorm,RmsNormConfig,Linear,LinearConfig,
    activation::{Activation,ActivationConfig},
    attention::{GroupedQueryAttention,GroupedQueryAttentionConfig,DenseAttentionMask,DenseAttentionOptions}};

/// Actual two/three projection feed-forward geometry and persisted activation.
#[derive(Config,Debug)]
pub struct DenseFeedForwardConfig {
    /// Input and final output width.
    pub d_model: usize,
    /// Actual intermediate width, independent of any model-family expansion rule.
    pub d_ff: usize,
    /// Add a separate gate projection; activation(gate)*up precedes down.
    #[config(default = false)]
    pub gated: bool,
    /// Bias in each selected projection.
    #[config(default = true)]
    pub bias: bool,
    /// All activation parameters participate in module visitation and records.
    #[config(default = "ActivationConfig::Gelu")]
    pub activation: ActivationConfig,
    /// Dropout on the activated intermediate, before the final projection.
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Native dense FFN/GeGLU/SwiGLU composition with real recorded parameters.
#[derive(Module,Debug)]
pub struct DenseFeedForward<B: Backend> {
    /// Value projection; its weight is always used.
    pub up: Linear<B>,
    /// Independent gate projection, present only for an explicitly gated FFN.
    pub gate: Option<Linear<B>>,
    /// Projection back to the declared residual width.
    pub down: Linear<B>,
    /// Stateful activation is not skipped during saving or optimizer visitation.
    pub activation: Activation<B>,
    /// Intermediate dropout using the backend's actual training mode.
    pub dropout: Dropout,
}

impl DenseFeedForwardConfig {
    /// Initialize only the explicitly requested feed-forward parameters.
    pub fn init<B: Backend>(&self,device: &B::Device) -> DenseFeedForward<B> {
        assert!(self.d_model > 0 && self.d_ff > 0,"feed-forward widths must be positive");
        assert!(self.dropout.is_finite() && (0.0..=1.0).contains(&self.dropout),"invalid feed-forward dropout");
        if let ActivationConfig::SwiGlu(config) = &self.activation {
            assert_eq!((config.d_input,config.d_output),(self.d_ff,self.d_ff),"stateful SwiGLU activation must retain intermediate width");
        }
        DenseFeedForward {
            up:LinearConfig::new(self.d_model,self.d_ff).with_bias(self.bias).init(device),
            gate:self.gated.then(||LinearConfig::new(self.d_model,self.d_ff).with_bias(self.bias).init(device)),
            down:LinearConfig::new(self.d_ff,self.d_model).with_bias(self.bias).init(device),
            activation:self.activation.init(device),dropout:DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> DenseFeedForward<B> {
    /// Connect actual loaded projections and activation without copying parameters.
    /// IDs, ties, frozen settings and existing records remain the caller's own.
    pub fn from_projections(up: Linear<B>,gate: Option<Linear<B>>,down: Linear<B>,
        activation: Activation<B>,dropout: Dropout) -> Self {
        let [width,inner] = up.weight.val().dims();
        assert_eq!(down.weight.val().dims(),[inner,width],"feed-forward projection geometry differs");
        if let Some(gate) = &gate { assert_eq!(gate.weight.val().dims(),[width,inner],"gate/value projection geometry differs"); }
        Self {up,gate,down,activation,dropout}
    }

    /// Ordinary dense FFN or explicitly gated FFN, with no residual added here.
    pub fn forward<const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {
        let up = self.up.forward(input.clone());
        let value = if let Some(gate) = &self.gate {
            let activated = self.activation.forward(gate.forward(input));
            assert_eq!(activated.dims(),up.dims(),"gate activation must preserve intermediate geometry");
            activated * up
        } else { self.activation.forward(up) };
        self.down.forward(self.dropout.forward(value))
    }
}

/// Last-axis transformer normalization with explicit affine parameters.
#[derive(Config,Debug)]
pub enum DenseTransformerNormConfig {
    /// Standard centered LayerNorm, including its explicit optional bias.
    Layer(LayerNormConfig),
    /// Uncentered RMSNorm.
    Rms(RmsNormConfig),
}

/// Actual native normalization module, recorded and optimized normally.
#[derive(Module,Debug)]
pub enum DenseTransformerNorm<B: Backend> {
    /// Centered last-axis normalization.
    Layer(LayerNorm<B>),
    /// RMS last-axis normalization.
    Rms(RmsNorm<B>),
}

impl DenseTransformerNormConfig {
    fn width(&self) -> usize { match self { Self::Layer(config)=>config.d_model,Self::Rms(config)=>config.d_model } }

    /// Initialize the declared normalization without changing its width or epsilon.
    pub fn init<B: Backend>(&self,device: &B::Device) -> DenseTransformerNorm<B> {
        match self {
            Self::Layer(config) => {
                assert!(config.epsilon.is_finite() && config.epsilon > 0.,"invalid LayerNorm epsilon");
                DenseTransformerNorm::Layer(config.init(device))
            }
            Self::Rms(config) => {
                assert!(config.epsilon.is_finite() && config.epsilon > 0.,"invalid RMSNorm epsilon");
                DenseTransformerNorm::Rms(config.init(device))
            }
        }
    }
}

impl<B: Backend> DenseTransformerNorm<B> {
    /// Actual affine width of the loaded normalization module.
    pub fn width(&self) -> usize {
        match self { Self::Layer(layer)=>layer.gamma.val().dims()[0],Self::Rms(layer)=>layer.gamma.val().dims()[0] }
    }

    /// FP32 statistics for half storage; explicit F64 inputs retain F64 arithmetic.
    pub fn forward<const D: usize>(&self,input: Tensor<B,D>) -> Tensor<B,D> {
        let dtype = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
        match self {
            Self::Layer(layer)=>layer.forward_with_compute_dtype(input,dtype),
            Self::Rms(layer)=>layer.forward_with_compute_dtype(input,dtype),
        }
    }
}

/// Native transformer block without model-specific position/visibility inference.
#[derive(Config,Debug)]
pub struct DenseTransformerBlockConfig {
    /// Actual Q/K/V/output projection and grouped-head geometry.
    pub attention: GroupedQueryAttentionConfig,
    /// Dense or gated feed-forward geometry and selected activation.
    pub feed_forward: DenseFeedForwardConfig,
    /// Two independent normalization modules initialized with this configuration.
    pub normalization: DenseTransformerNormConfig,
    /// Pre-norm rather than post-norm residual ordering.
    #[config(default = true)]
    pub norm_first: bool,
    /// Dropout on each branch's output, before residual addition.
    #[config(default = 0.0)]
    pub residual_dropout: f64,
}

/// One trainable self-attention block for caller-declared dense/causal visibility.
#[derive(Module,Debug)]
pub struct DenseTransformerBlock<B: Backend> {
    /// Native MHA/GQA/MQA and optional trainable score bias supplied at forward.
    pub attention: GroupedQueryAttention<B>,
    /// Actual stateful ordinary/gated FFN.
    pub feed_forward: DenseFeedForward<B>,
    /// Independent attention normalization parameters.
    pub attention_norm: DenseTransformerNorm<B>,
    /// Independent feed-forward normalization parameters.
    pub feed_forward_norm: DenseTransformerNorm<B>,
    /// Residual branch dropout, separate from probability/intermediate dropout.
    pub residual_dropout: Dropout,
    /// Explicit pre/post norm ordering retained by the module configuration.
    pub norm_first: bool,
}

impl DenseTransformerBlockConfig {
    /// Initialize the exact declared block, rejecting incompatible widths.
    pub fn init<B: Backend>(&self,device: &B::Device) -> DenseTransformerBlock<B> {
        assert_eq!(self.attention.d_model,self.feed_forward.d_model,"attention/feed-forward residual widths differ");
        assert_eq!(self.normalization.width(),self.attention.d_model,"normalization/residual widths differ");
        assert!(self.residual_dropout.is_finite() && (0.0..=1.0).contains(&self.residual_dropout),"invalid residual dropout");
        DenseTransformerBlock {
            attention:self.attention.init(device),feed_forward:self.feed_forward.init(device),
            attention_norm:self.normalization.init(device),feed_forward_norm:self.normalization.init(device),
            residual_dropout:DropoutConfig::new(self.residual_dropout).init(),norm_first:self.norm_first,
        }
    }
}

impl<B: Backend> DenseTransformerBlock<B> {
    /// Forward with explicit masks and causal/window rules; no guessed positional transform.
    pub fn forward(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_with_positions(input,masks,options,|query,key|(query,key))
    }

    /// Transform actual projected Q/K heads, for caller-owned RoPE or other positions.
    /// The closure must retain batch/head/token/feature geometry and actual device/dtype.
    pub fn forward_with_positions<F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        self.forward_feed_forward(self.forward_attention_with_positions(input,masks,options,positions))
    }

    /// Attention/residual/normalization stage alone, for encoder-decoder composition.
    /// Feed-forward parameters are not applied until forward_feed_forward is called.
    pub fn forward_attention_with_positions<F>(&self,input: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        residual_branch(input,&self.attention_norm,&self.residual_dropout,self.norm_first,|source| {
        let (query,key,value) = self.attention.project(source.clone(),source.clone(),source);
        let query_shape = query.dims();
        let key_shape = key.dims();
        let (query,key) = positions(query,key);
        assert_eq!(query.dims(),query_shape,"query position transform changed geometry");
        assert_eq!(key.dims(),key_shape,"key position transform changed geometry");
        let branch = self.attention.forward_projected(query,key,value,masks,options);
        branch
        })
    }

    /// Feed-forward/residual/normalization stage using the actual existing weights.
    pub fn forward_feed_forward(&self,hidden: Tensor<B,3>) -> Tensor<B,3> {
        residual_branch(hidden,&self.feed_forward_norm,&self.residual_dropout,self.norm_first,
            |source|self.feed_forward.forward(source))
    }
}

pub(super) fn residual_branch<B: Backend,F,const D: usize>(input: Tensor<B,D>,norm: &DenseTransformerNorm<B>,
    dropout: &Dropout,norm_first: bool,branch: F) -> Tensor<B,D>
where F: FnOnce(Tensor<B,D>)->Tensor<B,D> {
    let source = if norm_first { norm.forward(input.clone()) } else { input.clone() };
    let output = input + dropout.forward(branch(source));
    if norm_first { output } else { norm.forward(output) }
}

pub(super) fn try_residual_branch<B:Backend,F,E,const D:usize>(input:Tensor<B,D>,norm:&DenseTransformerNorm<B>,
    dropout:&Dropout,norm_first:bool,branch:F) -> Result<Tensor<B,D>,E>
    where F:FnOnce(Tensor<B,D>)->Result<Tensor<B,D>,E> {
    let source=if norm_first {norm.forward(input.clone())} else {input.clone()};
    let output=input+dropout.forward(branch(source)?);
    Ok(if norm_first {output} else {norm.forward(output)})
}

/// Residual cross-attention on actual encoder memory, including nonmatching widths.
#[derive(Module,Debug)]
pub struct DenseCrossAttentionBlock<B: Backend> {
    /// Loaded Q/K/V/output weights; query and memory input widths may differ.
    pub attention: GroupedQueryAttention<B>,
    /// Query/residual normalization, applied in the explicitly selected order.
    pub query_norm: DenseTransformerNorm<B>,
    /// Optional independent normalization of encoder memory before K/V projection.
    pub memory_norm: Option<DenseTransformerNorm<B>>,
    /// Dropout after the output projection, before residual addition.
    pub residual_dropout: Dropout,
    /// Pre-normalize queries rather than normalize the residual result.
    pub norm_first: bool,
}

impl<B: Backend> DenseCrossAttentionBlock<B> {
    /// Connect actual weights and optional memory norm without changing their IDs.
    pub fn new(attention: GroupedQueryAttention<B>,query_norm: DenseTransformerNorm<B>,
        memory_norm: Option<DenseTransformerNorm<B>>,residual_dropout: Dropout,norm_first: bool) -> Self {
        assert_eq!(query_norm.width(),attention.query.weight.val().dims()[0],"cross query/norm widths differ");
        if let Some(norm) = &memory_norm {
            assert_eq!(norm.width(),attention.key.weight.val().dims()[0],"cross memory/norm widths differ");
        }
        assert!(residual_dropout.prob.is_finite() && (0.0..=1.0).contains(&residual_dropout.prob),"invalid cross residual dropout");
        Self {attention,query_norm,memory_norm,residual_dropout,norm_first}
    }

    /// Explicit memory/query masks; causal/window alignment is never assumed.
    pub fn forward(&self,input: Tensor<B,3>,memory: Tensor<B,3>,masks: DenseAttentionMask<B>,
        options: DenseAttentionOptions) -> Tensor<B,3> {
        self.forward_with_positions(input,memory,masks,options,|query,key|(query,key))
    }

    /// Transform projected query and encoder key positions independently of payloads.
    pub fn forward_with_positions<F>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,
        masks: DenseAttentionMask<B>,options: DenseAttentionOptions,positions: F) -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let source = if self.norm_first { self.query_norm.forward(input.clone()) } else { input.clone() };
        let memory = if let Some(norm) = &self.memory_norm { norm.forward(memory) } else { memory };
        let (query,key,value) = self.attention.project(source,memory.clone(),memory);
        let query_shape = query.dims();
        let key_shape = key.dims();
        let (query,key) = positions(query,key);
        assert_eq!(query.dims(),query_shape,"cross query position transform changed geometry");
        assert_eq!(key.dims(),key_shape,"cross memory position transform changed geometry");
        let branch = self.attention.forward_projected(query,key,value,masks,options);
        let hidden = input + self.residual_dropout.forward(branch);
        if self.norm_first { hidden } else { self.query_norm.forward(hidden) }
    }
}

/// Self-attention, encoder-memory cross-attention and FFN in that exact order.
#[derive(Module,Debug)]
pub struct DenseEncoderDecoderLayer<B: Backend> {
    /// Actual self-attention and final feed-forward stages, with their independent norms.
    pub backbone: DenseTransformerBlock<B>,
    /// Actual encoder-memory stage inserted before the backbone's feed-forward.
    pub cross_attention: DenseCrossAttentionBlock<B>,
}

impl<B: Backend> DenseEncoderDecoderLayer<B> {
    /// Connect supplied stages; no new random weights or implicit extra residuals.
    pub fn new(backbone: DenseTransformerBlock<B>,cross_attention: DenseCrossAttentionBlock<B>) -> Self {
        assert_eq!(backbone.attention.query.weight.val().dims()[0],
            cross_attention.attention.query.weight.val().dims()[0],"decoder residual widths differ");
        Self {backbone,cross_attention}
    }

    /// Forward real target and encoder memory with distinct actual visibility rules.
    pub fn forward(&self,input: Tensor<B,3>,memory: Tensor<B,3>,self_masks: DenseAttentionMask<B>,
        self_options: DenseAttentionOptions,memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions)
        -> Tensor<B,3> {
        self.forward_with_positions(input,memory,self_masks,self_options,memory_masks,memory_options,
            |query,key|(query,key),|query,key|(query,key))
    }

    /// Caller-owned self/cross positional transforms, without guessing shared offsets.
    pub fn forward_with_positions<F,G>(&self,input: Tensor<B,3>,memory: Tensor<B,3>,
        self_masks: DenseAttentionMask<B>,self_options: DenseAttentionOptions,
        memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions,self_positions: F,cross_positions: G)
        -> Tensor<B,3>
    where F: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>),
        G: FnOnce(Tensor<B,4>,Tensor<B,4>)->(Tensor<B,4>,Tensor<B,4>) {
        let hidden = self.backbone.forward_attention_with_positions(input,self_masks,self_options,self_positions);
        let hidden = self.cross_attention.forward_with_positions(hidden,memory,memory_masks,memory_options,cross_positions);
        self.backbone.forward_feed_forward(hidden)
    }
}

/// Ordered encoder-decoder layers sharing the caller's actual memory tensor.
#[derive(Module,Debug)]
pub struct DenseEncoderDecoderStack<B: Backend> {
    /// Decoder layers in the actual architecture's order.
    pub layers: Vec<DenseEncoderDecoderLayer<B>>,
}

impl<B: Backend> DenseEncoderDecoderStack<B> {
    /// Connect actual loaded layers without resetting parameters, optimizers or IDs.
    pub fn new(layers: Vec<DenseEncoderDecoderLayer<B>>) -> Self { Self {layers} }

    /// Apply explicitly shared self/cross visibility rules to each actual layer.
    pub fn forward(&self,mut input: Tensor<B,3>,memory: Tensor<B,3>,self_masks: DenseAttentionMask<B>,
        self_options: DenseAttentionOptions,memory_masks: DenseAttentionMask<B>,memory_options: DenseAttentionOptions)
        -> Tensor<B,3> {
        for layer in &self.layers {
            input = layer.forward(input,memory.clone(),self_masks.clone(),self_options,memory_masks.clone(),memory_options);
        }
        input
    }

    /// Explicit per-layer positions/masks/options with no architecture-specific defaults.
    pub fn forward_with<F>(&self,mut input: Tensor<B,3>,memory: Tensor<B,3>,mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&DenseEncoderDecoderLayer<B>,Tensor<B,3>,Tensor<B,3>)->Tensor<B,3> {
        for (index,block) in self.layers.iter().enumerate() { input = layer(index,block,input,memory.clone()); }
        input
    }
}

/// Ordered actual blocks; no embedding, head, residual extras or final norm inferred.
#[derive(Module,Debug)]
pub struct DenseTransformerStack<B: Backend> {
    /// Blocks in the exact caller-selected order.
    pub blocks: Vec<DenseTransformerBlock<B>>,
}

impl<B: Backend> DenseTransformerStack<B> {
    /// Connect loaded blocks without rebuilding their parameters or changing IDs.
    pub fn new(blocks: Vec<DenseTransformerBlock<B>>) -> Self { Self {blocks} }

    /// Shared explicit visibility/options, appropriate when all layers use that contract.
    pub fn forward(&self,mut input: Tensor<B,3>,masks: DenseAttentionMask<B>,options: DenseAttentionOptions) -> Tensor<B,3> {
        for block in &self.blocks { input = block.forward(input,masks.clone(),options); }
        input
    }

    /// Per-layer masks/options/position transforms, supplied by the actual architecture.
    pub fn forward_with<F>(&self,mut input: Tensor<B,3>,mut layer: F) -> Tensor<B,3>
    where F: FnMut(usize,&DenseTransformerBlock<B>,Tensor<B,3>)->Tensor<B,3> {
        for (index,block) in self.blocks.iter().enumerate() { input = layer(index,block,input); }
        input
    }
}
