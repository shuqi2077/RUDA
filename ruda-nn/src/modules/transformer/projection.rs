use core::{convert::Infallible,fmt};
use ruda_model::{module::{Module,ModuleDisplay},tensor::{FrozenAwqOps,FrozenNf4Ops,Tensor,backend::Backend}};
use crate::{Linear,LoRALinear,FrozenNf4Linear,Nf4LoRALinear,FrozenAwqLinear,AwqLoRALinear};
use super::{AwqTransformerProjection,AwqGroupedQueryAttention,AwqFeedForward,AwqTransformerBlock,
    AwqTransformerStack,AwqTransformerHead,AwqTransformerModel};

/// Backend-bound identity for the actual stored projection type.
/// The blanket implementation retains the original module value and public field layout.
pub trait TransformerProjectionStorage<B:Backend>:Module<B> {
    /// The original stored module, without a wrapper or extra persistent field.
    type Stored:Module<B>;
}
impl<B:Backend,P:Module<B>> TransformerProjectionStorage<B> for P {type Stored=P;}
/// Actual projection value while retaining the backend type in generic field metadata.
pub type BackendProjection<B,P> = <P as TransformerProjectionStorage<B>>::Stored;

/// Logical projection geometry independent of its execution extension or storage format.
pub trait TransformerProjectionShape<B:Backend>:Module<B>+ModuleDisplay {
    /// Actual logical `[input,output]` widths, without decoding packed parameters.
    fn dimensions(&self) -> [usize;2];
}

/// Actual caller-selected native projection. Implementations retain all leading axes
/// and return the original failure; no dense substitution or quantizer is implied.
pub trait TransformerProjection<B:Backend>:TransformerProjectionShape<B> {
    /// Actual original native projection error.
    type Error:fmt::Debug;
    /// Apply the original selected projection and its original derivative contract.
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error>;
}

impl<B:Backend> TransformerProjectionShape<B> for AwqTransformerProjection<B> {
    fn dimensions(&self) -> [usize;2] {self.dimensions()}
}
impl<B:FrozenAwqOps> TransformerProjection<B> for AwqTransformerProjection<B> {
    type Error=B::AwqError;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {self.forward(input)}
}
impl<B:Backend> TransformerProjectionShape<B> for FrozenAwqLinear<B> {
    fn dimensions(&self) -> [usize;2] {self.dimensions()}
}
impl<B:FrozenAwqOps> TransformerProjection<B> for FrozenAwqLinear<B> {
    type Error=B::AwqError;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {self.forward(input)}
}
impl<B:Backend> TransformerProjectionShape<B> for AwqLoRALinear<B> {
    fn dimensions(&self) -> [usize;2] {self.base.dimensions()}
}
impl<B:FrozenAwqOps> TransformerProjection<B> for AwqLoRALinear<B> {
    type Error=B::AwqError;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {self.forward(input)}
}
impl<B:Backend> TransformerProjectionShape<B> for Linear<B> {
    fn dimensions(&self) -> [usize;2] {self.weight.val().dims()}
}
impl<B:Backend> TransformerProjection<B> for Linear<B> {
    type Error=Infallible;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {Ok(self.forward(input))}
}
impl<B:Backend> TransformerProjectionShape<B> for LoRALinear<B> {
    fn dimensions(&self) -> [usize;2] {self.base.weight.val().dims()}
}
impl<B:Backend> TransformerProjection<B> for LoRALinear<B> {
    type Error=Infallible;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {Ok(self.forward(input))}
}
impl<B:Backend> TransformerProjectionShape<B> for FrozenNf4Linear<B> {
    fn dimensions(&self) -> [usize;2] {[self.input_features,self.output_features]}
}
impl<B:FrozenNf4Ops> TransformerProjection<B> for FrozenNf4Linear<B> {
    type Error=B::Nf4Error;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {self.forward(input)}
}
impl<B:Backend> TransformerProjectionShape<B> for Nf4LoRALinear<B> {
    fn dimensions(&self) -> [usize;2] {[self.base.input_features,self.base.output_features]}
}
impl<B:FrozenNf4Ops> TransformerProjection<B> for Nf4LoRALinear<B> {
    type Error=B::Nf4Error;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {self.forward(input)}
}

/// Explicit dense, dense-LoRA, original NF4 or NF4-LoRA projection.
/// Unlike a mixed AWQ/NF4 graph, this requires only the native NF4 extension.
#[derive(Module,Debug)]
pub enum Nf4TransformerProjection<B:Backend> {
    /// Actual original dense module, including its frozen/trainable flags.
    Dense(Linear<B>),
    /// Actual original floating-base LoRA module.
    LoRA(LoRALinear<B>),
    /// Original immutable high-nibble-first U8 base and FP32 scales/codebook.
    Nf4(FrozenNf4Linear<B>),
    /// Original packed base and independently trainable floating adapters.
    Nf4LoRA(Nf4LoRALinear<B>),
}
impl<B:Backend> TransformerProjectionShape<B> for Nf4TransformerProjection<B> {
    fn dimensions(&self) -> [usize;2] {
        match self {Self::Dense(layer)=>layer.dimensions(),Self::LoRA(layer)=>layer.dimensions(),
            Self::Nf4(layer)=>layer.dimensions(),Self::Nf4LoRA(layer)=>layer.dimensions()}
    }
}
impl<B:FrozenNf4Ops> TransformerProjection<B> for Nf4TransformerProjection<B> {
    type Error=B::Nf4Error;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {
        match self {Self::Dense(layer)=>Ok(layer.forward(input)),Self::LoRA(layer)=>Ok(layer.forward(input)),
            Self::Nf4(layer)=>layer.forward(input),Self::Nf4LoRA(layer)=>layer.forward(input)}
    }
}

/// Original failure retaining which real packed representation was executed.
#[derive(Debug)]
pub enum MixedProjectionError<A:fmt::Debug,N:fmt::Debug> {
    /// Native AWQ execution or derivative failure.
    Awq(A),
    /// Native NF4 execution or derivative failure.
    Nf4(N),
}
impl<A:fmt::Debug,N:fmt::Debug> fmt::Display for MixedProjectionError<A,N> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {
        match self {Self::Awq(error)=>write!(f,"AWQ projection: {error:?}"),Self::Nf4(error)=>write!(f,"NF4 projection: {error:?}")}
    }
}
impl<A:fmt::Debug,N:fmt::Debug> core::error::Error for MixedProjectionError<A,N> {}

/// Explicit per-role mixed native storage; selecting a role never converts its base.
#[derive(Module,Debug)]
pub enum MixedTransformerProjection<B:Backend> {
    /// Actual original floating values and trainability.
    Dense(Linear<B>),
    /// Actual original floating base and floating adapters.
    LoRA(LoRALinear<B>),
    /// Actual loaded AWQ words, zero points and original scale storage.
    Awq(FrozenAwqLinear<B>),
    /// Actual loaded AWQ base and independent floating adapters.
    AwqLoRA(AwqLoRALinear<B>),
    /// Actual loaded NF4 bytes and original FP32 quantization metadata.
    Nf4(FrozenNf4Linear<B>),
    /// Actual loaded NF4 base and independent floating adapters.
    Nf4LoRA(Nf4LoRALinear<B>),
}
impl<B:Backend> TransformerProjectionShape<B> for MixedTransformerProjection<B> {
    fn dimensions(&self) -> [usize;2] {
        match self {Self::Dense(layer)=>layer.dimensions(),Self::LoRA(layer)=>layer.dimensions(),
            Self::Awq(layer)=>layer.dimensions(),Self::AwqLoRA(layer)=>layer.base.dimensions(),
            Self::Nf4(layer)=>layer.dimensions(),Self::Nf4LoRA(layer)=>layer.dimensions()}
    }
}
impl<B:FrozenAwqOps+FrozenNf4Ops> TransformerProjection<B> for MixedTransformerProjection<B> {
    type Error=MixedProjectionError<B::AwqError,B::Nf4Error>;
    fn forward<const D:usize>(&self,input:Tensor<B,D>) -> Result<Tensor<B,D>,Self::Error> {
        match self {Self::Dense(layer)=>Ok(layer.forward(input)),Self::LoRA(layer)=>Ok(layer.forward(input)),
            Self::Awq(layer)=>layer.forward(input).map_err(MixedProjectionError::Awq),
            Self::AwqLoRA(layer)=>layer.forward(input).map_err(MixedProjectionError::Awq),
            Self::Nf4(layer)=>layer.forward(input).map_err(MixedProjectionError::Nf4),
            Self::Nf4LoRA(layer)=>layer.forward(input).map_err(MixedProjectionError::Nf4)}
    }
}

/// Shared native GQA/MQA/self/cross attention graph over explicit projection storage.
pub type ProjectedGroupedQueryAttention<B,P> = AwqGroupedQueryAttention<B,P>;
/// Shared ordinary/gated FFN graph over explicit projection storage.
pub type ProjectedFeedForward<B,P> = AwqFeedForward<B,P>;
/// Shared native dense/packed/cached transformer block graph.
pub type ProjectedTransformerBlock<B,P> = AwqTransformerBlock<B,P>;
/// Shared original ordered block graph with native cache topology.
pub type ProjectedTransformerStack<B,P> = AwqTransformerStack<B,P>;
/// Shared native normalized/dropout output head.
pub type ProjectedTransformerHead<B,P> = AwqTransformerHead<B,P>;
/// Shared complete original embeddings/backbone/final norm/head graph.
pub type ProjectedTransformerModel<B,P> = AwqTransformerModel<B,P>;
/// Native NF4 graph requiring no AWQ backend capability.
pub type Nf4TransformerModel<B> = ProjectedTransformerModel<B,Nf4TransformerProjection<B>>;
/// Native independently selected dense/AWQ/NF4 graph.
pub type MixedTransformerModel<B> = ProjectedTransformerModel<B,MixedTransformerProjection<B>>;
