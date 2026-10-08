/// Attention module
pub mod attention;

/// Cache module
pub mod cache;

/// Convolution module
pub mod conv;

/// Pooling module
pub mod pool;

/// Transformer module
pub mod transformer;

/// Interpolate module
pub mod interpolate;

mod dropout;
mod embedding;
mod linear;
mod lora;
mod lora_record;
mod awq;
mod nf4;
mod nf4_experts;
mod packed_experts;
mod expert_lora;
mod floating_expert_lora;
mod mixed_expert_lora;
mod expert_lora_record;
mod moe;
/// Explicit tensor-parallel linear projections for replicated-loss model partitions.
#[cfg(feature = "tensor-parallel")]
pub mod tensor_parallel;
/// Native cross-rank routed experts with explicit global ownership and transport.
#[cfg(feature = "tensor-parallel")]
pub mod expert_parallel;
/// Element-sharded parameters with differentiable gather/reduce-scatter.
#[cfg(feature = "tensor-parallel")]
pub mod fully_sharded;
/// Joint data-sharded storage and tensor-parallel native projections.
#[cfg(feature = "tensor-parallel")]
pub mod hybrid_sharded;
/// Manifold-constrained hyper-connections and residual mixing.
pub mod mhc;
#[cfg(feature = "sparse")]
mod sparse_linear;
mod noise;
mod pos_encoding;
mod rnn;
mod rope_encoding;
mod unfold;

pub mod norm;
pub use norm::{batch::*, group::*, instance::*, layer::*, local_response::*, rms::*};

pub use dropout::*;
pub use embedding::*;
pub use linear::*;
pub use lora::*;
pub use lora_record::*;
pub use awq::*;
pub use nf4::*;
pub use nf4_experts::*;
pub use packed_experts::*;
pub use expert_lora::*;
pub use floating_expert_lora::*;
pub use mixed_expert_lora::*;
pub use expert_lora_record::*;
pub use moe::*;
pub use mhc::*;
#[cfg(feature = "sparse")]
pub use sparse_linear::*;
pub use noise::*;
pub use pos_encoding::*;
pub use rnn::*;
pub use rope_encoding::*;
pub use unfold::*;
