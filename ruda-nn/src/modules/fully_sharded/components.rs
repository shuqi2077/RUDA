use super::*;
use crate::{activation::Activation,attention::GroupedQueryAttention,
    transformer::{AdaptedProjection,AdaptedGroupedQueryAttention,AdaptedFeedForward,DenseFeedForward,DenseTransformerNorm}};

/// Original dense projection or independent native base/A/B storage, all locally sharded.
#[derive(Module,Debug)]
pub enum FullyShardedAdaptedProjection<B:Backend> {
    /// Original dense matrix and its actual optional bias.
    Dense(FullyShardedLinear<B>),
    /// Original base and adapters, without an implicit merge or new trainable flags.
    LoRA(FullyShardedLoRALinear<B>),
}

/// Original activation with every actual trainable parameter locally sharded.
#[derive(Module,Debug)]
pub enum FullyShardedActivation<B:Backend> {
    /// Parameter-free original activation, preserving all native scalar/approximation options.
    Stateless(Activation<B>),
    /// Original PReLU affine leaf and its native recorded initialization metadata.
    PRelu { alpha:ShardedParameter<B>,alpha_value:f64 },
    /// Original two independently loaded native SwiGLU projections.
    SwiGlu { inner:FullyShardedLinear<B>,outer:FullyShardedLinear<B> },
}

/// Actual last-axis normalization kind, including optional LayerNorm bias.
#[derive(Module,Debug)]
pub enum FullyShardedTransformerNorm<B:Backend> {
    /// Original centered normalization and epsilon.
    Layer(FullyShardedLayerNorm<B>),
    /// Original RMS normalization and epsilon.
    Rms(FullyShardedRmsNorm<B>),
}

/// Actual GQA/MQA/self/cross projections and head geometry with only local persistent parameters.
#[derive(Module,Debug)]
pub struct FullyShardedGroupedQueryAttention<B:Backend> {
    /// Original query projection, including its actual adapter choice.
    pub query:FullyShardedAdaptedProjection<B>,
    /// Original memory key projection; source width may differ from query width.
    pub key:FullyShardedAdaptedProjection<B>,
    /// Original memory value projection.
    pub value:FullyShardedAdaptedProjection<B>,
    /// Original context-to-residual projection.
    pub output:FullyShardedAdaptedProjection<B>,
    /// Original attention probability dropout.
    pub dropout:crate::Dropout,
    /// Actual original query head count.
    pub query_heads:usize,
    /// Actual original shared KV head count.
    pub kv_heads:usize,
    /// Original per-head feature width.
    pub head_dimension:usize,
}

/// Original ordinary/gated native FFN, including stateful activation parameters.
#[derive(Module,Debug)]
pub struct FullyShardedFeedForward<B:Backend> {
    /// Actual value projection.
    pub up:FullyShardedAdaptedProjection<B>,
    /// Actual optional gate; an absent gate remains absent.
    pub gate:Option<FullyShardedAdaptedProjection<B>>,
    /// Actual final projection.
    pub down:FullyShardedAdaptedProjection<B>,
    /// Original native activation, including locally sharded PReLU/SwiGLU weights.
    pub activation:FullyShardedActivation<B>,
    /// Original intermediate dropout.
    pub dropout:crate::Dropout,
}

impl<B:Backend> ShardingContext<B> {
    /// Partition the actual loaded dense/LoRA choice, not a guessed projection-role selection.
    pub fn adapted_projection(&mut self,projection:AdaptedProjection<B>) -> FullyShardedAdaptedProjection<B> {
        match projection {
            AdaptedProjection::Dense(layer)=>FullyShardedAdaptedProjection::Dense(self.linear(layer)),
            AdaptedProjection::LoRA(layer)=>FullyShardedAdaptedProjection::LoRA(self.lora(layer)),
        }
    }

    /// Preserve the exact original activation and shard every actually parameterized variant.
    pub fn activation(&mut self,activation:Activation<B>) -> FullyShardedActivation<B> {
        match activation {
            Activation::PRelu(layer)=>FullyShardedActivation::PRelu {alpha:self.parameter(layer.alpha),alpha_value:layer.alpha_value},
            Activation::SwiGlu(layer)=>FullyShardedActivation::SwiGlu {inner:self.linear(layer.linear_inner),outer:self.linear(layer.linear_outer)},
            other @ (Activation::Gelu(_) | Activation::Relu(_) | Activation::LeakyRelu(_) | Activation::Selu(_)
                | Activation::Sigmoid(_) | Activation::Tanh(_) | Activation::HardSigmoid(_) | Activation::HardSwish(_)
                | Activation::Softplus(_) | Activation::Softsign(_) | Activation::Elu(_) | Activation::Celu(_)
                | Activation::ThresholdedRelu(_) | Activation::HardShrink(_) | Activation::SoftShrink(_) | Activation::Shrink(_)
                | Activation::Silu(_))=>FullyShardedActivation::Stateless(other),
        }
    }

    /// Preserve the original norm kind, affine ties, optional bias and exact epsilon.
    pub fn normalization(&mut self,norm:DenseTransformerNorm<B>) -> FullyShardedTransformerNorm<B> {
        match norm {
            DenseTransformerNorm::Layer(layer)=>FullyShardedTransformerNorm::Layer(self.layer_norm(layer)),
            DenseTransformerNorm::Rms(layer)=>FullyShardedTransformerNorm::Rms(self.rms_norm(layer)),
        }
    }

    /// Partition loaded dense attention without adding adapters or changing head geometry.
    pub fn grouped_attention(&mut self,attention:GroupedQueryAttention<B>) -> FullyShardedGroupedQueryAttention<B> {
        self.adapted_attention(AdaptedGroupedQueryAttention {
            query:AdaptedProjection::Dense(attention.query),key:AdaptedProjection::Dense(attention.key),
            value:AdaptedProjection::Dense(attention.value),output:AdaptedProjection::Dense(attention.output),
            dropout:attention.dropout,query_heads:attention.query_heads,kv_heads:attention.kv_heads,head_dimension:attention.head_dimension,
        })
    }

    /// Partition actual selected attention base/A/B leaves using one shared alias context.
    pub fn adapted_attention(&mut self,attention:AdaptedGroupedQueryAttention<B>) -> FullyShardedGroupedQueryAttention<B> {
        FullyShardedGroupedQueryAttention {
            query:self.adapted_projection(attention.query),key:self.adapted_projection(attention.key),
            value:self.adapted_projection(attention.value),output:self.adapted_projection(attention.output),
            dropout:attention.dropout,query_heads:attention.query_heads,kv_heads:attention.kv_heads,head_dimension:attention.head_dimension,
        }
    }

    /// Partition loaded dense ordinary/gated FFN with its actual existing activation parameters.
    pub fn feed_forward(&mut self,feed:DenseFeedForward<B>) -> FullyShardedFeedForward<B> {
        self.adapted_feed_forward(AdaptedFeedForward {up:AdaptedProjection::Dense(feed.up),
            gate:feed.gate.map(AdaptedProjection::Dense),down:AdaptedProjection::Dense(feed.down),activation:feed.activation,dropout:feed.dropout})
    }

    /// Partition each original dense/adapter FFN role and retain the same actual activation.
    pub fn adapted_feed_forward(&mut self,feed:AdaptedFeedForward<B>) -> FullyShardedFeedForward<B> {
        FullyShardedFeedForward {up:self.adapted_projection(feed.up),gate:feed.gate.map(|gate|self.adapted_projection(gate)),
            down:self.adapted_projection(feed.down),activation:self.activation(feed.activation),dropout:feed.dropout}
    }
}

macro_rules! gathered_components {
    ($backend:ty,[$($generics:tt)*],$gather:ident) => {
        impl<$($generics)*> FullyShardedAdaptedProjection<$backend> {
            /// Transient original projection choice over actual gathered values; no merge or new leaf initialization.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<AdaptedProjection<$backend>,C::Error> {
                match self {
                    Self::Dense(layer)=>layer.$gather(communicator).map(AdaptedProjection::Dense),
                    Self::LoRA(layer)=>layer.$gather(communicator).map(AdaptedProjection::LoRA),
                }
            }
        }

        impl<$($generics)*> FullyShardedActivation<$backend> {
            /// Gather actual parameterized activation values; parameter-free native choices stay unchanged.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<Activation<$backend>,C::Error> {
                Ok(match self {
                    Self::Stateless(layer)=>{
                        assert!(!matches!(layer,Activation::PRelu(_)|Activation::SwiGlu(_)),"parameterized activation must use local shard storage");
                        layer.clone()
                    }
                    Self::PRelu {alpha,alpha_value}=>Activation::PRelu(crate::activation::PRelu {
                        alpha:Param::initialized(alpha.local.id,alpha.$gather::<C,1>(communicator)?),alpha_value:*alpha_value,
                    }),
                    Self::SwiGlu {inner,outer}=>Activation::SwiGlu(crate::activation::SwiGlu {
                        linear_inner:inner.$gather(communicator.clone())?,linear_outer:outer.$gather(communicator)?,
                    }),
                })
            }
        }

        impl<$($generics)*> FullyShardedTransformerNorm<$backend> {
            /// Transient actual native norm; its original statistics/epsilon implementation is reused.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<DenseTransformerNorm<$backend>,C::Error> {
                match self {
                    Self::Layer(layer)=>layer.$gather(communicator).map(DenseTransformerNorm::Layer),
                    Self::Rms(layer)=>layer.$gather(communicator).map(DenseTransformerNorm::Rms),
                }
            }
        }

        impl<$($generics)*> FullyShardedGroupedQueryAttention<$backend> {
            /// Materialize only this original attention's real projection values, retaining local persistent storage.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<AdaptedGroupedQueryAttention<$backend>,C::Error> {
                Ok(AdaptedGroupedQueryAttention {query:self.query.$gather(communicator.clone())?,key:self.key.$gather(communicator.clone())?,
                    value:self.value.$gather(communicator.clone())?,output:self.output.$gather(communicator)?,dropout:self.dropout.clone(),
                    query_heads:self.query_heads,kv_heads:self.kv_heads,head_dimension:self.head_dimension})
            }
        }

        impl<$($generics)*> FullyShardedFeedForward<$backend> {
            /// Transient actual native FFN values; original gated/ungated expression and activation are reused.
            pub fn $gather<C:BroadcastTensorCollective<B>>(&self,communicator:C) -> Result<AdaptedFeedForward<$backend>,C::Error> {
                Ok(AdaptedFeedForward {up:self.up.$gather(communicator.clone())?,
                    gate:self.gate.as_ref().map(|gate|gate.$gather(communicator.clone())).transpose()?,
                    down:self.down.$gather(communicator.clone())?,activation:self.activation.$gather(communicator)?,dropout:self.dropout.clone()})
            }
        }
    };
}
gathered_components!(B,[B:Backend],gather_inference);
gathered_components!(Autodiff<B,S>,[B:Backend,S:CheckpointStrategy],gather);

impl<B:Backend,S:CheckpointStrategy> FullyShardedFeedForward<Autodiff<B,S>> {
    /// Original native FFN expression with derivatives to each actual local sharded parameter.
    pub fn forward<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<Autodiff<B,S>,D>,communicator:C)
        -> Result<Tensor<Autodiff<B,S>,D>,C::Error> {
        Ok(self.gather(communicator)?.forward(input))
    }
}

impl<B:Backend> FullyShardedFeedForward<B> {
    /// Native inference through the exact original FFN and actual selected adapter/activation choices.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C)
        -> Result<Tensor<B,D>,C::Error> {
        Ok(self.gather_inference(communicator)?.forward(input))
    }
}
