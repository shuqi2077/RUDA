use ruda_model::{config::Config,module::Module,tensor::{Bool,Int,Tensor,backend::Backend}};
use crate::{Linear,LinearConfig,Dropout,DropoutConfig,
    pool::{pool_sequence,SequencePooling,SequencePoolOutput}};
use super::{DenseTransformerNorm,DenseTransformerNormConfig};

/// Native sequence/token classification projection without model-specific labels.
#[derive(Config,Debug)]
pub struct TransformerHeadConfig {
    /// Actual hidden feature width.
    pub d_model: usize,
    /// Caller-declared output class count; no vocabulary/family inference.
    pub classes: usize,
    /// Bias in the output projection.
    #[config(default = true)]
    pub bias: bool,
    /// Optional normalization before dropout and the actual output projection.
    pub normalization: Option<DenseTransformerNormConfig>,
    /// Dropout before the actual projection, separate from backbone dropout.
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Same actual projection can classify token states or explicitly pooled sequences.
#[derive(Module,Debug)]
pub struct TransformerHead<B: Backend> {
    /// Actual trainable class projection.
    pub projection: Linear<B>,
    /// Optional independently trainable affine normalization.
    pub normalization: Option<DenseTransformerNorm<B>>,
    /// Backend-mode dropout before projection.
    pub dropout: Dropout,
}

/// Sequence logits together with real-token visibility for caller-owned loss masks.
#[derive(Clone,Debug)]
pub struct SequenceHeadOutput<B: Backend> {
    /// [batch,classes]. Bias remains part of the projection even on empty rows.
    pub logits: Tensor<B,2>,
    /// True exactly for rows containing actual visible tokens.
    pub valid_rows: Tensor<B,1,Bool>,
    /// Actual I64 visible-token counts from pooling, not supervised-label counts.
    pub token_counts: Tensor<B,1,Int>,
}

impl TransformerHeadConfig {
    /// Initialize the declared output projection/norm only, not an entire model.
    pub fn init<B: Backend>(&self,device: &B::Device) -> TransformerHead<B> {
        assert!(self.d_model > 0 && self.classes > 0,"head hidden/class dimensions must be positive");
        assert!(self.dropout.is_finite() && (0.0..=1.0).contains(&self.dropout),"invalid head dropout");
        let normalization = self.normalization.as_ref().map(|config|config.init(device));
        if let Some(norm) = &normalization { assert_eq!(norm.width(),self.d_model,"head norm/input widths differ"); }
        TransformerHead {projection:LinearConfig::new(self.d_model,self.classes).with_bias(self.bias).init(device),
            normalization,dropout:DropoutConfig::new(self.dropout).init()}
    }
}

impl<B: Backend> TransformerHead<B> {
    /// Connect actual loaded output and norm weights without reinitializing their IDs.
    pub fn from_projection(projection: Linear<B>,normalization: Option<DenseTransformerNorm<B>>,dropout: Dropout) -> Self {
        if let Some(norm) = &normalization {
            assert_eq!(norm.width(),projection.weight.val().dims()[0],"head norm/projection widths differ");
        }
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid head dropout");
        Self {projection,normalization,dropout}
    }

    /// Classify actual hidden states, retaining all leading dimensions.
    /// This does not invent target labels, probabilities or padding-loss masks.
    pub fn forward<const D: usize>(&self,hidden: Tensor<B,D>) -> Tensor<B,D> {
        let hidden = if let Some(norm) = &self.normalization { norm.forward(hidden) } else { hidden };
        self.projection.forward(self.dropout.forward(hidden))
    }

    /// Pool real token states with the caller's explicit policy, then project logits.
    pub fn forward_sequence(&self,hidden: Tensor<B,3>,visible: Tensor<B,2,Bool>,pooling: SequencePooling)
        -> SequenceHeadOutput<B> {
        self.forward_pooled(pool_sequence(hidden,visible,pooling))
    }

    /// Use an already pooled actual tensor and retain its visibility/count metadata.
    pub fn forward_pooled(&self,pooled: SequencePoolOutput<B>) -> SequenceHeadOutput<B> {
        SequenceHeadOutput {logits:self.forward(pooled.values),valid_rows:pooled.valid_rows,token_counts:pooled.token_counts}
    }
}
