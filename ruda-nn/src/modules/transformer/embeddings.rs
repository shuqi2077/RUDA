use ruda_model::{config::Config,module::Module,tensor::{FloatDType,Int,Tensor,backend::Backend}};
use crate::{Embedding,EmbeddingConfig,Dropout,DropoutConfig};
use super::{DenseTransformerNorm,DenseTransformerNormConfig};

/// Explicit token/learned-position/type tables for a native transformer input.
#[derive(Config,Debug)]
pub struct TransformerEmbeddingsConfig {
    /// The actual tokenizer vocabulary and hidden width.
    pub token: EmbeddingConfig,
    /// Optional actual learned positions; absent for caller-owned RoPE, for example.
    pub position: Option<EmbeddingConfig>,
    /// Optional actual segment/type vocabulary; no automatic all-zero IDs.
    pub token_type: Option<EmbeddingConfig>,
    /// Optional independently trainable normalization after adding table outputs.
    pub normalization: Option<DenseTransformerNormConfig>,
    /// Backend-mode dropout after the optional normalization.
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Native transformer input tables with explicit per-token metadata.
#[derive(Module,Debug)]
pub struct TransformerEmbeddings<B: Backend> {
    /// Actual token embeddings, retained as their existing parameter identities.
    pub token: Embedding<B>,
    /// Actual learned positions, only when selected by the architecture.
    pub position: Option<Embedding<B>>,
    /// Actual type/segment embedding table, only when selected by the architecture.
    pub token_type: Option<Embedding<B>>,
    /// Normalization on the combined embeddings.
    pub normalization: Option<DenseTransformerNorm<B>>,
    /// Dropout on the actual combined/normalized input states.
    pub dropout: Dropout,
}

impl TransformerEmbeddingsConfig {
    /// Initialize only declared tables; optional position/type tables are not inferred.
    pub fn init<B: Backend>(&self,device: &B::Device) -> TransformerEmbeddings<B> {
        assert!(self.token.n_embedding > 0 && self.token.d_model > 0,"token embedding geometry must be positive");
        for config in [&self.position,&self.token_type].into_iter().flatten() {
            assert!(config.n_embedding > 0,"embedding table size must be positive");
            assert_eq!(config.d_model,self.token.d_model,"transformer embedding widths differ");
        }
        TransformerEmbeddings::from_tables(self.token.init(device),
            self.position.as_ref().map(|config|config.init(device)),
            self.token_type.as_ref().map(|config|config.init(device)),
            self.normalization.as_ref().map(|config|config.init(device)),DropoutConfig::new(self.dropout).init())
    }
}

impl<B: Backend> TransformerEmbeddings<B> {
    /// Connect actual loaded tables/norm, preserving ties, IDs and frozen settings.
    pub fn from_tables(token: Embedding<B>,position: Option<Embedding<B>>,token_type: Option<Embedding<B>>,
        normalization: Option<DenseTransformerNorm<B>>,dropout: Dropout) -> Self {
        let [vocabulary,width] = token.weight.val().dims();
        assert!(vocabulary > 0 && width > 0,"token embedding geometry must be positive");
        for table in [&position,&token_type].into_iter().flatten() {
            let [rows,features] = table.weight.val().dims();
            assert!(rows > 0 && features == width,"actual transformer table geometry differs");
        }
        if let Some(norm) = &normalization { assert_eq!(norm.width(),width,"embedding/norm widths differ"); }
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid embedding dropout");
        Self {token,position,token_type,normalization,dropout}
    }

    /// Read actual table rows in their existing storage, then sum/norm/dropout.
    /// Present tables require supplied IDs of the same actual [batch,tokens] shape.
    /// No pad ID, token type, sequence offset or positional scheme is guessed.
    pub fn forward(&self,input_ids: Tensor<B,2,Int>,position_ids: Option<Tensor<B,2,Int>>,
        token_type_ids: Option<Tensor<B,2,Int>>) -> Tensor<B,3> {
        self.forward_impl(input_ids,position_ids,token_type_ids,None)
    }

    /// Explicit arithmetic and output storage for mixed-storage input tables.
    /// Cast only looked-up rows, retaining derivatives without a full-table shadow.
    pub fn forward_with_compute_dtype(&self,input_ids: Tensor<B,2,Int>,position_ids: Option<Tensor<B,2,Int>>,
        token_type_ids: Option<Tensor<B,2,Int>>,compute: FloatDType,output: FloatDType) -> Tensor<B,3> {
        self.forward_impl(input_ids,position_ids,token_type_ids,Some(compute)).cast(output)
    }

    fn forward_impl(&self,input_ids: Tensor<B,2,Int>,position_ids: Option<Tensor<B,2,Int>>,
        token_type_ids: Option<Tensor<B,2,Int>>,compute: Option<FloatDType>) -> Tensor<B,3> {
        let geometry = input_ids.dims();
        let device = input_ids.device();
        let token_weight = self.token.weight.val();
        assert_eq!(token_weight.device(),device,"token IDs and embedding table must share a device");
        let storage = token_weight.dtype();
        // Metadata checks precede table launches, including optional-ID agreement.
        for (table,ids) in [(&self.position,&position_ids),(&self.token_type,&token_type_ids)] {
            assert_eq!(table.is_some(),ids.is_some(),"optional transformer table/ID presence differs");
            if let (Some(table),Some(ids)) = (table,ids) {
                assert_eq!(ids.dims(),geometry,"embedding metadata differs from token geometry");
                assert_eq!(ids.device(),device,"all transformer embedding IDs must share a device");
                let weight = table.weight.val();
                assert_eq!(weight.device(),device,"transformer input tables must share the token device");
                if compute.is_none() { assert_eq!(weight.dtype(),storage,"mixed table storage requires an explicit compute dtype"); }
            }
        }
        let hidden = ruda_model::tensor::module::embedding(token_weight,input_ids);
        let mut hidden = if let Some(dtype) = compute { hidden.cast(dtype) } else { hidden };
        for (table,ids) in [(&self.position,position_ids),(&self.token_type,token_type_ids)] {
            if let (Some(table),Some(ids)) = (table,ids) {
                let weight = table.weight.val();
                let rows = ruda_model::tensor::module::embedding(weight,ids);
                let rows = if let Some(dtype) = compute { rows.cast(dtype) } else { rows };
                hidden = hidden + rows;
            }
        }
        if let Some(norm) = &self.normalization { hidden = norm.forward(hidden); }
        self.dropout.forward(hidden)
    }
}
