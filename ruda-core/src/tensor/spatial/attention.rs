/// Options for attention.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AttentionModuleOptions {
    /// Custom scale factor applied to QK^T. When `None`, defaults to `1/sqrt(head_dim)`.
    pub scale: Option<f64>,

    /// Soft capping applied before softmax: `softcap * tanh(scores / softcap)`.
    /// Used by Gemma-2 and similar models. Must be positive when set.
    pub softcap: Option<f64>,

    /// When `true`, applies causal (autoregressive) masking so that each query position
    /// can only attend to key positions at or before it. This is more efficient than
    /// passing an explicit lower-triangular bool mask because backends can use optimized
    /// kernel paths (e.g. flash attention with causal mode).
    pub is_causal: bool,
}
