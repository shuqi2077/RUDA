//! Model loading, generation and scheduling on Ruda tensors and compute libraries.

mod ruda_inference;
mod generation;
mod huggingface;
mod llama;
mod continuous_batch;
mod paged_kv;
mod qwen35;

pub use qwen35::{Qwen35TextConfig, Qwen35TextModel, Qwen35Cache, Qwen35LayerType, Qwen35RopeConfig, LoadedQwen35Text, load_huggingface_qwen35_text};
pub use qwen35::{LoadedQwen35Vision, Qwen35VisionConfig, Qwen35VisionModel, Qwen35VisionOutput, load_huggingface_qwen35_vision};
pub use qwen35::{Qwen35MultimodalCache, Qwen35MultimodalModel, load_huggingface_qwen35_multimodal};
pub use qwen35::{Qwen35ImageProcessor, Qwen35PreparedImages, Qwen35RgbImage};

pub use generation::*;
pub use huggingface::*;
pub use llama::{
    LlamaAttention, LlamaConfig, LlamaConfigError, LlamaDecoderLayer, LlamaFeedForward,
    LlamaForCausalLm, LlamaKvCache, LlamaLayerCache, PackedLlamaForCausalLm,
};
pub use llama::awq::{AwqBackend, AwqLlamaForCausalLm};
pub use continuous_batch::{
    ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchScheduler, ContinuousBatchSnapshot,
    FinishedGeneration, ScheduledBatch, ScheduledBatchKind, ScheduledSequence,
    CancelledGeneration, ContinuousBatchOptions, KvAdmissionPolicy,
    BatchFence, BatchLaunchFailure, FencedBatchScheduler,
};
pub use paged_kv::*;

/// Optional backend adapters and host-side execution controls.
pub mod runtime;
pub mod backends;

/// Shared whole-generation calibration, enabled explicitly for native targets.
#[cfg(all(feature = "stack-autotune", any(target_os = "linux", target_os = "windows", target_os = "macos", target_os = "android")))]
pub mod autotune;
