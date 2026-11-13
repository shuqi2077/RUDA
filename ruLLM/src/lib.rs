//! Model loading, generation and scheduling on Ruda tensors and compute libraries.

/// Native inference plans and kernels shared with ruDNN, ruBLAS and ruRAND.
#[cfg(feature = "native-inference")]
pub mod native {
    pub use rublas::tensor_float_quant as quantization;
    pub use rudnn::inference::*;
    pub use rurand::tensor::sampling;
}

mod batch_kv;
mod batch_prefix;
mod continuous_batch;
mod generation;
mod huggingface;
mod llama;
mod prefix_transfer;
mod ruda_inference;
pub use prefix_transfer::{GpuPrefixBackend, PrefixShard, PrefixTensors};
#[cfg(feature = "tensor-parallel")]
mod parallel_batch;
pub use batch_kv::{DenseBatchCache, DenseBatchDelta, DenseBatchPrefix};
#[cfg(feature = "tensor-parallel")]
pub use parallel_batch::{ParallelBatchCache, ParallelBatchDelta, ParallelBatchPrefix};
mod causal_draft_batch;
pub use causal_draft_batch::{
    CausalBatchCache, CausalBatchDelta, CausalBatchModel, CausalBatchPrefix, CausalDraftBatch,
    CausalDraftBatchCache, CausalDraftBatchConfig, CausalDraftBatchPrefix, DenseCausalBatchModel,
    SpeculativeBatchModel,
};
mod deepseek2;
mod openai_server;
mod paged_attention;
mod paged_engine;
mod paged_kv;
mod qwen3;
mod qwen35;
pub mod training;
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::parallel::load_huggingface_mla_expert_parallel;
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::parallel::{
    DeepseekV2ExpertParallel, DeepseekV2ExpertParallelCache, DeepseekV2ExpertParallelCacheRecord,
    LoadedDeepseekV2ExpertParallel, load_huggingface_deepseek_v2_expert_parallel,
};
pub use deepseek2::{
    DeepseekMtp, DeepseekMtpCache, DeepseekMtpGenerationCache, DeepseekMtpGenerationCacheRecord,
    DeepseekMtpOutput, LoadedDeepseekMtp, load_huggingface_deepseek_mtp, load_huggingface_mla_mtp,
};
pub use deepseek2::{DeepseekMtpBatchCache, DeepseekMtpBatchPrefix};
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::{
    DeepseekMtpExpertParallel, DeepseekMtpExpertParallelCache,
    DeepseekMtpExpertParallelCacheRecord, LoadedDeepseekMtpExpertParallel,
    load_huggingface_deepseek_mtp_expert_parallel, load_huggingface_mla_mtp_expert_parallel,
};
pub use deepseek2::{
    DeepseekV2AttentionMode, DeepseekV2Cache, DeepseekV2CacheRecord, DeepseekV2Config,
    DeepseekV2Model, DeepseekV2Yarn, LoadedDeepseekV2, load_huggingface_deepseek_v2,
};
pub use deepseek2::{
    DeepseekV2Cache as MlaCache, DeepseekV2Config as MlaConfig, DeepseekV2Model as MlaModel,
    LoadedDeepseekV2 as LoadedMla,
};
pub use deepseek2::{
    KimiLinearAttentionConfig, KimiLinearBatchCache, KimiLinearBatchDelta, KimiLinearBatchPrefix,
    KimiLinearCache, KimiLinearCacheRecord, KimiLinearConfig, KimiLinearModel, LoadedKimiLinear,
    load_huggingface_kimi_linear,
};
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::{
    KimiLinearExpertParallel, KimiLinearExpertParallelCache, KimiLinearExpertParallelCacheRecord,
    LoadedKimiLinearExpertParallel, load_huggingface_kimi_linear_expert_parallel,
};
pub use deepseek2::{MlaFp8Backend, MlaFp8Projection, load_huggingface_mla};
#[cfg(feature = "tensor-parallel")]
pub use qwen3::parallel::{
    LoadedQwen3TensorParallel, Qwen3DraftTensorParallel, Qwen3TensorParallel,
    Qwen3TensorParallelCache, Qwen3TensorParallelCacheRecord,
    load_huggingface_qwen3_tensor_parallel,
};
pub use qwen3::{LoadedQwen3, Qwen3Config, Qwen3Model, Qwen3MoeConfig, load_huggingface_qwen3};
pub use qwen3::{Qwen3DraftBatchCache, Qwen3DraftBatchPrefix};
pub use qwen3::{
    Qwen3DraftCache, Qwen3DraftCacheRecord, Qwen3Drafted, load_huggingface_qwen3_draft,
};
pub mod disaggregated;
pub mod prefix_cache;
#[cfg(feature = "serving")]
pub mod serving;

pub use openai_server::{OpenAiServerConfig, serve_openai};
pub use paged_engine::{BatchStepReport, ContinuousBatchRunner, PagedLlamaEngine};
pub use training::{
    DpoTrainer, GrpoConfig, GrpoRollout, GrpoTrainer, PagedResponse, PostTrainingState, PpoConfig,
    PpoRollout, PpoTrainer, PreferencePair, ReferencePolicy, Response, ResponseRecord,
    RewardTrainer, ValueCritic,
};
pub use training::{
    LlamaLora, LoraConfig, LoraTargets, NativeLora, NativeSftModel, SftDataset, SftLoader,
    SftModel, SftTrainer, TrainConfig, export_huggingface_safetensors, load_huggingface_native_sft,
};

#[cfg(feature = "tensor-parallel")]
pub use qwen35::parallel::{
    LoadedQwen35TensorParallel, Qwen35TensorParallel, Qwen35TensorParallelCache,
    Qwen35TensorParallelCacheRecord, load_huggingface_qwen35_tensor_parallel,
    load_huggingface_qwen35_tensor_parallel_packed,
};
pub use qwen35::{
    LoadedQwen35Text, Qwen35Cache, Qwen35LayerType, Qwen35MoeConfig, Qwen35RopeConfig,
    Qwen35TextConfig, Qwen35TextModel, load_huggingface_qwen35_text,
    load_huggingface_qwen35_text_packed,
};
pub use qwen35::{
    LoadedQwen35Vision, Qwen35VisionConfig, Qwen35VisionModel, Qwen35VisionOutput,
    load_huggingface_qwen35_vision,
};
pub use qwen35::{Qwen35BatchCache, Qwen35BatchDelta};
pub use qwen35::{Qwen35CacheRecord, Qwen35LayerCacheState};
pub use qwen35::{
    Qwen35CachedMultimodalDecoder, Qwen35ImageConditioned, Qwen35MultimodalCacheRecord,
};
pub use qwen35::{Qwen35ImageProcessor, Qwen35PreparedImages, Qwen35RgbImage};
pub use qwen35::{
    Qwen35MultimodalCache, Qwen35MultimodalModel, load_huggingface_qwen35_multimodal,
};

pub use continuous_batch::{
    ActiveKvSwapConfig, ActiveKvSwapStats, BatchFence, BatchLaunchFailure, BatchSamplingState,
    CancelledGeneration, ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchOptions,
    ContinuousBatchScheduler, ContinuousBatchSnapshot, DeviceBatchExecutor, DeviceBatchModel,
    ExecutedBatch, FencedBatchScheduler, FinishedGeneration, KvAdmissionPolicy, PagedBatchSnapshot,
    PrefixBatchExecutor, PrefixBatchModel, PrefixBatchStep, ScheduledBatch, ScheduledBatchKind,
    ScheduledSequence, SpeculativeConfig,
};
pub use generation::*;
pub use huggingface::*;
pub use llama::awq::{AwqBackend, AwqLlamaForCausalLm};
pub use llama::gguf::{GgufBackend, GgufCausalLm, GgufLoadError, LoadedGguf, load_gguf};
#[cfg(feature = "tensor-parallel")]
pub use llama::parallel::{
    LlamaTensorParallel, LlamaTensorParallelCache, LlamaTensorParallelCacheRecord,
    LlamaTensorParallelRank, LoadedLlamaTensorParallel, load_huggingface_llama_tensor_parallel,
    load_huggingface_qwen2_tensor_parallel,
};
pub use llama::{
    LlamaAttention, LlamaConfig, LlamaConfigError, LlamaDecoderLayer, LlamaFeedForward,
    LlamaForCausalLm, LlamaKvCache, LlamaKvCacheRecord, LlamaLayerCache, PackedLlamaForCausalLm,
};
pub use paged_kv::*;
pub use ruda_store::gguf::{
    GgmlType, GgufEndian, GgufFile, GgufModel, GgufPackedTensor, GgufTensorInfo, GgufValue,
};

pub mod backends;
/// Optional backend adapters and host-side execution controls.
pub mod runtime;

/// Shared whole-generation calibration, enabled explicitly for native targets.
#[cfg(all(
    feature = "stack-autotune",
    any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "android"
    )
))]
pub mod autotune;

/// Shared device attention tensor transformations (candidate implementation).
pub mod gpu_inference;

/// Shared packed paged attention/MLA and local MoE building blocks.
/// Model adapters must provide their own projections, routing policy and cache ownership.
pub mod device_inference {
    pub use rudnn::moe::{
        GroupRoutingOptions, GroupedStrategy, RoutingOptions, SwiGluExperts, route_sigmoid_grouped,
    };
    pub use rudnn::paged_attention::{DevicePlan, HostPlan, PagedAttentionError};
}
