# ruDNN User Guide

[Compute libraries](README.md) · [ruBLAS](rublas.md) · [Tensors and frameworks](../tensor-framework.md) · [中文](../../zh/libraries/rudnn.md) | [日本語](../../ja/libraries/rudnn.md) | [Deutsch](../../de/libraries/rudnn.md) | [Русский](../../ru/libraries/rudnn.md)

## 1. Overview and features

ruDNN provides neural network operations. The Cargo package is `ruDNN` and the Rust crate is `rudnn`.

| Feature | Operations |
| --- | --- |
| `tensor-attention` | Tensor attention |
| `tensor-convolution` | Tensor convolution |
| `pooling`, `interpolation` | Pooling and interpolation |
| `grid-sample`, `ctc` | Grid sampling and CTC |
| `tensor-moe` | Device MoE routing and expert computation |
| `tensor-normalization` | General-purpose device normalization for model layers |
| `tensor-gated-delta` | Gated-delta computation for hybrid architectures such as Qwen3.5 |

Attention and convolution each have corresponding autotune features. See [Cargo.toml](../../../ruDNN/Cargo.toml) and [module exports](../../../ruDNN/src/lib.rs).

## 2. Attention and convolution

### Attention

`rudnn::attention::tensor::attention` takes query, key, value, optional mask, optional attn_bias, AttentionModuleOptions, and AttentionStrategy. It returns a device tensor or AttentionSetupError.

Strategies include FlashBlackboxAccelerated, FlashUnit, Fallback, and Autotune when its feature is enabled. The default is Fallback without autotuning and Autotune with it. Fallback uses multiple kernels on the same device, not a CPU backend.

Match layout, mask, and precision to the selected strategy. See the [attention interface](../../../ruDNN/src/attention/tensor/base.rs).

### Convolution

`rudnn::convolution::tensor::conv_forward` takes input, weight, optional bias, ConvOptions<N>, and ConvStrategy. It returns a device tensor or ConvSetupError. It converts inputs to channels-last layout for execution and converts the output back. `conv_forward_nhwc` uses channels-last layout directly.

Strategies include Direct, ImplicitGemm, and optional Autotune. The entry point uses Direct for three-dimensional F32 convolution. Grouped convolution also uses Direct when ImplicitGemm is selected. The strategy parameter is therefore not always a strict request to retain one algorithm.

The same module provides conv_data_backward and conv_weight_backward. Check shape, options, and execution requirements for each direction. See the [convolution interface](../../../ruDNN/src/convolution/tensor/base.rs).

## 3. MoE workflow

`rudnn::moe` provides this local computation sequence:

Input logits → softmax/top-k → compact dispatch → SwiGLU experts → weighted combine.

| API | Purpose |
| --- | --- |
| `route(logits, RoutingOptions)` | Creates a RoutingPlan |
| `RoutingPlan::expert_indices()`, `weights()` | Accesses selected experts and weights |
| `RoutingPlan::dispatch(input)` | Dispatches tokens by expert |
| `SwiGluExperts::new(gate, up, down)` | Creates bias-free expert weights |
| `SwiGluExperts::forward_dispatched` | Computes dispatched tokens |
| `DispatchedTokens::combine` | Restores token order and combines weighted results |
| `SwiGluExperts::forward(input, logits, options)` | Executes the complete local forward sequence |

These interfaces use `RudaTensor<R>`. Computation entry points return `Result` with `MoeError` for invalid shapes, dtypes, or other arguments.

## 4. Routing contract

Logits have shape [T, E] and use non-quantized F32, F16, or BF16. `RoutingOptions` contains `top_k: usize` and `renormalize: bool`, with 1 ≤ top_k ≤ E.

Softmax computes over all experts in FP32 before top-k selection. Ties favor lower expert IDs. With renormalize enabled, selected weights are renormalized, then cast to the logits dtype. Rows containing NaN, positive infinity, or only negative infinity retain NaN weights rather than using a uniform distribution.

Selected expert indices and weights both have shape [T, top_k]. Indices use U32.

## 5. Expert weights and dispatch

Input tokens have shape [T, H]; gate and up have shape [E, I, H]; down has shape [E, H, I]. Weights must share a floating-point dtype and device. Expert counts and dimensions must match routing and input.

Dispatch does not drop tokens to enforce capacity. Atomic assignment order within an expert is not fixed; a saved mapping restores token order. The forward entry point takes precomputed logits rather than performing the model's gate projection or loading weight files.

Source: [routing](../../../ruDNN/src/moe/routing.rs), [dispatch and combine](../../../ruDNN/src/moe/dispatch.rs), [experts](../../../ruDNN/src/moe/experts.rs), and [tests](../../../ruDNN/src/moe/tests.rs).

## 6. Call MoE

Enable `tensor-moe` on your `ruDNN` dependency. Prepare device tensors with the layouts above, then create expert weights and run the forward operation:

```rust
use rudnn::moe::{RoutingOptions, SwiGluExperts};

let experts = SwiGluExperts::new(gate, up, down)?;
let options = RoutingOptions {
    top_k: 2,
    renormalize: true,
};
let output = experts.forward(input, logits, options)?;
```

This selects two experts per token, so `E` must be at least 2. `input` has shape `[T, H]`, `logits` has shape `[T, E]`, and `output` has shape `[T, H]`. Reuse `experts` across subsequent input batches without recreating the weight object.

To inspect routing or insert your own processing between stages, call them separately:

```rust
use rudnn::moe::route;

let routing = route(logits, options)?;
let selected_experts = routing.expert_indices();
let selected_weights = routing.weights();
let dispatched = routing.dispatch(input)?;
let expert_output = experts.forward_dispatched(&dispatched)?;
let output = dispatched.combine(expert_output)?;
```

These are alternative forms; the second starts with a fresh batch of `input` and `logits`. `combine` uses the dispatch mapping to restore token order and merges expert outputs using routing weights.

Shape, dtype, or device mismatches return `MoeError`. To read results on the host, use `ruda_kernel::tensor::readback::into_data_sync(output)`, which waits for device results and returns `TensorData`.

## 7. LayerNorm, RMSNorm, and Softmax

Enable `tensor-normalization` and import these functions from `rudnn::normalization`. All operate on the final input axis and preserve shape:

| Function | Input | Parameters |
| --- | --- | --- |
| `layer_norm(input, gamma, beta, epsilon)` | F32/F16/BF16 | F32 vector `gamma`, optional F32 vector `beta` |
| `rms_norm(input, gamma, epsilon)` | F32/F16/BF16 | F32 vector `gamma` |
| `softmax_last_axis(input)` | F32 | No additional parameters |

Input must be unquantized with a nonempty final axis. For final-axis length H, `gamma` and `beta` must have shape `[H]`, be unquantized, and share the input device. `epsilon` must be finite and positive. Affine parameters remain F32 even for BF16/F16 inputs. Statistics and affine arithmetic use FP32, with the output cast to the input dtype.

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rudnn::normalization::{NormalizationError, layer_norm, rms_norm, softmax_last_axis};

fn normalize<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    beta: Option<RudaTensor<R>>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    layer_norm(input, gamma, beta, epsilon)
}

fn normalize_rms<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    rms_norm(input, gamma, epsilon)
}

fn probabilities<R: Runtime>(
    logits: RudaTensor<R>,
) -> Result<RudaTensor<R>, NormalizationError> {
    softmax_last_axis(logits)
}
```

## 8. Gated-delta prefill and recurrence

Enable `tensor-gated-delta`. Use `chunk_gated_delta_rule(input, chunk_size)` for chunked sequence prefill and `gated_delta_rule(input)` for token-by-token recurrence. Both take `GatedDeltaInput<R>`:

| Field | Shape/type |
| --- | --- |
| `query`, `key` | `[B, H, T, K]`, matching F32/F16/BF16 |
| `value` | `[B, H, T, V]`, same dtype as query |
| `beta` | `[B, H, T]`, same dtype as query |
| `log_decay` | `[B, H, T]`, F32 |
| `initial_state` | `[B, H, K, V]`, F32 |
| `query_scale` | Finite f32, supplied according to model configuration |

All tensors must be unquantized and on the same device. Supply Q/K after model-specific normalization; the entry point does not perform it for you. This function reuses the preceding `Runtime` and `RudaTensor` imports:

```rust
use rudnn::gated_delta::{
    GatedDeltaError, GatedDeltaInput, GatedDeltaOutput, chunk_gated_delta_rule,
};

fn delta_prefill<R: Runtime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    beta: RudaTensor<R>,
    log_decay: RudaTensor<R>,
    initial_state: RudaTensor<R>,
    query_scale: f32,
    chunk_size: usize,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    chunk_gated_delta_rule(
        GatedDeltaInput {
            query, key, value, beta, log_decay, initial_state, query_scale,
        },
        chunk_size,
    )
}
```

The returned `output` has shape `[B, H, T, V]` and query dtype. `final_state` is F32 with shape `[B, H, K, V]`. For the next segment of the same sequence, pass that `final_state` as `initial_state`. Keep separate state for different sequences. Initial state is not overwritten in place.

`chunk_size` must be positive, and the triangular workspace of `4 × (chunk_size² + chunk_size)` bytes must fit the device's per-workgroup shared-memory limit. The entry point pads the final chunk; padding is excluded from the output. Invalid arguments return `GatedDeltaError`.

For model-level text and image calls, see the [ruLLM inference guide](../model-inference.md).
