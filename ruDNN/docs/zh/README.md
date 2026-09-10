# ruDNN

[English](../../README.md) | **简体中文** | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

Ruda 神经网络算子库。

- Cargo package：`ruDNN`
- Rust crate：`rudnn`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor-attention` | 注意力 |
| `tensor-convolution` | 卷积 |
| `tensor-normalization` | LayerNorm、RMSNorm 与 softmax |
| `tensor-moe` | MoE 路由、分发与专家计算 |
| `tensor-gated-delta` | Gated-delta 运算 |
| `pooling` | 池化 |
| `interpolation` | 插值 |
| `grid-sample` | 网格采样 |
| `ctc` | CTC 损失 |

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruDNN --features tensor-normalization
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rudnn.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)

## ruDNN 用户指南

[计算库](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/README.md) · [ruBLAS](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rublas.md) · [张量框架](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/tensor-framework.md) · [English](../../README.md)

### 1. 概述与功能入口

ruDNN 组织神经网络计算算子，Cargo package 为 `ruDNN`，Rust crate 名为 `rudnn`。

| feature | 内容 |
| --- | --- |
| `tensor-attention` | 张量注意力路径 |
| `tensor-convolution` | 张量卷积路径 |
| `pooling`、`interpolation` | 池化、插值 |
| `grid-sample`、`ctc` | 网格采样、CTC |
| `tensor-moe` | 设备 MoE 路由与专家计算 |
| `tensor-normalization` | 通用设备归一化，供模型层复用 |
| `tensor-gated-delta` | gated-delta 计算，供 Qwen3.5 等混合结构接入 |

注意力和卷积还各有对应的 autotune feature。完整声明见 [Cargo.toml](../../Cargo.toml)，模块入口见 [lib.rs](../../src/lib.rs)。

### 2. 注意力与卷积

#### 注意力

`rudnn::attention::tensor::attention` 接收 query、key、value、可选 mask、可选 attn_bias、AttentionModuleOptions 和 AttentionStrategy，返回设备张量或 AttentionSetupError。

策略包括 FlashBlackboxAccelerated、FlashUnit、Fallback，以及启用对应 feature 后的 Autotune。未启用 autotune 时默认 Fallback，启用后默认 Autotune；Fallback 是同一设备计算的多 Kernel 实现，不是 CPU 后端切换。

数据布局、mask 和精度要求需要匹配所选策略。入口见[注意力实现](../../src/attention/tensor/base.rs)。

#### 卷积

`rudnn::convolution::tensor::conv_forward` 接收 input、weight、可选 bias、ConvOptions<N> 和 ConvStrategy，返回设备张量或 ConvSetupError。内部转换到通道末尾布局后执行，再转换输出。`conv_forward_nhwc` 则直接使用该布局。

策略包括 Direct、ImplicitGemm 及可选 Autotune。当前入口对三维 F32 使用 Direct；分组卷积选择 ImplicitGemm 时也会进入 Direct。因此策略参数不能一概理解为绝不替换算法的强制选项。

同模块还包含 conv_data_backward 与 conv_weight_backward。各方向需分别核对形状、选项和执行支持，定义见[卷积入口](../../src/convolution/tensor/base.rs)。

### 3. MoE 调用流程

`rudnn::moe` 的本地计算链为：

输入 logits → softmax／top-k → 紧凑分发 → SwiGLU 专家 → 加权合并。

| API | 用途 |
| --- | --- |
| `route(logits, RoutingOptions)` | 生成 RoutingPlan |
| `RoutingPlan::expert_indices()`、`weights()` | 访问选中专家及权重 |
| `RoutingPlan::dispatch(input)` | 按专家分发 token |
| `SwiGluExperts::new(gate, up, down)` | 建立无 bias 专家权重集合 |
| `SwiGluExperts::forward_dispatched` | 计算已分发的 token |
| `DispatchedTokens::combine` | 还原 token 顺序并加权合并 |
| `SwiGluExperts::forward(input, logits, options)` | 组合完整本地前向链 |

这些入口使用 `RudaTensor<R>`；计算入口通过 `Result` 报告形状、dtype 等参数错误，错误类型为 `MoeError`。

### 4. 路由契约

logits 的 shape 为 [T, E]，支持非量化 F32、F16、BF16。`RoutingOptions` 包含 `top_k: usize` 和 `renormalize: bool`，要求 1 ≤ top_k ≤ E。

softmax 以 FP32 在全部专家上计算，再选择 top-k；相等时优先较小专家编号。启用 renormalize 时重新归一化选中权重，之后转换为 logits dtype。NaN、正无穷或全负无穷行保留 NaN 权重，不切换到均匀分布。

选中专家与权重均为 [T, top_k]，专家索引为 U32。

### 5. 专家权重与分发

输入 token 为 [T, H]；gate 和 up 为 [E, I, H]；down 为 [E, H, I]。权重均须为相同浮点 dtype、同一设备。专家数量和维度需要与路由及输入一致。

分发不按容量丢弃 token；专家内部的原子排位顺序不保证固定，恢复顺序使用保存的映射。前向入口接收已有 logits，不负责模型 gate 投影或权重文件加载。

源码：[路由](../../src/moe/routing.rs)、[分发与合并](../../src/moe/dispatch.rs)、[专家](../../src/moe/experts.rs)、[测试](../../src/moe/tests.rs)。

### 6. 调用 MoE

在 `ruDNN` 依赖中启用 `tensor-moe` feature。准备上一节所列布局的设备张量后，创建专家权重对象并执行前向计算：

```rust
use rudnn::moe::{RoutingOptions, SwiGluExperts};

let experts = SwiGluExperts::new(gate, up, down)?;
let options = RoutingOptions {
    top_k: 2,
    renormalize: true,
};
let output = experts.forward(input, logits, options)?;
```

本例每个 token 选择两个专家，因此专家数量 `E` 至少为 2。`input` 为 `[T, H]`，`logits` 为 `[T, E]`，返回的 `output` 为 `[T, H]`。`experts` 可以复用于后续输入批次，调用者无需重复创建权重对象。

需要读取路由结果或在专家计算之间插入自己的处理时，可拆开调用：

```rust
use rudnn::moe::route;

let routing = route(logits, options)?;
let selected_experts = routing.expert_indices();
let selected_weights = routing.weights();
let dispatched = routing.dispatch(input)?;
let expert_output = experts.forward_dispatched(&dispatched)?;
let output = dispatched.combine(expert_output)?;
```

两段代码是替代用法，第二段从新一批 `input` 和 `logits` 开始。`combine` 根据分发映射还原 token 顺序，并使用路由权重合并专家输出。

形状、dtype 或设备不匹配时，接口返回 `MoeError`。在主机端读取结果可使用 `ruda_kernel::tensor::readback::into_data_sync(output)`；它会等待设备结果并返回 `TensorData`。

### 7. LayerNorm、RMSNorm 与 Softmax

启用 `tensor-normalization`，从 `rudnn::normalization` 导入以下接口。它们都沿输入的最后一维计算并保持 shape：

| 接口 | 输入 | 参数 |
| --- | --- | --- |
| `layer_norm(input, gamma, beta, epsilon)` | F32／F16／BF16 | `gamma` 为 F32 向量，`beta` 为可选 F32 向量 |
| `rms_norm(input, gamma, epsilon)` | F32／F16／BF16 | `gamma` 为 F32 向量 |
| `softmax_last_axis(input)` | F32 | 无额外参数 |

输入必须非量化且最后一维非空。该维长度为 H 时，`gamma` 和 `beta` 的 shape 必须为 `[H]`，与输入同设备且非量化。`epsilon` 必须有限且大于零。即使输入是 BF16／F16，仿射参数仍使用 F32；统计和仿射计算使用 FP32，输出再转换为输入 dtype。

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

### 8. Gated-delta 预填充与递推

启用 `tensor-gated-delta`。长序列预填充调用 `chunk_gated_delta_rule(input, chunk_size)`；逐 token 递推调用 `gated_delta_rule(input)`。两者接收相同的 `GatedDeltaInput<R>`：

| 字段 | shape／类型 |
| --- | --- |
| `query`、`key` | `[B, H, T, K]`，相同的 F32／F16／BF16 |
| `value` | `[B, H, T, V]`，与 query 相同 dtype |
| `beta` | `[B, H, T]`，与 query 相同 dtype |
| `log_decay` | `[B, H, T]`，F32 |
| `initial_state` | `[B, H, K, V]`，F32 |
| `query_scale` | 有限的 f32，按模型配置传入 |

所有张量非量化且位于同一设备。Q／K 应已完成模型要求的归一化；接口不替调用方做该步骤。以下函数复用上一节的 `Runtime` 和 `RudaTensor` 导入：

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

返回值的 `output` 为 `[B, H, T, V]`，dtype 与 query 相同；`final_state` 为 F32 的 `[B, H, K, V]`。处理同一序列的下一段时，将这个 `final_state` 传为下一次的 `initial_state`；不同序列使用各自的状态。初始状态不会被原地覆盖。

`chunk_size` 必须大于零，且三角工作区 `4 × (chunk_size² + chunk_size)` 字节不能超过设备的每工作组共享内存上限。尾块由入口补零，输出不包含 padding。参数不匹配时返回 `GatedDeltaError`。

模型级文本和图片调用见 [ruLLM 推理指南](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/model-inference.md)。
