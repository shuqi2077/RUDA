# ruDNN 用户指南

[计算库](README.md) · [ruBLAS](rublas.md) · [张量框架](../tensor-framework.md) · [English](../../en/libraries/rudnn.md) | [日本語](../../ja/libraries/rudnn.md) | [Deutsch](../../de/libraries/rudnn.md) | [Русский](../../ru/libraries/rudnn.md)

## 1. 概述与功能入口

ruDNN 组织神经网络计算算子，Cargo package 为 `ruDNN`，Rust crate 名为 `rudnn`。

| feature | 内容 |
| --- | --- |
| `tensor-attention` | 张量注意力路径 |
| `tensor-paged-attention` | 分页 MHA/GQA/MLA |
| `tensor-convolution` | 张量卷积路径 |
| `pooling`、`interpolation` | 池化、插值 |
| `grid-sample`、`ctc` | 网格采样、CTC |
| `tensor-moe` | 设备 MoE 路由与专家计算 |
| `tensor-normalization` | 通用设备归一化，供模型层复用 |
| `tensor-gated-delta` | gated-delta 计算，供 Qwen3.5 等混合结构接入 |

注意力和卷积还各有对应的 autotune feature。完整声明见 [Cargo.toml](../../../ruDNN/Cargo.toml)，模块入口见 [lib.rs](../../../ruDNN/src/lib.rs)。

## 2. 注意力与卷积

### 注意力

`rudnn::attention::tensor::attention` 接收 query、key、value、可选 mask、可选 attn_bias、AttentionModuleOptions 和 AttentionStrategy，返回设备张量或 AttentionSetupError。

策略包括 FlashBlackboxAccelerated、FlashUnit、Fallback，以及启用对应 feature 后的 Autotune。未启用 autotune 时默认 Fallback，启用后默认 Autotune；Fallback 是同一设备计算的多 Kernel 实现，不是 CPU 后端切换。

数据布局、mask 和精度要求需要匹配所选策略。入口见[注意力实现](../../../ruDNN/src/attention/tensor/base.rs)。

### 卷积

`rudnn::convolution::tensor::conv_forward` 接收 input、weight、可选 bias、ConvOptions<N> 和 ConvStrategy，返回设备张量或 ConvSetupError。内部转换到通道末尾布局后执行，再转换输出。`conv_forward_nhwc` 则直接使用该布局。

策略包括 Direct、ImplicitGemm 及可选 Autotune。当前入口对三维 F32 使用 Direct；分组卷积选择 ImplicitGemm 时也会进入 Direct。因此策略参数不能一概理解为绝不替换算法的强制选项。

同模块还包含 conv_data_backward 与 conv_weight_backward。各方向需分别核对形状、选项和执行支持，定义见[卷积入口](../../../ruDNN/src/convolution/tensor/base.rs)。

## 3. MoE 调用流程

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

## 4. 路由契约

logits 的 shape 为 [T, E]，支持非量化 F32、F16、BF16。`RoutingOptions` 包含 `top_k: usize` 和 `renormalize: bool`，要求 1 ≤ top_k ≤ E。

softmax 以 FP32 在全部专家上计算，再选择 top-k；相等时优先较小专家编号。启用 renormalize 时重新归一化选中权重，之后转换为 logits dtype。NaN、正无穷或全负无穷行保留 NaN 权重，不切换到均匀分布。

选中专家与权重均为 [T, top_k]，专家索引为 U32。

## 5. 专家权重与分发

输入 token 为 [T, H]；gate 和 up 为 [E, I, H]；down 为 [E, H, I]。权重均须为相同浮点 dtype、同一设备。专家数量和维度需要与路由及输入一致。

分发不按容量丢弃 token；专家内部的原子排位顺序不保证固定，恢复顺序使用保存的映射。前向入口接收已有 logits，不负责模型 gate 投影或权重文件加载。

源码：[路由](../../../ruDNN/src/moe/routing.rs)、[分发与合并](../../../ruDNN/src/moe/dispatch.rs)、[专家](../../../ruDNN/src/moe/experts.rs)、[测试](../../../ruDNN/src/moe/tests.rs)。

## 6. 调用 MoE

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

## 7. LayerNorm、RMSNorm 与 Softmax

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

## 8. Gated-delta 预填充与递推

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

模型级文本和图片调用见 [ruLLM 推理指南](../model-inference.md)。

## 9. 分页注意力与 MLA

启用 `tensor-paged-attention`，使用 `rudnn::paged_attention`。`HostPlan::new(page_size, pages, tables, lengths, sequence_ids, positions)` 校验主机端调度元数据；positions 是各序列内从零开始的绝对位置。`DevicePlan::upload(host, &q)` 将元数据上传至查询所在设备及执行队列。只有调度不变时才能复用该计划。

`DevicePlan::attention(q, k, v, scale, causal)` 直接读取物理缓存页，处理打包的变长预填充与解码。Q 为 `[queries, Hq, D]`，K 为 `[pages, page_size, Hkv, D]`，V 为 `[pages, page_size, Hkv, Dv]`，输出为 `[queries, Hq, Dv]`；`Hq` 必须能被 `Hkv` 整除。输入须连续、非量化，使用相同的 F32/F16/BF16 dtype、设备和队列。Q/K/V 须为有限值，scale 须有限且为正。`D`、`Dv` 范围为 `1..=1024`；设备需支持 32 或 64 lane 的 plane 及所需启动网格。支持前向与一阶反向，不支持任意外部掩码或量化 KV 缓存。

`DevicePlan::mla(q, qp, latent, kp, scale, causal)` 接收吸收投影后的查询 `[queries, H, R]`、位置查询 `[queries, H, P]`、共享潜在缓存 `[pages, page_size, 1, R]` 和位置缓存 `[pages, page_size, 1, P]`。输出压缩上下文 `[queries, H, R]`；`P` 范围为 `1..=256`。位置编码在调用前完成，value/output 投影在调用后完成。scale 使用原模型 QK 缩放，而非按压缩秩推算。

以上方法使用不分段路径。需要分段时，以 `2..=32` 个 splits 创建 `SplitWorkspace::new(&q, queries, heads, value_dim, splits)`，调用 `attention_with_workspace` 或 `mla_with_workspace`。局部统计和归并使用 FP32。每个工作区上限为 64 MiB，仅能在形状匹配、同一有序执行队列内复用。

`DevicePlan::append(k, v, key_cache, value_cache)` 返回后续调用应保留的缓存张量。共享缓存分配在修改前复制；共享前缀物理页还需要调度器执行写时复制。重复的物理位置写入会被拒绝。

## 10. 分组受限 sigmoid MoE 路由

启用 `tensor-moe` 后，`route_sigmoid_grouped(logits, bias, options)` 返回 `RoutingPlan`。logits 为 F32/F16/BF16 的 `[tokens, experts]`；可选校正 bias 为同设备、同队列的 FP32 `[experts]`。bias 只影响选择，返回权重使用原始 sigmoid 分数、可选归一化及 `scale`；分数相等时优先较小编号。

`GroupRoutingOptions` 包含 `top_k`、`groups`、`selected_groups`、`group_top_two`、`renormalize` 和 `scale`。专家数范围为 `1..=1024`，组数为 `1..=128`，top-k 为 `1..=64`；专家数必须能被组数整除，选中组数须有效，top-k 不得超过选中组的专家总数。`group_top_two=true` 将每组最大的两个校正分数相加，要求每组至少两个专家；否则使用组内最大分数。scale 必须有限且为正。

`SwiGluExperts::forward_sigmoid_grouped(input, logits, bias, options, strategy)` 完成路由、分发、专家计算与合并。`forward_dispatched_with_strategy` 可选择 `GroupedStrategy::Scalar`、`Auto` 或 `TensorCore`；现有 `forward`、`forward_dispatched` 仍使用 `Scalar`。Tensor Core 路径要求支持 F16/BF16 的相应硬件。`Auto` 仅在配置不受支持时回退，不会吞掉编译或执行失败。模型投影、共享专家和残差分支仍由调用方负责。

## 11. 分页注意力反向与有序历史梯度

`DevicePlan::attention_backward(q, k, v, grad_out, scale, causal)` 返回 `AttentionBackward { dq, dk, dv }`；`mla_backward(q, qp, latent, kp, grad_out, scale, causal)` 返回 `MlaBackward { dq, dqp, dlatent, dkp }`，其中 `dlatent` 同时包含 key 和 value 的贡献。反向重新计算概率，不保留完整分数矩阵。

不安全接口 `attention_backward_selected_into`、`mla_backward_selected_into` 接收分别可选的梯度缓冲。传入 `None` 不分配该输出及该分支专用的历史临时空间，但其他导数仍可能需要该输入值。输出不能与输入或彼此重叠，所有访问须在计划的同一有序队列中执行。这条路径的历史梯度采用 FP32 原子累加，不保证逐位确定性。

需要无浮点原子的历史归约时，创建 `OrderedBackwardWorkspace::new(&plan, &q)?`，以可变引用传给 `attention_backward_ordered_into` 或 `mla_backward_ordered_into`。同样适用输出不重叠约束，也不能与工作区存储重叠。只有不可变调度元数据、查询数／头数、设备及队列兼容时才能复用。统计量和反向索引共同受 64 MiB 工作区预算限制。固定累加顺序不保证跨设备或与原子路径逐位相同。

| 有序工作区选项 | 默认值 | 作用 |
| --- | --- | --- |
| `set_query_pruning(bool)` | `true` | 跳过可证明因果不可见的查询，保留其余项的累加顺序。 |
| `set_history_row_cache(bool)` | `false` | 在线程局部存储复用历史行，不增加设备张量，但可能增加寄存器压力。 |
| `set_history_compaction(bool)?` | `false` | 只对活跃物理页计算历史梯度，对非活跃页显式写零。 |

`RUDA_PAGED_ORDERED_CACHE_ROWS=1`、`RUDA_PAGED_ORDERED_COMPACT_HISTORY=1` 在工作区构造时分别启用后两项；未设置或 `0` 表示关闭，其他值报错。修改环境变量不会改变已有工作区。

压缩按“有查询的序列在有效 KV 长度内可达的页”划分，不使用页表容量，也不表示非零梯度数量。首次启用上传 `physical_pages * 4` 字节索引，计入同一预算；之后切换复用索引，关闭后仍保留分配。`history_compaction_pages()` 返回 `Option<(active_pages, inactive_pages)>`，`bytes()` 包含保留空间。预算不足时保留原工作区模式。这些选项不改变 PyTorch 适配器的原子反向默认值，也不保证提速。

## 12. MoE 一阶训练

`selected_router_weights(&logits, &indices, options)` 返回 FP32 的 `[T, top_k]` 权重；`selected_router_backward(&logits, &indices, &grad_weights, options)` 返回与 logits 同 dtype 的 `[T, E]` 梯度。logits 为连续 F32/F16/BF16，indices 为连续 U32/I32/I64，要求 `1 <= top_k <= min(E, 64)`，操作数同设备、同队列。`RouterWeightOptions` 选择 `RouterScoring::Softmax` 或 `Sigmoid`、可选的选中权重归一化，最后应用有限正数 `scale`。`grad_weights` 为 FP32。

重复索引按 gather 语义处理；非法索引使整行结果为 NaN，不越界读取，也不插入主机同步。这里只对固定选择下的连续权重求导，不对 top-k／分组决策或校正 bias 求导。`RoutingPlan::into_training(logits, options)` 保留选择并重新计算权重；通过返回的 `RouterTrainingPlan::routing()` 分发，通过 `backward(&grad_weights)` 取得 logits 梯度。反向前不得修改保存的输入值。

专家训练调用 `SwiGluExperts::forward_dispatched_training(&dispatched, strategy)`，保留其 `output`、`cache`，按以下顺序反向：

1. `dispatched.combine_backward(&expert_output, grad_output)` 返回 `dexpert` 和 FP32 的 `dweights`。
2. `cache.backward(dexpert)` 返回分发后行的 `dinput` 及 FP32 的 `dgate`、`dup`、`ddown`。
3. `dispatched.dispatch_backward(dinput)` 将选中行求和回原 token，不再次乘路由权重。
4. 将 `dweights` 传入路由训练反向，得到 logits 梯度。

`combine_backward` 默认使用 `CombineGradientStrategy::Serial`，`combine_backward_with_strategy` 可选择 `Plane`。专家 `backward` 独立于前向策略，默认 scalar；`backward_with_strategy` 接收 `GroupedStrategy::Scalar`、`Auto` 或 `TensorCore`。Tensor Core 反向要求受支持的 F16/BF16 硬件；`Auto` 只针对能力不支持回退，不吞掉编译或执行错误。
