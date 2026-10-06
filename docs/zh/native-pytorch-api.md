# 原生 PyTorch API 参考

[文档目录](README.md) · [包指南](../../ruda-torch/README.md) · [English](../en/native-pytorch-api.md)

## 设备与组件契约

先导入 `ruda_torch`，再创建 `ruda:0` 张量。同一进程不能已经注册其他 PrivateUse1 后端。`is_available()` 表示集成已初始化，不是 GPU 指令支持的完整清单；`device_count()` 为 1，`current_device()` 为 0。`synchronize(device=None)` 仅接受原生设备并等待完成。`execution_stats()` 返回累计内核、传输及部分库调用计数，不是计时器或 autotune 结果。

Rust／C++ 要求基础 ABI 10。可选功能各自协商版本：graph 3、training 4、router 1、sequence 1、NF4 decode 1、NF4 matmul 1。Paged backward 接受原生 API 1 或 2 及兼容桥接，ordered backward 要求 2；仅匹配基础 ABI 不表示具备全部扩展。使用匹配的[源码构建或预编译包](../../ruda-torch/README.md#build-and-install)。

原生 eager 后端不安装通用 CPU 回退。算子不支持、能力缺失及设备错误直接传播。CUDA 与 RUDA 张量使用不同分配，不能因为在同一 GPU 上就混用。

## 归一化与门控激活

下列融合函数要求连续、稠密、无未解析 conjugate／negative 视图的 FP32／FP16／BF16 `ruda:0` 张量。输出保留激活类型，归一化保存的统计量为 FP32，仅支持一阶导数。

| API | 参数、形状与结果 |
| --- | --- |
| `rms_norm(x, weight=None, eps=None)` | 对 `[...,D]` 的非空末轴归一化；可选同设备 `[D]` weight，输入类型或 FP32。`eps=None` 使用 `torch.finfo(x.dtype).eps`，显式 epsilon 转为 FP32 后须有限且为正。 |
| `RMSNorm(width, eps=1e-5, elementwise_affine=True, device=None, dtype=None)` | width 为正；仿射参数默认 FP32。允许 CPU 构造，但 forward 前须搬到 RUDA，末轴须等于 width。 |
| `layer_norm(x, weight=None, bias=None, eps=1e-5)` | 末轴归一化；每个仿射向量为同设备 `[D]`，输入类型或 FP32；epsilon 须为有限正 FP32。 |
| `LayerNorm(width, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None)` | 正 width，仿射默认 FP32；`elementwise_affine=False` 关闭两个参数，`bias=False` 仅关闭 bias。 |
| `silu_mul(gate, up)` | shape／dtype／device 相同，不广播；计算 `SiLU(gate) * up`，保留低精度存储舍入边界。 |

反向前不能修改保存的输入／权重。这些融合反向接口拒绝 `create_graph=True`。可选 FP32 仿射参数与显式 StaticGraph RMSNorm 的同类型权重契约不同。源码：[training.py](../../ruda-torch/python/ruda_torch/training.py)。

## 优化器与损失缩放

`AdamW(params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2, fused_step=False, max_grad_norm=None, hierarchical_stats=False)` 要求不重叠的普通 RUDA leaf 参数，并满足上述连续浮点契约。不支持稀疏梯度、`amsgrad`、`maximize`、`differentiable`、`capturable`。LR／decay 非负，epsilon 为正，两个 beta 转为 FP32 后均在 `[0,1)`。

- `step(closure=None, loss_scale=1.0)` 使用 FP32 主参数／动量更新活跃参数。下一累积窗口前清空梯度。默认原地反缩放梯度，更新前回读四字节有限性标记。
- `fused_step=True` 保留梯度不变，回读十二字节统计报告，每个活跃参数仍启动一个更新内核。`max_grad_norm` 为可选非负的全局反缩放后 L2 上限，要求该模式。
- `hierarchical_stats=True` 也要求融合模式，为较大统计工作区增加有界归约阶段。
- 非有限梯度跳过整个更新。查看 `last_step_skipped`、`last_step_had_grad`，融合模式还可查看 `last_grad_norm`／`last_clip_coef`；这条主机回读路径不是优化器图捕获。
- `state_dict()`／`load_state_dict()` 保留优化器状态与支持的 step 选项；恢复时参数形状与类型契约须匹配。

`GradScaler(init_scale=65536., growth_factor=2., backoff_factor=0.5, growth_interval=2000, min_scale=2**-24, max_scale=2**24)` 为 RUDA scaler，不是 `torch.amp.GradScaler`。尺度边界为正，growth factor 大于一，backoff 在 `(0,1)`，interval 为正整数。同一优化器周期依次执行 `scale(loss).backward()`、`step(optimizer)`、`update()`；loss 须为 RUDA 上的单元素 FP32。`step` 接受 RUDA AdamW、Muon、MuonAdamW，不接受 closure 或额外参数。没有 `unscale_()` 或多优化器周期；`get_scale()` 读取当前值，scaler 状态仅在完整周期之间存档。

Python Muon 分组、Newton–Schulz 选项及存档规则见[架构与 Python Muon 指南](architecture-training.md)和[优化器定义](../../ruda-torch/python/ruda_torch/optim.py)；独立的 [Rust Muon 指南](muon.md)介绍 Rust 张量优化器。不能将局部分片更新当成完整矩阵 Muon。

## 流与事件

| API | 执行／生命周期契约 |
| --- | --- |
| `Stream(priority=0)` | 仅支持优先级零，底层流池有界，应复用流。 |
| `current_stream(device=None)`、`default_stream(device=None)` | 仅原生设备，默认流 ID 为零。 |
| `stream(s)` | 上下文退出时恢复此前的流。 |
| `Stream.wait_stream(other)`、`wait_event(event)` | 建立设备侧顺序依赖，不能用主机就绪查询替代。 |
| `Stream.record_event(event=None)` | 在该流记录并返回事件。 |
| `Event(enable_timing=False)` | `record(stream=None)` 默认当前流，`wait(stream=None)` 插入事件依赖。 |
| `Stream.query()`、`Event.query()` | 查询是否就绪，不做主机完成等待。 |
| `Stream.synchronize()`、`Event.synchronize()` | 在主机等待完成。 |
| `start.elapsed_time(end)` | 返回毫秒；两个事件须开启计时，相关工作须完成。 |
| `record_stream(tensor, s)` | 为已提交工作保留分配，不建立生产／消费依赖。 |
| `Event.close()` | 显式释放原生事件，再用时须重新 record。 |

异步模式在首次原生提交前设置 `RUDA_TORCH_ASYNC=1`，进程内缓存该选项。取标量／回读及显式同步仍会等待。见[流示例](../../ruda-torch/README.md#streams-events-and-asynchronous-dispatch)。

## 分页注意力与固定路由权重

`PagedAttentionPlan(page_size=..., num_pages=..., block_tables=..., kv_lengths=..., sequence_ids=..., positions=..., splits=1, backward_strategy='atomic')` 保存不可变调度元数据。输入须为匹配的连续、同类型 FP32／FP16／BF16 原生张量与执行队列；调度改变时重建计划。

| 方法 | 张量契约 |
| --- | --- |
| `attention(q, k, v, scale=..., causal=True)` | Q `[queries,query_heads,features]`；K／V `[physical_pages,page_size,kv_heads,features]`，分别使用 key／value 特征宽度。query heads 能被 KV heads 整除；scale 有限且为正，数值有限，特征宽度不超过 1024。 |
| `mla(absorbed_query, position_query, latent_cache, position_cache, scale=..., causal=True)` | 查询 `[queries,heads,rank]`／`[queries,heads,position_dim]`，缓存 `[pages,page_size,1,rank]`／`[pages,page_size,1,position_dim]`，位置宽度不超过 256；返回尚未做 value／output 投影的 `[queries,heads,rank]` context。 |
| `workspace_bytes(query_heads, value_dim)` | 仅前向 split scratch，不含反向内存或总峰值显存。 |

`splits=1` 为不拆分，2–32 使用部分注意力与 FP32 归并，每个工作区上限 64 MiB。KV 内容／更新及位置编码由调用者负责，MLA 使用原模型 QK scale。支持一阶梯度，不支持任意外部 mask、量化 KV 或高阶导数。默认 atomic backward；`backward_strategy='ordered'` 显式选择无原子的历史归约，不是 autotune。所有权／压缩选项见[包注意力指南](../../ruda-torch/README.md#paged-gqa-and-mla)。

`selected_router_weights(logits, indices, scoring='softmax', renormalize=False, scale=1.)` 输入连续原生 `[tokens,experts]` FP32／FP16／BF16 logits 和 `[tokens,top_k]` int32／int64 indices，`1 <= top_k <= min(experts,64)`。返回 FP32 已选择槽位权重及一阶 logits 梯度，不负责专家选择。Softmax 先覆盖全部专家再 gather，sigmoid 逐点计算；可选重归一化覆盖已 gather 槽位，最后乘 scale。重复索引累加梯度；无效索引使相应前向／反向整行输出 NaN，仍保持越界安全，不通过主机同步检查索引。见[路由实现](../../ruda-torch/python/ruda_torch/_router.py)。

## 序列训练与可学习伪量化

`solve_triangular(a, b, upper=False, left=True, unitriangular=False)` 接收同设备、同浮点类型矩阵。A 为 `[...,N,N]`；左求解 B 为 `[...,N,K]`，右求解为 `[...,K,N]`，批次维度可广播。flag 须为 Python bool；RUDA 使用 sequence API 1，CPU／CUDA 为显式参考设备，支持一阶自动微分。

`gated_delta_rule(query, key, value, beta, log_decay, initial_state=None, query_scale=None, normalize_qk=False, norm_eps=1e-6, chunk_size=64, checkpoint_chunks=True, native_forward=True)` 要求同一浮点设备：Q／K `[B,H,T,Dk]`，V `[B,H,T,Dv]`，beta／decay `[B,H,T]`，可选初始状态 `[B,H,Dk,Dv]`，Q／K／V 存储类型相同。`chunk_size` 为正；返回 `(output, final_state)`，输出 `[B,H,T,Dv]`。状态／中间值为 FP32，显式 CPU FP64 输入使用 FP64。默认 RUDA 前向复用 ruDNN chunk 路径，反向重算同设备分块操作，不是独立融合 DeltaNet backward；输出与最终状态均传播梯度。准确 scale／归一化及空序列行为见[序列实现](../../ruda-torch/python/ruda_torch/sequence_training.py)。

`learned_fake_quantize(x, scales, bits=4, block_shape=None, symmetric=True)` 要求非空 FP32／FP16／BF16 输入及同设备 FP32 scales。bits 为 2、4、8；没有 block shape 时只有一个 scale，否则每轴一个正块大小，scale 数量为 `product(ceil(shape/block_shape))`，包括边缘块。scale 下限为 `1e-8`，低于下限的尺度梯度为零；返回具有一阶 straight-through 导数的**浮点**张量，不是打包推理权重。`LearnedFakeQuantize(initial_scales, ...)` 保存可训练、可存档的 FP32 尺度参数，移动模块时仍须保持 FP32。见[伪量化实现](../../ruda-torch/python/ruda_torch/quantization.py)和不同格式的 [NF4 微调](finetuning.md)。

## 编译、图与模型

- [多进程副本训练](distributed-training.md#python-多进程副本训练与-nccl)：`ReplicaGroup`、显式 CUDA ordinal、NCCL 存储互操作、token 加权同步及 SFT 接入。

- [架构训练指南](architecture-training.md)：mHC、DSA／CSA／HCA、函数式压缩 KV 缓存、混合模型构造及 Python Muon 分组。

- [通用模型编译](model-compiler.md)：`compile`、`make_backend`、`CompiledModel`、`CompiledFunction`，准确原生重载、shape／stream 缓存及错误策略。
- [固定地址静态图](static-pytorch-graphs.md)：`StaticGraph`、`GraphOp`、重放／输出生命周期和显式一阶训练。
- [微调 API](finetuning.md)：公开的打包、LoRA／NF4 加载、因果监督、适配器及训练 checkpoint 入口。
- [混合模型源码](../../ruda-torch/python/ruda_torch/hybrid_model.py)：`MHCTransformerBlock`、`HybridAttentionLanguageModel`、`next_token_loss`，属于模型组合，不是预训练权重加载器。
- [mHC 源码](../../ruda-torch/python/ruda_torch/mhc.py)：`MHC`、`MHCResidual`、`MHCSequential`、`MHCCoefficients`、`sinkhorn`。
- [压缩注意力源码](../../ruda-torch/python/ruda_torch/sparse_attention.py)：`LightningIndexer`／`DSAIndexer`、`indexer_kl_loss`、`LearnedKVCompressor`、`CompressedSparseAttention`／`CSA`、`HeavilyCompressedAttention`／`HCA`、`RotaryEmbedding`、`AttentionOutput`、`IndexerOutput`、`CompressedAttentionCache`、`CompressionState`。

架构组件组合的是同设备张量操作，名称不代表兼容全部预训练 DeepSeek 架构。API 元数据错误通常抛出 `ValueError`／`TypeError`，原生执行错误直接传播，异步错误可能在同步／回读时出现。
