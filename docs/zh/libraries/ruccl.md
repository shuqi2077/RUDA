# ruCCL 用户指南

[计算库](README.md) · [张量框架](../tensor-framework.md) · [English](../../en/libraries/ruccl.md) | [日本語](../../ja/libraries/ruccl.md) | [Deutsch](../../de/libraries/ruccl.md) | [Русский](../../ru/libraries/ruccl.md)

## 1. 层级与入口

Cargo package 为 `ruCCL`，Rust crate 名为 `ruccl`。

ruCCL 包含面向张量 Backend 的集合操作、rank 核心与进程内实现。`ruda-communication` 承担通信基础设施。`orchestrator` feature 用于编排相关入口。

显式 rank／设备映射、TCP rendezvous、初始化顺序、token 加权副本和逐 rank 恢复见[分布式训练指南](../distributed-training.md)。

## 2. 张量集合 API

| 函数 | 行为 |
| --- | --- |
| `register<B>` | 注册 peer、设备和 CollectiveConfig |
| `all_reduce<B>` | 归约后向参与方返回结果 |
| `broadcast<B>` | 发送方传 Some(tensor)，接收方传 None |
| `reduce<B>` | 向指定 root 归约；非 root 返回 None |
| `finish_collective<B>` | 结束该 peer 的集合会话 |
| `reset_collective<B>` | 重置本地集合服务并丢弃注册及进行中操作状态 |

这些注册接口基于 `B: ruda_tensor::Backend`，数据类型是 `B::FloatTensorPrimitive`。低层调用注册内层 Backend；需要显式 rank 的张量图操作时，使用 `ruda_autodiff::collective`。

## 3. 注册与调用契约

`CollectiveConfig::default()` 创建配置，`with_num_devices` 指定本地参与设备数；策略和多节点地址通过相应配置方法设置。

参与方的设备数配置应一致，peer ID 必须唯一。各方需要按匹配的集合操作序列调用，并保持 shape、归约操作及 root 等参数一致。broadcast 每次应有且只有一个发送方。

涉及多节点时，节点数量、全局地址、本地地址及数据服务端口等参数需成组配置。

## 4. 错误与生命周期

接口返回 `CollectiveError`，包含注册重复／缺失、shape 不一致、归约操作不一致、root 不一致和广播发送方数量错误等情况。

正常结束使用 `finish_collective`。`reset_collective` 会遗忘进行中的状态，不是完成当前集合操作、设备任务 checkpoint 或无损恢复的替代接口。

## 5. CUDA 示例

`cuda` feature 启用 CUDA 张量后端。运行 `cargo run --locked -p ruCCL --features cuda --example all_reduce`：在 GPU 0 上执行四个逻辑 rank 的 Ring AllReduce，检查 257 个 FP32 元素的 Sum／Mean、输入保持与会话退出。

设备适配位于 [tensor_device](../../../ruCCL/src/tensor_device)，优化器接口见[显式 rank 梯度归约](../../../ruda-optim/src/optim/grads/collective.rs)。传输包含 host-staged 路径，不是零拷贝 P2P。

源码：[集合 API](../../../ruCCL/src/api.rs)、[配置](../../../ruCCL/src/config.rs)、[rank](../../../ruCCL/src/rank/mod.rs)、[进程内实现](../../../ruCCL/src/in_process/mod.rs)。

## 6. 集合通信训练

在 `ruda-optim` 中启用 `collective`。显式持有 rank 通信器时，先将反向梯度转换成 `GradientsParams`，调用 `grads.all_reduce_with::<InnerBackend>(&communicator, ReduceOperation::Mean)?`，再将返回的梯度交给 `optimizer.step`。各 rank 的参数 ID、梯度 shape、dtype 和调用顺序必须一致；自动微分训练中的 `InnerBackend` 是未包装 `Autodiff` 的后端。

源码中的两 rank 训练示例可以直接运行：

```powershell
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- run ./collective-training-state
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- resume ./collective-training-state
```

`run` 要求保存目录尚不存在；它在第一次更新后保存各 rank 的模型和优化器，再执行第二次更新。`resume` 从该目录恢复并执行第二次更新。启用 CUDA 时，这个示例的两个逻辑 rank 使用同一默认设备。

完整调用见[集合通信训练示例](../../../ruda-optim/examples/collective_training.rs)；需要连同调度器和待累积梯度一起保存时，使用[训练状态保存](../training.md)中的 `TrainingRecord`。

## 7. 显式 rank 张量集合通信

`RankCommunicator<TensorDevice<B>>` 提供浮点与 I32/I64 的 broadcast、all-reduce、all-gather、reduce-scatter。Gather 按 rank 顺序拼接等形第零轴分片；scatter 要求第零轴能被 world size 整除。整数保留存储宽度，不经浮点转换；此传输仍为 host-staged。

已跟踪张量使用 `ruda_autodiff::collective` 中的 `all_gather`、`reduce_scatter_sum`、`reduce_scatter_mean`、`all_reduce_sum`、`all_reduce_mean`、`broadcast`；聚合／分片的 `_dim` 版本接受任意正轴或负轴。各 rank 的前向和反向须保持匹配的调用顺序、shape、dtype、梯度跟踪和 root。[训练指南](../training.md#副本训练与可微分张量集合通信)列出反向规则及按 token 加权的 `DataParallel::reduce_fp32`。

`DataParallel<B, C>` 接受 `DataParallelCommunicator<B::InnerBackend>`；[rust-ascend](https://github.com/shuqi2077/rust-ascend) 提供原生 HCCL 张量传输，TCP 仅传递训练 metadata。替换通信器不改变参数 ID、冻结／共享语义或 token／样本加权。
