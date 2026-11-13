# 原生推理算子

接口语义参考固定版本 [FlashInfer v0.7.4](https://github.com/flashinfer-ai/flashinfer/tree/v0.7.4)，
提交 `77039051b0be921788dc2784ec41423b558878ec`。这里提供 Rust/RUDA 原生接口，
不是 FlashInfer Python 包的直接替代，也不调用其 CUDA 内核。

## 接入方式

- `ruDNN/tensor-inference`：`rudnn::inference`。
- `ruDNN/inference-cuda`：显式拥有生命周期的 CUDA IPC 通信组。
- `ruDNN/inference-autotune`：原生 RUDA 内核/图调优会话与持久化选择。
- `ruBLAS/tensor-float-quant`：FP4/FP8 数据与缩放布局。
- `ruPRIM/tensor-topk`、`tensor-bitpack`：选择和掩码打包。
- `ruRAND/tensor`：设备采样。
- `ruda-llm/native-inference`：`rullm::native`。
- `ruda-tensor-device/native-inference`：共享原生张量算子接口。

这些显式 feature 不改变默认 ruLLM 生成行为。完整类型索引见[英文接口说明](../en/flashinfer.md)。

## 调用约定

`prepare` 分配并保留输出和工作区，`run` / `*_into` 使用绑定的设备缓冲区。
不安全的原地接口要求调用方保证输出独占、队列依赖和分配生命周期；便捷的函数式缓存
接口使用 copy-on-write 保留已有别名。`PreparedPack` 可将带 stride 的视图打包，
更新后通过 `write_back` 显式写回。数值计算和设备调度元数据不回读 CPU。

`Matrix` 支持浮点、带解码缩放的 FP8、以及原生块缩放 FP4/FP8。FP8 负载使用 U8 字节存储。
全局缩放采用**解码乘数**，使用上游编码乘数时应先求倒数。F32/F16/BF16 通用计算路径
采用 FP32 累积；类型/算子名称不意味着 PTX、随机序列或性能与上游相同。

ReplaySSM 的 ring cursor 和接受数由调用方管理，内核只读这些 tracker。预分配路径只
重建被选择请求的状态，不转换整个状态池。前缀物化先快照源状态再写目标，零长度前缀
保留原始状态与缩放字节。每 token 的 SSM 状态历史按需开启，容量为 `O(T*H*D*N)`。

稀疏流水线提供 score/select/attend 分阶段入口，必须按数据依赖调用。分页和状态池的
并行写入目标 ID 必须唯一。多卡映射及同步对象必须保留到设备操作完成，不能把进程内
地址作为可移植 checkpoint 数据。

设备采样显式接收随机数或 Philox 状态；无效行返回 token -1 和无效标记。FP32 nucleus
边界同值整体保留，不保证与旧 CPU/F64 采样逐 token 一致。

## NVFP4 注意力存储格式互操作

`nvfp4_attention::{PreparedQuantize, PackedQkv, PreparedAttention}` 实现 Q 与 KV
独立补齐到 128 个 token、全序列 K 去均值、分块/全局 Q 去均值、紧凑 FP32 QK
修正项、K token 置换、64 行 E4M3 scale 重排以及转置 V。打包字段采用 U8 保存原始
E2M1/E4M3 字节。前向支持 GQA、可选 LSE 及调用者提供输出；因果对齐以补齐后的
物理长度为准，`unpadded_k_len` 屏蔽逻辑尾部。此原生可移植实现不要求 SM120
指令，不承诺与厂商内核相同的中间概率舍入或性能。

`trace::ApplyRegistry` 提供显式注册、形状约束、确定优先级和默认回退的强类型
原生算子替换，实际选择执行替代算子而非只记录 trace；不安装 Python 装饰器，
不修改进程全局包函数。
