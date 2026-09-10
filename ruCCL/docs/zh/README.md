# ruCCL

[English](../../README.md) | **简体中文**

本仓库是源码镜像。以下命令需在 [RUDA 主仓库](https://github.com/shuqi2077/RUDA)根目录运行。

Ruda 的集合通信库。公共张量入口复用 Ruda 张量后端及其计算库；`rank` 和 `in_process` 提供独立于具体设备的通信协议、调度与设备适配契约。

## CUDA 张量示例

```sh
cargo run -p ruCCL --features cuda --example all_reduce
```

需要可用的 NVIDIA 驱动与 CUDA Toolkit。示例在 GPU 0 上创建四个逻辑 rank，经张量计算、Ring AllReduce、结果回读和会话退出，校验 257 个 FP32 元素的 Sum／Mean 以及原输入保持不变。

## Feature

| Feature | 范围 |
| --- | --- |
| 默认 | 通用通信核心与张量后端接口，不自动启用 GPU 后端 |
| `cuda` | CUDA 张量后端，沿用该后端的默认融合与调优配置 |
| `test-cuda` | 在 `cuda` 基础上选择现有 CUDA 测试后端 |
| `test-wgpu`／`test-metal`／`test-vulkan` | 现有 WGPU 测试入口；与 CUDA 测试 feature 分别运行 |
| `tracing` | 已有跨层 tracing 接线 |

公共张量 API 包括 `register`、`all_reduce`、`reduce`、`broadcast` 和 `finish_collective`；各 rank 须按一致顺序调用匹配的集合操作。自动微分调用者使用内部后端，梯度同步由优化器层负责。
