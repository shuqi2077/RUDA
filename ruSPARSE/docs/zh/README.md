# ruSPARSE

[English](../../README.md) | **简体中文**

本仓库是源码镜像。以下命令需在 [RUDA 主仓库](https://github.com/shuqi2077/RUDA)根目录运行。

Ruda 稀疏计算库，复用公共张量、运行时和设备后端。

## Feature 与入口

| Feature | 入口与依赖 |
| --- | --- |
| 默认（空） | 稀疏格式、校验及主机转换；不引入设备实现 |
| `tensor` | `CsrTensor<R>`、SpMV、SpMM、SpVV、SDDMM、SpGEAM、SpGEMM；由调用者选择公共运行时 |
| `cuda` | `tensor` 加 NVIDIA 驱动，供 CUDA 程序与示例使用 |
| `cuda-tests` | `cuda` 的测试兼容入口 |

## CUDA 示例

```sh
cargo run -p ruSPARSE --features cuda --example csrmv
```

示例执行 `alpha * A * x + beta * y`，同步读回并校验 `[7.0, 2.0, 18.5]`。

## 当前计算契约

- 公共设备路径使用 FP32 值及 U32 索引；数值内核要求 32-lane 子组。
- 通过 `CsrTensor::from_csr` 上传已校验的结构；`with_values` 更新数值，`transpose` 重排当前设备值。操作返回新结果，不覆盖输入。
- SpGEAM／SpGEMM 共享主机符号算法，保留显式零结构；数值阶段在设备执行。结构索引保留主机副本，数值不回读到主机计算。
