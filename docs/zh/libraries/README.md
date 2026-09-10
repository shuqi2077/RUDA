# 计算库参考

[文档首页](../README.md) · [张量框架](../tensor-framework.md) · [English](../../en/libraries/README.md) | [日本語](../../ja/libraries/README.md) | [Deutsch](../../de/libraries/README.md) | [Русский](../../ru/libraries/README.md)

领域库负责具体计算，运行时负责设备执行，张量框架负责上层组合。库名代表职责划分，不表示与对应 CUDA 库拥有相同 API 或完整功能覆盖。

## 选择计算库

| 库 | Cargo package | Rust crate | 主要内容 |
| --- | --- | --- | --- |
| [ruBLAS](rublas.md) | `rublas` | `rublas` | 矩阵乘、向量运算、分组矩阵乘与 INT4 路径 |
| [ruDNN](rudnn.md) | `ruDNN` | `rudnn` | 注意力、卷积、池化、MoE |
| [ruTENSOR](rutensor.md) | `ruTENSOR` | `rutensor` | 通用张量收缩、einsum、归约、置换与逐元素运算 |
| [ruPRIM](ruprim.md) | `ruPRIM` | `ruprim` | 归约、扫描、逐元素与索引 |
| [ruFFT](rufft.md) | `ruFFT` | `rufft` | 实数 FFT 与逆变换 |
| [ruRAND](rurand.md) | `ruRAND` | `rurand` | 均匀、正态、伯努利分布 |
| [ruSPARSE](rusparse.md) | `ruSPARSE` | `rusparse` | 稀疏矩阵格式与运算 |
| [ruCCL](ruccl.md) | `ruCCL` | `ruccl` | 集合通信与编排 |

## 接口层级

- Kernel／launch 接口接收设备绑定和执行配置，供内核及库开发使用。
- 张量接口负责分配、布局处理和调用，常使用 `RudaTensor<R>`。
- 框架接口通过 `ruda-tensor-device` 等组件分发到领域库。

同名操作在不同层的参数、返回值和错误处理可能不同。使用时同时确认 package、模块路径和 feature，不将 Kernel 入口签名套用到张量入口。

## 阅读方式

每本库手册分别说明用途、feature、接口、数据契约及支持边界。通用路径应明确选择所需功能，不能假设默认 features 适合所有设备。
