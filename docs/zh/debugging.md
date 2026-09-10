# 调试与诊断

[文档首页](README.md) · [编译器](compiler-guide.md) · [Runtime API](runtime-api.md) · [English](../en/debugging.md)

## 1. 按阶段定位

| 阶段／现象 | 首先检查 |
| --- | --- |
| Cargo 清单或依赖解析失败 | 路径、features、锁文件、依赖缓存 |
| Rust 构建失败 | 首个实际编译错误、工具链及 feature 组合 |
| CUDA 工具包定位失败 | CUDA_PATH、实际安装目录、头文件 |
| 驱动或设备初始化失败 | 驱动可用性、设备索引及动态库加载 |
| 直接 PTX 配置失败 | direct-ptx feature、编译器选择、PTX 版本 |
| Direct PTX 编译错误 | 错误中的不支持操作、目标或参数布局 |
| 回读／同步失败 | 先前异步提交、输入绑定、设备错误 |
| 数值错误 | shape、strides、dtype、边界、同步及算法契约 |

保留首个错误及其上下文，不只复制最终的构建失败摘要。

## 2. 编译与缓存日志

运行时配置位于 `ruda::runtime::config`。`CompilationConfig` 提供 logger、cache 和 check_mode。`CompilationLogLevel` 的序列化名称为 disabled、basic、full；full 包含源码级编译信息。

`ptx-runtime` 示例自带日志计数器，统计编译与 PTX 磁盘缓存命中。`RUDA_PTX_TEST_CACHE` 仅由该示例读取，不是所有应用自动支持的全局环境变量。

配置定义见 [compilation.rs](../../ruda/src/runtime/config/compilation.rs)，示例见 [ptx_runtime.rs](../../ruda-driver-cuda/examples/ptx_runtime.rs)。

## 3. 边界检查

`BoundsCheckMode` 有三种配置：

| 配置 | 运行时定义 |
| --- | --- |
| auto | 普通启动使用检查，显式 unchecked 启动允许跳过检查 |
| enforce | 对启动强制使用检查 |
| validate | 普通启动保持检查；unchecked 路径选择验证模式 |

这是运行时选择的执行模式，具体检测能力仍取决于编译器和后端，不代表自动发现所有越界、竞争或生命周期错误。不要为了让失败用例继续运行而关闭检查。

## 4. 异步错误

`ComputeClient::launch` 不返回数值结果。使用返回 `Result` 的回读或等待同步 future 观察完成情况；区分参数检查失败、Kernel 编译失败和设备执行失败。

## 5. 定向复现

[示例页](samples.md)列出共享内存限制、张量、位运算和半精度用例。

提交问题时附源码版本、features、完整命令、后端、PTX 版本、GPU／驱动／工具包、输入与实际输出；去除敏感内容后按[贡献指南](CONTRIBUTING.md)反馈。
