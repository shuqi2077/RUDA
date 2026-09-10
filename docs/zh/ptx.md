# PTX 后端参考

[文档首页](README.md) · [编译器指南](compiler-guide.md) · [示例](samples.md) · [English](../en/ptx.md)

## 1. 使用范围

`ruda_compiler::ptx` 将 Ruda Kernel IR 生成 PTX 文本。它不是任意 PTX 程序的解释器，也不是 NVIDIA 最终机器码生成器。

独立编译入口由 `ruda-compiler/ptx` feature 启用；通过 CUDA 运行时选择此路径使用 `ruda-driver-cuda/direct-ptx`。

## 2. 目标类型

| 类型／字段 | 含义 |
| --- | --- |
| `PtxTarget::version: (u32, u32)` | PTX 主／次版本 |
| `PtxTarget::sm: u32` | SM 目标 |
| `PtxCompilationOptions::target: Option<PtxTarget>` | 显式目标配置 |
| `PtxCompiler` | 实现公共 Compiler trait 的直接后端 |

`target` 为空时编译返回验证错误；独立编译器不会根据运行编译器的主机猜测 GPU 架构。

环境变量解析接受主版本不低于 6、次版本数值不高于 9 的 `major.minor`。通过此语法检查并不证明版本已被当前驱动或生成器支持。

## 3. 编译产物

`PtxKernel` 保存：

- `source`：PTX 文本。
- `entrypoint`：入口名称。
- `ruda_dim`：原始 Kernel 的工作组尺寸。
- `shared_memory_bytes`：共享内存需求。
- `dynamic_metadata_index`：需要时的动态元数据指针参数位置。

调用执行层时需保持参数布局、入口和共享内存需求一致，不能只提取文本而丢弃启动契约。

## 4. 错误处理

不支持的 IR 返回 `CompilationError::UnsupportedInstruction`；配置或结构验证错误返回 `CompilationError::Validation`，诊断包含 `Direct PTX:` 前缀。此路径不自动回退 NVRTC。

源码定义：[ptx 模块](../../ruda-compiler/src/ptx/mod.rs)、[编译器测试](../../ruda-compiler/src/ptx/tests.rs)、[运行时后端选择](../../ruda-driver-cuda/src/compiler_backend.rs)。
