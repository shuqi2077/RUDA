# 编译器指南

[文档首页](README.md) · [PTX 参考](ptx.md) · [编程指南](programming-guide.md) · [English](../en/compiler-guide.md) | [日本語](../ja/compiler-guide.md) | [Deutsch](../de/compiler-guide.md) | [Русский](../ru/compiler-guide.md)

## 1. 编译流程

通用 Kernel 路径从 `ruda-kernel::dsl` 的宏与类型出发，生成 Kernel IR，再由后端完成 lowering 和代码生成。`ruda-compiler` 保存编译实现；设备驱动负责将编译产物交给对应执行环境。

对于 NVIDIA，当前有两条可选路径：

| 选择 | Kernel 编译流程 |
| --- | --- |
| `nvrtc`，默认 | Rust Kernel 前端 → IR → CUDA C++ → NVRTC → PTX |
| `ptx` | Rust Kernel 前端 → IR → PTX |

两条路径最终都通过 NVIDIA 驱动执行。保留 CUDA C++ 编译链不意味着公开了通用 C++ 工程导入或完整 CUDA 源码兼容接口。

## 2. Cargo features

| 组件／feature | 用途 |
| --- | --- |
| `ruda-kernel/frontend` | Kernel DSL 前端 |
| `ruda-kernel/lowering-cpp` | C++ lowering 接入 |
| `ruda-compiler/cpp` | C++ 后端实现 |
| `ruda-compiler/ptx` | 直接 PTX 编译器 |
| `ruda-driver-cuda/direct-ptx` | 在 CUDA 驱动后端启用直接 PTX 选择 |

features 控制代码是否参与构建，环境变量控制当前选择；两者不是同一个开关。具体依赖见 [CUDA 清单](../../ruda-driver-cuda/Cargo.toml) 与 [编译器清单](../../ruda-compiler/Cargo.toml)。

## 3. 环境变量

| 变量 | 行为 |
| --- | --- |
| `RUDA_CUDA_COMPILER` | 未设置时使用 nvrtc；接受 `nvrtc` 或 `ptx` |
| `RUDA_PTX_VERSION` | 直接 PTX 路径要求显式提供 `major.minor` |
| `CUDA_PATH` | CUDA Toolkit 安装根目录 |
| `RUDA_PTX_TEST_CACHE` | 仅 ptx-runtime 示例读取的缓存位置覆盖变量 |

未知编译器值报错。选择 `ptx` 但未启用 `direct-ptx` 也报错，不会切回 NVRTC。具体解析见 [compiler_backend.rs](../../ruda-driver-cuda/src/compiler_backend.rs)。

## 4. 目标与缓存

独立调用直接 PTX 编译器时，需要显式提供 PTX 版本和 SM 目标。PTX 版本描述指令集版本，SM 描述目标架构，两者不能互相替代。

CUDA 驱动中的直接 PTX 缓存命名空间包含后端标识、SM 与 PTX 版本，和 NVRTC 缓存分开。

## 5. 编译失败

不支持的 IR、参数元数据或目标条件需要报告错误。直接 PTX 的既定语义是“不支持即报错”，不是自动改用另一条编译路径。诊断步骤见[调试指南](debugging.md)。

`ruda-compiler` 还包含 WGSL、SPIR-V 与 MLIR 相关模块。
