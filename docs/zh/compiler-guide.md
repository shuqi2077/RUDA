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

features 控制代码是否参与构建，环境变量控制当前选择；两者不是同一个开关。具体依赖见 [CUDA 清单](../../ruda-driver/cuda/Cargo.toml) 与 [编译器清单](../../ruda-compiler/Cargo.toml)。

## 3. 环境变量

| 变量 | 行为 |
| --- | --- |
| `RUDA_CUDA_COMPILER` | 未设置时使用 nvrtc；接受 `nvrtc` 或 `ptx` |
| `RUDA_PTX_VERSION` | 直接 PTX 路径要求显式提供 `major.minor` |
| `CUDA_PATH` | CUDA Toolkit 安装根目录 |
| `RUDA_PTX_TEST_CACHE` | 仅 ptx-runtime 示例读取的缓存位置覆盖变量 |

未知编译器值报错。选择 `ptx` 但未启用 `direct-ptx` 也报错，不会切回 NVRTC。具体解析见 [compiler_backend.rs](../../ruda-driver/cuda/src/compiler_backend.rs)。

## 4. 目标与缓存

独立调用直接 PTX 编译器时，需要显式提供 PTX 版本和 SM 目标。PTX 版本描述指令集版本，SM 描述目标架构，两者不能互相替代。

CUDA 驱动中的直接 PTX 缓存命名空间包含后端标识、SM 与 PTX 版本，和 NVRTC 缓存分开。

## 5. 编译失败

不支持的 IR、参数元数据或目标条件需要报告错误。直接 PTX 的既定语义是“不支持即报错”，不是自动改用另一条编译路径。诊断步骤见[调试指南](debugging.md)。

`ruda-compiler` 还包含 WGSL、SPIR-V 与 MLIR 相关模块。

## 6. Triton 前端

源码构建组件 `ruda-triton` 将 Triton TTIR 经 SIMT 描述和 Ruda IR 编译为直接 PTX。先执行 `cargo build --locked --release -p ruda-triton`，再用 pip 安装 `./ruda-triton/python`。Python 包声明依赖 Python 3.10+、Triton 3.8+ 和 PyTorch。将 `RUDA_TRITON_PTX` 指向生成的 `target/release/ruda-triton-ptx` 可执行文件（Windows 加 `.exe`），或将其放入 PATH。

导入 Triton 前设置 `TRITON_DEFAULT_BACKEND=ruda`。`RUDA_TRITON_DEVICE=cuda` 为默认模式，使用 `torch.cuda` 存储及当前流；`RUDA_TRITON_DEVICE=ruda` 使用 `ruda_torch` PrivateUse1 存储和 RUDA 当前原生流，启动前 flush 同流上排队的 RUDA 工作。两种模式不使 CUDA 与 RUDA 指针可互换。

此前端接收 Triton 而非 Gluon，不支持的 TTIR 操作会返回 lowering 错误。它独立于 Rust Kernel DSL，选择该后端不表示兼容全部 Triton Kernel。GX1 模式需要此软件 workspace 未包含的独立 GX1 库。

源码：[Python 安装元数据](../../ruda-triton/python/pyproject.toml)、[编译阶段](../../ruda-triton/python/ruda_triton/compiler.py)、[设备／流适配](../../ruda-triton/python/ruda_triton/driver.py)。

## 7. 直接 PTX 的优化边界

直接后端自动处理满足条件的 U32／U64 常量除法和取余、向量算术／转换、对齐访存以及 warp 归约。[PTX 参考](ptx.md)说明精确整数语义、SM／PTX 条件、对齐回退和参与掩码规则。

协作行排布探索是作用于调用方 IR 的显式 API，不是自动算子识别或运行时自动调优。低精度是另一个独立的显式选项。PTX 指令更少本身不证明执行更快，还应检查寄存器压力和设备计时。默认的 `nvrtc` 编译器选择不变。
