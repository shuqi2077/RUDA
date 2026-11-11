# PTX 后端参考

[文档首页](README.md) · [编译器指南](compiler-guide.md) · [示例](samples.md) · [English](../en/ptx.md) | [日本語](../ja/ptx.md) | [Deutsch](../de/ptx.md) | [Русский](../ru/ptx.md)

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

源码定义：[ptx 模块](../../ruda-compiler/src/ptx/mod.rs)、[编译器测试](../../ruda-compiler/src/ptx/tests.rs)、[运行时后端选择](../../ruda-driver/cuda/src/compiler_backend.rs)。

## 5. Packed FP4 传输与 NVFP4 缩放 MMA

`StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)` 每字节存储两个 FP4 值。直接 PTX 支持保持位模式的复制、索引、select、reinterpret 和内存传输，不将 packed 字节当成标量 FP4 算术值；不支持的算术或转换会报错。单纯传输不要求 `sm_120a` 目标。

`CoopMma::ExecuteScaled` 当前仅接受 SM 120、PTX 8.7 或更新版本、`16×8×64` tile、packed E2M1x2 A／B、F32 C／D、E4M3 scale，以及 `scales_factor = 4`。它生成 `mxf4nvf4.block_scale.scale_vec::4X` 指令，并将输出 target 改为 `sm_120a`。其他架构、tile、scale 格式或倍数会被拒绝，同时要求足够的 fragment 寄存器容量。

源码：[packed 操作处理](../../ruda-compiler/src/ptx/packed.rs)、[缩放 MMA lowering](../../ruda-compiler/src/ptx/matrix/scaled.rs)、[packed 传输／布局示例](../../ruda-driver/cuda/examples/ptx_packed.rs)。该示例读取已导出的 PTX，检查字节传输和 fragment 坐标，不执行 NVFP4 MMA 算术。

## 6. 自动 lowering 与显式排布选择

U32／U64 对编译时非零常量的除法和取余，在完整无符号输入范围内使用精确整数强度削减。2 的幂使用移位或掩码；满足条件的其他除数使用高位乘法加移位；商只可能为 0 或 1 的大除数使用比较；其余除数使用倒数乘法和一次整数修正。余数计算可使用低位乘加。输入和输出寄存器可以重叠。除数在 IR 中是常量时自动应用，不按算子名称识别、不添加特定形状实现、不假设分子有额外范围限制，也不经浮点数计算索引。动态除数、零除数及有符号除法保持原有 lowering。见[整数 lowering](../../ruda-compiler/src/ptx/constant_division.rs)。

其他自动直接 PTX 路径包括：

- 当类型和目标支持时使用成对 FP16／BF16 算术和转换；SM100+、PTX 8.6+ 还可对 FP32 加、减、乘和显式 FMA 使用成对指令。分开的乘法与加法仍分别舍入，不隐式启用低精度。
- 连续向量访存使用 packed／vector 指令；无法证明对齐时保留对齐检查和标量回退。SM100+、PTX 8.8+ 的全局访存每条指令最多处理 32 字节，其他支持目标仍以 16 字节为上限；不假设任意切片已对齐。
- 仅当线程块尺寸与控制流能证明参与关系时，warp 归约才省略动态参与掩码处理；不完整或分歧的 warp 保留活动 lane 路径。

[行排布辅助接口](../../ruda-compiler/src/ptx/row.rs)中的 `compile_row_variants` 编译调用方提供的 IR，`select_row_variant` 通过调用方的正确性与计时回调选择候选。它们不根据 Kernel 名称推断算子，也不在普通编译中自动跑分。排布和精度是独立决策；`FastMath::ReducedPrecision` 必须显式请求。

PTX 不是最终机器码。经 NVIDIA 汇编后，指令数和寄存器分配仍会变化，指令更少不保证执行更快。应对目标工作负载比较汇编产物和设备计时；编译或汇编通过不等于性能结果。默认 NVRTC 选择、启动和参数契约均不改变。
