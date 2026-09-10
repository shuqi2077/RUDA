# 参与 Ruda

[English](../../CONTRIBUTING.md) | **简体中文** | [日本語](../ja/CONTRIBUTING.md) | [Deutsch](../de/CONTRIBUTING.md) | [Русский](../ru/CONTRIBUTING.md)

## 范围

贡献面向计算软件栈：编译器、运行时、Kernel、领域库、张量、模型接入、文档和测试。

保持库职责独立，不在张量框架层重复实现领域库算子。不因重构删功能、改变 dtype、计算顺序、错误处理或资源生命周期。不支持的路径应准确报告，不能悄悄换成 CPU、另一编译后端或更低精度。

## 报告问题

提供使用的源码版本、操作系统、Rust 工具链、GPU/驱动/工具包版本、启用的 features、复现命令、最小输入及实际/预期结果。附日志前移除令牌、个人路径和未授权公开的模型或数据；不提交模型权重和构建缓存。

## 提交修改

- 修改前核对所属组件与现有测试；跨层变更说明依赖和调用契约。
- 保留第三方作者、版权、许可证及源码来源；不要统一覆盖所有文件的许可声明。
- 附与行为对应的测试。区分新增测试源码、语法检查、编译通过与真实设备运行通过。
- 没有执行的验证明确列出，不能将 Cargo metadata 或格式检查当作编译测试。
- 对性能改动报告相同输入、dtype、设备和配置下的正确性及前后测量，不用模拟器或缩小配置代替目标硬件验收。

## 本地检查

从 workspace 根目录执行。下面的命令只读取清单，不编译或运行 Kernel：

```powershell
cargo metadata --no-deps --format-version 1 --offline --locked
cargo tree -p ruda-driver-cuda --no-default-features --features direct-ptx --edges normal,build --offline --locked
```

`--offline` 需要本地已有解析所需的依赖缓存。

实际构建与演示入口见[快速开始](getting-started.md)。

提交者应确认有权提交相关代码。Ruda 原创贡献采用 Apache-2.0；第三方代码保持其原许可和适用条件。
