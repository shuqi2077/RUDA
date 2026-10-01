# RUDA 昇腾 Rust 设备内核重写 · v38 候选源码

## 修正 v37 的方向

v37 是 Rust 主机封装，计算仍依赖原 DeepGEMM-Ascend C++ 内核。本包不再沿用
这条做法：移除原来的九份设备头文件，将 BF16 GEMM 的设备端调度、地址计算、
数据搬运、流水同步、矩阵乘加和输出流程写为 Rust 结构化设备程序。

**完成范围：v37 使用的 direct-store BF16 普通、批量、M 分组矩阵路径。不是
整个 DeepGEMM-Ascend 已完成 Rust 化。FP8、FP4、MQA、MegaMoE、非 Identity
后处理、驻留操作数和通用 RUDA 内核后端没有在本包实现。**

## 源码入口

| 文件（相对 `source/ruda-ascend-kernels/src/`） | 内容 |
|---|---|
| `kernel.rs` | Rust 设备程序：外层分块、K 维归约、L1/L0 轮换、乘加与写出 |
| `scheduler.rs` | 持久块分发、XOR/snake 排列、批次与空专家处理 |
| `layout.rs` | 分块大小、本地存储分配、字节偏移与容量检查 |
| `ir.rs` | 结构化循环、分支、算术、设备读写和有类型的流水操作 |
| `ascend.rs` | 把上述操作降为 CANN 指令接口；没有外部 GEMM 调用 |
| `tests.rs` | 同一 Rust 程序的测试专用内存/事件模型，不能充当 NPU 验收 |

代码中不提供 Raw C++ statement；`kernel.rs` 也不包含 CANN 函数名字符串。
算法由 Rust 构造，后端仅打印目标指令接口。旧 `bf16_gemm_impl` 和
`#include <deep_gemm/...>` 不再属于生产构建。

## 必须明确的编译边界

```
Rust 编写的设备程序 → Rust 结构化中间表示 → 自动生成 CCE 设备源码
                 → CANN Bisheng 编译 → LLD 链接 → 昇腾原生代码对象
```

**本版不是纯 Rust 直接生成昇腾 ISA，也没有消除 CANN 的 CCE/C++ 编译阶段。**
设备程序的唯一维护源是 Rust；生成 `.asc` 是后端产物，不是人工保留的原 C++
算法。这与 v37 的“Rust 加载未修改 C++ 内核”不同，但不能混称为完整 Rust
原生编译器。

宿主仍使用 RUDA 原有 Rust CANN 驱动和 `rublas::cann::DeepGemm`。新构建物
采用 `ruda.ascend.rust.bf16.v2` 清单，加载器拒绝 v37 的旧模板构建物，避免
表面升级 Rust 源码、运行时仍执行旧二进制。

## 安装与回滚

本包包含完整累计 `source/`。更新器和增量补丁针对已核验的完整 v37，
不声称兼容未经读取的最新远程版本。

```bash
python tools/verify_package.py
python tools/update.py /path/to/RUDA --check
python tools/update.py /path/to/RUDA --apply
```

更新器备份受管文件，包括将删除的原 C++ 头文件。未知本地修改会拒绝覆盖。
回滚使用打印出的仓库外备份目录：

```bash
python tools/update.py /path/to/RUDA --rollback /path/to/backup
```

也可使用 `patches/from-v37-rust-kernels.patch`，与更新器二选一。

## 先验证真正的 Rust 程序

在更新后的仓库根目录：

```bash
# 只需 Rust 编译器，不需要 CANN。编译并执行生产 IR 的测试模型。
python tools/ascend/test_rust_kernels.py --output ./rust-kernel-host-tests

# 或使用 Cargo 工作区。
cargo test --locked -p ruda-ascend-kernels

# 生成 18 种源配置，不会发布可加载清单或伪二进制。
python tools/ascend/build_deepgemm.py --emit-only --out ./rust-generated
```

生成器自身也是 Rust，缺少 rustc 时 `--emit-only` 也会明确失败。不会使用
Python 把原 C++ 模板替换几个参数冒充 Rust 内核生成。

## 构建和设备验收

```bash
# 先加载实际安装的 CANN 环境。
python tools/ascend/validate.py --toolkit "$ASCEND_HOME_PATH" \
  --output ./ascend-rust-validation
```

该入口执行 SDK 接口检查、Rust 程序测试、驱动/公共库检查、18 种设备源码
编译链接，最后运行矩阵与梯度参考对照。Rust 主机模型通过不代表设备编译通过。
需要重新编译设备构建物和 Rust 运行时，不能继续使用 v37 二进制目录。

契约沿用 v37：Ascend950DT/dav-c310、32 个计算核；连续 BF16 输入，输出
BF16/FP32；M/N/K 是正的 16 倍数；M 分组为 NT，物理前缀是 256 的倍数；
原有同步与所有权规则不变。不是新增 910B/910C 支持，也不支持 M=1 解码。

本包是 BF16 主路径的 Rust 重写候选，不是整个 RUDA 或 DeepGEMM 的全量移植。
