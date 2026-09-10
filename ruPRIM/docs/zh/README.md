# ruPRIM

[English](../../README.md) | **简体中文**

Ruda 并行基础算子库，提供归约、扫描与索引操作。

- Cargo package：`ruPRIM`
- Rust crate：`ruprim`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor-reduce` | 全张量与按轴归约 |
| `tensor-reduce-autotune` | 归约自动调优 |
| `tensor-scan` | 累积和、积、最小值与最大值 |
| `elementwise` | 逐元素运算 |
| `indexing` | 选择、切片、gather 与 scatter |

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruPRIM --features tensor-reduce,tensor-scan,indexing
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/ruprim.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)
