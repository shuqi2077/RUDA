# ruRAND

[English](../../README.md) | **简体中文**

Ruda 随机数生成库。

- Cargo package：`ruRAND`
- Rust crate：`rurand`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor` | 设备张量上的均匀、正态与伯努利分布 |

通过 `rurand::tensor::{random_uniform, random_normal, random_bernoulli}` 分配张量，或使用 crate 根模块中的函数填充已有设备存储。`rurand::seed` 设置共享随机种子。

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruRAND --no-default-features --features std,tensor
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rurand.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)
