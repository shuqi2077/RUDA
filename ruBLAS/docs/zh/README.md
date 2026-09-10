# ruBLAS

[English](../../README.md) | **简体中文**

Ruda 线性代数与矩阵乘法库。

- Cargo package：`rublas`
- Rust crate：`rublas`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor-matmul` | 矩阵乘法 |
| `tensor-matmul-autotune` | 矩阵乘法自动调优 |
| `tensor-vector` | 向量运算 |
| `tensor-grouped` | 分组矩阵乘法 |
| `tensor-int4` | INT4 权重运算 |

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p rublas --features tensor-matmul
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rublas.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)
