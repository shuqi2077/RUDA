# ruDNN

[English](../../README.md) | **简体中文**

Ruda 神经网络算子库。

- Cargo package：`ruDNN`
- Rust crate：`rudnn`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor-attention` | 注意力 |
| `tensor-convolution` | 卷积 |
| `tensor-normalization` | LayerNorm、RMSNorm 与 softmax |
| `tensor-moe` | MoE 路由、分发与专家计算 |
| `tensor-gated-delta` | Gated-delta 运算 |
| `pooling` | 池化 |
| `interpolation` | 插值 |
| `grid-sample` | 网格采样 |
| `ctc` | CTC 损失 |

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruDNN --features tensor-normalization
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rudnn.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)
