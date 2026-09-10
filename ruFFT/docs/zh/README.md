# ruFFT

[English](../../README.md) | **简体中文**

Ruda 快速傅里叶变换库。

- Cargo package：`ruFFT`
- Rust crate：`rufft`

## 功能

| Feature | 算子 |
| --- | --- |
| `tensor` | 设备张量上的实数 FFT 与逆变换 |

通过 `rufft::tensor::{rfft, irfft}` 分配变换输出，或使用 `rfft_launch`、`irfft_launch` 直接操作设备绑定。

## 快速开始

在 RUDA 工作区中构建：

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruFFT --no-default-features --features std,tensor
```

## 文档

- [使用手册](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rufft.md)
- [环境配置](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/getting-started.md)
- [Cargo features](../../Cargo.toml) · [模块入口](../../src/lib.rs)
