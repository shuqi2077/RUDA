# ruFFT-host

CPU real Fourier transforms for Ruda host tensors. This package provides dtype-specific forward and inverse transforms plus host dispatch, separate from the device `ruda-fft` implementation.

## Interfaces

- `rfft_f32`, `rfft_f64`, `rfft_f16`, and `rfft_bf16` expose forward transforms.
- `irfft_f32`, `irfft_f64`, `irfft_f16`, and `irfft_bf16` expose inverse transforms.
- `dispatch` contains host transform dispatch interfaces.

## Usage

Cargo package: `ruFFT-host`. Rust import: `rufft_host`.

```toml
[dependencies]
ruFFT-host = "0.1"
```

## Features

Default features: `std`, `simd`, `rayon`.

| Feature | Purpose |
| --- | --- |
| `simd` | Enable SIMD transform support. |
| `rayon` | Enable Rayon parallel transforms. |
| `std` | Enable standard-library numeric support. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruFFT/host/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruFFT/host/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rufft.md)
