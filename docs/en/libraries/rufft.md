# ruFFT User Guide

[Compute libraries](README.md) · [Tensor framework](../tensor-framework.md) · [中文](../../zh/libraries/rufft.md) | [日本語](../../ja/libraries/rufft.md) | [Deutsch](../../de/libraries/rufft.md) | [Русский](../../ru/libraries/rufft.md)

ruFFT computes real-signal FFTs and inverse transforms on the device. The `rufft::tensor` interface allocates outputs; use `rfft_launch` and `irfft_launch` when managing device bindings directly.

## 1. Configure dependencies

The Cargo package is `ruFFT`; its Rust import name is `rufft`. Feature `tensor` enables device tensor interfaces. This configuration places the application directory alongside the `RUDA` source directory. See [Getting started](../getting-started.md) for NVIDIA setup.

```toml
[dependencies]
rufft = { package = "ruFFT", path = "../RUDA/ruFFT", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

## 2. Forward and inverse transforms

This complete `src/main.rs` uploads a four-point F32 signal, computes its spectrum, and reconstructs the input. Run `cargo run` from the application directory:

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rufft::tensor::{irfft, rfft};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = vec![1.0f32, 0.0, -1.0, 0.0];
    let signal = from_data::<CudaRuntime>(TensorData::new(input.clone(), [4]), &device);
    let (real, imag) = rfft(signal, 0, None);

    let re = into_data_sync(real.clone()).to_vec::<f32>()?;
    let im = into_data_sync(imag.clone()).to_vec::<f32>()?;
    for (actual, expected) in re.iter().zip([0.0f32, 2.0, 0.0]) {
        assert!((actual - expected).abs() < 1e-5);
    }
    assert!(im.iter().all(|value| value.abs() < 1e-5));

    let restored = irfft(real, imag, 0, Some(input.len()));
    let restored = into_data_sync(restored).to_vec::<f32>()?;
    for (actual, expected) in restored.iter().zip(input) {
        assert!((actual - expected).abs() < 1e-5);
    }
    println!("real={re:?}, imag={im:?}, restored={restored:?}");
    Ok(())
}
```

A four-point signal produces three bins: DC, one positive-frequency bin, and Nyquist. Real and imaginary components are separate tensors, not interleaved complex values. The forward transform is unnormalized; the inverse includes normalization for the actual FFT length, so do not divide by length again.

The example uses `clone()` to retain tensor handles for the inverse transform. `into_data_sync` waits for readback and returns host data; it panics on readback failure.

## 3. Parameters and output shapes

| Function | Parameters | Returns |
| --- | --- | --- |
| `rfft(signal, dim, n)` | Signal, zero-based axis, optional requested length | Two device tensors `(real, imag)` |
| `irfft(real, imag, dim, n)` | Real part, imaginary part, axis, optional output length | A real-valued device tensor |

Device computation uses F32; supply F32 signals and spectra. `dim` must be smaller than the input rank. Inverse-transform components must have matching shape, dtype, and device. These interfaces return tensors rather than `Result`; failed argument assertions or launches panic.

For requested length n, the actual FFT length N is the smallest power of two greater than or equal to n:

- With `n = None`, `rfft` uses the input length along dim. Otherwise it first truncates or zero-pads to n, then zero-pads to N.
- The forward output length along dim is `N / 2 + 1`; other dimensions are unchanged.
- With `n = None`, `irfft` uses `2 × (bin count - 1)`. An explicit n specifies the returned length.
- The inverse first truncates or zero-pads both spectrum components to `N / 2 + 1` bins, computes the N-point inverse, then crops to n.
- Requested length must be at least 2; a one-point transform fails the device kernel's length check.

| Input/call | Actual transform length | Output length along dim |
| --- | --- | --- |
| Length 8, `rfft(..., None)` | 8 | 5 |
| Length 8, `rfft(..., Some(6))` | 8; retain only the first 6 input points and zero-pad | 5 |
| Length 6, `rfft(..., None)` | 8, zero-padded | 5 |
| 5 bins, `irfft(..., None)` | 8 | 8 |
| 5 bins, `irfft(..., Some(6))` | 8 | 6 |

Thus `Some(6)` does not request a six-point DFT. For a round trip of an original six-point signal, pass `Some(6)` to the inverse to remove the two padded tail points.

## 4. Batches and multidimensional input

Axes other than dim are batch dimensions. For example, transforming `[batch, channels, 1024]` along dim=2 produces real and imaginary tensors of shape `[batch, channels, 513]`.

This function reuses the preceding `rfft` import and accepts a device tensor without reading it back:

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

fn batched_rfft<R: Runtime>(
    signals: RudaTensor<R>,
    dim: usize,
    length: usize,
) -> (RudaTensor<R>, RudaTensor<R>) {
    rfft(signals, dim, Some(length))
}
```

A single `rfft` call transforms only the selected axis, not an entire multidimensional FFT. Do not treat the separated real and imaginary components as complete complex signals by passing each independently to `rfft`.

## 5. Buffers and execution

Tensor interfaces allocate outputs and perform padding and cropping as needed. Actual lengths above 4096 automatically use the staged path without changing the call. To manage output buffers yourself, the launch interfaces take a client, input/output `TensorBinding` values, dim, and `StorageType`, returning `Result<(), LaunchError>`. Allocate matching output layouts for the actual N.

To avoid materializing zero-padded input when managing buffers, use these entry points:

| Function | Additional length parameter |
| --- | --- |
| `rfft_launch_padded(client, signal, real, imag, dim, signal_len, dtype)` | Reads only the first signal_len signal elements and treats the rest as zero; N is inferred from output spectrum length |
| `irfft_launch_padded(client, real, imag, signal, dim, spec_bins, dtype)` | Reads only the first spec_bins frequency bins and treats the rest as zero; N comes from output signal length |

N must be a power of two of at least 2, and real/imaginary shapes must match. `signal_len` cannot exceed the input axis length or N. `spec_bins` must be at least 1 and cannot exceed the input spectrum axis length or `N / 2 + 1`. Output buffers still require full allocation; these interfaces avoid materializing the zero-padded input tail.

API reference: [Tensor interface](../../../ruFFT/src/tensor.rs), [Forward launch](../../../ruFFT/src/fft/rfft.rs), [Inverse launch](../../../ruFFT/src/fft/irfft.rs).
