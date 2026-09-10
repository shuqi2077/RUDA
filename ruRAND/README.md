# ruRAND

**English** | [简体中文](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/docs/zh/README.md) | [日本語](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/docs/ja/README.md) | [Deutsch](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/docs/de/README.md) | [Русский](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/docs/ru/README.md)

Random number generation for Ruda.

- Cargo package: `ruRAND`
- Rust crate: `rurand`

## Features

| Feature | Operations |
| --- | --- |
| `tensor` | Uniform, normal, and Bernoulli distributions on device tensors |

Use `rurand::tensor::{random_uniform, random_normal, random_bernoulli}` to allocate tensors, or the functions at the crate root to fill existing device storage. `rurand::seed` sets the shared random seed.

## Quick Start

Build from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruRAND --no-default-features --features std,tensor
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rurand.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/Cargo.toml) · [Module exports](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/src/lib.rs)

## ruRAND User Guide

[Compute libraries](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/README.md) · [Runtime API](https://github.com/shuqi2077/RUDA/blob/main/docs/en/runtime-api.md) · [中文](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/docs/zh/README.md)

ruRAND generates device tensors with uniform, normal, and Bernoulli distributions. Use `rurand::tensor` to allocate new tensors, or the same-named functions at the crate root to fill existing device storage.

### 1. Configure dependencies

The Cargo package is `ruRAND`; its Rust import name is `rurand`. Feature `tensor` enables device tensor interfaces. This configuration places the application directory alongside the `RUDA` source directory. See [Getting started](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md) for NVIDIA setup.

```toml
[dependencies]
rurand = { package = "ruRAND", path = "../RUDA/ruRAND", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Generate random tensors

This complete `src/main.rs` generates F32 tensors of shape `[2, 3]` and replays a call by resetting the seed. Run `cargo run` from the application directory:

```rust
use ruda_core::tensor::DType;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::readback::into_data_sync;
use rurand::tensor::{
    random_bernoulli, random_like_uniform, random_normal, random_uniform,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    rurand::seed(42);
    let uniform = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let first = into_data_sync(uniform.clone()).to_vec::<f32>()?;

    rurand::seed(42);
    let repeated = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let repeated = into_data_sync(repeated).to_vec::<f32>()?;
    assert_eq!(first, repeated);
    assert!(first.iter().all(|&x| (0.0..1.0).contains(&x)));

    let normal = random_normal::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let mask = random_bernoulli::<CudaRuntime>([2, 3].into(), &device, 0.25, DType::F32);
    let noise = random_like_uniform(&uniform, -0.1, 0.1, DType::F32);

    let mask = into_data_sync(mask).to_vec::<f32>()?;
    assert!(mask.iter().all(|&x| x == 0.0 || x == 1.0));
    println!("uniform={first:?}");
    println!("normal={:?}", into_data_sync(normal).to_vec::<f32>()?);
    println!("mask={mask:?}");
    println!("noise={:?}", into_data_sync(noise).to_vec::<f32>()?);
    Ok(())
}
```

`random_like_uniform` takes only shape and device from the reference. It allocates a new tensor without changing or copying the reference values. Its final argument explicitly selects output dtype.

`random_bernoulli` returns zeros and ones in the selected dtype, not an automatically created Boolean mask. This example returns F32 values 0.0 and 1.0.

### 3. Parameters and distributions

The first three tensor functions share `shape: Shape`, `device: &R::Device`, and a final `dtype: DType` parameter. The returned `RudaTensor<R>` has the specified shape, dtype, and device.

| Function and parameter order | Distribution parameters |
| --- | --- |
| `random_uniform(shape, device, lower_bound, upper_bound, dtype)` | f32 lower and upper bounds |
| `random_normal(shape, device, mean, std, dtype)` | f32 mean and standard deviation; std is not variance |
| `random_bernoulli(shape, device, probability, dtype)` | f32 probability of returning 1 |
| `random_like_uniform(&reference, lower_bound, upper_bound, dtype)` | Reference tensor, f32 bounds, output dtype |

Supply finite parameters consistent with the distribution: lower bound below upper bound, nonnegative normal standard deviation, and Bernoulli probability in `[0, 1]`. These interfaces do not validate these ranges through `Result`; do not use invalid parameters to request an error return.

Uniform generation first produces F32 `u ∈ [0, 1)`, computes `lower_bound + (upper_bound - lower_bound) × u`, then casts to the output dtype. The F32 `[0, 1)` case excludes 1. Scaling to other intervals or converting to lower precision can round a value near the upper bound to that bound.

Normal generation uses an FP32 Box–Muller transform before casting to the output dtype. Choosing F64 storage does not increase internal random floating-point precision. Bernoulli compares `u < probability`, so probability 0 produces all zeros and probability 1 produces all ones.

### 4. Seeds and call order

`rurand::seed(seed: u64)` resets shared host random state. Each generation call draws new seeds from that state and advances it:

- Set the seed once to initialize a sequence, then generate successive batches.
- To replay a sequence, reset the same seed and preserve call order, shape, dtype, and device configuration.
- Do not reset the same seed before every training batch unless repeated random inputs are intended.
- Concurrent threads share this state; interleaving affects which seeds each call receives. It is not a separate generator object per thread or device.

The replay comparison above uses one process, one device, and matching arguments. Matching seeds alone do not imply elementwise equality with another random library or backend.

### 5. Fill an existing buffer

Use crate-root launch functions to reuse an output allocation. This function takes an already allocated F32 device tensor, writes random values into its storage, and returns it:

```rust
use ruda_kernel::{
    dsl::{Runtime, prelude::LaunchError},
    tensor::RudaTensor,
};

fn fill_uniform<R: Runtime>(
    output: RudaTensor<R>,
    lower: f32,
    upper: f32,
) -> Result<RudaTensor<R>, LaunchError> {
    rurand::random_uniform(
        &output.client,
        lower,
        upper,
        output.clone().binding(),
        output.dtype.into(),
    )?;
    Ok(output)
}
```

The function does not invoke a tensor allocator. Other handles sharing that storage also observe the writes. Corresponding functions for normal and Bernoulli values are `rurand::random_normal(&client, mean, std, binding, dtype)` and `rurand::random_bernoulli(&client, probability, binding, dtype)`.

Tensor-level functions panic if launching returns an error; the underlying entry point above returns `Result<(), LaunchError>`. Successful submission does not mean GPU execution has finished. Read the tensor back or await client synchronization. `into_data_sync` panics on readback failure; use asynchronous `into_data(tensor).await` to propagate readback errors.

API reference: [Tensor interfaces](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/src/tensor/mod.rs), [Seed state](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/src/state.rs), [Distributions](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/src/distributions/mod.rs).
