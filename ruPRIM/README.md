# ruPRIM

**English** | [简体中文](docs/zh/README.md) | [日本語](docs/ja/README.md) | [Deutsch](docs/de/README.md) | [Русский](docs/ru/README.md)

Parallel primitives, reductions, scans, and indexing for Ruda.

- Cargo package: `ruPRIM`
- Rust crate: `ruprim`

## Features

| Feature | Operations |
| --- | --- |
| `tensor-reduce` | Whole-tensor and axis reductions |
| `tensor-reduce-autotune` | Reduction autotuning |
| `tensor-scan` | Cumulative sum, product, minimum, and maximum |
| `elementwise` | Elementwise operations |
| `indexing` | Selection, slicing, gather, and scatter |

## Quick Start

Build from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruPRIM --features tensor-reduce,tensor-scan,indexing
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/ruprim.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](Cargo.toml) · [Module exports](src/lib.rs)

## ruPRIM User Guide

[Compute libraries](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/README.md) · [Tensor framework](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md) · [中文](docs/zh/README.md)

ruPRIM provides device tensor reductions, cumulative scans, elementwise operations, and indexing. This page uses `RudaTensor<R>`, where R is a device Runtime.

### 1. Configure dependencies

The Cargo package is `ruPRIM`; its Rust import name is `ruprim`. Enable features for the operations you need:

| Feature | Interface |
| --- | --- |
| `tensor-reduce` | `ruprim::reduce::tensor`: whole-tensor and axis reductions |
| `tensor-reduce-autotune` | Reduction autotuning; also enables tensor-reduce |
| `tensor-scan` | `ruprim::scan`: cumulative sum, product, minimum, maximum |
| `elementwise` | `ruprim::elementwise`: elementwise computation |
| `indexing` | `ruprim::indexing`: selection, slicing, gather, scatter; also enables elementwise |

This configuration places the application directory alongside the `RUDA` source directory. See [Getting started](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md) for NVIDIA setup.

```toml
[dependencies]
ruprim = { package = "ruPRIM", path = "../RUDA/ruPRIM", default-features = false, features = ["std", "tensor-reduce", "tensor-scan", "indexing"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Sum, row reductions, and scans

This complete `src/main.rs` uses F32 matrix `[[1, 2, 3], [4, 5, 6]]` to compute its total, row means, row argmax indices, and row prefix sums. Run `cargo run` from the application directory:

```rust
use ruda_core::tensor::{DType, TensorMetadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use ruprim::reduce::{
    components::instructions::ReduceOperationConfig,
    tensor::{KernelReduceStrategy, SumStrategy, reduce_dim, sum},
};
use ruprim::scan::cumsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        &device,
    );
    let total = sum(
        input.clone(),
        SumStrategy::Chained(KernelReduceStrategy::Unspecified),
    )?;
    assert_eq!(total.shape().as_slice(), &[1]);
    assert_eq!(into_data_sync(total).to_vec::<f32>()?, [21.0]);

    let means = reduce_dim(
        input.clone(), None, 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::Mean,
    )?;
    assert_eq!(means.shape().as_slice(), &[2, 1]);
    assert_eq!(into_data_sync(means).to_vec::<f32>()?, [2.0, 5.0]);

    let indices = reduce_dim(
        input.clone(), Some(DType::I32), 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::ArgMax,
    )?;
    assert_eq!(into_data_sync(indices).to_vec::<i32>()?, [2, 2]);

    let prefix = cumsum(input, 1);
    assert_eq!(into_data_sync(prefix).to_vec::<f32>()?, [1., 3., 6., 4., 9., 15.]);
    Ok(())
}
```

`sum` reduces all elements and returns shape `[1]`. `reduce_dim` handles only the selected axis, normally preserving rank and setting that axis length to 1. Row means therefore have shape `[2, 1]`, not `[2]`.

### 3. Reduction operations and strategies

Full parameter order for tensor interfaces:

| Function | Purpose |
| --- | --- |
| `sum(tensor, strategy)` | Whole-tensor sum |
| `sum_fallback(tensor, strategy)` | Whole-tensor sum; switches OneShot to Chained when the required atomic add is unavailable |
| `reduce(tensor, output_dtype, strategy, config)` | Reduces every axis in turn and returns shape [1] |
| `reduce_dim(tensor, output_dtype, dim, strategy, config)` | Reduces a selected axis |

Select `config` with `ReduceOperationConfig`:

| Operation | `reduce_dim` output |
| --- | --- |
| `Sum`, `Prod`, `Mean` | Sum, product, mean; selected axis length 1 |
| `Min`, `Max`, `MaxAbs` | Minimum, maximum, maximum absolute value; selected axis length 1 |
| `ArgMin`, `ArgMax` | Zero-based indices within the selected axis; axis length 1 |
| `TopK(k)` | Top k values along the selected axis; axis length k |
| `ArgTopK(k)` | Corresponding indices within the axis; axis length k |

Index reductions require an explicit integer output dtype such as `Some(DType::I32)`. For value reductions, pass `None`; output retains input dtype. Do not use `output_dtype` as a general cast parameter. F16/BF16 Sum, Prod, and Mean use FP32 accumulation in this path before writing the input dtype. The axis must be valid, and TopK k should be in `1..=axis length`. Use `reduce_dim` for TopK, not whole-tensor `reduce`, which resets the final shape to `[1]`.

Choose `SumStrategy` as follows:

- `OneShot(ruda_count)`: explicitly sets a positive workgroup count and requires atomic add for the input dtype.
- `Chained(KernelReduceStrategy::Unspecified)`: uses staged reductions without requiring the whole-tensor sum's global atomic add; used in the example.
- `Autotune`: available with `tensor-reduce-autotune`. Without that feature the default is `OneShot(4)`; with it the default is `Autotune`.

`KernelReduceStrategy` offers `Unspecified`, `Specific(ReduceStrategy)`, and feature-gated `Autotune`. Use Specific to fix a low-level strategy. Without autotuning, the default is Unspecified.

Reduction interfaces return `Result<RudaTensor<R>, ReduceError>`. An out-of-range axis returns `InvalidAxis`; missing atomic add for OneShot returns `MissingAtomicAdd`. `sum_fallback` replaces only the unsupported OneShot atomic-add case; it does not switch to CPU execution.

### 4. Cumulative scans

All four functions take `(tensor, dim)`:

| Function | Result for one input row [3, 1, 2] |
| --- | --- |
| `cumsum` | [3, 4, 6] |
| `cumprod` | [3, 3, 6] |
| `cummin` | [3, 1, 1] |
| `cummax` | [3, 3, 3] |

These are inclusive prefix operations. Output shape and dtype match the input; batch rows are processed independently. dim is zero-based and must be within the input rank. Scans return tensors directly, not `Result`.

The current algorithm reads the corresponding prefix for each output position, giving O(n²) total reads for an axis of length n. Account for this cost on long scan axes rather than estimating work from input element count alone.

### 5. Selection, gather, slicing, and integer powers

Add this function to the same file and call `indexing_example(&device)?;` before main returns. It reuses the earlier imports:

```rust
fn indexing_example(device: &CudaDevice) -> Result<(), Box<dyn std::error::Error>> {
    use ruda_core::tensor::element::Scalar;
    use ruprim::{
        elementwise::binary::integer_power,
        indexing::{gather, select, slice},
    };

    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        device,
    );
    let columns = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0], [2]), device,
    );
    let selected = select(input.clone(), 1, columns);
    assert_eq!(into_data_sync(selected).to_vec::<f32>()?, [3., 1., 6., 4.]);

    let positions = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0, 1, 1], [2, 2]), device,
    );
    let gathered = gather(1, input.clone(), positions);
    assert_eq!(into_data_sync(gathered).to_vec::<f32>()?, [3., 1., 5., 5.]);

    let sliced = slice(input.clone(), &[0..2, 1..3]);
    assert_eq!(into_data_sync(sliced).to_vec::<f32>()?, [2., 3., 5., 6.]);

    let squared = integer_power::scalar(input, Scalar::Int(2));
    assert_eq!(into_data_sync(squared).to_vec::<f32>()?, [1., 4., 9., 16., 25., 36.]);
    Ok(())
}
```

| Function | Indices and output |
| --- | --- |
| `select(tensor, dim, indices)` | One-dimensional integer indices; applies the same positions across other dimensions, replacing selected-axis length with index count |
| `gather(dim, tensor, indices)` | dim is the first argument; chooses an input value per output position, with output shape equal to indices shape |
| `slice(tensor, ranges)` | Half-open `start..end` ranges per axis; the example selects columns 1 and 2 from all rows |
| `slice_assign(tensor, slices, value)` | Uses `ruda_core::tensor::Slice`; writes value into the selected region and returns the updated tensor |

Indices and data must share a device; indices must be valid zero-based integers. For gather, use index tensors with the same rank and matching non-selected dimensions. Each example row can select different columns. Slice ranges must lie within their axes, and assignment value shape must match the selected region.

Integer powers are in `ruprim::elementwise::binary::integer_power`:

- `scalar(input, Scalar::Int(exponent))`: uses one integer exponent for all elements.
- `tensor(base, exponents)`: uses floating-point bases and integer exponent tensors with broadcast-compatible shapes; output retains base dtype.

Use `Scalar::Int(-2)` for a negative exponent rather than converting the exponent to floating point. Indexing and elementwise interfaces return device tensors directly. To retain an input for other operations, pass cloned handles as in the example and keep the tensor returned by each operation.

API reference: [Reductions](src/reduce/tensor/base.rs), [Scans](src/scan/tensor.rs), [Indexing](src/indexing/mod.rs), [Integer powers](src/elementwise/binary/integer_power.rs).
