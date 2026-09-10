# ruSPARSE

**English** | [简体中文](docs/zh/README.md) | [日本語](docs/ja/README.md) | [Deutsch](docs/de/README.md) | [Русский](docs/ru/README.md)

This repository is a source mirror. Run the commands below from the [RUDA monorepo](https://github.com/shuqi2077/RUDA) root.

Ruda's sparse-computing library, reusing shared tensors, runtimes, and device backends.

## Features and entry points

| Feature | Entry points and dependencies |
| --- | --- |
| Default (empty) | Sparse formats, validation, and host conversions; does not include device implementations |
| `tensor` | `CsrTensor<R>`, SpMV, SpMM, SpVV, SDDMM, SpGEAM, and SpGEMM; callers select the shared runtime |
| `cuda` | `tensor` plus the NVIDIA driver, for CUDA programs and examples |
| `cuda-tests` | Compatibility entry point for `cuda` tests |

## CUDA example

```sh
cargo run -p ruSPARSE --features cuda --example csrmv
```

The example computes `alpha * A * x + beta * y`, reads the result back synchronously, and checks it against `[7.0, 2.0, 18.5]`.

## Current computation contract

- The shared device path uses FP32 values and U32 indices; numerical kernels require 32-lane subgroups.
- Upload validated structures with `CsrTensor::from_csr`; `with_values` updates values, and `transpose` rearranges current device values. Operations return new results rather than overwriting inputs.
- SpGEAM/SpGEMM share host-side symbolic algorithms and preserve explicit-zero structure; the numerical stage runs on the device. Structural indices retain host copies; values are not read back for host-side computation.

## ruSPARSE User Guide

[Compute libraries](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/README.md) · [中文](docs/zh/README.md)

### 1. Overview

ruSPARSE provides sparse computation.

### 2. Data descriptors

| Type | Purpose |
| --- | --- |
| `CsrMatrix`, `CsrMatrixOwned` | Borrowed CSR views and owned storage |
| `CooMatrix`, `CooMatrixOwned` | COO representation |
| `CscMatrix`, `CscMatrixOwned` | CSC representation |
| `BsrMatrix`, `EllMatrix` | Block sparse and ELL representations |
| `DenseMatrix`, `DenseMatrixOwned` | Dense matrix descriptors |
| `SparseVector` | Sparse vector |
| `IndexBase` | Zero or One index base |
| `DenseOrder` | RowMajor or ColumnMajor |
| `Operation` | None, Transpose, or ConjugateTranspose |

Index base, layout, and operation type are explicit contracts and cannot be inferred from array lengths alone. Owned distinguishes owned data from borrowed views; it does not automatically allocate general-purpose GPU storage.

### 3. Operations

Operations are organized into matrix-vector multiplication, matrix multiplication, sparse matrix multiplication/addition, sparse vector dot products, SDDMM, and format conversion. The execution layer contains `SparsePlan`, `SparseError`, and operation-specific result types.

CSR matrix-vector multiplication uses the alpha, op(A), x, beta, y contract. When porting a call, match index base, transpose mode, and data layout rather than only changing the library prefix.

### 4. CUDA CSR matrix-vector multiplication

The Cargo package is `ruSPARSE`; the Rust import name is `rusparse`. Feature `tensor` enables `rusparse::tensor`; `cuda` enables both device tensors and CUDA dependencies. Run from the source root:

```powershell
cargo run --locked -p ruSPARSE --features cuda --example csrmv
```

This example computes `0.5 × A × x + 2 × y`:

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, IndexBase, Operation,
    tensor::{CsrTensor, csrmv},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let host = CsrMatrix::new(
        3, 3,
        &[0, 2, 3, 4],
        &[0, 1, 1, 2],
        &[2., 1., -1., 5.],
        IndexBase::Zero,
    )?;
    let matrix = CsrTensor::<CudaRuntime>::from_csr(host, Operation::None, &device)?;
    let x = from_data(TensorData::new(vec![3f32, 4., 5.], [3]), &device);
    let y = from_data(TensorData::new(vec![1f32, 2., 3.], [3]), &device);
    let output = into_data_sync(csrmv(&matrix, 0.5, x, 2., y)?).to_vec::<f32>()?;
    assert_eq!(output, [7., 2., 18.5]);
    println!("alpha * A * x + beta * y = {output:?}");
    Ok(())
}
```

The arguments to `CsrMatrix::new` are rows, columns, row offsets, column indices, nonzero values, and index base. The first row stores two entries; the second and third store one each. The result is `[7.0, 2.0, 18.5]`.

`CsrTensor::from_csr` uploads structure and values. Specify `Operation::Transpose` here to transpose the matrix. For `csrmv(&matrix, alpha, x, beta, y)`, x length equals the column count and y length equals the row count. Dense inputs must be unquantized F32 tensors on the matrix's device. The call returns a new device tensor; reusing the matrix does not require uploading its CSR structure again.

For sparse/dense matrix multiplication, use `csrmm(&matrix, operation_b, alpha, b, beta, c, output_order)`. Here c is optional, and `DenseOrder` specifies output layout. To change only nonzero values, call `matrix.with_values(values)` with an F32 device tensor of length nnz; the structure is preserved.

### 5. CSR in the tensor framework

`ruda_tensor::api::CsrTensor<B>` targets Backends implementing `SparseOps`. It is a different type from `rusparse::tensor::CsrTensor<R>` above. Create it with `CsrTensor::<B>::from_data(&data, &device)`, where data has the Backend's `B::CsrData` type.

| Method | Usage |
| --- | --- |
| `matmul(rhs)`, `transpose_matmul(rhs)` | Multiply the sparse matrix or its transpose by a two-dimensional dense tensor |
| `sparse_matmul(&rhs)` | Multiply sparse matrices and construct the output pattern |
| `add(&rhs)`, `add_scaled(&rhs, alpha, beta)` | Add sparse matrices, optionally with coefficients |
| `gather(dense)` | Gather dense values at the CSR positions into a one-dimensional tensor |
| `scatter_add()` | Add stored values to their dense positions |
| `mul_dense(rhs)` | Multiply by dense values at the CSR positions |
| `sampled_matmul(lhs, rhs)` | Evaluate a dense matrix product only at the current CSR positions |
| `sampled_sparse_matmul(&lhs, &rhs)` | Evaluate a sparse matrix product only at the current CSR positions |
| `transpose()`, `to_dense()` | Transpose or convert to a dense tensor |
| `with_values(values)` | Replace the one-dimensional value tensor while retaining structure |
| `to_data().await` | Asynchronously read back `B::CsrData` |

These computation methods return `Result`. Value tensors used in framework sparse operations must be unquantized.

API reference: [Device CSR](src/tensor/mod.rs), [Framework CSR](https://github.com/shuqi2077/RUDA/blob/main/ruda-tensor/src/api/sparse.rs).
