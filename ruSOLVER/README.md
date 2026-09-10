# ruSOLVER

**English** | [简体中文](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rusolver.md)

Numerical linear solvers and matrix factorizations for Ruda.

- Cargo package: `ruda-solver`
- Rust crate: `rusolver`

This repository is a source mirror. Build from the [RUDA workspace](https://github.com/shuqi2077/RUDA), which provides shared configuration and optional dependencies.

## Features

| Feature | Operations |
| --- | --- |
| Default (empty) | Host FP64 real/complex factorizations, sparse LU, CG, and analytic reverse-mode pullbacks; no external dependencies |
| `sparse` | Adapters for existing ruSPARSE CSR structures |
| `tensor` | Native FP32 small-batch LU, Cholesky, QR, symmetric eigen, and dense CG on device tensors |
| `cuda` | `tensor` plus the NVIDIA driver for CUDA programs and examples |
| `collective` | ruCCL TCP transport adapter for host row-partitioned CG |

The library is experimental. Host, device, sparse, and distributed interfaces have separate execution contracts; enabling a feature does not move host algorithms onto the GPU.

## Quick Start

Build and run from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --locked -p ruda-solver
cargo run --locked -p ruda-solver --example solver-demo
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rusolver.md)
- [Extended numerical science](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/science-extended.md)
- [Environment setup](https://github.com/shuqi2077/RUDA/blob/main/docs/en/getting-started.md)
- [Cargo features](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/Cargo.toml) · [Module exports](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/lib.rs)

## ruSOLVER User Guide

[Compute libraries](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/README.md) · [中文](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/science-extended.md)

### 1. Host operations

| Interface | Purpose |
| --- | --- |
| `Lu` | Partial-row-pivot LU, multiple-RHS solve, transpose solve, inverse, and signed log determinant |
| `Cholesky` | Symmetric positive-definite factorization, solve, and log determinant |
| `Qr` | Column-pivoted Householder QR and full-column-rank least squares |
| `Svd` | Thin Jacobi SVD, pseudoinverse, and minimum-norm solve |
| `symmetric_eigen` | Real symmetric eigenvalues and eigenvectors |
| `conjugate_gradient` | Matrix-free preconditioned CG for symmetric positive-definite systems |
| `complex` | Complex LU/adjoint solve, Hermitian Cholesky, and Householder QR/least squares |
| `sparse_direct::SparseLu` | Sparse-row LU with partial pivoting, multiple RHS, transpose solve, and a fill budget |

`Matrix` owns nonempty, finite, row-major FP64 data. `MatrixView` supports immutable positive-stride views and explicit transposition. `Complex64` stores two FP64 components. Inputs are not implicitly downloaded from devices.

### 2. Solve a linear system

Factor A once and reuse the factor for subsequent right-hand sides. This example solves `A x = b` for `x = [1, 2]`:

```rust
use rusolver::{Lu, Matrix, relative_residual};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Matrix::new(2, 2, vec![4.0, 1.0, 1.0, 3.0])?;
    let b = Matrix::new(2, 1, vec![6.0, 7.0])?;
    let factor = Lu::factor(a.view(), Default::default())?;
    let x = factor.solve(b.view())?;
    println!("x = {:?}", x.values());
    println!("relative residual = {}",
        relative_residual(a.view(), x.values(), b.values())?);
    Ok(())
}
```

LU uses `P A = L U`; pivot entries record sequential row swaps. QR uses `A[:, permutation] = Q R`. Put multiple right-hand sides in the columns of B instead of forming an inverse to solve each system.

Tolerances control numerical pivot and rank decisions; they are not condition-number estimates. QR rejects rank-deficient least squares. SVD supports rank truncation, so its pseudoinverse and minimum-norm solve use the retained singular values. CG requires a fixed symmetric positive-definite operator and positive-definite preconditioner; check `report.converged()` before treating its output as a solution.

Source: [LU](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/lu.rs), [QR](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/qr.rs), [SVD](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/svd.rs), and [CG](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/iterative.rs).

### 3. Complex and sparse systems

The [advanced example](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/examples/advanced_solver.rs) covers SVD, complex LU, sparse LU, and a solve pullback. The [CSR example](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/examples/sparse_poisson.rs) uses the optional ruSPARSE adapter:

```sh
cargo run --locked -p ruda-solver --example advanced-solver
cargo run --locked -p ruda-solver --features sparse --example sparse-poisson
```

`SparseLu::factor_csr` accepts zero-based CSR arrays, including duplicate entries that are summed. Factors stay in sparse row storage rather than a dense matrix. `max_factor_nonzeros` limits factor entries, not total process memory. Feature `sparse` enables `SparseLu::from_rusparse` and `sparse::CsrF32Operator` for existing ruSPARSE structures.

### 4. Device tensors

Enable `tensor` for runtime-generic interfaces or `cuda` for NVIDIA examples:

```sh
cargo run --locked -p ruda-solver --features cuda --example solver-cuda
cargo run --locked -p ruda-solver --features cuda --example solver-cuda-advanced
```

| Interface in `rusolver::tensor` | Per-system dimensions |
| --- | --- |
| `cholesky_solve_batched` | Square order 1..32; 1..8 right-hand sides |
| `lu_solve_batched` | Square order 1..32; 1..8 right-hand sides |
| `qr_batched` | 1..32 columns; columns <= rows <= 64; no column pivoting |
| `symmetric_eigen_batched` | Square order 1..32 |
| `conjugate_gradient_batched` | Square order 1..128; one RHS; zero initial guess |

Inputs must be unquantized, contiguous, row-major FP32 tensors on the same device and execution queue. Default paths use one device thread per system. Explicit `*_batched_warp` LU and Cholesky paths cooperate within a 32-thread block using shared memory. They require the opt-in `warp-solvers` feature and are experimental, not enabled by default. This is not a blocked, warp-cooperative large-matrix implementation or a cuSOLVER performance claim. Inputs are unchanged, and outputs/workspace use new device allocations.

[Warp-kernel optimization and A/B benchmark](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/solver-kernel-optimization.md) ·
[优化说明与验证范围](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/solver-kernel-optimization.md)

Call `check_status_sync()` before using results. It synchronizes and reads each system's status; successful submission alone does not establish numerical success. There is no implicit host numerical fallback or LAPACK/cuSOLVER ABI compatibility.

Source: [Device Cholesky](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/tensor/mod.rs) and [additional device solvers](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/tensor/advanced.rs).

### 5. Distributed computation and autodiff

`distributed::distributed_cg` keeps CSR matrix rows local while gathering search vectors and reducing scalar quantities. Feature `collective` provides `RucclCommunicator` over existing ruCCL TCP. This is host FP64 CG, not multi-GPU/RDMA or distributed sparse-direct factorization. Use a dedicated communication session; failed sessions must be replaced before retrying.

`adjoint` provides analytic pullbacks for solve, Cholesky, SVD, symmetric eigen, and fixed-pivot QR. Spectral derivatives require separated spectra; SVD derivatives also require full rank. The optional [`ruda-autodiff/solver-host`](https://github.com/shuqi2077/RUDA/blob/main/ruda-autodiff/src/solver_host.rs) feature connects solve, Cholesky, singular values, and symmetric eigenvalues to the existing first-order FP64 Host graph. It does not provide complex, sparse, distributed, GPU, or higher-order graph derivatives.

Source: [Distributed CG](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/distributed.rs) and [pullbacks](https://github.com/shuqi2077/RUDA/blob/main/ruSOLVER/src/adjoint.rs).
