# ruTENSOR

**English** | [简体中文](docs/zh/README.md) | [日本語](docs/ja/README.md) | [Deutsch](docs/de/README.md) | [Русский](docs/ru/README.md)

Tensor linear algebra for Ruda device tensors: contractions and einsum, reductions, physical permutations, and elementwise operations.

- Cargo package: `ruTENSOR`
- Rust crate: `rutensor`
- [English user guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rutensor.md)
- [中文使用文档](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/rutensor.md)

## ruTENSOR User Guide

[Compute libraries](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/README.md) · [Tensor framework](https://github.com/shuqi2077/RUDA/blob/main/docs/en/tensor-framework.md) · [中文](docs/zh/README.md)

ruTENSOR provides named-axis tensor contractions, reductions, physical permutations, and elementwise operations. Inputs use `RudaTensor<R>`; the application selects a device Runtime. This library is distinct from the higher-level `ruda-tensor` framework.

### 1. Configure dependencies

The Cargo package is `ruTENSOR`; the Rust import name is `rutensor`. Defaults enable `std` and device tensor computation without selecting a driver. The following configuration places the application directory alongside the `RUDA` source directory:

```toml
[dependencies]
rutensor = { package = "ruTENSOR", path = "../RUDA/ruTENSOR" }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Contraction, reduction, and permutation

Save the following as the application's `src/main.rs`:

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rutensor::{einsum, permute, reduce, ReductionOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let a = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]), &device,
    );
    let b = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 0., 0., 1., 1., 1.], [3, 2]), &device,
    );
    let product = einsum("ik,kj->ij", &[&a, &b])?;
    let row_sums = reduce(&a, &[0, 1], &[0], ReductionOp::Sum)?;
    let transposed = permute(&a, &[1, 0])?;

    println!("product: {:?}", into_data_sync(product).to_vec::<f32>()?);
    println!("row sums: {:?}", into_data_sync(row_sums).to_vec::<f32>()?);
    println!("transpose: {:?}", into_data_sync(transposed).to_vec::<f32>()?);
    Ok(())
}
```

`einsum("ik,kj->ij", ...)` sums over k and returns shape `[2, 2]`. `reduce` retains mode 0 and reduces mode 1, returning shape `[2]`. `permute` returns a newly allocated `[3, 2]` tensor, not a view sharing the input storage.

### 3. Einsum expressions

`einsum(expression, inputs)` accepts one or more inputs. Case-sensitive letters identify axes; the right side of the arrow selects and orders output axes.

| Expression | Operation |
| --- | --- |
| `ik,kj->ij` | Matrix multiplication |
| `...ik,...kj->...ij` | Matrix multiplication with broadcast batch axes |
| `abc,cde->abde` | Multidimensional tensor contraction |
| `ij,jk,kl->il` | Three-input contraction |
| `i,j->ij` | Outer product |
| `ii->i` | Diagonal |
| `ii->` | Trace, returning a rank-zero scalar |
| `ijk->ki` | Reduce j and reorder the remaining axes |
| `...i->i` | Reduce all axes represented by the ellipsis |

- Matching modes across inputs must have equal extents or an extent of 1. Ellipsis axes broadcast with right alignment.
- Repeated labels within an input select a diagonal; those axes must have exactly equal extents.
- Output labels must be unique and present in the inputs.
- Without `->`, output axes start with the ellipsis axes, followed by alphabetically sorted labels that occur exactly once.
- A scalar input uses an empty label segment; for example, `,ij->ij` multiplies the first scalar input into the matrix.
- Inputs can be noncontiguous. Computation does not read tensor values back to the host.

For fixed expressions, shapes, strides, and dtypes, construct `EinsumPlan::new(expression, descriptors)` and reuse `execute(inputs)`. To specify output and arithmetic precision, use `EinsumPlan::with_options` or `einsum_with_options`.

### 4. Descriptors and execution plans

A `Mode` is an `i32` label. The same label identifies the same logical axis across tensors, regardless of its physical position. `TensorDescriptor` stores extents, element strides, and storage dtype. `OperandDescriptor` adds labels and an input unary transform.

These functions create and execute a plan for `D = alpha * A @ B + beta * C`:

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rutensor::{
    ComputeType, DType, OperandDescriptor, OperationDescriptor,
    Plan, Result, TensorDescriptor,
};

fn make_plan<R: Runtime>(
    a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    m: usize, n: usize,
) -> Result<Plan> {
    let operation = OperationDescriptor::contraction(
        OperandDescriptor::from_tensor(a, &[0, 2])?,
        OperandDescriptor::from_tensor(b, &[2, 1])?,
        Some(OperandDescriptor::from_tensor(c, &[0, 1])?),
        TensorDescriptor::contiguous(&[m, n], DType::F32)?,
        &[0, 1],
        ComputeType::F32,
    )?;
    Plan::new(operation)
}

fn execute<R: Runtime>(
    plan: &Plan, a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    alpha: f64, beta: f64,
) -> Result<RudaTensor<R>> {
    plan.execute(&[a, b, c], &[alpha, beta])
}
```

Plans do not retain input buffers. Subsequent executions can use different tensors with matching shapes, strides, and dtypes. All inputs must reside on the same device.

| Constructor | Execution input order | Scalar order |
| --- | --- | --- |
| `contraction` | A, B; optional C | alpha; beta when C is present |
| `sum_product` | All product inputs; optional C | alpha; beta when C is present |
| `reduction` | A; optional C | alpha; beta when C is present |
| `permutation` | A | alpha |
| `elementwise_binary` | A, B | alpha, beta |
| `elementwise_trinary` | A, B, C | alpha, beta, gamma |

`Plan::execute` allocates the output. `Plan::execute_into` accepts and returns an output tensor whose shape, strides, and dtype match the output descriptor. Its buffer must be exclusively owned, without sharing with inputs or other views.

Use `TensorDescriptor::new(extents, strides, dtype)` for padded or reordered output layouts; output axes must not overlap. Explicit operation descriptors can introduce additional broadcast axes through the output shape.

### 5. Reduction and elementwise operations

`reduce(input, input_modes, output_modes, operation)` reduces labels absent from the output; reduced axes are removed:

| ReductionOp | Operation | Empty reduction |
| --- | --- | --- |
| `Sum` | Sum | 0 |
| `Product` | Product | 1 |
| `Min` | Minimum | Positive infinity |
| `Max` | Maximum | Negative infinity |

`elementwise_binary` and `elementwise_trinary` align and broadcast named axes, using `BinaryOp::{Add, Mul, Min, Max}`. Trinary operations evaluate `(alpha * op(A) op_ab beta * op(B)) op_abc gamma * op(C)`.

Select `Identity`, `Negate`, `Abs`, `Sqrt`, `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Relu`, `Reciprocal`, or `Conjugate` through `OperandDescriptor::with_unary`. `Log` is the natural logarithm; `Conjugate` equals `Identity` for real values. Min and Max propagate NaNs.

### 6. Precision, storage, and errors

- Storage types are F16, BF16, F32, and F64. Quantized, integer, and complex storage are not accepted.
- `ComputeType::F32` or `F64` controls input conversion, unary transforms, products, reductions, and scalar arithmetic. F64 inputs require F64 computation.
- Defaults use F64 arithmetic if any input is F64, otherwise F32. Identical input storage types preserve that output type; mixed inputs produce F64 if any input is F64, otherwise F32.
- Request an explicit output dtype for higher-precision storage; increasing storage precision cannot recover precision already lost in an input.
- Output conversion happens at the final write. Inputs with a different dtype from the compute type require device conversion buffers; matching inputs are read in their original layout.
- Reductions use parallel workgroup tree merging. Floating-point addition and multiplication order can differ from serial evaluation.
- Empty outputs submit no computation; rank-zero outputs contain one scalar. Shape, stride, and address calculations use platform `usize`; dispatches also obey backend resource limits.
- Returned `Result` values report expression, descriptor, shape, device, and dispatch-range errors. Device submission follows the Runtime's error handling; asynchronous execution errors surface at synchronization or readback.
- General contractions directly traverse reduction coordinates. They do not search multi-input contraction paths or automatically lower to Tensor Core matrix multiplication. Storage conversion requires additional device memory.

ruTENSOR exposes a Ruda Rust API, not the NVIDIA cuTENSOR C ABI. The device must support the selected storage and compute types.
