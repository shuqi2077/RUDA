# ruBLAS User Guide

[Compute libraries](README.md) · [Runtime API](../runtime-api.md) · [中文](../../zh/libraries/rublas.md)

## 1. Overview and features

ruBLAS provides linear algebra operations and device tensor interfaces.

| Feature | Module |
| --- | --- |
| `tensor-vector` | `rublas::tensor_vector` |
| `tensor-matmul` | `rublas::tensor_matmul` |
| `tensor-matmul-autotune` | Tensor matrix multiplication autotuning |
| `tensor-int4` | `rublas::tensor_int4` |
| `tensor-grouped` | `rublas::tensor_grouped` |

The Cargo package is `rublas`. Select the required features and disable unnecessary defaults for general-purpose paths. See [Cargo.toml](../../../ruBLAS/Cargo.toml).

## 2. Matrix multiplication, vectors, and INT4

### Tensor matrix multiplication

`rublas::tensor_matmul::matmul` takes lhs, rhs, optional output out, MatmulStrategy, and output DType. It returns a device tensor or MatmulSetupError. Pass an existing output through out, or omit it to allocate one.

Strategies include Ruda, CmmaResidueFirst, Naive, and Autotune when `tensor-matmul-autotune` is enabled. The default is Ruda without that feature and Autotune with it. For quantized inputs, Naive may dequantize and compute after its initial path fails; this is not native INT4 computation.

`CmmaResidueFirst` explicitly selects the Tensor Core path that processes a partial K32 tile first, without requiring a materialized padded matrix. lhs and rhs must have matching unquantized BF16 or F16 dtype and compatible matrix dimensions. The device must support the path's matrix instructions. This function returns F32 output:

```rust
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rublas::{
    kernel_ir::definition::MatmulSetupError,
    tensor_matmul::{MatmulStrategy, matmul},
};

fn residue_matmul<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    matmul(lhs, rhs, None, MatmulStrategy::CmmaResidueFirst, DType::F32)
}
```

Inputs `[M, K]` and `[K, N]` produce `[M, N]`. Invalid setup or missing device capabilities return `MatmulSetupError` rather than switching to Naive.

See the [matrix multiplication entry point](../../../ruBLAS/src/tensor_matmul/base.rs).

### Vector cross product

`rublas::tensor_vector::cross(lhs, rhs, dim)` requires the selected dimension to have length 3 and returns a device tensor. Computing along a non-final dimension involves permutation and contiguous conversion. See [cross.rs](../../../ruBLAS/src/tensor_vector/cross.rs).

### AWQ INT4

`rublas::tensor_int4::AwqGemm::new` takes qweight, qzeros, scales, optional bias, and group_size to construct packed weights. `forward` accepts F16 input and returns F16 output or Int4Error.

For input dimension K, output dimension N, and group size G:

| Data | Dtype | Shape |
| --- | --- | --- |
| qweight | Packed I32 | [K, N/8] |
| qzeros | Packed I32 | [K/G, N/8] |
| scales | F16 | [K/G, N] |
| bias (optional) | F16 | [N] |

K, N, and G must be nonzero; K must be divisible by G and N by 8. The input's final dimension is K, replaced by N in the output. All operands must share a device. Packed bit order must match the [AWQ kernel](../../../ruBLAS/src/tensor_int4/kernel.rs); an arbitrary INT4 file cannot be used directly as qweight.

See the [INT4 interface](../../../ruBLAS/src/tensor_int4/mod.rs) for the object and its checks.

## 3. Grouped matrix multiplication

`rublas::tensor_grouped::grouped_matmul_nt<R: Runtime>` takes input, weights, and row_experts as `RudaTensor<R>` values. It returns `Result<RudaTensor<R>, GroupedMatmulError>`.

| Argument | Shape | Requirements |
| --- | --- | --- |
| input | [M, K] | Non-quantized F32, F16, or BF16 |
| weights | [E, N, K] | Same dtype and device as input |
| row_experts | [M] | Non-quantized U32 on the same device |
| Output | [M, N] | Same dtype as input |

Row m selects the weight matrix indexed by row_experts[m] and computes dot products between the input row and each row of that matrix. The final two weight dimensions participate in transposed form without requiring a materialized transpose.

## 4. Grouped multiplication semantics

- K, N, and E must be positive; M may be zero.
- Expert indices outside the valid range indicate padding and produce zero output rows.
- The kernel accumulates in FP32, then casts to the input dtype.
- The entry point makes input and weights contiguous when necessary, which may copy data.
- Relevant element counts must fit U32 indexing.
- Invalid arguments return `GroupedMatmulError`. Handle asynchronous execution errors during readback or synchronization.

Source: [grouped interface](../../../ruBLAS/src/tensor_grouped/mod.rs) and [kernel](../../../ruBLAS/src/tensor_grouped/kernel.rs).

## 5. Integration with other libraries

[ruDNN MoE](rudnn.md) uses grouped multiplication for expert projections. Quantized weights use the separate `tensor_int4` module; ordinary floating-point grouped multiplication is not INT4 expert computation.

The grouped kernel uses scalar accumulation. Measure matrix multiplication strategies for your dtype, shape, and backend.
