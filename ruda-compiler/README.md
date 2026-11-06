# ruda-compiler

Compiler backends and IR optimization for Ruda kernels. This crate emits target code; it is separate from the driver packages that load and execute that code.

## Interfaces

- `ptx`: direct kernel-IR-to-PTX compilation with explicit target options.
- `cpp`: CUDA C++ and Metal C++ code generation.
- `optimizer`, `wgsl`, `spirv`, and `mlir`: optimization and feature-selected target backends.

## Usage

Cargo package: `ruda-compiler`. Rust import: `ruda_compiler`.

```toml
[dependencies]
ruda-compiler = "0.1"
```

## Features

Default features: `ptx`, `cpp-default`.

| Feature | Purpose |
| --- | --- |
| `ptx` | Enable direct PTX generation. |
| `cpp` | Enable C++ code generation. |
| `wgsl` | Enable WGSL generation. |
| `spirv` | Enable SPIR-V generation. |
| `mlir` | Enable the MLIR backend and its LLVM dependencies. |

## Vector and warp lowering on NVIDIA

Use vector IR for independent adjacent elements, and plane reductions for
cross-lane communication. The direct PTX backend selects instructions from
the element types, vector width, requested PTX version and target SM:

- FP16/BF16 arithmetic is paired where supported; Blackwell (SM100+ with
  PTX 8.6+) can also pair FP32 add, subtract, multiply and explicit FMA.
  A scalar operand broadcasts; an odd final lane uses scalar arithmetic.
  Separate multiply and add operations keep their original rounding.
- Contiguous register lanes can narrow FP32 to FP16/BF16 or FP32/FP16 to
  FP8 in pairs. FP8-to-FP32 conversion uses both lanes of the packed input.
  The backend retains the scalar rounding and FP8 saturation contracts;
  enabling these optimizations does not opt the kernel into fast math.
- Reductions omit the dynamic mask path when the required block dimensions
  prove participation. Inside branches the mask is dynamic; after a structured
  join it can be reused only if no lane exited or executed opaque assembly.
  Short blocks, tail warps and masks with holes retain active-lane order.
- Vector loads/stores reuse a per-buffer alignment check when the element
  stride preserves alignment. Sliced or misaligned arguments still take the
  scalar fallback; callers need not make a global alignment promise.

These are register and collective optimizations, not a memory alignment
promise or a choice of tile size. They do not change kernel arguments,
bounds checks, or the selected CUDA backend.

For operators that already assign whole warps to independent rows,
`ptx::row::compile_row_variants` accepts a closure producing your own IR
for each legal row/block layout. It does not infer an operator, change its
mathematics, or select by kernel name. Respect the device block limit and skip
layouts that exceed your scratch/register budget. Use `RowLayout::blocks` for
the grid, keep tail rows masked across barriers, check output correctness,
then use `select_row_variant` with your own validation/timing callback on
representative inputs and cache the winner by
device, operator, dtype and geometry. Layout selection never opts into
lower precision. When the application's numerical contract permits it, an
inverse-square-root instruction may separately opt into the existing
`FastMath::ReducedPrecision` mode; validate that mode independently.
The default scalar/NVRTC path is unchanged.

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-compiler/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-compiler/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/compiler-guide.md)

## Experimental Ascend row lowering (`ascend` feature)

The Ascend backend accepts real `KernelDefinition` input; `row_programs` defines
FP32 row sum/mean/max, Softmax/LogSoftmax, RMSNorm, two Softmax input backwards,
and RMSNorm **input** backward in that same public IR. This is not an ACLNN call
wrapper. Width must be 32..4096 in multiples of 32; all buffers are contiguous.
A single logical 32-lane plane owns a row, with explicit strided column loads.
Only this exact plane contract is lowered; arbitrary warp operations are not
reinterpreted as whole-row reductions.

```rust,ignore
use ruda_compiler::ascend::{AscendCompiler, AscendOptions, AscendTarget,
    row_programs::{self, RowProgram}};
use ruda_core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode};

let ir = row_programs::definition(RowProgram::RmsNorm, 4096, 1e-5)?;
let compiled = AscendCompiler.compile(ir, &AscendOptions {
    target: Some(AscendTarget::Ascend950DT),
    elements: 3 * 4096,
    row_width: Some(4096),
    ..Default::default()
}, ExecutionMode::Checked, UIntKind::U64.into())?;
// Bindings: X[3,4096], weight[4096], Y[3,4096], rstd[3].
assert_eq!(compiled.bindings()[3].bytes, 12);
```

The output is generated CCE device source, still requiring Bisheng/LLD and a
trusted CANN artifact loaded through `CannProgram`. This is not direct Rust ISA,
not a PyTorch/autograd backend, and not a validated performance implementation.
There is no affine-weight gradient, generic axis/strides, FP16/BF16 row lowering,
or long-row tiling beyond the explicit width limit. Non-finite/subnormal
numerical equivalence to other backends has not been established.

`AscendKernel::build_contract` uses `ruda.ascend.common-row.v1` for these programs;
map mode retains its original schema. Map `tile_elements` is independent of row
width. Row statistics use their own buffer length and no CPU math fallback exists.
See `tools/ascend/validate_common_rows.py` for a strict real-toolchain/device run.
