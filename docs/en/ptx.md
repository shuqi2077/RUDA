# PTX Backend Reference

[Documentation](README.md) · [Compiler guide](compiler-guide.md) · [Examples](samples.md) · [中文](../zh/ptx.md) | [日本語](../ja/ptx.md) | [Deutsch](../de/ptx.md) | [Русский](../ru/ptx.md)

## 1. Scope

`ruda_compiler::ptx` generates PTX text from Ruda Kernel IR. It is neither an interpreter for arbitrary PTX programs nor a generator of final NVIDIA machine code.

Enable `ruda-compiler/ptx` for standalone compilation. To select this path through the CUDA runtime, enable `ruda-driver-cuda/direct-ptx`.

## 2. Target types

| Type/field | Meaning |
| --- | --- |
| `PtxTarget::version: (u32, u32)` | PTX major/minor version |
| `PtxTarget::sm: u32` | SM target |
| `PtxCompilationOptions::target: Option<PtxTarget>` | Explicit target configuration |
| `PtxCompiler` | Direct backend implementing the public Compiler trait |

Compilation returns a validation error when `target` is absent. The standalone compiler does not infer GPU architecture from its host machine.

Environment variable parsing accepts `major.minor` with a major version of at least 6 and a minor version no greater than 9. Syntactic validity alone does not establish driver or generator support for that version.

## 3. Compilation output

`PtxKernel` contains:

- `source`: PTX text.
- `entrypoint`: entry point name.
- `ruda_dim`: the original kernel's workgroup dimensions.
- `shared_memory_bytes`: required shared memory.
- `dynamic_metadata_index`: the dynamic metadata pointer argument position, when needed.

Preserve the argument layout, entry point, and shared memory requirements when calling the execution layer. The text alone does not contain the complete launch contract.

## 4. Error handling

Unsupported IR returns `CompilationError::UnsupportedInstruction`. Invalid configuration or structure returns `CompilationError::Validation`. Diagnostics include the `Direct PTX:` prefix. The backend does not automatically fall back to NVRTC.

Definitions: [PTX module](../../ruda-compiler/src/ptx/mod.rs), [compiler tests](../../ruda-compiler/src/ptx/tests.rs), and [runtime backend selection](../../ruda-driver/cuda/src/compiler_backend.rs).

## 5. Packed FP4 transport and NVFP4 scaled MMA

`StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)` stores two FP4 values per byte. Direct PTX supports bit-preserving copy/index/select/reinterpret and memory transport for this storage. It does not treat a packed byte as a scalar FP4 arithmetic value; unsupported arithmetic or conversion returns an error. Transport alone does not require an `sm_120a` target.

`CoopMma::ExecuteScaled` currently accepts only SM 120 with PTX 8.7 or newer, a `16×8×64` tile, packed E2M1x2 A/B, F32 C/D, E4M3 scales, and `scales_factor = 4`. It emits the `mxf4nvf4.block_scale.scale_vec::4X` instruction and changes the generated target to `sm_120a`. Other architectures, tile shapes, scale formats or factors are rejected; sufficient fragment register capacity is also required.

See [packed operation handling](../../ruda-compiler/src/ptx/packed.rs), [scaled MMA lowering](../../ruda-compiler/src/ptx/matrix/scaled.rs), and the [packed transport/layout example](../../ruda-driver/cuda/examples/ptx_packed.rs). That example consumes exported PTX files and checks byte transport and fragment coordinates; it does not execute NVFP4 MMA arithmetic.

## 6. Automatic lowering and explicit layout choices

U32/U64 division and remainder by nonzero compile-time constants use exact integer strength reduction over the full unsigned input range. Powers of two use a shift or mask; eligible other divisors use a high-word multiply and shift; large divisors admitting only a zero-or-one quotient use a comparison. Remaining divisors use a reciprocal multiply and one integer correction. Remainder formation can use low-word multiply-add. Input and output registers may alias. This applies when the divisor is an IR constant, not by operator name or fixed shape; no numerator-range assumption or floating-point conversion is needed. Dynamic and zero divisors and signed division retain their existing lowering. See [integer lowering](../../ruda-compiler/src/ptx/constant_division.rs).

Other automatic direct-PTX paths include:

- Paired FP16/BF16 arithmetic and conversions where supported by the type and target; paired FP32 add/subtract/multiply and explicit FMA on SM100+ with PTX 8.6+. Separate multiply and add keep separate rounding. Reduced precision is not implicitly enabled.
- Packed/vector instructions for contiguous vector memory accesses, retaining alignment checks and scalar fallback when alignment is not proven. Global memory can use up to 32 bytes per instruction on SM100+ with PTX 8.8+; other supported targets retain the 16-byte cap. Arbitrary slices are not assumed aligned.
- Warp reductions can omit dynamic participation-mask handling only when block geometry and control flow prove participation. Partial or divergent warps retain the active-lane path.

The [row-layout helpers](../../ruda-compiler/src/ptx/row.rs) `compile_row_variants` and `select_row_variant` compile caller-provided IR and select using the caller's correctness-and-timing callback. They do not infer an operator from a kernel name or automatically benchmark during normal compilation. Layout and precision are independent decisions; `FastMath::ReducedPrecision` must be requested explicitly.

PTX is not final machine code. NVIDIA assembly can change instruction counts and register allocation; fewer instructions do not guarantee faster execution. Compare assembled code and device timings for the intended workload. Compilation or assembly alone is not a performance result. The default NVRTC selection and launch/argument contracts are unchanged.
