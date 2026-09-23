"""Direct, dependency-free PTX emission. No CUDA C++, NVCC, or NVRTC.

These kernels establish an inspectable PTX boundary, not a tuned BLAS replacement.
All shapes are specialized and validated before generating pointer arithmetic.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import cached_property
import hashlib
import json
import math
import re
import struct

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
MAX_ELEMENTS = (1 << 31) - 1


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if not self.shape or any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError("Only nonempty static positive shapes are supported")
        if self.numel > MAX_ELEMENTS:
            raise ValueError("Tensor exceeds the validated 31-bit element-index range")

    @property
    def numel(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.numel * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class Kernel:
    name: str
    operation: str
    ptx: str
    parameters: tuple[str, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    target_sm: int
    shared_bytes: int = 0  # Dynamic shared memory only; static arrays live in PTX.

    @cached_property
    def digest(self):
        # Launch geometry and parameter order are part of the cache identity.
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _hex(x: float) -> str:
    return "0f" + struct.pack(">f", x).hex()


def _start(name, dtype, parameters, shared=""):
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
        raise ValueError("Invalid PTX entry name")
    sm = 80 if dtype == "bfloat16" else 70
    p = ",\n".join(f"    .param .u64 {v}" for v in parameters)
    return f"""// Generated directly by ruda_ptx; candidate, hardware validation required.
.version 7.0
.target sm_{sm}
.address_size 64
.visible .entry {name}(\n{p}\n)
{{
    .reg .pred %p<16>;
    .reg .b32 %r<48>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<32>;
    .reg .b16 %h<4>;
    {shared}
""", sm


def _load(dtype, dst, addr):
    if dtype == "float32":
        return f"ld.global.f32 {dst}, [{addr}];"
    if dtype == "float16":
        return f"ld.global.b16 %h0, [{addr}];\n    cvt.f32.f16 {dst}, %h0;"
    # BF16 widening is exact bit placement; no newer cvt ISA needed.
    return (f"ld.global.b16 %h0, [{addr}];\n    cvt.u32.u16 %r46, %h0;\n"
            f"    shl.b32 %r46, %r46, 16;\n    mov.b32 {dst}, %r46;")


def _store(dtype, addr, src):
    if dtype == "float32":
        return f"st.global.f32 [{addr}], {src};"
    scalar = "f16" if dtype == "float16" else "bf16"
    return f"cvt.rn.{scalar}.f32 %h1, {src};\n    st.global.b16 [{addr}], %h1;"


def _round(dtype, reg):
    if dtype == "float32":
        return ""
    if dtype == "float16":
        return f"cvt.rn.f16.f32 %h2, {reg};\n    cvt.f32.f16 {reg}, %h2;"
    return (f"cvt.rn.bf16.f32 %h2, {reg};\n    cvt.u32.u16 %r46, %h2;\n"
            f"    shl.b32 %r46, %r46, 16;\n    mov.b32 {reg}, %r46;")


def elementwise(name: str, operation: str, spec: TensorSpec) -> Kernel:
    if operation not in {"add", "mul", "silu", "silu_mul"}:
        raise ValueError(f"Unsupported elementwise operation: {operation}")
    binary = operation != "silu"
    params = ("x", "y", "out") if binary else ("x", "out")
    body, sm = _start(name, spec.dtype, params)
    body += f"""
    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd2, [out];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    setp.ge.u32 %p0, %r3, {spec.numel};
    @%p0 bra DONE;
    mul.wide.u32 %rd3, %r3, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd4, %rd0, %rd3;
    {_load(spec.dtype, '%f0', '%rd4')}
"""
    if binary:
        body += f"""    ld.param.u64 %rd1, [y];
    add.u64 %rd4, %rd1, %rd3;
    {_load(spec.dtype, '%f1', '%rd4')}
"""
    if operation in {"silu", "silu_mul"}:
        body += f"""    neg.f32 %f2, %f0;
    mul.f32 %f2, %f2, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f2, %f2;
    add.f32 %f2, %f2, {_hex(1.0)};
    div.rn.f32 %f0, %f0, %f2;
"""
        if operation == "silu_mul":
            # Preserve the low-precision SiLU intermediate rounding of eager PyTorch.
            body += "    " + _round(spec.dtype, "%f0") + "\n"
    if operation == "add":
        body += "    add.f32 %f0, %f0, %f1;\n"
    elif operation in {"mul", "silu_mul"}:
        body += "    mul.f32 %f0, %f0, %f1;\n"
    body += f"""    add.u64 %rd4, %rd2, %rd3;
    {_store(spec.dtype, '%rd4', '%f0')}
DONE:
    ret;
}}
"""
    return Kernel(name, operation, body, params, ((spec.numel + 255) // 256, 1, 1), (256, 1, 1), sm)


def rms_norm_reference(name: str, spec: TensorSpec, eps: float) -> Kernel:
    if type(eps) not in (int, float) or not math.isfinite(eps) or not 0 <= eps <= 3.4028234663852886e38:
        raise ValueError("RMSNorm epsilon must be a finite nonnegative FP32 value")
    width, rows = spec.shape[-1], spec.numel // spec.shape[-1]
    body, sm = _start(name, spec.dtype, ("x", "weight", "out"), ".shared .align 4 .b8 sums[1024];")
    body += f"""
    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [weight];
    ld.param.u64 %rd2, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mul.lo.u32 %r2, %r1, {width};
    mov.u32 %r3, %r0;
    mov.f32 %f0, {_hex(0.0)};
ACCUM:
    setp.ge.u32 %p0, %r3, {width};
    @%p0 bra REDUCE;
    add.u32 %r4, %r2, %r3;
    mul.wide.u32 %rd3, %r4, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd4, %rd0, %rd3;
    {_load(spec.dtype, '%f1', '%rd4')}
    fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r3, %r3, 256;
    bra ACCUM;
REDUCE:
    mov.u64 %rd5, sums;
    mul.wide.u32 %rd6, %r0, 4;
    add.u64 %rd7, %rd5, %rd6;
    st.shared.f32 [%rd7], %f0;
    bar.sync 0;
"""
    for offset in (128, 64, 32, 16, 8, 4, 2, 1):
        body += f"""    setp.ge.u32 %p1, %r0, {offset};
    @%p1 bra REDUCE_{offset};
    ld.shared.f32 %f0, [%rd7];
    add.u64 %rd8, %rd7, {4 * offset};
    ld.shared.f32 %f1, [%rd8];
    add.f32 %f0, %f0, %f1;
    st.shared.f32 [%rd7], %f0;
REDUCE_{offset}:
    bar.sync 0;
"""
    body += f"""    ld.shared.f32 %f2, [%rd5];
    div.rn.f32 %f2, %f2, {_hex(float(width))};
    add.f32 %f2, %f2, {_hex(float(eps))};
    sqrt.rn.f32 %f2, %f2;
    mov.u32 %r3, %r0;
WRITE:
    setp.ge.u32 %p0, %r3, {width};
    @%p0 bra DONE;
    add.u32 %r4, %r2, %r3;
    mul.wide.u32 %rd3, %r4, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd4, %rd0, %rd3;
    {_load(spec.dtype, '%f0', '%rd4')}
    mul.wide.u32 %rd9, %r3, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd4, %rd1, %rd9;
    {_load(spec.dtype, '%f1', '%rd4')}
    div.rn.f32 %f0, %f0, %f2;
    mul.f32 %f0, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3;
    {_store(spec.dtype, '%rd4', '%f0')}
    add.u32 %r3, %r3, 256;
    bra WRITE;
DONE:
    ret;
}}
"""
    return Kernel(name, "rms_norm", body, ("x", "weight", "out"), (rows, 1, 1), (256, 1, 1), sm)


def matmul(name: str, a: TensorSpec, b: TensorSpec, *, transpose_b=False, bias=False) -> Kernel:
    if len(a.shape) != 2 or len(b.shape) != 2 or a.dtype != b.dtype:
        raise ValueError("Matmul requires same-dtype, contiguous 2D matrices")
    if type(transpose_b) is not bool or type(bias) is not bool:
        raise ValueError("Matmul flags must be bool")
    m, k = a.shape
    n, kb = b.shape if transpose_b else (b.shape[1], b.shape[0])
    if k != kb:
        raise ValueError("Matmul contraction dimensions do not match")
    TensorSpec((m, n), a.dtype)
    if (m + 15) // 16 > 65535:
        raise ValueError("Matrix row count exceeds the configured grid.y limit")
    params = ("a", "b", "bias", "out") if bias else ("a", "b", "out")
    body, sm = _start(name, a.dtype, params, ".shared .align 16 .b8 tile_a[1024];\n    .shared .align 16 .b8 tile_b[1024];")
    s = DTYPE_BYTES[a.dtype]
    body += f"""
    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %tid.y;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ctaid.y;
    mad.lo.u32 %r4, %r3, 16, %r1;
    mad.lo.u32 %r5, %r2, 16, %r0;
    mad.lo.u32 %r6, %r1, 16, %r0;
    mul.wide.u32 %rd3, %r6, 4;
    mov.u64 %rd4, tile_a;
    mov.u64 %rd5, tile_b;
    add.u64 %rd6, %rd4, %rd3;
    add.u64 %rd7, %rd5, %rd3;
    mov.u32 %r7, 0;
    mov.f32 %f0, {_hex(0.0)};
TILE:
    setp.ge.u32 %p0, %r7, {k};
    @%p0 bra WRITE;
    add.u32 %r8, %r7, %r0;
    add.u32 %r9, %r7, %r1;
    mov.f32 %f1, {_hex(0.0)};
    mov.f32 %f2, {_hex(0.0)};
    setp.ge.u32 %p1, %r4, {m};
    setp.ge.u32 %p2, %r8, {k};
    or.pred %p3, %p1, %p2;
    @%p3 bra SKIP_A;
    mad.lo.u32 %r10, %r4, {k}, %r8;
    mul.wide.u32 %rd8, %r10, {s};
    add.u64 %rd9, %rd0, %rd8;
    {_load(a.dtype, '%f1', '%rd9')}
SKIP_A:
    setp.ge.u32 %p1, %r9, {k};
    setp.ge.u32 %p2, %r5, {n};
    or.pred %p3, %p1, %p2;
    @%p3 bra SKIP_B;
"""
    if transpose_b:
        body += f"    mad.lo.u32 %r10, %r5, {k}, %r9;\n"
    else:
        body += f"    mad.lo.u32 %r10, %r9, {n}, %r5;\n"
    body += f"""    mul.wide.u32 %rd8, %r10, {s};
    add.u64 %rd9, %rd1, %rd8;
    {_load(a.dtype, '%f2', '%rd9')}
SKIP_B:
    st.shared.f32 [%rd6], %f1;
    st.shared.f32 [%rd7], %f2;
    bar.sync 0;
    mov.u32 %r11, 0;
DOT:
    mad.lo.u32 %r12, %r1, 16, %r11;
    mad.lo.u32 %r13, %r11, 16, %r0;
    mul.wide.u32 %rd10, %r12, 4;
    mul.wide.u32 %rd11, %r13, 4;
    add.u64 %rd12, %rd4, %rd10;
    add.u64 %rd13, %rd5, %rd11;
    ld.shared.f32 %f1, [%rd12];
    ld.shared.f32 %f2, [%rd13];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u32 %r11, %r11, 1;
    setp.lt.u32 %p4, %r11, 16;
    @%p4 bra DOT;
    bar.sync 0;
    add.u32 %r7, %r7, 16;
    bra TILE;
WRITE:
    setp.ge.u32 %p1, %r4, {m};
    setp.ge.u32 %p2, %r5, {n};
    or.pred %p3, %p1, %p2;
    @%p3 bra DONE;
"""
    if bias:
        body += f"""    ld.param.u64 %rd14, [bias];
    mul.wide.u32 %rd15, %r5, {s};
    add.u64 %rd16, %rd14, %rd15;
    {_load(a.dtype, '%f1', '%rd16')}
    add.f32 %f0, %f0, %f1;
"""
    body += f"""    mad.lo.u32 %r10, %r4, {n}, %r5;
    mul.wide.u32 %rd8, %r10, {s};
    add.u64 %rd9, %rd2, %rd8;
    {_store(a.dtype, '%rd9', '%f0')}
DONE:
    ret;
}}
"""
    return Kernel(name, "linear" if transpose_b else "matmul", body, params,
                  ((n + 15) // 16, (m + 15) // 16, 1), (16, 16, 1), sm)


def rms_norm(name: str, spec: TensorSpec, eps: float) -> Kernel:
    """Warp-first v8 path; the v7 kernel remains available for A/B validation."""
    from .reductions import rms_norm as optimized
    return optimized(name, spec, eps)
