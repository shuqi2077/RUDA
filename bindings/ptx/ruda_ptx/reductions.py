"""Warp-first PTX reductions. No CUDA source generation or runtime dependency.

FP32 accumulation. The fused residual path rounds x + residual to the input
storage dtype before normalization, preserving the unfused tensor boundary.
"""
from __future__ import annotations
import math
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, _hex, _start, _load, _store, _round


def threads_for(width: int) -> int:
    return 32 if width <= 128 else (128 if width <= 1024 else 256)


def warp_reduce(reg: str, operation: str = "add") -> str:
    if operation not in {"add", "max"}:
        raise ValueError("Unsupported reduction")
    return "\n".join(
        f"""    mov.b32 %r40, {reg};
    shfl.sync.bfly.b32 %r41, %r40, {offset}, 31, -1;
    mov.b32 %f30, %r41;
    {operation}.f32 {reg}, {reg}, %f30;"""
        for offset in (16, 8, 4, 2, 1)) + "\n"


def block_reduce(reg: str, threads: int, tag: str, operation="add") -> str:
    """All threads must call; reg receives the total in every lane.

    r0 is the linear thread ID. Scratch: r20/r21/r40/r41, rd20..22, f30.
    Exactly two CTA barriers for a multi-warp block; none for one warp.
    """
    code = warp_reduce(reg, operation)
    if threads == 32:
        return code
    identity = 0.0 if operation == "add" else -math.inf
    warps = threads // 32
    code += f"""    and.b32 %r20, %r0, 31;
    shr.u32 %r21, %r0, 5;
    mov.u64 %rd20, partials;
    mul.wide.u32 %rd21, %r21, 4;
    add.u64 %rd21, %rd20, %rd21;
    setp.eq.u32 %p10, %r20, 0;
    @%p10 st.shared.f32 [%rd21], {reg};
    bar.sync 0;
    mov.f32 {reg}, {_hex(identity)};
    setp.ge.u32 %p10, %r0, {warps};
    @%p10 bra {tag}_SKIP_PARTIAL;
    mul.wide.u32 %rd22, %r0, 4;
    add.u64 %rd22, %rd20, %rd22;
    ld.shared.f32 {reg}, [%rd22];
{tag}_SKIP_PARTIAL:
"""
    code += warp_reduce(reg, operation)
    code += f"""    setp.eq.u32 %p10, %r0, 0;
    @%p10 st.shared.f32 [%rd20], {reg};
    bar.sync 0;
    ld.shared.f32 {reg}, [%rd20];
"""
    return code


def rms_norm(name: str, spec: TensorSpec, eps: float, *, residual=False) -> Kernel:
    if type(eps) not in (int, float) or not math.isfinite(eps) or not 0 <= eps <= 3.4028234663852886e38:
        raise ValueError("RMSNorm epsilon must be a finite nonnegative FP32 value")
    if type(residual) is not bool:
        raise TypeError("residual must be bool")
    width, rows = spec.shape[-1], spec.numel // spec.shape[-1]
    threads = threads_for(width)
    params = ("x", "residual", "weight", "out") if residual else ("x", "weight", "out")
    body, sm = _start(name, spec.dtype, params, ".shared .align 4 .b8 partials[32];" if threads > 32 else "")
    body += f"""
    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [weight];
    ld.param.u64 %rd2, [out];
"""
    if residual:
        body += "    ld.param.u64 %rd10, [residual];\n"
    body += f"""    mov.u32 %r0, %tid.x;
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
"""
    if residual:
        body += f"""    add.u64 %rd4, %rd10, %rd3;
    {_load(spec.dtype, '%f3', '%rd4')}
    add.f32 %f1, %f1, %f3;
    {_round(spec.dtype, '%f1')}
"""
    body += f"""    fma.rn.f32 %f0, %f1, %f1, %f0;
    add.u32 %r3, %r3, {threads};
    bra ACCUM;
REDUCE:
""" + block_reduce("%f0", threads, "NORM")
    body += f"""    div.rn.f32 %f2, %f0, {_hex(float(width))};
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
"""
    if residual:
        body += f"""    add.u64 %rd4, %rd10, %rd3;
    {_load(spec.dtype, '%f3', '%rd4')}
    add.f32 %f0, %f0, %f3;
    {_round(spec.dtype, '%f0')}
"""
    body += f"""    mul.wide.u32 %rd9, %r3, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd4, %rd1, %rd9;
    {_load(spec.dtype, '%f1', '%rd4')}
    div.rn.f32 %f0, %f0, %f2;
    mul.f32 %f0, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3;
    {_store(spec.dtype, '%rd4', '%f0')}
    add.u32 %r3, %r3, {threads};
    bra WRITE;
DONE:
    ret;
}}
"""
    return Kernel(name, "residual_rms_norm" if residual else "rms_norm", body, params,
                  (rows, 1, 1), (threads, 1, 1), sm)


def softmax(name: str, spec: TensorSpec) -> Kernel:
    """Stable last-axis softmax, no full-size temporary buffer.

    Three input passes trade bandwidth for a constant-size workspace. This is
    a baseline candidate; no unconditional speedup over optimized PyTorch claimed.
    """
    width, rows = spec.shape[-1], spec.numel // spec.shape[-1]
    threads = threads_for(width)
    body, sm = _start(name, spec.dtype, ("x", "out"), ".shared .align 4 .b8 partials[32];" if threads > 32 else "")
    body += f"""    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mul.lo.u32 %r2, %r1, {width};
    mov.u32 %r3, %r0;
    mov.f32 %f0, {_hex(-math.inf)};
MAX_LOOP:
    setp.ge.u32 %p0, %r3, {width};
    @%p0 bra MAX_REDUCE;
    add.u32 %r4, %r2, %r3;
    mul.wide.u32 %rd2, %r4, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd3, %rd0, %rd2;
    {_load(spec.dtype, '%f1', '%rd3')}
    max.f32 %f0, %f0, %f1;
    add.u32 %r3, %r3, {threads};
    bra MAX_LOOP;
MAX_REDUCE:
""" + block_reduce("%f0", threads, "MAX", "max")
    if threads > 32:
        body += "    bar.sync 0;\n"  # Every warp consumed MAX before SUM overwrites scratch.
    body += f"""    mov.f32 %f2, %f0;
    mov.f32 %f0, {_hex(0.0)};
    mov.u32 %r3, %r0;
SUM_LOOP:
    setp.ge.u32 %p0, %r3, {width};
    @%p0 bra SUM_REDUCE;
    add.u32 %r4, %r2, %r3;
    mul.wide.u32 %rd2, %r4, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd3, %rd0, %rd2;
    {_load(spec.dtype, '%f1', '%rd3')}
    sub.f32 %f1, %f1, %f2;
    mul.f32 %f1, %f1, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f1, %f1;
    add.f32 %f0, %f0, %f1;
    add.u32 %r3, %r3, {threads};
    bra SUM_LOOP;
SUM_REDUCE:
""" + block_reduce("%f0", threads, "SUM")
    body += f"""    mov.f32 %f3, %f0;
    mov.u32 %r3, %r0;
WRITE:
    setp.ge.u32 %p0, %r3, {width};
    @%p0 bra DONE;
    add.u32 %r4, %r2, %r3;
    mul.wide.u32 %rd2, %r4, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd3, %rd0, %rd2;
    {_load(spec.dtype, '%f1', '%rd3')}
    sub.f32 %f1, %f1, %f2;
    mul.f32 %f1, %f1, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f1, %f1;
    div.rn.f32 %f1, %f1, %f3;
    add.u64 %rd3, %rd1, %rd2;
    {_store(spec.dtype, '%rd3', '%f1')}
    add.u32 %r3, %r3, {threads};
    bra WRITE;
DONE:
    ret;
}}
"""
    return Kernel(name, "softmax", body, ("x", "out"), (rows, 1, 1), (threads, 1, 1), sm)
