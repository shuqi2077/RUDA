"""Warp-per-output linear projection for small-row decode workloads.

Weights remain in contiguous [out_features, in_features] layout. Unlike the
16x16 reference GEMM, M=1 does not leave 15 of 16 row threads without outputs.
A candidate shape policy, not a measured tuning result. FP32 accumulation.
"""
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, _hex, _load, _store, _start
from .reductions import warp_reduce


def linear_decode(name: str, a: TensorSpec, weight: TensorSpec, *, bias=False) -> Kernel:
    if len(a.shape) != 2 or len(weight.shape) != 2 or a.dtype != weight.dtype:
        raise ValueError("Decode linear expects same-dtype 2D input and weights")
    m, k = a.shape
    n, kw = weight.shape
    if k != kw or m > 65535:
        raise ValueError("Decode linear shape mismatch or grid.y overflow")
    if type(bias) is not bool:
        raise TypeError("bias must be bool")
    TensorSpec((m, n), a.dtype)
    params = ("a", "b", "bias", "out") if bias else ("a", "b", "out")
    body, sm = _start(name, a.dtype, params)
    size = DTYPE_BYTES[a.dtype]
    body += f"""    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [b];
    ld.param.u64 %rd2, [out];
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    shr.u32 %r2, %r0, 5;
    mov.u32 %r3, %ctaid.x;
    mad.lo.u32 %r3, %r3, 4, %r2;
    mov.u32 %r4, %ctaid.y;
    setp.ge.u32 %p0, %r3, {n};
    @%p0 bra DONE;
    mov.u32 %r5, %r1;
    mov.f32 %f0, {_hex(0.0)};
DOT:
    setp.ge.u32 %p0, %r5, {k};
    @%p0 bra REDUCE;
    mad.lo.u32 %r6, %r4, {k}, %r5;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd0, %rd3;
    {_load(a.dtype, '%f1', '%rd4')}
    mad.lo.u32 %r6, %r3, {k}, %r5;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd1, %rd3;
    {_load(a.dtype, '%f2', '%rd4')}
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u32 %r5, %r5, 32;
    bra DOT;
REDUCE:
""" + warp_reduce("%f0")
    body += "    setp.ne.u32 %p0, %r1, 0;\n    @%p0 bra DONE;\n"
    if bias:
        body += f"""    ld.param.u64 %rd5, [bias];
    mul.wide.u32 %rd3, %r3, {size};
    add.u64 %rd4, %rd5, %rd3;
    {_load(a.dtype, '%f1', '%rd4')}
    add.f32 %f0, %f0, %f1;
"""
    body += f"""    mad.lo.u32 %r6, %r4, {n}, %r3;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd2, %rd3;
    {_store(a.dtype, '%rd4', '%f0')}
DONE:
    ret;
}}
"""
    return Kernel(name, "linear", body, params, ((n+3)//4, m, 1), (128, 1, 1), sm)
