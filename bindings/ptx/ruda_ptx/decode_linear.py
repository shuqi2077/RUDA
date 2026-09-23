"""Small-row PTX linear kernels with input reuse and optional gate fusion.

Contiguous [N,K] weights, FP32 dot-product accumulation. No Tensor Core claim.
The retained v8 implementation is in decode_linear_v8 for A/B comparisons.
"""
import math
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, _hex, _load, _store, _round, _start
from .reductions import warp_reduce


def _shape(a, weight):
    if len(a.shape) != 2 or len(weight.shape) != 2 or a.dtype != weight.dtype:
        raise ValueError("Decode linear expects same-dtype 2D input and weights")
    m, k = a.shape
    n, kw = weight.shape
    if k != kw or m > 65535:
        raise ValueError("Decode linear shape mismatch or grid.y overflow")
    TensorSpec((m, n), a.dtype)
    return m, k, n


def linear_decode(name: str, a: TensorSpec, weight: TensorSpec, *, bias=False,
                  outputs_per_warp: int = 1) -> Kernel:
    """Reuse one input value for 1, 2 or 4 output accumulators per warp.

    Larger tiles reduce redundant input loads but also reduce block count and
    increase register demand. They are tuning candidates, not always faster.
    """
    m, k, n = _shape(a, weight)
    if type(bias) is not bool:
        raise TypeError("bias must be bool")
    if type(outputs_per_warp) is not int or outputs_per_warp not in (1, 2, 4):
        raise ValueError("outputs_per_warp must be 1, 2 or 4")
    tile = outputs_per_warp
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
    mul.lo.u32 %r3, %r3, {tile};
    mov.u32 %r4, %ctaid.y;
    setp.ge.u32 %p0, %r3, {n};
    @%p0 bra DONE;
    mov.u32 %r5, %r1;
"""
    for j in range(tile):
        body += f"    mov.f32 %f{j}, {_hex(0.0)};\n"
    body += f"""DOT:
    setp.ge.u32 %p0, %r5, {k};
    @%p0 bra REDUCE;
    mad.lo.u32 %r6, %r4, {k}, %r5;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd0, %rd3;
    {_load(a.dtype, '%f8', '%rd4')}
"""
    for j in range(tile):
        body += f"""    add.u32 %r7, %r3, {j};
    setp.ge.u32 %p0, %r7, {n};
    @%p0 bra SKIP_DOT_{j};
    mad.lo.u32 %r6, %r7, {k}, %r5;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd1, %rd3;
    {_load(a.dtype, '%f9', '%rd4')}
    fma.rn.f32 %f{j}, %f8, %f9, %f{j};
SKIP_DOT_{j}:
"""
    body += "    add.u32 %r5, %r5, 32;\n    bra DOT;\nREDUCE:\n"
    for j in range(tile):
        body += warp_reduce(f"%f{j}")
    body += "    setp.ne.u32 %p0, %r1, 0;\n    @%p0 bra DONE;\n"
    if bias:
        body += "    ld.param.u64 %rd5, [bias];\n"
    for j in range(tile):
        body += f"""    add.u32 %r7, %r3, {j};
    setp.ge.u32 %p0, %r7, {n};
    @%p0 bra DONE;
"""
        if bias:
            body += f"""    mul.wide.u32 %rd3, %r7, {size};
    add.u64 %rd4, %rd5, %rd3;
    {_load(a.dtype, '%f9', '%rd4')}
    add.f32 %f{j}, %f{j}, %f9;
"""
        body += f"""    mad.lo.u32 %r6, %r4, {n}, %r7;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd2, %rd3;
    {_store(a.dtype, '%rd4', f'%f{j}')}
"""
    body += "DONE:\n    ret;\n}\n"
    return Kernel(name, "linear", body, params, ((n + 4*tile - 1)//(4*tile), m, 1), (128, 1, 1), sm)


def gated_linear(name: str, a: TensorSpec, weight: TensorSpec, *,
                 gate_bias: bool = False, up_bias: bool = False) -> Kernel:
    """Compute SiLU(linear(x, gate)) * linear(x, up) in one decode kernel.

    Both weights must have the supplied [N,K] shape/dtype. FP32 dot accumulation;
    round EACH projection and the SiLU intermediate to the storage dtype before
    the next operation, preserving the unfused low-precision tensor boundaries.
    """
    m, k, n = _shape(a, weight)
    if m > 4:
        raise ValueError("Fused gated projection is a decode-only candidate (M <= 4)")
    if type(gate_bias) is not bool or type(up_bias) is not bool:
        raise TypeError("Bias flags must be bool")
    params = (("a", "gate_weight", "up_weight") + (("gate_bias",) if gate_bias else ())
              + (("up_bias",) if up_bias else ()) + ("out",))
    body, sm = _start(name, a.dtype, params)
    size = DTYPE_BYTES[a.dtype]
    body += f"""    ld.param.u64 %rd0, [a];
    ld.param.u64 %rd1, [gate_weight];
    ld.param.u64 %rd2, [up_weight];
    ld.param.u64 %rd6, [out];
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
    mov.f32 %f1, {_hex(0.0)};
DOT:
    setp.ge.u32 %p0, %r5, {k};
    @%p0 bra REDUCE;
    mad.lo.u32 %r6, %r4, {k}, %r5;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd0, %rd3;
    {_load(a.dtype, '%f8', '%rd4')}
    mad.lo.u32 %r6, %r3, {k}, %r5;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd1, %rd3;
    {_load(a.dtype, '%f9', '%rd4')}
    fma.rn.f32 %f0, %f8, %f9, %f0;
    add.u64 %rd4, %rd2, %rd3;
    {_load(a.dtype, '%f9', '%rd4')}
    fma.rn.f32 %f1, %f8, %f9, %f1;
    add.u32 %r5, %r5, 32;
    bra DOT;
REDUCE:
""" + warp_reduce("%f0") + warp_reduce("%f1")
    body += "    setp.ne.u32 %p0, %r1, 0;\n    @%p0 bra DONE;\n"
    for flag, parameter, reg in ((gate_bias, "gate_bias", "%f0"), (up_bias, "up_bias", "%f1")):
        if flag:
            body += f"""    ld.param.u64 %rd5, [{parameter}];
    mul.wide.u32 %rd3, %r3, {size};
    add.u64 %rd4, %rd5, %rd3;
    {_load(a.dtype, '%f9', '%rd4')}
    add.f32 {reg}, {reg}, %f9;
"""
    body += "    " + _round(a.dtype, "%f0") + "\n    " + _round(a.dtype, "%f1") + "\n"
    body += f"""    neg.f32 %f2, %f0;
    mul.f32 %f2, %f2, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f2, %f2;
    add.f32 %f2, %f2, {_hex(1.0)};
    div.rn.f32 %f0, %f0, %f2;
    {_round(a.dtype, '%f0')}
    mul.f32 %f0, %f0, %f1;
    mad.lo.u32 %r6, %r4, {n}, %r3;
    mul.wide.u32 %rd3, %r6, {size};
    add.u64 %rd4, %rd6, %rd3;
    {_store(a.dtype, '%rd4', '%f0')}
DONE:
    ret;
}}
"""
    return Kernel(name, "gated_linear", body, params, ((n+3)//4, m, 1), (128, 1, 1), sm)
