"""RoPE with model-provided, already position-selected cosine/sine tables.

Input layout [B,H,S,D], tables [S,rotary_dim/2] in the same storage dtype.
Supports split-half and adjacent-pair rotations; unrotated suffix is copied.
No model-specific frequency, context scaling, or position policy is assumed.
"""
from __future__ import annotations
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, _start, _load, _store


def rope(name: str, spec: TensorSpec, rotary_dim: int, *, interleaved=False) -> Kernel:
    if len(spec.shape) != 4:
        raise ValueError("RoPE requires [batch, heads, sequence, dimension]")
    if type(rotary_dim) is not int or rotary_dim < 2 or rotary_dim % 2 or rotary_dim > spec.shape[-1]:
        raise ValueError("rotary_dim must be positive, even and no larger than the head dimension")
    if type(interleaved) is not bool:
        raise TypeError("interleaved must be bool")
    _, _, seq, dim = spec.shape
    half, size = rotary_dim // 2, DTYPE_BYTES[spec.dtype]
    TensorSpec((seq, half), spec.dtype)
    params = ("x", "cos", "sin", "out")
    body, sm = _start(name, spec.dtype, params)
    for i, p in enumerate(params):
        body += f"    ld.param.u64 %rd{i}, [{p}];\n"
    body += f"""    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;
    setp.ge.u32 %p0, %r0, {spec.numel};
    @%p0 bra DONE;
    rem.u32 %r1, %r0, {dim};
    div.u32 %r2, %r0, {dim};
    rem.u32 %r2, %r2, {seq};
    mul.wide.u32 %rd4, %r0, {size};
    add.u64 %rd5, %rd0, %rd4;
    {_load(spec.dtype, '%f0', '%rd5')}
    setp.ge.u32 %p0, %r1, {rotary_dim};
    @%p0 bra WRITE;
"""
    if interleaved:
        body += """    xor.b32 %r3, %r1, 1;
    div.u32 %r4, %r1, 2;
    and.b32 %r5, %r1, 1;
    setp.eq.u32 %p1, %r5, 0;
"""
    else:
        body += f"""    rem.u32 %r4, %r1, {half};
    setp.lt.u32 %p1, %r1, {half};
    add.u32 %r3, %r1, {half};
    @!%p1 sub.u32 %r3, %r1, {half};
"""
    body += f"""    sub.u32 %r5, %r0, %r1;
    add.u32 %r5, %r5, %r3;
    mul.wide.u32 %rd5, %r5, {size};
    add.u64 %rd5, %rd0, %rd5;
    {_load(spec.dtype, '%f1', '%rd5')}
    @%p1 neg.f32 %f1, %f1;
    mad.lo.u32 %r6, %r2, {half}, %r4;
    mul.wide.u32 %rd5, %r6, {size};
    add.u64 %rd6, %rd1, %rd5;
    {_load(spec.dtype, '%f2', '%rd6')}
    add.u64 %rd6, %rd2, %rd5;
    {_load(spec.dtype, '%f3', '%rd6')}
    mul.f32 %f0, %f0, %f2;
    fma.rn.f32 %f0, %f1, %f3, %f0;
WRITE:
    add.u64 %rd5, %rd3, %rd4;
    {_store(spec.dtype, '%rd5', '%f0')}
DONE:
    ret;
}}
"""
    return Kernel(name, "rope_interleaved" if interleaved else "rope_split_half", body, params,
                  ((spec.numel+255)//256, 1, 1), (256, 1, 1), sm)
