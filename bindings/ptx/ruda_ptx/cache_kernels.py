"""Bit-preserving PTX KV append into fixed-capacity device allocations.

The length update is a separate ordered launch: updating it in the copy kernel
would race other blocks that still read the old prefix length.
"""
from __future__ import annotations
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, _start


def kv_append(name: str, k: TensorSpec, v: TensorSpec, capacity: int) -> tuple[Kernel, Kernel]:
    if len(k.shape) != 4 or len(v.shape) != 4 or k.shape[:3] != v.shape[:3] or k.dtype != v.dtype:
        raise ValueError("New K/V must be matching [batch, kv_heads, tokens, dimension] tensors")
    if type(capacity) is not int or capacity < k.shape[2]:
        raise ValueError("Capacity must fit all new tokens")
    b, h, tokens, dk = k.shape
    dv = v.shape[-1]
    TensorSpec((b, h, capacity, dk), k.dtype)
    TensorSpec((b, h, capacity, dv), v.dtype)
    params = ("k_new", "v_new", "k_cache", "v_cache", "length")
    body, sm = _start(name, k.dtype, params)
    for i, p in enumerate(params):
        body += f"    ld.param.u64 %rd{i}, [{p}];\n"
    body += f"""    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;
    ld.global.u32 %r1, [%rd4];
    setp.gt.u32 %p0, %r1, {capacity-tokens};
    @%p0 bra DONE;
"""
    size = DTYPE_BYTES[k.dtype]
    raw_type, reg = ("b32", "%r18") if size == 4 else ("b16", "%h3")
    for label, spec, src, dst in (("K", k, "%rd0", "%rd2"), ("V", v, "%rd1", "%rd3")):
        d = spec.shape[-1]
        body += f"""    setp.ge.u32 %p0, %r0, {spec.numel};
    @%p0 bra {label}_DONE;
    div.u32 %r3, %r0, {tokens*d};
    div.u32 %r4, %r0, {d};
    rem.u32 %r4, %r4, {tokens};
    rem.u32 %r5, %r0, {d};
    add.u32 %r4, %r4, %r1;
    mad.lo.u32 %r6, %r3, {capacity}, %r4;
    mad.lo.u32 %r6, %r6, {d}, %r5;
    mul.wide.u32 %rd5, %r0, {size};
    add.u64 %rd6, {src}, %rd5;
    ld.global.{raw_type} {reg}, [%rd6];
    mul.wide.u32 %rd5, %r6, {size};
    add.u64 %rd6, {dst}, %rd5;
    st.global.{raw_type} [%rd6], {reg};
{label}_DONE:
"""
    body += "DONE:\n    ret;\n}\n"
    copy = Kernel(name, "kv_append_copy", body, params, ((max(k.numel, v.numel)+255)//256, 1, 1), (256, 1, 1), sm)
    advance, sm = _start(name + "_advance", k.dtype, ("length",))
    advance += f"""    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra DONE;
    ld.param.u64 %rd0, [length];
    ld.global.u32 %r0, [%rd0];
    setp.gt.u32 %p0, %r0, {capacity-tokens};
    @%p0 bra DONE;
    add.u32 %r0, %r0, {tokens};
    st.global.u32 [%rd0], %r0;
DONE:
    ret;
}}
"""
    return copy, Kernel(name+"_advance", "kv_append_advance", advance, ("length",), (1, 1, 1), (32, 1, 1), sm)
