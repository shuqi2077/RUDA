"""Direct-PTX online attention and split-context decode candidates.

No repeated KV heads, no [B,H,Sq,Sk] score tensor. FP32 statistics and weighted
accumulators. This is not a tuned FlashAttention implementation: one warp per
query/partition, no Tensor Core instructions. Finite-input inference only.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, MAX_ELEMENTS, _start, _load, _store, _hex
from .reductions import warp_reduce


@dataclass(frozen=True)
class AttentionProgram:
    first: Kernel
    merge: Kernel | None
    output: TensorSpec
    partial: TensorSpec | None
    partitions: int
    dynamic_length: bool

    @property
    def workspace_bytes(self):
        return self.partial.nbytes if self.partial else 0

    @property
    def kernels(self):
        return (self.first, self.merge) if self.merge else (self.first,)


def attention_program(name: str, q: TensorSpec, k: TensorSpec, v: TensorSpec, *,
                      causal: bool = False, alignment: str = "upper_left",
                      scale: float | None = None, partitions: int = 1,
                      dynamic_length: bool = False) -> AttentionProgram:
    if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
        raise ValueError("Attention layout must be contiguous [batch, heads, sequence, dimension]")
    b, hq, sq, d = q.shape
    bk, hk, capacity, dk = k.shape
    bv, hv, sv, dv = v.shape
    if (b != bk or b != bv or hk != hv or capacity != sv or d != dk
            or hq % hk or q.dtype != k.dtype or k.dtype != v.dtype):
        raise ValueError("Attention batch, KV heads, sequence, contraction or dtype mismatch")
    if d > 256 or dv > 256:
        raise ValueError("This register-tiled attention variant supports dimensions up to 256")
    if type(causal) is not bool or type(dynamic_length) is not bool:
        raise TypeError("Attention flags must be bool")
    if alignment not in {"upper_left", "lower_right"}:
        raise ValueError("Unknown causal alignment")
    if type(partitions) is not int or not 1 <= partitions <= 32:
        raise ValueError("partitions must be an integer in [1, 32]")
    if partitions > 1 and sq != 1:
        raise ValueError("Split-context kernels are decode-only (query length must be one)")
    if dynamic_length and sq != 1:
        raise ValueError("Device-side prefix length is currently decode-only")
    scale = 1 / math.sqrt(d) if scale is None else scale
    if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("Attention scale must be finite FP32")
    rows, size = b * hq * sq, DTYPE_BYTES[q.dtype]
    if rows * partitions > MAX_ELEMENTS:
        raise ValueError("Attention grid exceeds the validated index range")
    out = TensorSpec((b, hq, sq, dv), q.dtype)
    partial = TensorSpec((rows, partitions, dv + 2), "float32") if partitions > 1 else None
    params = ("q", "k", "v") + (("length",) if dynamic_length else ()) + ("out",)
    body, sm = _start(name, q.dtype, params)
    body = body.replace(".f32 %f<32>", ".f32 %f<64>")
    chunk = (capacity + partitions - 1) // partitions
    body += f"""    ld.param.u64 %rd0, [q];
    ld.param.u64 %rd1, [k];
    ld.param.u64 %rd2, [v];
    ld.param.u64 %rd3, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    rem.u32 %r15, %r1, {partitions};
    div.u32 %r1, %r1, {partitions};
    rem.u32 %r4, %r1, {sq};
    div.u32 %r3, %r1, {sq};
    div.u32 %r2, %r3, {hq};
    rem.u32 %r3, %r3, {hq};
    div.u32 %r5, %r3, {hq // hk};
    mad.lo.u32 %r6, %r2, {hk}, %r5;
    mul.lo.u32 %r6, %r6, {capacity};
    mov.u32 %r14, {capacity};
"""
    if dynamic_length:
        body += f"""    ld.param.u64 %rd4, [length];
    ld.global.u32 %r14, [%rd4];
    min.u32 %r14, %r14, {capacity};
"""
    if causal:
        if alignment == "upper_left":
            body += "    add.u32 %r13, %r4, 1;\n"
        else:
            body += f"""    sub.s32 %r13, %r14, {sq};
    add.s32 %r13, %r13, %r4;
    add.s32 %r13, %r13, 1;
    max.s32 %r13, %r13, 0;
"""
        body += "    min.u32 %r14, %r14, %r13;\n"
    body += f"""    mul.lo.u32 %r8, %r15, {chunk};
    add.u32 %r9, %r8, {chunk};
    min.u32 %r9, %r9, %r14;
    mov.f32 %f5, {_hex(-math.inf)};
    mov.f32 %f6, {_hex(0.0)};
"""
    for i in range((d + 31) // 32):
        body += f"""    mov.f32 %f{32+i}, {_hex(0.0)};
    add.u32 %r10, %r0, {i*32};
    setp.ge.u32 %p0, %r10, {d};
    @%p0 bra Q_SKIP_{i};
    mad.lo.u32 %r11, %r1, {d}, %r10;
    mul.wide.u32 %rd5, %r11, {size};
    add.u64 %rd9, %rd0, %rd5;
    {_load(q.dtype, f'%f{32+i}', '%rd9')}
Q_SKIP_{i}:
"""
    for i in range((dv + 31) // 32):
        body += f"    mov.f32 %f{48+i}, {_hex(0.0)};\n"
    body += f"""KEY_LOOP:
    setp.ge.u32 %p0, %r8, %r9;
    @%p0 bra WRITE;
    add.u32 %r12, %r6, %r8;
    mov.f32 %f0, {_hex(0.0)};
"""
    for i in range((d + 31) // 32):
        body += f"""    add.u32 %r10, %r0, {i*32};
    setp.ge.u32 %p0, %r10, {d};
    @%p0 bra K_SKIP_{i};
    mad.lo.u32 %r11, %r12, {d}, %r10;
    mul.wide.u32 %rd5, %r11, {size};
    add.u64 %rd9, %rd1, %rd5;
    {_load(k.dtype, '%f1', '%rd9')}
    fma.rn.f32 %f0, %f{32+i}, %f1, %f0;
K_SKIP_{i}:
"""
    body += warp_reduce("%f0")
    body += f"""    mul.f32 %f0, %f0, {_hex(float(scale))};
    max.f32 %f2, %f5, %f0;
    sub.f32 %f3, %f5, %f2;
    mul.f32 %f3, %f3, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f3, %f3;
    sub.f32 %f4, %f0, %f2;
    mul.f32 %f4, %f4, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f4, %f4;
    fma.rn.f32 %f6, %f6, %f3, %f4;
    mov.f32 %f5, %f2;
"""
    for i in range((dv + 31) // 32):
        body += f"""    add.u32 %r10, %r0, {i*32};
    setp.ge.u32 %p0, %r10, {dv};
    @%p0 bra V_SKIP_{i};
    mad.lo.u32 %r11, %r12, {dv}, %r10;
    mul.wide.u32 %rd5, %r11, {size};
    add.u64 %rd9, %rd2, %rd5;
    {_load(v.dtype, '%f1', '%rd9')}
    mul.f32 %f{48+i}, %f{48+i}, %f3;
    fma.rn.f32 %f{48+i}, %f1, %f4, %f{48+i};
V_SKIP_{i}:
"""
    body += "    add.u32 %r8, %r8, 1;\n    bra KEY_LOOP;\nWRITE:\n"
    if partial:
        body += f"""    mad.lo.u32 %r16, %r1, {partitions}, %r15;
    mul.lo.u32 %r16, %r16, {dv + 2};
    mul.wide.u32 %rd5, %r16, 4;
    add.u64 %rd9, %rd3, %rd5;
    setp.eq.u32 %p0, %r0, 0;
    @%p0 st.global.f32 [%rd9], %f5;
    add.u64 %rd9, %rd9, 4;
    @%p0 st.global.f32 [%rd9], %f6;
"""
    for i in range((dv + 31) // 32):
        body += f"""    add.u32 %r10, %r0, {i*32};
    setp.ge.u32 %p0, %r10, {dv};
    @%p0 bra OUT_SKIP_{i};
"""
        if partial:
            body += f"""    add.u32 %r11, %r16, %r10;
    add.u32 %r11, %r11, 2;
    mul.wide.u32 %rd5, %r11, 4;
    add.u64 %rd9, %rd3, %rd5;
    st.global.f32 [%rd9], %f{48+i};
"""
        else:
            body += f"""    setp.eq.f32 %p1, %f6, {_hex(0.0)};
    @!%p1 div.rn.f32 %f{48+i}, %f{48+i}, %f6;
    mad.lo.u32 %r11, %r1, {dv}, %r10;
    mul.wide.u32 %rd5, %r11, {size};
    add.u64 %rd9, %rd3, %rd5;
    {_store(q.dtype, '%rd9', f'%f{48+i}')}
"""
        body += f"OUT_SKIP_{i}:\n"
    body += "    ret;\n}\n"
    first = Kernel(name, "split_decode_partial" if partial else "online_attention", body, params,
                   (rows * partitions, 1, 1), (32, 1, 1), sm)
    merge = _merge(name + "_merge", rows, dv, partitions, q.dtype) if partial else None
    return AttentionProgram(first, merge, out, partial, partitions, dynamic_length)


def _merge(name, rows, width, partitions, dtype):
    body, sm = _start(name, dtype, ("partial", "out"))
    body = body.replace(".f32 %f<32>", ".f32 %f<64>")
    body += f"""    ld.param.u64 %rd0, [partial];
    ld.param.u64 %rd1, [out];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, 0;
    mov.f32 %f5, {_hex(-math.inf)};
    mov.f32 %f6, {_hex(0.0)};
"""
    for i in range((width + 31) // 32):
        body += f"    mov.f32 %f{48+i}, {_hex(0.0)};\n"
    body += f"""PART_LOOP:
    setp.ge.u32 %p0, %r2, {partitions};
    @%p0 bra WRITE;
    mad.lo.u32 %r3, %r1, {partitions}, %r2;
    mul.lo.u32 %r3, %r3, {width+2};
    mul.wide.u32 %rd2, %r3, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    add.u64 %rd3, %rd3, 4;
    ld.global.f32 %f1, [%rd3];
    setp.eq.f32 %p1, %f1, {_hex(0.0)};
    @%p1 bra NEXT_PART;
    max.f32 %f2, %f5, %f0;
    sub.f32 %f3, %f5, %f2;
    mul.f32 %f3, %f3, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f3, %f3;
    sub.f32 %f4, %f0, %f2;
    mul.f32 %f4, %f4, {_hex(math.log2(math.e))};
    ex2.approx.f32 %f4, %f4;
    mul.f32 %f1, %f1, %f4;
    fma.rn.f32 %f6, %f6, %f3, %f1;
    mov.f32 %f5, %f2;
"""
    for i in range((width + 31) // 32):
        body += f"""    add.u32 %r4, %r0, {32*i};
    setp.ge.u32 %p0, %r4, {width};
    @%p0 bra PART_SKIP_{i};
    add.u32 %r5, %r3, %r4;
    add.u32 %r5, %r5, 2;
    mul.wide.u32 %rd2, %r5, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];
    mul.f32 %f{48+i}, %f{48+i}, %f3;
    fma.rn.f32 %f{48+i}, %f1, %f4, %f{48+i};
PART_SKIP_{i}:
"""
    body += "NEXT_PART:\n    add.u32 %r2, %r2, 1;\n    bra PART_LOOP;\nWRITE:\n"
    for i in range((width + 31) // 32):
        body += f"""    add.u32 %r4, %r0, {32*i};
    setp.ge.u32 %p0, %r4, {width};
    @%p0 bra OUT_SKIP_{i};
    setp.eq.f32 %p1, %f6, {_hex(0.0)};
    @!%p1 div.rn.f32 %f{48+i}, %f{48+i}, %f6;
    mad.lo.u32 %r5, %r1, {width}, %r4;
    mul.wide.u32 %rd2, %r5, {DTYPE_BYTES[dtype]};
    add.u64 %rd3, %rd1, %rd2;
    {_store(dtype, '%rd3', f'%f{48+i}')}
OUT_SKIP_{i}:
"""
    body += "    ret;\n}\n"
    return Kernel(name, "split_decode_merge", body, ("partial", "out"),
                  (rows, 1, 1), (32, 1, 1), sm)
