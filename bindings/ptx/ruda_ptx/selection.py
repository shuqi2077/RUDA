"""Small-k PTX selection for expert routing / greedy token selection.

Returns device int32 indices and float32 values, descending with lowest-index
stable ties. Finite input contract; k <= 8. This is NOT an entire MoE backend,
Top-P sampler, or a replacement for model-specific expert grouping policies.
"""
from dataclasses import dataclass
import math
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, _start, _load, _hex
from .reductions import threads_for


@dataclass(frozen=True)
class TopKProgram:
    kernel: Kernel
    rows: int
    k: int

    @property
    def indices_nbytes(self):
        return self.rows * self.k * 4

    @property
    def values_nbytes(self):
        return self.rows * self.k * 4


def _pair_reduce():
    code = ""
    for offset in (16, 8, 4, 2, 1):
        code += f"""    mov.b32 %r40, %f0;
    shfl.sync.bfly.b32 %r41, %r40, {offset}, 31, -1;
    mov.b32 %f1, %r41;
    shfl.sync.bfly.b32 %r11, %r10, {offset}, 31, -1;
    setp.gt.f32 %p2, %f1, %f0;
    setp.eq.f32 %p3, %f1, %f0;
    setp.lt.u32 %p4, %r11, %r10;
    and.pred %p3, %p3, %p4;
    or.pred %p2, %p2, %p3;
    selp.f32 %f0, %f1, %f0, %p2;
    selp.u32 %r10, %r11, %r10, %p2;
"""
    return code


def topk(name: str, spec: TensorSpec, k: int = 1) -> TopKProgram:
    width, rows = spec.shape[-1], spec.numel // spec.shape[-1]
    if type(k) is not int or not 1 <= k <= min(8, width):
        raise ValueError("This selection variant supports 1 <= k <= min(8, width)")
    threads = threads_for(width)
    warps = threads // 32
    shared = ".shared .align 4 .b8 pair_values[32];\n    .shared .align 4 .b8 pair_indices[32];" if warps > 1 else ""
    body, sm = _start(name, spec.dtype, ("x", "indices", "values"), shared)
    body += f"""    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [values];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    and.b32 %r20, %r0, 31;
    shr.u32 %r21, %r0, 5;
"""
    for round_ in range(k):
        body += f"""    mov.f32 %f0, {_hex(-math.inf)};
    mov.u32 %r10, 2147483647;
    mov.u32 %r2, %r0;
SCAN_{round_}:
    setp.ge.u32 %p0, %r2, {width};
    @%p0 bra REDUCE_{round_};
"""
        for prior in range(round_):
            body += f"    setp.eq.u32 %p0, %r2, %r{24+prior};\n    @%p0 bra NEXT_{round_};\n"
        body += f"""    mad.lo.u32 %r3, %r1, {width}, %r2;
    mul.wide.u32 %rd3, %r3, {DTYPE_BYTES[spec.dtype]};
    add.u64 %rd4, %rd0, %rd3;
    {_load(spec.dtype, '%f1', '%rd4')}
    setp.gt.f32 %p2, %f1, %f0;
    setp.eq.f32 %p3, %f1, %f0;
    setp.lt.u32 %p4, %r2, %r10;
    and.pred %p3, %p3, %p4;
    or.pred %p2, %p2, %p3;
    selp.f32 %f0, %f1, %f0, %p2;
    selp.u32 %r10, %r2, %r10, %p2;
NEXT_{round_}:
    add.u32 %r2, %r2, {threads};
    bra SCAN_{round_};
REDUCE_{round_}:
""" + _pair_reduce()
        if warps > 1:
            body += f"""    mov.u64 %rd10, pair_values;
    mov.u64 %rd11, pair_indices;
    mul.wide.u32 %rd12, %r21, 4;
    add.u64 %rd13, %rd10, %rd12;
    add.u64 %rd14, %rd11, %rd12;
    setp.eq.u32 %p0, %r20, 0;
    @%p0 st.shared.f32 [%rd13], %f0;
    @%p0 st.shared.u32 [%rd14], %r10;
    bar.sync 0;
    mov.f32 %f0, {_hex(-math.inf)};
    mov.u32 %r10, 2147483647;
    setp.ge.u32 %p0, %r0, {warps};
    @%p0 bra SKIP_PART_{round_};
    mul.wide.u32 %rd12, %r0, 4;
    add.u64 %rd13, %rd10, %rd12;
    add.u64 %rd14, %rd11, %rd12;
    ld.shared.f32 %f0, [%rd13];
    ld.shared.u32 %r10, [%rd14];
SKIP_PART_{round_}:
""" + _pair_reduce()
            body += """    setp.eq.u32 %p0, %r0, 0;
    @%p0 st.shared.f32 [%rd10], %f0;
    @%p0 st.shared.u32 [%rd11], %r10;
    bar.sync 0;
    ld.shared.f32 %f0, [%rd10];
    ld.shared.u32 %r10, [%rd11];
"""
        body += f"""    mov.u32 %r{24+round_}, %r10;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra ROUND_END_{round_};
    mad.lo.u32 %r3, %r1, {k}, {round_};
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd1, %rd3;
    st.global.u32 [%rd4], %r10;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f0;
ROUND_END_{round_}:
"""
        # Do not let faster warps overwrite scratch before slower warps loaded it.
        if warps > 1 and round_ < k - 1:
            body += "    bar.sync 0;\n"
    body += "    ret;\n}\n"
    kernel = Kernel(name, "argmax" if k == 1 else "topk", body, ("x", "indices", "values"),
                    (rows, 1, 1), (threads, 1, 1), sm)
    return TopKProgram(kernel, rows, k)
