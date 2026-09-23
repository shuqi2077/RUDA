"""Hierarchical PTX Top-K for wide rows (vocabulary logits, generic routing).

Each partition produces K candidates; a second kernel merges only P*K values.
No dense logits readback, no CPU selection, no scratch allocation on warm runs.
Finite inputs only; ties use the lower ORIGINAL index. This is not Top-P or a
complete MoE router. Extra launch/workspace may hurt small rows: explicit opt-in.
"""
from dataclasses import dataclass
import math
import threading
from .emitter import Kernel, TensorSpec, DTYPE_BYTES, MAX_ELEMENTS, _start, _hex, _load
from .selection import _pair_reduce
from .reductions import threads_for
from .device import DeviceTensor, validate_tensor, submit
from .runtime import Buffer


def _emit(name, dtype, rows, width, k, partitions, *, indexed):
    chunk = (width + partitions - 1)//partitions
    threads = threads_for(chunk)
    warps = threads//32
    shared = ".shared .align 4 .b8 pair_values[32];\n    .shared .align 4 .b8 pair_indices[32];" if warps > 1 else ""
    params = ("x", "source_indices", "indices", "values") if indexed else ("x", "indices", "values")
    body, sm = _start(name, dtype, params, shared)
    body += f"""    ld.param.u64 %rd0, [x];
    ld.param.u64 %rd1, [indices];
    ld.param.u64 %rd2, [values];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r12, %ctaid.y;
    mul.lo.u32 %r13, %r12, {chunk};
    mad.lo.u32 %r14, %r1, {partitions}, %r12;
    and.b32 %r20, %r0, 31;
    shr.u32 %r21, %r0, 5;
"""
    if indexed:
        body += "    ld.param.u64 %rd5, [source_indices];\n"
    for turn in range(k):
        body += f"""    mov.f32 %f0, {_hex(-math.inf)};
    mov.u32 %r10, 2147483647;
    mov.u32 %r2, %r0;
SCAN_{turn}:
    setp.ge.u32 %p0, %r2, {chunk};
    @%p0 bra REDUCE_{turn};
    add.u32 %r15, %r13, %r2;
    setp.ge.u32 %p0, %r15, {width};
    @%p0 bra REDUCE_{turn};
    mad.lo.u32 %r3, %r1, {width}, %r15;
"""
        if indexed:
            body += """    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd5, %rd3;
    ld.global.u32 %r16, [%rd4];
"""
        else:
            body += "    mov.u32 %r16, %r15;\n"
        for prior in range(turn):
            body += f"    setp.eq.u32 %p0, %r16, %r{24+prior};\n    @%p0 bra NEXT_{turn};\n"
        body += f"""    mul.wide.u32 %rd3, %r3, {DTYPE_BYTES[dtype]};
    add.u64 %rd4, %rd0, %rd3;
    {_load(dtype, '%f1', '%rd4')}
    setp.gt.f32 %p2, %f1, %f0;
    setp.eq.f32 %p3, %f1, %f0;
    setp.lt.u32 %p4, %r16, %r10;
    and.pred %p3, %p3, %p4;
    or.pred %p2, %p2, %p3;
    selp.f32 %f0, %f1, %f0, %p2;
    selp.u32 %r10, %r16, %r10, %p2;
NEXT_{turn}:
    add.u32 %r2, %r2, {threads};
    bra SCAN_{turn};
REDUCE_{turn}:
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
    @%p0 bra SKIP_PART_{turn};
    mul.wide.u32 %rd12, %r0, 4;
    add.u64 %rd13, %rd10, %rd12;
    add.u64 %rd14, %rd11, %rd12;
    ld.shared.f32 %f0, [%rd13];
    ld.shared.u32 %r10, [%rd14];
SKIP_PART_{turn}:
""" + _pair_reduce()
            body += """    setp.eq.u32 %p0, %r0, 0;
    @%p0 st.shared.f32 [%rd10], %f0;
    @%p0 st.shared.u32 [%rd11], %r10;
    bar.sync 0;
    ld.shared.f32 %f0, [%rd10];
    ld.shared.u32 %r10, [%rd11];
"""
        body += f"""    mov.u32 %r{24+turn}, %r10;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra ROUND_END_{turn};
    mad.lo.u32 %r3, %r14, {k}, {turn};
    mul.wide.u32 %rd3, %r3, 4;
    add.u64 %rd4, %rd1, %rd3;
    st.global.u32 [%rd4], %r10;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f0;
ROUND_END_{turn}:
"""
        if warps > 1 and turn < k-1:
            body += "    bar.sync 0;\n"
    body += "    ret;\n}\n"
    return Kernel(name, "topk_merge" if indexed else "topk_partition", body, params,
                  (rows, partitions, 1), (threads, 1, 1), sm)


@dataclass(frozen=True)
class PartitionedTopKProgram:
    first: Kernel
    merge: Kernel
    spec: TensorSpec
    k: int
    partitions: int

    @property
    def rows(self):
        return self.spec.numel//self.spec.shape[-1]

    @property
    def output_nbytes(self):
        """Size of each output array (int32 indices OR FP32 values)."""
        return self.rows*self.k*4

    @property
    def partial_nbytes(self):
        """Size of each partial array (indices OR values)."""
        return self.output_nbytes*self.partitions

    @property
    def workspace_nbytes(self):
        return 2*self.partial_nbytes


def partitioned_topk(name: str, spec: TensorSpec, k: int = 1, *, partitions: int = 8):
    width, rows = spec.shape[-1], spec.numel//spec.shape[-1]
    if type(k) is not int or not 1 <= k <= min(8, width):
        raise ValueError("1 <= k <= min(8, width) is required")
    if type(partitions) is not int or not 1 <= partitions <= 64:
        raise ValueError("partitions must be an integer in [1, 64]")
    if rows*partitions*k > MAX_ELEMENTS:
        raise ValueError("Partial selection workspace exceeds 31-bit indexing")
    first = _emit(name + "_parts", spec.dtype, rows, width, k, partitions, indexed=False)
    merge = _emit(name + "_merge", "float32", rows, partitions*k, k, 1, indexed=True)
    return PartitionedTopKProgram(first, merge, spec, k, partitions)


@dataclass(frozen=True)
class TopKResult:
    """Borrowed device outputs, overwritten on the next session run.

    indices is a packed int32 Buffer of shape [rows,k]; values is a float32
    DeviceTensor. Reading them back is a separate, explicit caller action.
    """
    indices: Buffer
    values: DeviceTensor
    rows: int
    k: int


class TopKSession:
    """Fixed-shape, same-runtime, ordered-stream Top-K with reusable scratch."""
    def __init__(self, runtime, spec: TensorSpec, k: int = 1, *, partitions: int = 8):
        if runtime is None or not callable(getattr(runtime, "validate_buffer", None)):
            raise TypeError("An explicit validating PTX runtime is required")
        self.runtime = runtime
        self.program = partitioned_topk("ruda_topk", spec, k, partitions=partitions)
        self._owned, self._lock = [], threading.RLock()
        self._closed = self._poisoned = False
        try:
            self.loaded = tuple(runtime.load(x) for x in (self.program.first, self.program.merge))
            def alloc(size):
                b = runtime.allocate(size)
                self._owned.append(b)
                return b
            self._indices = alloc(self.program.partial_nbytes)
            self._values = alloc(self.program.partial_nbytes)
            indices, values = alloc(self.program.output_nbytes), alloc(self.program.output_nbytes)
            self.result = TopKResult(indices, DeviceTensor(values, TensorSpec((self.program.rows, k))), self.program.rows, k)
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note(f"Selection construction cleanup failed: {cleanup}")
            raise

    def run(self, x: DeviceTensor, *, synchronize: bool = False):
        if type(synchronize) is not bool:
            raise TypeError("synchronize must be bool")
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError("Top-K session is closed or failed")
            validate_tensor(self.runtime, x, self.program.spec)
            if any(x.buffer is b for b in self._owned):
                raise ValueError("Input may not alias selection scratch/output")
            try:
                submit(self.runtime, (
                    (self.loaded[0], self.program.first, (x.buffer, self._indices, self._values)),
                    (self.loaded[1], self.program.merge, (self._values, self._indices, self.result.indices, self.result.values.buffer)),
                ))
                if synchronize:
                    self.runtime.synchronize()
            except BaseException:
                self._poisoned = True
                raise
            return self.result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.runtime.synchronize()
            while self._owned:
                self.runtime.free(self._owned[-1])
                self._owned.pop()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
