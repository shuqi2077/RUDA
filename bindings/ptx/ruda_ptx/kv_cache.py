"""Device-resident append-only KV cache with explicit reusable decode sessions.

Uniform prefix length across batch; fixed capacity and contiguous [B,H,T,D]
physical layout. No paging, eviction, prefix sharing, or quantized KV storage.
"""
from __future__ import annotations
import struct
import threading
from .emitter import TensorSpec
from .device import DeviceTensor, validate_tensor, submit
from .cache_kernels import kv_append
from .attention import attention_program


class StaticKVCache:
    def __init__(self, runtime, *, batch: int, kv_heads: int, capacity: int,
                 head_dim: int, value_dim: int | None = None, dtype: str = "float16"):
        if runtime is None or not callable(getattr(runtime, "validate_buffer", None)):
            raise TypeError("An explicit validating PTX runtime is required")
        value_dim = head_dim if value_dim is None else value_dim
        self.k_spec = TensorSpec((batch, kv_heads, capacity, head_dim), dtype)
        self.v_spec = TensorSpec((batch, kv_heads, capacity, value_dim), dtype)
        self.runtime, self.capacity = runtime, capacity
        self._owned, self._append_programs, self._sessions = [], {}, []
        self._lock, self._closed, self._poisoned = threading.RLock(), False, False
        self._length = 0
        self.stats = {"appends": 0, "copied_new_kv_bytes": 0, "history_copy_bytes": 0}
        try:
            self.k_buffer = self._alloc(self.k_spec.nbytes)
            self.v_buffer = self._alloc(self.v_spec.nbytes)
            self.length_buffer = self._alloc(4)
            runtime.write(self.length_buffer, struct.pack("<I", 0))
            self._prepare_append(1)  # No per-token JIT once single-token decode starts.
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note(f"Cache construction cleanup failed: {cleanup}")
            raise

    @property
    def length(self):
        return self._length

    @property
    def nbytes(self):
        return self.k_spec.nbytes + self.v_spec.nbytes + 4

    def _alloc(self, size):
        buf = self.runtime.allocate(size)
        self._owned.append(buf)
        return buf

    def _check_open(self):
        if self._closed or self._poisoned:
            raise RuntimeError("KV cache is closed or failed; create a new cache")

    def _prepare_append(self, tokens):
        if tokens not in self._append_programs:
            b, h, _, d = self.k_spec.shape
            kn = TensorSpec((b, h, tokens, d), self.k_spec.dtype)
            vn = TensorSpec((b, h, tokens, self.v_spec.shape[-1]), self.v_spec.dtype)
            kernels = kv_append(f"ruda_kv_append_{tokens}", kn, vn, self.capacity)
            loaded = tuple(self.runtime.load(k) for k in kernels)
            self._append_programs[tokens] = kn, vn, kernels, loaded
        return self._append_programs[tokens]

    def append(self, k: DeviceTensor, v: DeviceTensor):
        """Enqueue copying NEW tokens only, then advance the device prefix length.

        No allocation/upload/readback/synchronization on a warm append path.
        Callers must not mutate length_buffer; asynchronous faults poison the
        cache only when surfaced by the runtime. Use synchronize at checkpoints.
        """
        with self._lock:
            self._check_open()
            if not isinstance(k, DeviceTensor) or len(k.spec.shape) != 4:
                raise ValueError("K must be a 4D DeviceTensor")
            tokens = k.spec.shape[2]
            if self.length + tokens > self.capacity:
                raise ValueError("KV capacity exceeded; no writes were submitted")
            b, h, _, d = self.k_spec.shape
            expected_k = TensorSpec((b, h, tokens, d), self.k_spec.dtype)
            expected_v = TensorSpec((b, h, tokens, self.v_spec.shape[-1]), self.v_spec.dtype)
            validate_tensor(self.runtime, k, expected_k)
            validate_tensor(self.runtime, v, expected_v)
            if any(x.buffer is owned for x in (k, v) for owned in self._owned):
                raise ValueError("New KV tensors may not alias cache storage")
            _, _, kernels, loaded = self._prepare_append(tokens)
            args = (k.buffer, v.buffer, self.k_buffer, self.v_buffer, self.length_buffer)
            try:
                submit(self.runtime, ((loaded[0], kernels[0], args),
                                      (loaded[1], kernels[1], (self.length_buffer,))))
            except BaseException:
                self._poisoned = True
                raise
            self._length += tokens
            self.stats["appends"] += 1
            self.stats["copied_new_kv_bytes"] += k.spec.nbytes + v.spec.nbytes

    def prepare_decode(self, query_heads: int, *, partitions: int = 1, scale=None):
        with self._lock:
            self._check_open()
            session = DecodeSession(self, query_heads, partitions=partitions, scale=scale)
            self._sessions.append(session)
            return session

    def reset(self):
        """Start a new sequence. Old KV bytes are not cleared, but become invisible.

        Synchronizes intentionally; not intended as a per-token operation.
        """
        with self._lock:
            self._check_open()
            self.runtime.synchronize()
            self.runtime.write(self.length_buffer, struct.pack("<I", 0))
            self._length = 0

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.runtime.synchronize()
            for session in self._sessions:
                session.close()
            while self._owned:
                self.runtime.free(self._owned[-1])
                self._owned.pop()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class DecodeSession:
    """Fixed query shape, stable output/workspace, dynamic length on device.

    Reads every valid historical key, including the just-appended token. This is
    cached decode semantics, not PyTorch's upper-left rectangular causal mask.
    """
    def __init__(self, cache, query_heads, *, partitions, scale):
        self.cache, self.runtime = cache, cache.runtime
        b, _, _, d = cache.k_spec.shape
        self.q_spec = TensorSpec((b, query_heads, 1, d), cache.k_spec.dtype)
        self.program = attention_program("ruda_cached_decode", self.q_spec, cache.k_spec, cache.v_spec,
                                         dynamic_length=True, partitions=partitions, scale=scale)
        self._owned, self._closed = [], False
        try:
            self.loaded = tuple(self.runtime.load(k) for k in self.program.kernels)
            output = self.runtime.allocate(self.program.output.nbytes)
            self._owned.append(output)
            self.output = DeviceTensor(output, self.program.output)
            self.partial = None
            if self.program.partial:
                self.partial = self.runtime.allocate(self.program.partial.nbytes)
                self._owned.append(self.partial)
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note(f"Decode construction cleanup failed: {cleanup}")
            raise

    def run(self, q: DeviceTensor, *, synchronize: bool = False) -> DeviceTensor:
        if type(synchronize) is not bool:
            raise TypeError("synchronize must be bool")
        with self.cache._lock:
            self.cache._check_open()
            if self._closed:
                raise RuntimeError("Decode session is closed")
            validate_tensor(self.runtime, q, self.q_spec)
            if any(q.buffer is b for b in self._owned + self.cache._owned):
                raise ValueError("Query aliases cache/session storage")
            first_out = self.partial if self.partial is not None else self.output.buffer
            args = (q.buffer, self.cache.k_buffer, self.cache.v_buffer, self.cache.length_buffer, first_out)
            calls = [(self.loaded[0], self.program.first, args)]
            if self.program.merge:
                calls.append((self.loaded[1], self.program.merge, (self.partial, self.output.buffer)))
            try:
                submit(self.runtime, calls)
                if synchronize:
                    self.runtime.synchronize()
            except BaseException:
                self.cache._poisoned = True
                raise
            return self.output

    def close(self):
        with self.cache._lock:
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
