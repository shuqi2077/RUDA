"""Optional native-PTX reference executor using only the NVIDIA Driver API.

Explicit opt-in. No CUDA toolkit headers, nvcc, NVRTC, libcudart or torch.cuda.
This is NOT RUDA's Rust runtime, and must never be reported as one.
Linux, 64-bit host only. Requires libcuda.so.1 supplied by the installed driver.
"""
from __future__ import annotations
import ctypes as C
from contextlib import contextmanager
import sys
import threading
from .emitter import Kernel
from .runtime import Buffer


class DriverError(RuntimeError):
    pass


class NvidiaDriverRuntime:
    name = "nvidia_driver_reference_not_ruda_rust"

    def __init__(self, device: int = 0):
        if sys.platform != "linux" or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError("The reference driver adapter currently supports 64-bit Linux only")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a nonnegative integer")
        try:
            self._lib = C.CDLL("libcuda.so.1")
        except OSError as exc:
            raise DriverError("NVIDIA driver library libcuda.so.1 is unavailable; no CPU fallback is used") from exc
        self._lock, self._owner = threading.RLock(), object()
        self._buffers, self._modules = {}, {}
        self._closed = False
        self._bind()
        self._check(self._cuInit(0), "cuInit")
        dev = C.c_int()
        self._check(self._cuDeviceGet(C.byref(dev), device), "cuDeviceGet")
        self._device = dev
        major, minor = C.c_int(), C.c_int()
        self._check(self._cuDeviceGetAttribute(C.byref(major), 75, dev), "compute-capability-major")
        self._check(self._cuDeviceGetAttribute(C.byref(minor), 76, dev), "compute-capability-minor")
        self.sm = major.value * 10 + minor.value
        self._ctx, self._stream = C.c_void_p(), C.c_void_p()
        self._check(self._cuDevicePrimaryCtxRetain(C.byref(self._ctx), dev), "retain-primary-context")
        try:
            with self._guard():
                self._check(self._cuStreamCreate(C.byref(self._stream), 0), "cuStreamCreate")
        except BaseException:
            self._cuDevicePrimaryCtxRelease_v2(dev)
            raise

    def _bind(self):
        P, U, I, Z, Q = C.c_void_p, C.c_uint, C.c_int, C.c_size_t, C.c_uint64
        signatures = {
            "cuInit": [U], "cuDeviceGet": [C.POINTER(I), I],
            "cuDeviceGetAttribute": [C.POINTER(I), I, I],
            "cuDevicePrimaryCtxRetain": [C.POINTER(P), I],
            "cuDevicePrimaryCtxRelease_v2": [I],
            "cuCtxPushCurrent_v2": [P], "cuCtxPopCurrent_v2": [C.POINTER(P)],
            "cuStreamCreate": [C.POINTER(P), U], "cuStreamDestroy_v2": [P],
            "cuStreamSynchronize": [P], "cuMemAlloc_v2": [C.POINTER(Q), Z],
            "cuMemFree_v2": [Q], "cuMemcpyHtoD_v2": [Q, P, Z],
            "cuMemcpyDtoH_v2": [P, Q, Z],
            "cuModuleLoadDataEx": [C.POINTER(P), P, U, C.POINTER(I), C.POINTER(P)],
            "cuModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
            "cuModuleUnload": [P],
            "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, C.POINTER(P), C.POINTER(P)],
            "cuGetErrorString": [I, C.POINTER(C.c_char_p)],
        }
        for name, args in signatures.items():
            try:
                fn = getattr(self._lib, name)
            except AttributeError as exc:
                raise DriverError(f"Installed driver does not export {name}") from exc
            fn.argtypes, fn.restype = args, I
            setattr(self, "_" + name, fn)

    def _check(self, result, operation):
        if result:
            text = C.c_char_p()
            self._cuGetErrorString(result, C.byref(text))
            message = text.value.decode(errors="replace") if text.value else "unknown driver error"
            raise DriverError(f"{operation}: {result}: {message}")

    @contextmanager
    def _guard(self):
        with self._lock:
            if self._closed:
                raise DriverError("PTX runtime is closed")
            self._check(self._cuCtxPushCurrent_v2(self._ctx), "push-context")
            try:
                yield
            finally:
                old = C.c_void_p()
                self._check(self._cuCtxPopCurrent_v2(C.byref(old)), "pop-context")

    def _validate(self, buffer):
        if (not isinstance(buffer, Buffer) or buffer.owner is not self._owner
                or self._buffers.get(buffer.handle) is not buffer):
            raise ValueError("Buffer is freed, foreign, or invalid")

    def allocate(self, nbytes):
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Allocation size must be a positive integer")
        with self._guard():
            ptr = C.c_uint64()
            self._check(self._cuMemAlloc_v2(C.byref(ptr), nbytes), "cuMemAlloc_v2")
            buffer = Buffer(ptr.value, nbytes, self._owner)
            self._buffers[ptr.value] = buffer
            return buffer

    def free(self, buffer):
        with self._guard():
            self._validate(buffer)
            self._check(self._cuStreamSynchronize(self._stream), "synchronize-before-free")
            self._check(self._cuMemFree_v2(buffer.handle), "cuMemFree_v2")
            del self._buffers[buffer.handle]

    def write(self, buffer, data):
        if not isinstance(data, bytes):
            raise TypeError("Host upload must be bytes")
        with self._guard():
            self._validate(buffer)
            if len(data) > buffer.nbytes:
                raise ValueError("Upload exceeds allocation")
            if not data:
                return
            self._check(self._cuStreamSynchronize(self._stream), "synchronize-before-upload")
            raw = C.create_string_buffer(data)
            self._check(self._cuMemcpyHtoD_v2(buffer.handle, C.cast(raw, C.c_void_p), len(data)), "cuMemcpyHtoD_v2")

    def read(self, buffer, nbytes):
        with self._guard():
            self._validate(buffer)
            if type(nbytes) is not int or not 0 <= nbytes <= buffer.nbytes:
                raise ValueError("Invalid readback length")
            if nbytes == 0:
                return b""
            self._check(self._cuStreamSynchronize(self._stream), "synchronize-before-readback")
            raw = C.create_string_buffer(nbytes)
            self._check(self._cuMemcpyDtoH_v2(C.cast(raw, C.c_void_p), buffer.handle, nbytes), "cuMemcpyDtoH_v2")
            return raw.raw

    def load(self, kernel):
        if not isinstance(kernel, Kernel):
            raise TypeError("Expected Kernel")
        with self._guard():
            if self.sm < kernel.target_sm:
                raise DriverError(f"PTX requires sm_{kernel.target_sm}; device reports sm_{self.sm}")
            key = kernel.digest
            if key in self._modules:
                return key
            if "\0" in kernel.ptx:
                raise ValueError("PTX source contains an embedded NUL")
            source = C.create_string_buffer(kernel.ptx.encode("utf-8"))
            log = C.create_string_buffer(16384)
            options = (C.c_int * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER, *_SIZE_BYTES
            values = (C.c_void_p * 2)(C.cast(log, C.c_void_p), C.c_void_p(len(log)))
            module, function = C.c_void_p(), C.c_void_p()
            result = self._cuModuleLoadDataEx(C.byref(module), C.cast(source, C.c_void_p), 2, options, values)
            if result:
                try:
                    self._check(result, "PTX JIT")
                except DriverError as exc:
                    raise DriverError(f"{exc}\n{log.value.decode(errors='replace')}") from exc
            try:
                self._check(self._cuModuleGetFunction(C.byref(function), module, kernel.name.encode()), "cuModuleGetFunction")
            except BaseException:
                self._cuModuleUnload(module)
                raise
            self._modules[key] = (module, function)
            return key

    def validate_buffer(self, buffer):
        with self._guard():
            self._validate(buffer)

    def _validate_launch(self, loaded, kernel, buffers):
        if loaded != kernel.digest or loaded not in self._modules:
            raise ValueError("Kernel handle does not match this runtime/kernel")
        if len(buffers) != len(kernel.parameters):
            raise ValueError("Wrong kernel argument count")
        for buffer in buffers:
            self._validate(buffer)

    def _launch_unlocked(self, loaded, kernel, buffers):
        scalars = [C.c_uint64(buffer.handle) for buffer in buffers]
        pointers = (C.c_void_p * len(scalars))(*(C.cast(C.byref(v), C.c_void_p) for v in scalars))
        function = self._modules[loaded][1]
        self._check(self._cuLaunchKernel(function, *kernel.grid, *kernel.block,
                                        kernel.shared_bytes, self._stream, pointers, None), "cuLaunchKernel")

    def launch(self, loaded, kernel, buffers):
        with self._guard():
            self._validate_launch(loaded, kernel, buffers)
            self._launch_unlocked(loaded, kernel, buffers)

    def launch_many(self, calls):
        """One context push/pop per sequence, rather than per kernel.

        This is ordered submission, NOT driver-graph capture or one GPU launch.
        All handles are validated before the first launch. Runtime device faults
        can still occur mid-sequence and surface asynchronously.
        """
        calls = tuple(calls)
        with self._guard():
            for loaded, kernel, buffers in calls:
                self._validate_launch(loaded, kernel, buffers)
            for loaded, kernel, buffers in calls:
                self._launch_unlocked(loaded, kernel, buffers)

    def synchronize(self):
        with self._guard():
            self._check(self._cuStreamSynchronize(self._stream), "cuStreamSynchronize")

    def close(self):
        if self._closed:
            return
        with self._guard():
            self._check(self._cuStreamSynchronize(self._stream), "close-synchronize")
            for ptr in list(self._buffers):
                self._check(self._cuMemFree_v2(ptr), "close-free")
                del self._buffers[ptr]
            for key, (module, _) in list(self._modules.items()):
                self._check(self._cuModuleUnload(module), "close-module")
                del self._modules[key]
            self._check(self._cuStreamDestroy_v2(self._stream), "close-stream")
        self._check(self._cuDevicePrimaryCtxRelease_v2(self._device), "release-primary-context")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()
