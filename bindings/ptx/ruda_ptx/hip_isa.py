"""Explicit AMD native code-object reference runtime through HIP module APIs.

Consumes linked .hsaco, never PTX or CUDA/HIP C++ source. Requires the installed
AMD ROCm/HIP runtime/driver; it is not RUDA's Rust runtime. A target is explicitly
supplied; compatibility with the physical GPU is ultimately checked by the HIP
code-object loader (this adapter does not guess struct layouts to probe gfx ID).
"""
from __future__ import annotations
import ctypes as C
from contextlib import contextmanager
import sys
import threading
from .isa import AMD_TARGETS, NativeImage
from .runtime import Buffer


class HipIsaError(RuntimeError):
    pass


class HipIsaRuntime:
    name = "amd_hsaco_reference_not_ruda_rust"

    def __init__(self, target: str, device: int = 0):
        if target not in AMD_TARGETS:
            raise ValueError("Unsupported AMD target; no alternate ISA is selected")
        if sys.platform != "linux" or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError("AMD ISA reference runtime supports 64-bit Linux only")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a nonnegative integer")
        self.target, self.device = target, device
        self.target_probe = "explicit_target; physical_compatibility_checked_by_hip_module_loader"
        self._closed, self._lock, self._owner = False, threading.RLock(), object()
        self._buffers, self._modules = {}, {}
        self._stream = C.c_void_p()
        try:
            self._lib = C.CDLL("libamdhip64.so")
        except OSError as exc:
            raise HipIsaError("AMD HIP runtime library is unavailable; no PTX/CUDA/CPU fallback") from exc
        self._bind()
        self._check(self._hipInit(0), "hipInit")
        count = C.c_int()
        self._check(self._hipGetDeviceCount(C.byref(count)), "hipGetDeviceCount")
        if device >= count.value:
            raise HipIsaError("Requested AMD device is unavailable")
        with self._guard():
            self._check(self._hipStreamCreate(C.byref(self._stream)), "hipStreamCreate")

    def _bind(self):
        P, I, U, Z = C.c_void_p, C.c_int, C.c_uint, C.c_size_t
        signatures = {
            "hipInit": [U], "hipGetDeviceCount": [C.POINTER(I)],
            "hipGetDevice": [C.POINTER(I)], "hipSetDevice": [I],
            "hipMalloc": [C.POINTER(P), Z], "hipFree": [P],
            "hipMemcpyHtoD": [P, P, Z], "hipMemcpyDtoH": [P, P, Z],
            "hipStreamCreate": [C.POINTER(P)], "hipStreamDestroy": [P], "hipStreamSynchronize": [P],
            "hipModuleLoadData": [C.POINTER(P), P], "hipModuleUnload": [P],
            "hipModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
            "hipModuleLaunchKernel": [P, U, U, U, U, U, U, U, P, C.POINTER(P), C.POINTER(P)],
        }
        for name, args in signatures.items():
            try:
                fn = getattr(self._lib, name)
            except AttributeError as exc:
                raise HipIsaError(f"HIP runtime does not provide {name}") from exc
            fn.argtypes, fn.restype = args, I
            setattr(self, "_" + name, fn)
        self._hipGetErrorString = self._lib.hipGetErrorString
        self._hipGetErrorString.argtypes, self._hipGetErrorString.restype = [I], C.c_char_p

    def _check(self, result, operation):
        if result:
            message = self._hipGetErrorString(result)
            raise HipIsaError(f"{operation}: {result}: {(message or b'unknown error').decode(errors='replace')}")

    @contextmanager
    def _guard(self):
        with self._lock:
            if self._closed:
                raise HipIsaError("AMD ISA runtime is closed")
            old = C.c_int()
            self._check(self._hipGetDevice(C.byref(old)), "hipGetDevice")
            self._check(self._hipSetDevice(self.device), "hipSetDevice")
            try:
                yield
            finally:
                self._check(self._hipSetDevice(old.value), "restore-hip-device")

    def _validate(self, buffer):
        if (not isinstance(buffer, Buffer) or buffer.owner is not self._owner
                or self._buffers.get(buffer.handle) is not buffer):
            raise ValueError("Foreign, freed, or invalid AMD buffer")

    def validate_buffer(self, buffer):
        with self._guard():
            self._validate(buffer)

    def allocate(self, nbytes):
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Allocation must have a positive integer size")
        with self._guard():
            ptr = C.c_void_p()
            self._check(self._hipMalloc(C.byref(ptr), nbytes), "hipMalloc")
            buffer = Buffer(ptr.value, nbytes, self._owner)
            self._buffers[ptr.value] = buffer
            return buffer

    def free(self, buffer):
        with self._guard():
            self._validate(buffer)
            self._check(self._hipStreamSynchronize(self._stream), "synchronize-before-free")
            self._check(self._hipFree(buffer.handle), "hipFree")
            del self._buffers[buffer.handle]

    def write(self, buffer, data):
        if not isinstance(data, bytes):
            raise TypeError("AMD upload requires bytes")
        with self._guard():
            self._validate(buffer)
            if len(data) > buffer.nbytes:
                raise ValueError("Upload exceeds allocation")
            if data:
                self._check(self._hipStreamSynchronize(self._stream), "synchronize-before-upload")
                raw = C.create_string_buffer(data)
                self._check(self._hipMemcpyHtoD(buffer.handle, C.cast(raw, C.c_void_p), len(data)), "hipMemcpyHtoD")

    def read(self, buffer, nbytes):
        with self._guard():
            self._validate(buffer)
            if type(nbytes) is not int or not 0 <= nbytes <= buffer.nbytes:
                raise ValueError("Invalid readback size")
            if nbytes == 0:
                return b""
            self._check(self._hipStreamSynchronize(self._stream), "synchronize-before-readback")
            out = C.create_string_buffer(nbytes)
            self._check(self._hipMemcpyDtoH(C.cast(out, C.c_void_p), buffer.handle, nbytes), "hipMemcpyDtoH")
            return out.raw

    def load_image(self, image: NativeImage):
        if not isinstance(image, NativeImage) or image.isa != "amdgcn" or image.target != self.target:
            raise ValueError("HSACO image does not match this explicitly selected ISA target")
        with self._guard():
            if image.digest in self._modules:
                return image.digest
            raw = C.create_string_buffer(image.code, len(image.code))
            module, function = C.c_void_p(), C.c_void_p()
            self._check(self._hipModuleLoadData(C.byref(module), C.cast(raw, C.c_void_p)), "hipModuleLoadData(hsaco)")
            try:
                self._check(self._hipModuleGetFunction(C.byref(function), module, image.entry.encode()), "hip-native-entry")
            except BaseException:
                self._hipModuleUnload(module)
                raise
            self._modules[image.digest] = (module, function)
            return image.digest

    def _validate_launch(self, loaded, image, buffers):
        if loaded != image.digest or loaded not in self._modules:
            raise ValueError("Native image was not loaded in this runtime")
        if len(buffers) != len(image.parameters):
            raise ValueError("Wrong native argument count")
        for index, buffer in enumerate(buffers):
            self._validate(buffer)
            if buffer.handle % image.pointer_alignment:
                raise ValueError("Buffer does not meet the native vector-load alignment requirement")
            if image.required_bytes and buffer.nbytes < image.required_bytes[index]:
                raise ValueError("Buffer is smaller than the native kernel contract")

    def _launch_unlocked(self, loaded, image, buffers):
        scalars = [C.c_uint64(buffer.handle) for buffer in buffers]
        pointers = (C.c_void_p * len(scalars))(*(C.cast(C.byref(v), C.c_void_p) for v in scalars))
        self._check(self._hipModuleLaunchKernel(self._modules[loaded][1], *image.grid, *image.block,
                    image.shared_bytes, self._stream, pointers, None), "hipModuleLaunchKernel")

    def launch_image(self, loaded, image, buffers):
        with self._guard():
            self._validate_launch(loaded, image, buffers)
            self._launch_unlocked(loaded, image, buffers)

    def launch_many_images(self, calls):
        calls = tuple((loaded, image, tuple(buffers)) for loaded, image, buffers in calls)
        with self._guard():
            for loaded, image, buffers in calls:
                self._validate_launch(loaded, image, buffers)
            for loaded, image, buffers in calls:
                self._launch_unlocked(loaded, image, buffers)

    def synchronize(self):
        with self._guard():
            self._check(self._hipStreamSynchronize(self._stream), "hipStreamSynchronize")

    def close(self):
        if self._closed:
            return
        with self._guard():
            self._check(self._hipStreamSynchronize(self._stream), "close-synchronize")
            for ptr in list(self._buffers):
                self._check(self._hipFree(ptr), "close-free")
                del self._buffers[ptr]
            for key, (module, _) in list(self._modules.items()):
                self._check(self._hipModuleUnload(module), "close-module")
                del self._modules[key]
            self._check(self._hipStreamDestroy(self._stream), "close-stream")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
