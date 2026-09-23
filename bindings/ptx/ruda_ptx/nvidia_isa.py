"""PTX -> native NVIDIA cubin using driver linking, with an opt-in disk cache.

No CUDA C++, nvcc, ptxas, NVRTC, or CUDA Runtime library. The NVIDIA driver JIT
compiler is still required for a cache miss. This remains a reference runtime,
NOT an implementation of the unavailable RUDA Rust runtime bridge.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import uuid
from .emitter import Kernel
from .isa import NativeCache, NativeImage, digest_json, MAX_IMAGE_BYTES
from .nvidia_driver import NvidiaDriverRuntime, DriverError


class NvidiaIsaRuntime(NvidiaDriverRuntime):
    name = "nvidia_native_isa_reference_not_ruda_rust"

    def __init__(self, device: int = 0, *, cache_dir=None, optimization: int = 4):
        if type(optimization) is not int or not 0 <= optimization <= 4:
            raise ValueError("Driver optimization level must be an integer from 0 to 4")
        self.optimization = optimization
        self.native_cache = NativeCache(cache_dir) if cache_dir is not None else None
        self._images = {}
        self.isa_stats = {"driver_compiles": 0, "native_module_loads": 0}
        super().__init__(device)
        try:
            version = C.c_int()
            self._check(self._cuDriverGetVersion(C.byref(version)), "cuDriverGetVersion")
            # cuDriverGetVersion is only the API level. Include the actual driver
            # build; without that information prohibit cross-process cache reuse.
            try:
                driver_build = Path("/proc/driver/nvidia/version").read_text()[:4096]
            except OSError:
                driver_build = "unknown-driver-build-process-" + uuid.uuid4().hex
            self.compiler_id = "nvidia-driver-link:" + digest_json({
                "driver_api": version.value, "driver_build": driver_build,
                "optimization": optimization, "adapter": "ruda-native-v1"})
        except BaseException:
            self.close()
            raise

    def _bind(self):
        super()._bind()
        P, I, U, Z = C.c_void_p, C.c_int, C.c_uint, C.c_size_t
        signatures = {
            "cuDriverGetVersion": ([C.POINTER(I)], ("cuDriverGetVersion",)),
            "cuLinkCreate": ([U, C.POINTER(I), C.POINTER(P), C.POINTER(P)], ("cuLinkCreate_v2", "cuLinkCreate")),
            "cuLinkAddData": ([P, I, P, Z, C.c_char_p, U, C.POINTER(I), C.POINTER(P)], ("cuLinkAddData_v2", "cuLinkAddData")),
            "cuLinkComplete": ([P, C.POINTER(P), C.POINTER(Z)], ("cuLinkComplete",)),
            "cuLinkDestroy": ([P], ("cuLinkDestroy",)),
        }
        for name, (args, aliases) in signatures.items():
            fn = next((getattr(self._lib, alias, None) for alias in aliases if hasattr(self._lib, alias)), None)
            if fn is None:
                raise DriverError(f"Native ISA compilation requires driver symbol {name}")
            fn.argtypes, fn.restype = args, I
            setattr(self, "_" + name, fn)

    def _compile_unlocked(self, kernel: Kernel) -> NativeImage:
        if not isinstance(kernel, Kernel) or kernel.target_sm > self.sm or "\0" in kernel.ptx:
            raise ValueError("PTX kernel is invalid or incompatible with this device")
        raw = C.create_string_buffer(kernel.ptx.encode("utf-8"))
        log = C.create_string_buffer(32768)
        # CU_JIT_ERROR_LOG_BUFFER, *_SIZE_BYTES, OPTIMIZATION_LEVEL, TARGET.
        options = (C.c_int * 4)(5, 6, 7, 9)
        values = (C.c_void_p * 4)(C.cast(log, C.c_void_p), C.c_void_p(len(log)),
                                  C.c_void_p(self.optimization), C.c_void_p(self.sm))
        state = C.c_void_p()
        self._check(self._cuLinkCreate(4, options, values, C.byref(state)), "cuLinkCreate")
        original = None
        try:
            self._check(self._cuLinkAddData(state, 1, C.cast(raw, C.c_void_p), len(raw),
                                           (kernel.name + ".ptx").encode(), 0, None, None), "cuLinkAddData(PTX)")
            pointer, size = C.c_void_p(), C.c_size_t()
            self._check(self._cuLinkComplete(state, C.byref(pointer), C.byref(size)), "cuLinkComplete")
            if not pointer.value or not 64 <= size.value <= MAX_IMAGE_BYTES:
                raise DriverError("Driver returned an invalid or oversized cubin image")
            # Driver-owned storage is valid ONLY until cuLinkDestroy.
            code = C.string_at(pointer, size.value)
            image = NativeImage("nvidia-sass", f"sm_{self.sm}", kernel.name, kernel.parameters,
                                kernel.grid, kernel.block, kernel.digest, self.compiler_id,
                                code, shared_bytes=kernel.shared_bytes)
            self.isa_stats["driver_compiles"] += 1
            return image
        except BaseException as exc:
            original = exc
            if isinstance(exc, DriverError):
                raise DriverError(f"{exc}\n{log.value.decode(errors='replace')}") from exc
            raise
        finally:
            result = self._cuLinkDestroy(state)
            if original is None:
                self._check(result, "cuLinkDestroy")

    def compile_native(self, kernel: Kernel) -> NativeImage:
        if not isinstance(kernel, Kernel):
            raise TypeError("Expected a PTX Kernel")
        with self._guard():
            if kernel.target_sm > self.sm:
                raise DriverError(f"PTX requires sm_{kernel.target_sm}; device is sm_{self.sm}")
            if kernel.digest in self._images:
                return self._images[kernel.digest]
            identity = {"isa": "nvidia-sass", "target": f"sm_{self.sm}",
                        "source_digest": kernel.digest, "compiler_id": self.compiler_id}
            image = (self.native_cache.get_or_compile(identity, lambda: self._compile_unlocked(kernel))
                     if self.native_cache is not None else self._compile_unlocked(kernel))
            self._images[kernel.digest] = image
            return image

    def load_image(self, kernel: Kernel, image: NativeImage):
        """Load an explicitly supplied cubin. No PTX fallback on image rejection."""
        if not isinstance(kernel, Kernel) or not isinstance(image, NativeImage):
            raise TypeError("Expected Kernel and NativeImage")
        if (image.isa != "nvidia-sass" or image.target != f"sm_{self.sm}"
                or image.source_digest != kernel.digest or image.entry != kernel.name
                or image.parameters != kernel.parameters or image.grid != kernel.grid
                or image.block != kernel.block or image.shared_bytes != kernel.shared_bytes):
            raise ValueError("Native image does not match this kernel, launch ABI, or exact device target")
        with self._guard():
            if kernel.digest in self._modules:
                return kernel.digest
            raw = C.create_string_buffer(image.code, len(image.code))
            module, function = C.c_void_p(), C.c_void_p()
            self._check(self._cuModuleLoadDataEx(C.byref(module), C.cast(raw, C.c_void_p),
                                                0, None, None), "cuModuleLoadDataEx(cubin)")
            try:
                self._check(self._cuModuleGetFunction(C.byref(function), module, image.entry.encode()), "native-entry")
            except BaseException:
                self._cuModuleUnload(module)
                raise
            self._modules[kernel.digest] = (module, function)
            self.isa_stats["native_module_loads"] += 1
            return kernel.digest

    def load(self, kernel):
        return self.load_image(kernel, self.compile_native(kernel))

    def prepare_launches(self, calls):
        return PreparedLaunches(self, calls)


class PreparedLaunches:
    """Reuse ctypes argument arrays for a fixed-address sequence on one stream.

    This is NOT a GPU graph or a single device launch. Every replay still issues
    one driver launch per kernel. Freed/foreign buffers are checked each time.
    """
    def __init__(self, runtime: NvidiaDriverRuntime, calls):
        self.runtime = runtime
        self.calls = tuple((loaded, kernel, tuple(buffers)) for loaded, kernel, buffers in calls)
        self._closed = False
        self.replays = 0
        self._packed = []
        with runtime._guard():
            for loaded, kernel, buffers in self.calls:
                runtime._validate_launch(loaded, kernel, buffers)
            for loaded, kernel, buffers in self.calls:
                values = [C.c_uint64(buffer.handle) for buffer in buffers]
                pointers = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
                self._packed.append((values, pointers))  # keep scalar storage alive

    def replay(self, *, synchronize=False):
        if type(synchronize) is not bool:
            raise TypeError("synchronize must be bool")
        rt = self.runtime
        with rt._guard():
            if self._closed:
                raise RuntimeError("Prepared sequence is closed")
            for loaded, kernel, buffers in self.calls:
                rt._validate_launch(loaded, kernel, buffers)
            for (loaded, kernel, _), (_, pointers) in zip(self.calls, self._packed):
                rt._check(rt._cuLaunchKernel(rt._modules[loaded][1], *kernel.grid, *kernel.block,
                                            kernel.shared_bytes, rt._stream, pointers, None), "prepared-launch")
            if synchronize:
                rt._check(rt._cuStreamSynchronize(rt._stream), "prepared-synchronize")
            self.replays += 1

    def close(self):
        with self.runtime._lock:
            self._closed = True
            self._packed.clear()
