"""Explicit native-image adapter for the existing tensor plan/executor.

The source plan provides operator/buffer identities. Backend-specific images
provide their OWN launch geometry. No PTX interpretation or fallback occurs.
"""
from __future__ import annotations
from typing import Protocol
from .isa import NativeImage
from .runtime import Buffer


class NativeIsaRuntime(Protocol):
    name: str
    def allocate(self, nbytes: int) -> Buffer: ...
    def free(self, buffer: Buffer) -> None: ...
    def write(self, buffer: Buffer, data: bytes) -> None: ...
    def read(self, buffer: Buffer, nbytes: int) -> bytes: ...
    def validate_buffer(self, buffer: Buffer) -> None: ...
    def load_image(self, image: NativeImage): ...
    def launch_image(self, loaded, image: NativeImage, buffers: tuple[Buffer, ...]) -> None: ...
    def synchronize(self) -> None: ...


class NativePlanRuntime:
    def __init__(self, runtime: NativeIsaRuntime, images: dict[str, NativeImage]):
        if runtime is None:
            raise ValueError("An explicit native ISA runtime is required")
        self.runtime, self.images = runtime, dict(images)
        self.name = "native-plan:" + runtime.name
        self._loaded = {}
        for identity, image in self.images.items():
            if not isinstance(image, NativeImage) or identity != image.source_digest:
                raise ValueError("Native image map contains an invalid source identity")

    def allocate(self, nbytes): return self.runtime.allocate(nbytes)
    def free(self, buffer): return self.runtime.free(buffer)
    def write(self, buffer, data): return self.runtime.write(buffer, data)
    def read(self, buffer, nbytes): return self.runtime.read(buffer, nbytes)
    def validate_buffer(self, buffer): return self.runtime.validate_buffer(buffer)
    def synchronize(self): return self.runtime.synchronize()

    def load(self, kernel):
        image = self.images.get(kernel.digest)
        if image is None:
            raise ValueError(f"Missing native ISA lowering for {kernel.name}; no PTX/CPU fallback")
        if image.entry != kernel.name or image.parameters != kernel.parameters:
            raise ValueError("Native image entry or pointer ABI does not match the source plan")
        self._loaded[kernel.digest] = self.runtime.load_image(image)
        return kernel.digest

    def _call(self, loaded, kernel, buffers):
        if loaded != kernel.digest or loaded not in self._loaded:
            raise ValueError("Source plan kernel was not loaded in this native adapter")
        image = self.images[loaded]
        buffers = tuple(buffers)
        if len(buffers) != len(image.parameters):
            raise ValueError("Wrong native argument count")
        for b in buffers:
            self.runtime.validate_buffer(b)
        return self._loaded[loaded], image, buffers

    def launch(self, loaded, kernel, buffers):
        self.runtime.launch_image(*self._call(loaded, kernel, buffers))

    def launch_many(self, calls):
        converted = tuple(self._call(*call) for call in calls)
        batch = getattr(self.runtime, "launch_many_images", None)
        if batch is not None:
            batch(converted)
        else:
            for call in converted:
                self.runtime.launch_image(*call)
