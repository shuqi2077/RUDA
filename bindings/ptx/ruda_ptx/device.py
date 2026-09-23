"""Device-resident execution helpers; explicit PTX runtime, no host compute fallback."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from .emitter import Kernel, TensorSpec
from .runtime import Buffer


@dataclass(frozen=True)
class DeviceTensor:
    """A contiguous tensor descriptor, not a torch.Tensor or an owning allocation.

    The creator owns the buffer. Executor/cache-session outputs are borrowed
    until the next submission to that executor/session, or until it is closed.
    Cross-runtime buffers and asynchronous cross-stream use are not supported.
    """
    buffer: Buffer
    spec: TensorSpec

    def __post_init__(self):
        if not isinstance(self.buffer, Buffer) or not isinstance(self.spec, TensorSpec):
            raise TypeError("DeviceTensor requires a Buffer and TensorSpec")
        if self.buffer.nbytes < self.spec.nbytes:
            raise ValueError("Device buffer is smaller than its tensor descriptor")


def validate_tensor(runtime, value: DeviceTensor, expected: TensorSpec):
    if not isinstance(value, DeviceTensor) or value.spec != expected:
        raise ValueError(f"Expected device tensor {expected}")
    validate = getattr(runtime, "validate_buffer", None)
    if not callable(validate):
        raise TypeError("Device APIs require runtime.validate_buffer to reject stale/foreign buffers")
    validate(value.buffer)
    if value.buffer.nbytes < expected.nbytes:
        raise ValueError("Undersized device buffer")


def submit(runtime, calls: Iterable[tuple[object, Kernel, tuple[Buffer, ...]]]):
    calls = tuple(calls)
    if callable(getattr(runtime, "launch_many", None)):
        runtime.launch_many(calls)
    else:
        # Still executes PTX through the supplied runtime; not a compute fallback.
        for loaded, kernel, buffers in calls:
            runtime.launch(loaded, kernel, buffers)
