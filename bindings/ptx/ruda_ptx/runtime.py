"""Backend-independent PTX execution contract and frozen inference executor.

A RUDA Rust adapter must implement this contract using its real buffers/streams.
This file never imports torch.cuda, libcudart, cuBLAS, cuDNN, or an eager backend.
"""
from __future__ import annotations
from dataclasses import dataclass
import threading
from typing import Any, Protocol
from .emitter import Kernel, TensorSpec
from .frontend import Plan


@dataclass(frozen=True)
class Buffer:
    handle: Any
    nbytes: int
    owner: object


class PtxRuntime(Protocol):
    name: str
    def allocate(self, nbytes: int) -> Buffer: ...
    def free(self, buffer: Buffer) -> None: ...
    def write(self, buffer: Buffer, data: bytes) -> None: ...
    def read(self, buffer: Buffer, nbytes: int) -> bytes: ...
    def load(self, kernel: Kernel) -> Any: ...
    def launch(self, loaded: Any, kernel: Kernel, buffers: tuple[Buffer, ...]) -> None: ...
    def synchronize(self) -> None: ...
    def validate_buffer(self, buffer: Buffer) -> None: ...


class Executor:
    """A single-stream, nonconcurrent executable with reusable device storage.

    In host mode, CPU tensors are an explicit I/O boundary, not an operator
    fallback. Weights upload once; host calls transfer inputs and final outputs.
    run_device instead accepts existing device buffers and returns borrowed
    device outputs without those transfers. The supplied runtime is owned by the
    caller. This is NOT a PyTorch PrivateUse1 device integration.
    """
    def __init__(self, plan: Plan, runtime: PtxRuntime, *, input_mode: str = "host"):
        if runtime is None:
            raise ValueError("An explicit PTX runtime is required; no CUDA or CPU fallback exists")
        if input_mode not in {"host", "device"}:
            raise ValueError("input_mode must be host or device")
        self.input_mode = input_mode
        self.plan, self.runtime = plan, runtime
        self._lock = threading.RLock()
        self._owned, self._buffers, self._loaded = [], {}, []
        self._closed, self._poisoned = False, False
        self._prepared = None
        self.stats = {"calls": 0, "launches": 0, "input_upload_bytes": 0,
                      "output_download_bytes": 0, "constant_upload_bytes": 0, "device_calls": 0}
        try:
            # Reject assembly/target failures before uploading any model weights.
            self._loaded = [runtime.load(step.kernel) for step in plan.steps]
            for name in tuple(plan.constants) + (plan.inputs if input_mode == "host" else ()):
                buffer = runtime.allocate(plan.specs[name].nbytes)
                self._owned.append(buffer)
                self._buffers[name] = buffer
            assignment, sizes = plan.workspace()
            slots = []
            for size in sizes:
                buffer = runtime.allocate(size)
                self._owned.append(buffer)
                slots.append(buffer)
            self._buffers.update({name: slots[index] for name, index in assignment.items()})
            for name, tensor in plan.constants.items():
                data = self._tensor_bytes(tensor)
                runtime.write(self._buffers[name], data)
                self.stats["constant_upload_bytes"] += len(data)
            prepare = getattr(runtime, "prepare_launches", None)
            if input_mode == "host" and prepare is not None:
                self._prepared = prepare((loaded, step.kernel,
                    tuple(self._buffers[n] for n in step.inputs + (step.output,)))
                    for step, loaded in zip(self.plan.steps, self._loaded))
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note(f"PTX cleanup also failed: {cleanup}")
            raise

    @staticmethod
    def _tensor_bytes(tensor):
        import torch
        # Byte view only; BF16 never goes through a lossy numerical NumPy cast.
        return tensor.detach().reshape(-1).view(torch.uint8).numpy().tobytes()

    def __call__(self, *args, **kwargs):
        import torch
        from torch.utils._pytree import tree_flatten, tree_unflatten
        if self.input_mode != "host":
            raise RuntimeError("This executor was built for device inputs; use run_device")
        if torch.is_grad_enabled():
            raise RuntimeError("This executable is inference-only; use torch.inference_mode()")
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError("This PTX executable is closed or failed; construct a new instance")
            flat, in_spec = tree_flatten((args, kwargs))
            if in_spec != self.plan.in_spec or len(flat) != len(self.plan.inputs):
                raise ValueError("Input structure differs from the exported program")
            for name, value in zip(self.plan.inputs, flat):
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.layout != torch.strided or not value.is_contiguous()):
                    raise ValueError("This adapter accepts contiguous CPU tensors at its explicit I/O boundary")
                actual = TensorSpec(tuple(value.shape), str(value.dtype).removeprefix("torch."))
                if actual != self.plan.specs[name]:
                    raise ValueError(f"{name}: expected {self.plan.specs[name]}, received {actual}")
            try:
                for name, value in zip(self.plan.inputs, flat):
                    data = self._tensor_bytes(value)
                    self.runtime.write(self._buffers[name], data)
                    self.stats["input_upload_bytes"] += len(data)
                from .device import submit
                if self._prepared is not None:
                    self._prepared.replay()
                else:
                    calls = [(loaded, step.kernel,
                              tuple(self._buffers[n] for n in step.inputs + (step.output,)))
                             for step, loaded in zip(self.plan.steps, self._loaded)]
                    submit(self.runtime, calls)
                self.stats["launches"] += len(self.plan.steps)
                self.runtime.synchronize()
                outputs, downloaded = [], {}
                for name in self.plan.outputs:
                    if name in downloaded:
                        outputs.append(downloaded[name])
                        continue
                    spec = self.plan.specs[name]
                    raw = self.runtime.read(self._buffers[name], spec.nbytes)
                    if len(raw) != spec.nbytes:
                        raise RuntimeError("PTX runtime returned an incorrect readback length")
                    tensor = torch.frombuffer(bytearray(raw), dtype=getattr(torch, spec.dtype)).reshape(spec.shape)
                    outputs.append(tensor)
                    downloaded[name] = tensor
                    self.stats["output_download_bytes"] += len(raw)
                self.stats["calls"] += 1
                return tree_unflatten(outputs, self.plan.out_spec)
            except BaseException:
                self._poisoned = True
                raise

    def run_device(self, inputs, *, synchronize: bool = False):
        """Enqueue with existing device tensors; no upload/readback/allocation.

        Inputs is a mapping of plan input names to DeviceTensor descriptors.
        Outputs have the exported pytree structure and are borrowed until this
        executor's next run/close. Dependencies must use the same runtime stream.
        This is not PyTorch CUDA tensor interop or a PrivateUse1 device backend.
        """
        from .device import DeviceTensor, validate_tensor, submit
        from torch.utils._pytree import tree_unflatten
        from collections.abc import Mapping
        if type(synchronize) is not bool:
            raise TypeError("synchronize must be bool")
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError("This PTX executable is closed or failed; construct a new instance")
            if not isinstance(inputs, Mapping) or set(inputs) != set(self.plan.inputs):
                raise ValueError("Device input names must exactly match plan.inputs")
            for name in self.plan.inputs:
                validate_tensor(self.runtime, inputs[name], self.plan.specs[name])
                # Reusing this executor's own output as input would violate its
                # statically planned lifetimes. Use a separate executor/buffer.
                if any(inputs[name].buffer is b for b in self._owned):
                    raise ValueError("Input aliases this executor's managed storage")
            buffers = dict(self._buffers)
            buffers.update({n: v.buffer for n, v in inputs.items()})
            try:
                calls = [(loaded, step.kernel,
                          tuple(buffers[n] for n in step.inputs + (step.output,)))
                         for step, loaded in zip(self.plan.steps, self._loaded)]
                submit(self.runtime, calls)
                if synchronize:
                    self.runtime.synchronize()
                self.stats["launches"] += len(calls)
                self.stats["calls"] += 1
                self.stats["device_calls"] += 1
                outputs = {}
                for name in self.plan.outputs:
                    outputs.setdefault(name, DeviceTensor(buffers[name], self.plan.specs[name]))
                return tree_unflatten([outputs[n] for n in self.plan.outputs], self.plan.out_spec)
            except BaseException:
                self._poisoned = True
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            # Never free buffers while prior launches could still be using them.
            self.runtime.synchronize()
            if self._prepared is not None:
                self._prepared.close()
                self._prepared = None
            while self._owned:
                self.runtime.free(self._owned[-1])
                self._owned.pop()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()
