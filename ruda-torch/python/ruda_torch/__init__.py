import ctypes
import os
from pathlib import Path
import sys

import torch

_name = torch._C._get_privateuse1_backend_name()
if _name != "privateuseone":
    raise RuntimeError(f"PrivateUse1 is already registered as {_name}; use a separate process for RUDA")

_root = Path(__file__).resolve().parents[3]
_filename = "ruda_torch_native.dll" if sys.platform == "win32" else "libruda_torch_native.so"
_packaged_library = Path(__file__).resolve().with_name(_filename)
_default_library = _packaged_library if _packaged_library.is_file() else _root / "target" / "debug" / _filename
_library = Path(os.environ.get("RUDA_TORCH_LIBRARY", _default_library))
_dll_directories = []
if sys.platform == "win32":
    for directory in os.environ.get("RUDA_TORCH_DLL_DIR", "").split(os.pathsep):
        if directory:
            _dll_directories.append(os.add_dll_directory(directory))
_native = ctypes.CDLL(str(_library))

from . import _C

if not hasattr(_native, "ruda_torch_abi_version"):
    raise RuntimeError("RUDA native library is outdated; rebuild Rust and C++ extensions")
_native.ruda_torch_abi_version.restype = ctypes.c_uint32
if _native.ruda_torch_abi_version() != 9 or getattr(_C, "abi_version", None) != 9:
    raise RuntimeError("RUDA native ABI mismatch; rebuild Rust and C++ extensions")

_C.initialize([ctypes.cast(getattr(_native, "ruda_torch_" + name), ctypes.c_void_p).value
               for name in ("alloc", "free", "error", "execute", "transfer", "sync", "fill", "spatial", "addmm", "layer_norm", "rms_norm", "stream", "paged")])
torch.utils.rename_privateuse1_backend("ruda")

def is_available():
    return True

def device_count():
    return 1

def current_device():
    return 0

def _is_in_bad_fork():
    return False

def synchronize(device=None):
    if device not in (None, 0, "ruda", "ruda:0", torch.device("ruda:0")):
        raise ValueError("RUDA currently exposes only ruda:0")
    _C.synchronize()

torch._register_device_module("ruda", sys.modules[__name__])
torch.utils.generate_methods_for_privateuse1_backend()

_native.ruda_torch_counter.argtypes = [ctypes.c_uint32]
_native.ruda_torch_counter.restype = ctypes.c_uint64

def execution_stats():
    return {name: _native.ruda_torch_counter(index) for index, name in enumerate(
        ("kernel_launches", "host_to_device_bytes", "device_to_host_bytes",
         "rublas_calls", "scalar_matmul_calls", "direct_pointwise_calls",
         "addmm_epilogues", "addmm_workspace_bytes_total", "legacy_fp32_temp_bytes_total",
         "warp_softmax_calls", "scalar_softmax_calls", "fused_layer_norm_calls",
         "trusted_index_calls", "storage_reduction_calls", "warp_reduction_calls",
         "fused_rms_norm_calls", "async_dispatches", "paged_split_calls",
         "paged_workspace_allocations", "paged_workspace_bytes_total"))}

from . import _ops

from ._streams import Stream, Event, stream, current_stream, default_stream, record_stream
from ._paged import PagedAttentionPlan
