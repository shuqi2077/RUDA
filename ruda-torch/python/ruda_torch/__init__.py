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
if _native.ruda_torch_abi_version() != 10 or getattr(_C, "abi_version", None) != 10:
    raise RuntimeError("RUDA native ABI mismatch; rebuild Rust and C++ extensions")

_C.initialize([ctypes.cast(getattr(_native, "ruda_torch_" + name), ctypes.c_void_p).value
               for name in ("alloc", "free", "error", "execute", "transfer", "sync", "fill", "spatial", "addmm", "layer_norm", "rms_norm", "stream", "paged")])
torch.utils.rename_privateuse1_backend("ruda")

def is_available():
    return True

def is_initialized():
    """The native bridge is initialized before the device module is registered."""
    return True

def _lazy_init():
    # Importing this module already performs native ABI checks/initialization.
    return None

def device_count():
    return 1

def current_device():
    return 0

def _is_in_bad_fork():
    return False

def get_amp_supported_dtype():
    """Dtypes accepted by torch.autocast(device_type="ruda")."""
    return [torch.float16, torch.bfloat16]

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
         "paged_workspace_allocations", "paged_workspace_bytes_total",
         "static_graph_builds", "static_graph_replays", "static_graph_eager_runs",
         "paged_backward_calls", "paged_backward_history_workspace_bytes_total",
         "paged_backward_placeholder_bytes_total", "paged_backward_ordered_calls",
         "paged_backward_ordered_workspace_allocations", "paged_backward_ordered_workspace_bytes_total"))}

from . import _ops

from ._streams import Stream, Event, stream, current_stream, default_stream, record_stream
from ._paged import PagedAttentionPlan

# Additive extension API: the old eager tensor ABI still loads without it.
_graph_available = False
if hasattr(_native, "ruda_torch_graph_api_version") and hasattr(_C, "initialize_graph"):
    _native.ruda_torch_graph_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_graph_api_version() != 3 or getattr(_C,"graph_api_version",None) != 3:
        raise RuntimeError("RUDA static graph API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_graph(ctypes.cast(_native.ruda_torch_graph,ctypes.c_void_p).value)
    _graph_available = True
from ._graph import StaticGraph, GraphOp

# Training is a separately negotiated extension. Explicit StaticGraph training
# and model compilation remain opt-in. No CPU fallback is installed.
_training_available = False
_C.storage_mean_api = 0
if hasattr(_native, "ruda_torch_training_api_version") and hasattr(_C, "initialize_training"):
    _native.ruda_torch_training_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_training_api_version() != 4 or getattr(_C, "training_api_version", None) != 4:
        raise RuntimeError("RUDA training API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_training(ctypes.cast(_native.ruda_torch_training, ctypes.c_void_p).value)
    _training_available = True
    _C.storage_mean_api = 1
from .training import LayerNorm, layer_norm, RMSNorm, rms_norm, silu_mul, AdamW, GradScaler

# Optional router-weight extension. Existing base ABI 10/training API 4 remain
# loadable without it; a router operation fails clearly instead of falling back.
_router_available = False
if hasattr(_native, "ruda_torch_router_api_version") and hasattr(_C, "initialize_router"):
    _native.ruda_torch_router_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_router_api_version() != 1 or getattr(_C, "router_api_version", None) != 1:
        raise RuntimeError("RUDA router API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_router(ctypes.cast(_native.ruda_torch_router, ctypes.c_void_p).value)
    _router_available = True
from ._router import selected_router_weights

# Selected paged backward has an additive capability check. Old eager tensor
# ABI 10 remains loadable, but the new autograd path never silently falls back
# to materializing all gradients with an older runtime.
_paged_backward_available = False
if hasattr(_native, "ruda_torch_paged_backward_api_version") and hasattr(_C, "initialize_paged_backward"):
    _native.ruda_torch_paged_backward_api_version.restype=ctypes.c_uint32
    version=_native.ruda_torch_paged_backward_api_version()
    if version not in (1,2) or getattr(_C,"paged_backward_api_version",0)<version:
        raise RuntimeError("RUDA paged backward API mismatch; rebuild Rust and C++ extensions")
    _C.initialize_paged_backward(version)
    _paged_backward_available=True

# General model capture is layered above the additive native graph API.
from .compiler import compile, make_backend, CompiledModel, CompiledFunction, NativeCoverageError, GraphExecutionError

from .quantization import LearnedFakeQuantize, learned_fake_quantize

_nf4_available = False
if hasattr(_native, "ruda_torch_nf4_api_version") and hasattr(_C, "initialize_nf4"):
    _native.ruda_torch_nf4_api_version.restype = ctypes.c_uint32
    if _native.ruda_torch_nf4_api_version() != 1 or getattr(_C, "nf4_api_version", None) != 1:
        raise RuntimeError("RUDA NF4 API mismatch; rebuild Rust and C++ libraries")
    _C.initialize_nf4(ctypes.cast(_native.ruda_torch_nf4_decode, ctypes.c_void_p).value)
    _nf4_available = True
from .finetuning import (LoRALinear, NF4Linear, inject_lora, quantize_nf4,
                         adapter_state_dict, load_adapter_state_dict, merge_lora,
                         load_nf4_safetensors, finetune_state_dict, load_finetune_state_dict)
