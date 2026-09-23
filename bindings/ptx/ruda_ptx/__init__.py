"""PTX-first inference compiler candidate; no implicit device/runtime selection."""
from .emitter import Kernel, TensorSpec
from .frontend import Plan, UnsupportedGraph, compile_exported
from .runtime import Buffer, Executor, PtxRuntime

__version__ = "0.10.0"
__all__ = ["Buffer", "Executor", "Kernel", "Plan", "PtxRuntime", "TensorSpec", "UnsupportedGraph", "compile_exported"]

from .device import DeviceTensor
from .kv_cache import StaticKVCache, DecodeSession
from .attention import AttentionProgram, attention_program
from .rotary import rope
from .selection import TopKProgram, topk
__all__ += ["DeviceTensor", "StaticKVCache", "DecodeSession", "AttentionProgram", "attention_program", "rope", "TopKProgram", "topk"]
from .partitioned_selection import PartitionedTopKProgram, TopKSession, TopKResult, partitioned_topk
__all__ += ["PartitionedTopKProgram", "TopKSession", "TopKResult", "partitioned_topk"]

from .isa import NativeImage, NativeCache
from .native_plan import NativeIsaRuntime, NativePlanRuntime
__all__ += ["NativeImage", "NativeCache", "NativeIsaRuntime", "NativePlanRuntime"]
