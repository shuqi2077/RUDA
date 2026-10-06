"""Model-independent LoRA and frozen, packed NF4 linear layers.

NF4 weights use row-major bytes (first value in the high nibble), FP32
absolute maxima per flat block, and the published NF4 codebook. This is a
versioned RUDA format, not a bitsandbytes/PEFT checkpoint reader. Runtime
decoding stays on the input device: the FP16/BF16 fused path decodes shared
matrix tiles inside GEMM; the reference/FP32 path decodes bounded row tiles.
"""
from __future__ import annotations

import math
import copy
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


NF4_CODEBOOK = (-1.0, -0.6961928009986877, -0.5250730514526367,
                -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0)
_FLOATS = (torch.float32, torch.float16, torch.bfloat16)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


@torch.no_grad()
def pack_nf4(weight, *, block_size=64, chunk_blocks=1024):
    """Explicit CPU preprocessing; bounded FP32 temporary, no GPU download.

    Quantizes nearest codebook value, with lower-code tie breaking. Partial
    final blocks and odd element counts are supported. Returns bytes/scales.
    """
    _positive_int(block_size, 'block_size')
    _positive_int(chunk_blocks, 'chunk_blocks')
    if block_size % 2:
        raise ValueError('block_size must be even')
    if weight.device.type != 'cpu' or weight.dtype not in _FLOATS:
        raise ValueError('pack_nf4 requires an explicit CPU floating weight')
    if weight.ndim != 2 or not all(weight.shape) or not weight.is_contiguous():
        raise ValueError('weight must be a nonempty contiguous [out,in] matrix')
    flat = weight.detach().view(-1)
    packed = torch.empty((flat.numel() + 1) // 2, dtype=torch.uint8, device='cpu')
    scales = torch.empty((flat.numel() + block_size - 1) // block_size, dtype=torch.float32, device='cpu')
    table = torch.tensor(NF4_CODEBOOK, dtype=torch.float32, device='cpu')
    boundaries = (table[:-1] + table[1:]) * .5
    chunk = block_size * chunk_blocks
    for begin in range(0, flat.numel(), chunk):
        end = min(begin + chunk, flat.numel())
        values = flat[begin:end].float()
        if not torch.isfinite(values).all():
            raise ValueError('NF4 weights must be finite')
        padded = F.pad(values, (0, (-values.numel()) % block_size)).view(-1, block_size)
        maximum = padded.abs().amax(1)
        normalized = padded / torch.where(maximum == 0, 1., maximum).unsqueeze(1)
        codes = torch.bucketize(normalized, boundaries).to(torch.uint8).flatten()[:end-begin]
        if codes.numel() % 2:
            codes = F.pad(codes, (0, 1), value=7)
        packed[begin//2:(end+1)//2] = (codes[::2] << 4) | codes[1::2]
        scales[begin//block_size:(end+block_size-1)//block_size] = maximum
    return packed, scales


def _decode(packed, scales, table, begin, rows, width, block_size, dtype):
    if packed.device.type == 'ruda':
        from . import _C, _nf4_available
        if not _nf4_available:
            raise RuntimeError('RUDA NF4 API 1 required; rebuild Rust and C++ libraries')
        return torch.ops.ruda.nf4_decode(packed, scales, table, begin, rows, width, block_size, _FLOATS.index(dtype))
    # Explicit same-device reference, never a fallback from a failed RUDA call.
    if packed.device.type not in ('cpu', 'cuda'):
        raise ValueError('NF4 supports ruda, cpu reference, or cuda reference devices')
    indices = torch.arange(begin * width, (begin + rows) * width, device=packed.device)
    byte = packed[indices // 2].to(torch.int64)
    codes = torch.where(indices % 2 == 0, byte >> 4, byte & 15)
    return (table[codes] * scales[indices // block_size]).to(dtype).view(rows, width)


class _NF4LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scales, table, outputs, width, block_size, tile_rows):
        flat = x.reshape(-1, width)
        fused = False
        if x.device.type == 'ruda' and x.dtype in (torch.float16, torch.bfloat16):
            from . import _nf4_matmul_available
            fused = _nf4_matmul_available
        if fused:
            result = torch.ops.ruda.nf4_matmul(flat.contiguous(), packed, scales, table, outputs, width, block_size, False)
        else:
            result = x.new_empty((flat.shape[0], outputs))
            for start in range(0, outputs, tile_rows):
                rows = min(tile_rows, outputs-start)
                weight = _decode(packed, scales, table, start, rows, width, block_size, x.dtype)
                result[:, start:start+rows].copy_(F.linear(flat, weight))
        ctx.fused = fused
        ctx.save_for_backward(packed, scales, table)
        ctx.geometry = outputs, width, block_size, tile_rows
        ctx.input_shape, ctx.input_dtype = x.shape, x.dtype
        return result.view(*x.shape[:-1], outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        if not ctx.needs_input_grad[0]:
            return (None,) * 8
        packed, scales, table = ctx.saved_tensors
        outputs, width, block_size, tile_rows = ctx.geometry
        grad = gradient.reshape(-1, outputs)
        if ctx.fused:
            result = torch.ops.ruda.nf4_matmul(grad.to(ctx.input_dtype).contiguous(), packed, scales, table,
                                 outputs, width, block_size, True)
            return (result.to(ctx.input_dtype).view(ctx.input_shape),) + (None,) * 7
        result = torch.zeros((grad.shape[0], width), dtype=torch.float32, device=grad.device)
        for start in range(0, outputs, tile_rows):
            rows = min(tile_rows, outputs-start)
            weight = _decode(packed, scales, table, start, rows, width, block_size, ctx.input_dtype)
            left = grad[:, start:start+rows].to(ctx.input_dtype).contiguous()
            if grad.device.type == 'ruda':
                partial = torch.ops.ruda.mixed_mm(left, weight)
            else:
                partial = left.float() @ weight.float()
            result.add_(partial)
        return (result.to(ctx.input_dtype).view(ctx.input_shape),) + (None,) * 7


class NF4Linear(nn.Module):
    """Frozen NF4 base with first-order input gradients and no dense shadow.

    Native NF4 matmul API 1 uses fused shared-tile dequantization and Tensor Core
    GEMM for FP16/BF16 forward/input gradients, with FP32 accumulators. FP32 and
    older native libraries retain the bounded tiled-decode path. No failed
    fused operation is retried through that path.
    """
    def __init__(self, in_features, out_features, packed, scales, *,
                 block_size=64, tile_rows=128, bias=None):
        super().__init__()
        self.in_features = _positive_int(in_features, 'in_features')
        self.out_features = _positive_int(out_features, 'out_features')
        self.block_size = _positive_int(block_size, 'block_size')
        self.tile_rows = _positive_int(tile_rows, 'tile_rows')
        size = in_features * out_features
        if block_size % 2 or block_size > 2**32-1 or size > 2**32-1:
            raise ValueError('even block_size and at most uint32 weight elements required')
        if packed.dtype != torch.uint8 or packed.shape != ((size+1)//2,):
            raise ValueError('packed must contain two row-major NF4 codes per byte')
        if scales.dtype != torch.float32 or scales.shape != ((size+block_size-1)//block_size,):
            raise ValueError('scales must be one FP32 absmax per block')
        if packed.device != scales.device or not packed.is_contiguous() or not scales.is_contiguous():
            raise ValueError('packed/scales must be contiguous and on the same device')
        if scales.requires_grad:
            raise ValueError('NF4 base scales are frozen, not learned QAT parameters')
        if bias is not None and (bias.shape != (out_features,) or bias.dtype not in _FLOATS or bias.device != packed.device):
            raise ValueError('bias must be a matching floating vector on the weight device')
        self.register_buffer('packed', packed.detach())
        self.register_buffer('scales', scales.detach())
        self.register_buffer('codebook', torch.tensor(NF4_CODEBOOK, dtype=torch.float32, device=packed.device))
        self.register_buffer('bias', None if bias is None else bias.detach())

    @classmethod
    def from_linear(cls, linear, *, block_size=64, tile_rows=128):
        if not isinstance(linear, nn.Linear):
            raise TypeError('expected nn.Linear')
        packed, scales = pack_nf4(linear.weight, block_size=block_size)
        result = cls(linear.in_features, linear.out_features, packed, scales,
                     block_size=block_size, tile_rows=tile_rows,
                     bias=None if linear.bias is None else linear.bias.detach().clone())
        return result.train(linear.training)

    def _apply(self, fn, recurse=True):
        # Device/dtype conversion must not round the FP32 quantization metadata.
        scales, table = self.scales, self.codebook
        result = super()._apply(fn, recurse)
        self.scales = scales.to(device=self.packed.device, dtype=torch.float32)
        self.codebook = table.to(device=self.packed.device, dtype=torch.float32)
        return result

    def get_extra_state(self):
        return {'version': 1, 'format': 'ruda-nf4', 'in_features': self.in_features,
                'out_features': self.out_features, 'block_size': self.block_size}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('NF4 checkpoint format/geometry mismatch')

    def forward(self, x):
        if x.dtype not in _FLOATS or x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError('NF4 input must be floating and end in in_features')
        if x.device != self.packed.device:
            raise ValueError('NF4 input and weight must use the same device')
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = _NF4LinearFunction.apply(x, self.packed, self.scales, self.codebook,
                                        self.out_features, self.in_features, self.block_size, self.tile_rows)
            return y if self.bias is None else y + self.bias.to(x.dtype)


class LoRALinear(nn.Module):
    """Frozen dense/NF4 base plus alpha/rank * B(A(x))."""
    def __init__(self, base, *, rank=16, alpha=16., adapter_dtype=torch.float32, dropout=0., use_rslora=False):
        super().__init__()
        if not isinstance(base, (nn.Linear, NF4Linear)):
            raise TypeError('LoRA requires nn.Linear or NF4Linear')
        _positive_int(rank, 'rank')
        if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        if adapter_dtype not in _FLOATS:
            raise ValueError('adapter_dtype must be FP32, FP16 or BF16')
        if type(dropout) not in (int,float) or not 0<=dropout<=1 or type(use_rslora) is not bool:
            raise ValueError('LoRA dropout must be in [0,1] and use_rslora must be bool')
        self.base = base
        self.rank, self.alpha = rank, float(alpha)
        self.dropout, self.use_rslora = float(dropout), use_rslora
        self.in_features, self.out_features = base.in_features, base.out_features
        device = base.weight.device if isinstance(base, nn.Linear) else base.packed.device
        # Initialization is an explicit setup operation, not a runtime CPU fallback.
        a = torch.empty((rank, self.in_features), dtype=adapter_dtype, device='cpu')
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(a.to(device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank), dtype=adapter_dtype, device='cpu').to(device))
        for parameter in base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.train(base.training)

    def get_extra_state(self):
        result = {'version': 1, 'rank': self.rank, 'alpha': self.alpha}
        if self.dropout or self.use_rslora:
            result.update(version=2,dropout=self.dropout,use_rslora=self.use_rslora)
        return result

    @property
    def scaling(self):
        return self.alpha/(math.sqrt(self.rank) if self.use_rslora else self.rank)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('LoRA checkpoint configuration mismatch')

    def forward(self, x):
        result = self.base(x)
        adapted = x.to(self.lora_A.dtype)
        if self.training and self.dropout:
            if x.device.type=='ruda':
                from .random import native_dropout
                adapted = native_dropout(adapted,self.dropout)[0]
            else:
                adapted = F.dropout(adapted,p=self.dropout,training=True)
        update = F.linear(F.linear(adapted, self.lora_A), self.lora_B)
        return result + (update * self.scaling).to(result.dtype)


def _selected(model, target_modules, types):
    if target_modules != 'all-linear':
        if isinstance(target_modules, str) or not isinstance(target_modules, Sequence) or not target_modules:
            raise ValueError("target_modules must be 'all-linear' or a nonempty sequence of full module names")
        if any(not isinstance(name, str) for name in target_modules) or len(set(target_modules)) != len(target_modules):
            raise ValueError('target module names must be unique strings')
    modules = list(model.named_modules(remove_duplicate=False))
    selected = [(name, module) for name, module in modules
                if isinstance(module, types) and (target_modules == 'all-linear' or name in target_modules)]
    if not selected or any(not name for name, _ in selected):
        raise ValueError('select non-root linear modules (wrap a root Linear in a container)')
    if target_modules != 'all-linear' and set(target_modules) != {name for name, _ in selected}:
        raise ValueError('target module missing or has unsupported type')
    selected_ids = {id(module) for _, module in selected}
    return [(name, module) for name, module in modules if id(module) in selected_ids]


def _replace(model, replacements):
    for path, value in replacements:
        parent_name, _, child = path.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, value)


def inject_lora(model, *, target_modules, rank=16, alpha=16., adapter_dtype=torch.float32,dropout=0.,use_rslora=False):
    """In-place injection; freeze base model, preserving shared module aliases.

    Names are exact qualified module paths. No model family or head-selection
    heuristic is used. Construct the optimizer AFTER injection.
    """
    if __package__:
        from .parallel_adapters import (_ParallelLoRA,_ParallelNF4,lora_from_parallel_base)
        from .parallel_training import ColumnParallelLinear,RowParallelLinear
        parallel_types=(ColumnParallelLinear,RowParallelLinear,_ParallelNF4)
    else:
        _ParallelLoRA=()
        parallel_types=()
    if any(isinstance(m, (LoRALinear,_ParallelLoRA)) for m in model.modules()):
        raise ValueError('model already contains LoRA adapters')
    selected = _selected(model, target_modules, (nn.Linear, NF4Linear)+parallel_types)
    parallel_paths=[name for name,module in model.named_modules(remove_duplicate=False) if isinstance(module,parallel_types)]
    selected=[(name,module) for name,module in selected if not any(name.startswith(parent+'.') for parent in parallel_paths)]
    if not selected or (target_modules!='all-linear' and not set(target_modules).issubset({name for name,_ in selected})):
        raise ValueError('select complete parallel projections, not their local implementation modules')
    wrappers = {}
    for _, base in selected:
        if id(base) not in wrappers:
            factory=lora_from_parallel_base if isinstance(base,parallel_types) else LoRALinear
            wrappers[id(base)] = factory(base, rank=rank, alpha=alpha, adapter_dtype=adapter_dtype,
                                        dropout=dropout,use_rslora=use_rslora)
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    _replace(model, [(name, wrappers[id(base)]) for name, base in selected])
    return model


def quantize_nf4(model, *, target_modules, block_size=64, tile_rows=128):
    """Explicit CPU layer-by-layer conversion before moving a model to RUDA.

    Tied weights are rejected rather than silently breaking embedding/head ties.
    This does not implement a sharded Hugging Face checkpoint loader.
    """
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('quantize the frozen base before injecting adapters')
    _positive_int(block_size, 'block_size')
    _positive_int(tile_rows, 'tile_rows')
    if block_size % 2:
        raise ValueError('block_size must be even')
    selected = _selected(model, target_modules, nn.Linear)
    owners = {}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), set()).add(id(module))
    for _, linear in selected:
        if len(owners[id(linear.weight)]) != 1:
            raise ValueError('exclude tied weights from NF4 conversion')
        if linear.weight.device.type != 'cpu' or linear.weight.dtype not in _FLOATS or not linear.weight.is_contiguous():
            raise ValueError('NF4 conversion requires contiguous CPU floating weights')
    # Commit each completed layer to avoid retaining the whole dense model plus
    # the whole quantized model. On failure, completed layers remain converted.
    aliases = {}
    for name, linear in selected:
        aliases.setdefault(id(linear), []).append(name)
    del selected, linear
    for names in aliases.values():
        linear = model.get_submodule(names[0])
        layer = NF4Linear.from_linear(linear, block_size=block_size, tile_rows=tile_rows)
        _replace(model, [(name, layer) for name in names])
        del linear
    return model


def adapter_state_dict(model):
    """Portable CPU adapter tensors plus a strict, versioned layout contract."""
    if __package__:
        from .sharded_adapter_interop import has_sharded_units,sharded_adapter_state_dict
        if has_sharded_units(model):return sharded_adapter_state_dict(model)
    layers = {}
    if __package__:
        from .parallel_adapters import _ParallelLoRA,adapter_base_kind,full_adapter_matrix
    else:
        _ParallelLoRA=()
        adapter_base_kind=lambda layer:'nf4' if isinstance(layer.base,NF4Linear) else 'dense'
    for name, layer in model.named_modules():
        if isinstance(layer, (LoRALinear,_ParallelLoRA)):
            layers[name] = {'rank': layer.rank, 'alpha': layer.alpha,
                            'in_features': layer.in_features, 'out_features': layer.out_features,
                            'base': adapter_base_kind(layer),
                            'lora_A': full_adapter_matrix(layer,'lora_A') if isinstance(layer,_ParallelLoRA) else layer.lora_A.detach().cpu().clone(),
                            'lora_B': full_adapter_matrix(layer,'lora_B') if isinstance(layer,_ParallelLoRA) else layer.lora_B.detach().cpu().clone()}
    if not layers:
        raise ValueError('model has no LoRA adapters')
    version = 1
    if any(layer.dropout or layer.use_rslora for layer in model.modules() if isinstance(layer,(LoRALinear,_ParallelLoRA))):
        version = 2
        for name,layer in model.named_modules():
            if isinstance(layer,(LoRALinear,_ParallelLoRA)):
                layers[name].update(dropout=layer.dropout,use_rslora=layer.use_rslora)
    return {'format': 'ruda-lora', 'version': version, 'layers': layers}


@torch.no_grad()
def load_adapter_state_dict(model, state):
    if __package__:
        from .sharded_adapter_interop import has_sharded_units,load_sharded_adapter_state_dict
        if has_sharded_units(model):return load_sharded_adapter_state_dict(model,state)
    if __package__:
        from .parallel_adapters import _ParallelLoRA,adapter_base_kind,local_adapter_matrix
    else:
        _ParallelLoRA=()
        adapter_base_kind=lambda layer:'nf4' if isinstance(layer.base,NF4Linear) else 'dense'
    if not isinstance(state, dict) or set(state) != {'format', 'version', 'layers'} or state['format'] != 'ruda-lora' or state['version'] not in (1,2):
        raise ValueError('unsupported adapter checkpoint')
    layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, (LoRALinear,_ParallelLoRA))}
    if not layers or not isinstance(state['layers'], dict) or set(layers) != set(state['layers']):
        raise ValueError('adapter target names mismatch')
    copies = []
    shared = {}
    for name, layer in layers.items():
        entry = state['layers'][name]
        metadata = {'rank': layer.rank, 'alpha': layer.alpha, 'in_features': layer.in_features,
                    'out_features': layer.out_features, 'base': adapter_base_kind(layer)}
        if state['version']==2:
            metadata.update(dropout=layer.dropout,use_rslora=layer.use_rslora)
        elif layer.dropout or layer.use_rslora:
            raise ValueError('legacy adapter checkpoint does not contain dropout/RSLoRA semantics')
        if not isinstance(entry, dict) or set(entry) != set(metadata) | {'lora_A', 'lora_B'} or any(entry[k] != v for k, v in metadata.items()):
            raise ValueError(f'adapter configuration mismatch: {name}')
        for key in ('lora_A', 'lora_B'):
            src, dst = entry[key], getattr(layer, key)
            shape=(layer.rank,layer.in_features) if key=='lora_A' else (layer.out_features,layer.rank)
            if not isinstance(src, torch.Tensor) or src.shape != shape or src.dtype not in _FLOATS:
                raise ValueError(f'adapter tensor mismatch: {name}.{key}')
            value=local_adapter_matrix(layer,key,src) if isinstance(layer,_ParallelLoRA) else src
            if id(dst) in shared and not torch.equal(shared[id(dst)],value):
                raise ValueError('shared adapter parameters have different checkpoint values')
            shared[id(dst)]=value
            copies.append((dst,value))
    for dst, src in copies:
        dst.copy_(src)


@torch.no_grad()
def merge_lora(model):
    """Permanently replace eval-mode dense adapters with merged Linear layers.

    Packed bases are not silently expanded or requantized. Keep adapters for
    NF4 deployment. Shared base weights are rejected to preserve tie semantics.
    """
    dense_types=(nn.Linear,)
    if __package__:
        from .parallel_adapters import _ParallelLoRA,merge_parallel_lora
        from .parallel_training import ColumnParallelLinear,RowParallelLinear
        dense_types+=(ColumnParallelLinear,RowParallelLinear)
    else:
        _ParallelLoRA=()
    layers = [(name, layer) for name, layer in model.named_modules(remove_duplicate=False) if isinstance(layer, (LoRALinear,_ParallelLoRA))]
    if not layers:
        raise ValueError('model has no adapters')
    owners = {}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            owners.setdefault(id(parameter), set()).add(id(module))
    for name, layer in layers:
        if not name or layer.training or not isinstance(layer.base, dense_types):
            raise ValueError('merge requires non-root, eval-mode dense LoRA layers')
        if len(owners[id(layer.base.weight)]) != 1:
            raise ValueError('cannot merge a base weight shared with another module')
    merged = {}
    for _, layer in layers:
        if id(layer) in merged:
            continue
        if isinstance(layer,_ParallelLoRA):
            merged[id(layer)]=merge_parallel_lora(layer)
            continue
        base = layer.base
        result = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None, device='meta', dtype=base.weight.dtype)
        result.weight = nn.Parameter((base.weight.float() + (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling).to(base.weight.dtype), requires_grad=False)
        if base.bias is not None:
            result.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        merged[id(layer)] = result.eval()
    _replace(model, [(name, merged[id(layer)]) for name, layer in layers])
    return model


def load_nf4_safetensors(model, directory, *, target_modules, device,
                         dtype=torch.bfloat16, block_size=64, tile_rows=128,
                         parameter_dtypes=None, buffer_dtypes=None):
    """Load exact model tensor names from HF-style safetensors, one tensor at a time.

    Construct the model on meta first. Selected Linear weights are quantized on
    CPU and uploaded packed; remaining tensors are uploaded individually. No
    architecture, tokenizer, remote code or key-renaming rules are inferred.
    Nonpersistent meta buffers must be materialized by the model's constructor.
    Failure leaves already-loaded tensors in place; use a fresh meta model to retry.
    """
    from safetensors import safe_open
    directory = Path(directory).resolve()
    if dtype not in _FLOATS:
        raise ValueError('unsupported base dtype')
    _positive_int(block_size, 'block_size')
    _positive_int(tile_rows, 'tile_rows')
    if block_size % 2:
        raise ValueError('block_size must be even')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda', 'ruda'):
        raise ValueError('unsupported execution device')
    selected = _selected(model, target_modules, nn.Linear)
    targets = {name for name, _ in selected}
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    parameter_dtypes = {} if parameter_dtypes is None else dict(parameter_dtypes)
    buffer_dtypes = {} if buffer_dtypes is None else dict(buffer_dtypes)
    for overrides, tensors in ((parameter_dtypes, parameters), (buffer_dtypes, buffers)):
        known = dict(tensors)
        if any(name not in known or not known[name].is_floating_point() or kind not in _FLOATS
               for name, kind in overrides.items()):
            raise ValueError('dtype overrides require exact floating parameter/buffer names')
        tied_dtypes = {}
        for name, tensor in tensors:
            if name in overrides:
                previous = tied_dtypes.setdefault(id(tensor), overrides[name])
                if previous != overrides[name]:
                    raise ValueError('conflicting dtype overrides for tied tensors')
        for name, tensor in tensors:
            if id(tensor) in tied_dtypes:
                overrides[name] = tied_dtypes[id(tensor)]
    if any(p.device.type != 'meta' for _, p in parameters):
        raise ValueError('construct the base parameters on meta before streaming load')
    owners = {}
    for name, parameter in parameters:
        owners.setdefault(id(parameter), []).append(name)
    for name, layer in selected:
        if name + '.weight' in parameter_dtypes:
            raise ValueError(f'exclude dtype-preserved weights from NF4 targets: {name}')
        target_aliases = {path + '.weight' for path, other in selected if other is layer}
        if set(owners[id(layer.weight)]) != target_aliases:
            raise ValueError(f'exclude tied weights from NF4 targets: {name}')
    index_path = directory / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))['weight_map']
        if not isinstance(index, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in index.items()):
            raise ValueError('invalid safetensors weight map')
    else:
        with safe_open(str(directory / 'model.safetensors'), framework='pt', device='cpu') as source:
            index = {name: 'model.safetensors' for name in source.keys()}
    files = {}
    for name, relative in index.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('checkpoint shard must be a file inside the checkpoint directory')
        files[name] = path
    aliases = {}
    for name, tensor in parameters + buffers:
        aliases.setdefault(id(tensor), []).append(name)
    def key_for(name, tensor):
        candidates = [key for key in aliases[id(tensor)] if key in files]
        return name if name in files else (candidates[0] if candidates else None)
    persistent = set(model.state_dict())
    for name, tensor in parameters + buffers:
        key = key_for(name, tensor)
        if key is None:
            if name in persistent or tensor.device.type == 'meta':
                raise ValueError(f'checkpoint missing tensor or uninitialized buffer: {name}')
            continue
        with safe_open(str(files[key]), framework='pt', device='cpu') as source:
            if tuple(source.get_slice(key).get_shape()) != tuple(tensor.shape):
                raise ValueError(f'checkpoint shape mismatch: {name}')
    def read(name, tensor):
        key = key_for(name, tensor)
        if key is None:
            return tensor.detach().cpu()
        with safe_open(str(files[key]), framework='pt', device='cpu') as source:
            value = source.get_tensor(key)
        if value.is_floating_point() != tensor.is_floating_point():
            raise ValueError(f'checkpoint tensor kind mismatch: {name}')
        return value
    replacements = {}
    for name, base in selected:
        if id(base) not in replacements:
            dense_weight = read(name + '.weight', base.weight).contiguous()
            packed, scales = pack_nf4(dense_weight, block_size=block_size)
            del dense_weight
            bias = None if base.bias is None else read(name + '.bias', base.bias).to(dtype=parameter_dtypes.get(name + '.bias', dtype))
            replacements[id(base)] = NF4Linear(base.in_features, base.out_features, packed, scales,
                block_size=block_size, tile_rows=tile_rows, bias=bias).to(device=device).train(base.training)
        _replace(model, [(name, replacements[id(base)])])
    loaded = {}
    # Parameter identities remain available in the captured meta list for ties.
    for name, original in parameters + buffers:
        if any(name == target + '.weight' or name == target + '.bias' for target in targets):
            continue
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        if id(original) not in loaded:
            storage_dtype = (parameter_dtypes.get(name, dtype) if isinstance(original, nn.Parameter)
                             else buffer_dtypes.get(name, original.dtype))
            value = read(name, original).to(device=device, dtype=storage_dtype if original.is_floating_point() else original.dtype)
            loaded[id(original)] = nn.Parameter(value, requires_grad=False) if isinstance(original, nn.Parameter) else value
        setattr(parent, leaf, loaded[id(original)])
    return model


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _optimizer_layout(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    if not names or any(n.rsplit('.', 1)[-1] not in ('lora_A', 'lora_B') for n in names.values()):
        raise ValueError('fine-tuning checkpoint requires adapter-only trainable parameters')
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        row = []
        for p in group['params']:
            if id(p) not in names or id(p) in seen:
                raise ValueError('optimizer must contain each trainable adapter exactly once')
            row.append(names[id(p)])
            seen.add(id(p))
        groups.append(row)
    if seen != set(names):
        raise ValueError('optimizer is missing adapter parameters')
    return groups


def finetune_state_dict(model, optimizer, *, base_id, step, data_state, scaler=None):
    """Snapshot at an optimizer-step boundary, after clearing gradients.

    base_id must identify the exact frozen checkpoint/quantization configuration.
    data_state is caller-owned sampler/cursor state. No frozen base is duplicated.
    Save the returned state with torch.save; this function performs no file writes.
    """
    if not isinstance(base_id, str) or not base_id or type(step) is not int or step < 0:
        raise ValueError('nonempty base_id and nonnegative completed step required')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('clear gradients before snapshot; partial accumulation is not saved')
    layout = _optimizer_layout(model, optimizer)
    cuda_devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    state = {'format': 'ruda-finetune', 'version': 1, 'base_id': base_id, 'step': step,
            'adapter': adapter_state_dict(model), 'optimizer': _cpu_tree(optimizer.state_dict()),
            'optimizer_type': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
            'optimizer_layout': layout, 'data_state': _cpu_tree(data_state),
            'scaler': None if scaler is None else _cpu_tree(scaler.state_dict()),
            'cpu_rng': torch.get_rng_state().clone(),
            'cuda_rng': {i: torch.cuda.get_rng_state(i).cpu() for i in cuda_devices}}
    if any(p.device.type=='ruda' for p in model.parameters()):
        from . import get_rng_state
        state['ruda_rng'] = get_rng_state()
    return state


def load_finetune_state_dict(model, optimizer, state, *, base_id, scaler=None):
    """Restore adapter/optimizer/scaler/RNG; return completed step and data cursor."""
    if state.get('format') != 'ruda-finetune' or state.get('version') != 1 or state.get('base_id') != base_id:
        raise ValueError('fine-tuning checkpoint format/base mismatch')
    optimizer_type = type(optimizer).__module__ + '.' + type(optimizer).__qualname__
    if state['optimizer_layout'] != _optimizer_layout(model, optimizer) or state['optimizer_type'] != optimizer_type:
        raise ValueError('optimizer type/parameter layout mismatch')
    if (scaler is None) != (state['scaler'] is None):
        raise ValueError('scaler configuration mismatch')
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    if devices != sorted(state['cuda_rng']):
        raise ValueError('CUDA RNG device set mismatch')
    load_adapter_state_dict(model, state['adapter'])
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if scaler is not None:
        scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['cpu_rng'])
    for i, rng in state['cuda_rng'].items():
        torch.cuda.set_rng_state(rng, i)
    if 'ruda_rng' in state:
        from . import set_rng_state
        set_rng_state(state['ruda_rng'])
    optimizer.zero_grad(set_to_none=True)
    return state['step'], copy.deepcopy(state['data_state'])
