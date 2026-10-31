# RUDA SOL-ExecBench launcher: loads bundled sm_100 PTX through the CUDA
# driver and launches RUDA kernels on the current PyTorch stream.
import base64
import ctypes
import json
import lzma
import re
import struct

import torch

_DRIVER = []
_KERNELS = {}


def _decode_bundle(encoded):
    data = lzma.decompress(base64.b85decode(encoded))
    if data.startswith(b'RDN1'):
        previous = [0, 0]
        pieces, offset = [], 4
        for match in re.finditer(rb'([\xfe\xff])([\x80-\xff]*[\x00-\x7f])', data[4:]):
            start, end = match.start() + 4, match.end() + 4
            pieces.append(data[offset:start])
            value = 0
            for shift, byte in enumerate(match[2]):
                value |= (byte & 127) << (7 * shift)
            kind = match[1][0] - 254
            previous[kind] += (value >> 1) ^ -(value & 1)
            pieces.append((b'L' if kind == 0 else b'%r') + str(previous[kind]).encode())
            offset = end
        pieces.append(data[offset:])
        data = b''.join(pieces)
    return json.loads(data)


def _check(code, what='CUDA driver call'):
    if code:
        raise RuntimeError(what + ' failed with CUDA error ' + str(code))


def _driver():
    if not _DRIVER:
        driver = ctypes.CDLL('libcuda.so.1')
        signatures = {
            'cuCtxGetCurrent': [ctypes.POINTER(ctypes.c_void_p)],
            'cuModuleLoadData': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            'cuModuleGetFunction': [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
            'cuFuncSetAttribute': [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
            'cuTensorMapEncodeTiled': [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)] + [ctypes.c_int] * 4,
            'cuLaunchKernel': [ctypes.c_void_p] + [ctypes.c_uint] * 7
                + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)],
        }
        for name, args in signatures.items():
            fn = getattr(driver, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
        _DRIVER.append(driver)
    return _DRIVER[0]


class _Kernel:
    __slots__ = ('function', 'block', 'shared', 'widths', 'scalars', 'module')

    def __init__(self, name):
        driver = _driver()
        spec = _BUNDLE[name]
        source = ctypes.create_string_buffer(spec['ptx'].encode())
        module = ctypes.c_void_p()
        _check(driver.cuModuleLoadData(ctypes.byref(module), source), 'cuModuleLoadData(' + name + ')')
        function = ctypes.c_void_p()
        _check(driver.cuModuleGetFunction(ctypes.byref(function), module, name.encode()), 'cuModuleGetFunction(' + name + ')')
        if spec['shared'] > 0:
            _check(driver.cuFuncSetAttribute(function, 8, spec['shared']), 'cuFuncSetAttribute(' + name + ')')
        self.module = module
        self.function = function
        self.block = tuple(spec['block'])
        self.shared = spec['shared']
        self.widths = tuple(spec['widths'])
        self.scalars = tuple((kind, count) for kind, count in spec['scalars'])


def _kernel(name):
    context = ctypes.c_void_p()
    _check(_driver().cuCtxGetCurrent(ctypes.byref(context)))
    key = (context.value, name)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = _Kernel(name)
    return kernel


_MAP_TYPES = {torch.float16: 6, torch.bfloat16: 9, torch.float32: 7, torch.uint8: 0, torch.int8: 0}
if hasattr(torch, 'float8_e4m3fn'):
    _MAP_TYPES[torch.float8_e4m3fn] = 0
    _MAP_TYPES[torch.float8_e5m2] = 0


def _tmap(tensor, dims, strides, box, atomic32=False):
    """128B-swizzled tiled tensor map. dims/box innermost first (elements);
    strides: byte strides of dims[1:]."""
    storage = ctypes.create_string_buffer(128 + 63)
    address = (ctypes.addressof(storage) + 63) & ~63
    rank = len(dims)
    if tensor.data_ptr() % 16 or any(s % 16 for s in strides):
        raise ValueError('tensor map needs 16-byte aligned base and strides')
    _check(_driver().cuTensorMapEncodeTiled(
        address, _MAP_TYPES[tensor.dtype], rank, ctypes.c_void_p(tensor.data_ptr()),
        (ctypes.c_uint64 * rank)(*dims), (ctypes.c_uint64 * max(1, rank - 1))(*strides),
        (ctypes.c_uint32 * rank)(*box), (ctypes.c_uint32 * rank)(*([1] * rank)), 0, 4 if atomic32 else 3, 2, 0), 'cuTensorMapEncodeTiled')
    return storage, address


def _aligned(t):
    """Contiguous, 16-byte aligned view or copy (kernels assume aligned vectors)."""
    if not t.is_contiguous():
        t = t.contiguous()
    if t.data_ptr() % 16:
        t = t.clone()
    return t


def _launch(name, buffers, grid, scalars=None, maps=()):
    """Launch `name` on `grid` with tensors `buffers` (kernel order) and
    scalars {'I': [...], 'i': [...], 'f': [...]} in declaration order.
    `maps` are encoded tensor maps (from `_tmap`) for the leading tensor-map
    arguments; their tensors are also the leading entries of `buffers`."""
    kernel = _kernel(name)
    if len(buffers) != len(kernel.widths):
        raise ValueError('buffer ABI mismatch: ' + name)
    for t, w in zip(buffers[len(maps):], kernel.widths[len(maps):]):
        if w > 1 and t.data_ptr() % min(16, w * t.element_size()):
            raise ValueError(name + ': vector buffer is not aligned')
    info = bytearray()
    for kind, count in kernel.scalars:
        values = scalars[kind]
        if len(values) != count:
            raise ValueError('scalar ABI mismatch: ' + name)
        info += struct.pack('<' + kind * count, *values)
        info += b'\0' * (-len(info) % 8)
    sizes = [t.numel() // w for t, w in zip(buffers, kernel.widths)]
    info += struct.pack('<' + 'I' * (2 * len(sizes)), *(sizes + sizes))
    info += b'\0' * (-len(info) % 8)
    packed = ctypes.create_string_buffer(bytes(info), len(info))
    pointers = (ctypes.c_uint64 * len(buffers))(*[t.data_ptr() for t in buffers])
    base = ctypes.addressof(pointers)
    parameters = (ctypes.c_void_p * (len(maps) + len(buffers) + 1))(
        *[address for _, address in maps], *[base + 8 * i for i in range(len(buffers))], ctypes.addressof(packed))
    stream = torch.cuda.current_stream(buffers[0].device).cuda_stream
    _check(_driver().cuLaunchKernel(kernel.function, grid[0], grid[1], grid[2], *kernel.block,
                                    kernel.shared, stream, parameters, None), 'cuLaunchKernel(' + name + ')')


def _empty(shape, like, dtype=None):
    return torch.empty(shape, dtype=dtype or like.dtype, device=like.device)


def _blocks(count, per_block):
    return (max(1, (count + per_block - 1) // per_block), 1, 1)


def _sm_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count
