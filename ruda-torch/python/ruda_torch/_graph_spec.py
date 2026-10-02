"""Pure metadata planner for the native static graph. No tensor execution here."""
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math
import struct

NO_WEIGHT = 2**32-1
MAX_NODES = 256
MAX_TENSORS = 512

@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

@dataclass(frozen=True)
class GraphOp:
    """One immutable output; names refer to inputs or earlier node outputs.

    copy/silu: right=None, scalar=0.0. add: scalar is alpha. mul: scalar=0.0.
    rms_norm: right is an optional weight, scalar is explicit epsilon.
    silu_mul: round SiLU to the storage dtype before multiplication, like two nodes.
    This plan never infers epsilon from dtype or substitutes other operations.
    """
    kind: str
    output: str
    left: str
    right: str | None = None
    scalar: float = 0.0

    @classmethod
    def add(cls, output, left, right, *, alpha=1.0): return cls('add',output,left,right,alpha)
    @classmethod
    def mul(cls, output, left, right): return cls('mul',output,left,right)
    @classmethod
    def silu(cls, output, value): return cls('silu',output,value)
    @classmethod
    def silu_mul(cls, output, gate, up): return cls('silu_mul',output,gate,up)
    @classmethod
    def copy(cls, output, value): return cls('copy',output,value)
    @classmethod
    def rms_norm(cls, output, value, weight=None, *, eps=1e-5): return cls('rms_norm',output,value,weight,eps)

# API 3 operators. Codes reuse the production pointwise dispatch where possible.
UNARY_CODES = {'copy': 0, 'relu': 3, 'exp': 9, 'log': 10, 'sqrt': 11,
    'rsqrt': 12, 'sigmoid': 13, 'silu': 14, 'tanh': 17, 'sin': 19,
    'cos': 20, 'abs': 21, 'sign': 22, 'floor': 23, 'ceil': 24, 'trunc': 25,
    'reciprocal': 27, 'log1p': 35, 'sinh': 36, 'cosh': 37, 'asinh': 38,
    'acosh': 39, 'atanh': 40, 'hardsigmoid': 43, 'hardswish': 45, 'neg': 108}
BINARY_CODES = {'add': 1, 'mul': 2, 'div': 8, 'silu_backward': 15,
    'sigmoid_backward': 16, 'tanh_backward': 18, 'silu_mul': 101}
SCALAR_CODES = {'add_scalar': 109, 'mul_scalar': 110, 'div_scalar': 111}
CODES = {**UNARY_CODES, **BINARY_CODES, **SCALAR_CODES, 'mm': 7, 'bmm': 30,
    'rms_norm': 100, 'softmax': 102, 'log_softmax': 103, 'sum_keepdim': 104,
    'mean_keepdim': 105, 'softmax_backward': 106, 'log_softmax_backward': 107}

@dataclass(frozen=True)
class Layout:
    names: tuple[str,...]
    specs: tuple[TensorSpec,...]
    words: tuple[int,...]
    scalars: tuple[float,...]
    output_indices: tuple[int,...]
    inputs: int

    @property
    def workspace_bytes(self):
        return sum(math.prod(s.shape)*(4 if s.dtype=='float32' else 2) for s in self.specs[self.inputs:])


def plan_layout(inputs: Mapping[str,TensorSpec], nodes: Sequence[GraphOp], outputs=None) -> Layout:
    if not isinstance(inputs,Mapping) or not 1<=len(inputs)<=MAX_TENSORS:
        raise ValueError('static graph needs named inputs')
    nodes=tuple(nodes)
    if not 1<=len(nodes)<=MAX_NODES or len(nodes)+len(inputs)>MAX_TENSORS:
        raise ValueError('static graph accepts 1..256 nodes and at most 512 tensors')
    names=list(inputs); specs=list(inputs.values()); ids={name:i for i,name in enumerate(names)}
    for name,spec in inputs.items():
        if not isinstance(name,str) or not name or not isinstance(spec,TensorSpec):
            raise TypeError('expected nonempty names and TensorSpec values')
        if spec.dtype not in ('float32','float16','bfloat16'):
            raise ValueError('static graph supports float32/float16/bfloat16')
        if not 1<=len(spec.shape)<=8 or any(type(d) is not int or d<=0 for d in spec.shape):
            raise ValueError('static graph requires nonempty rank 1..8 shapes')
        if math.prod(spec.shape)>2**32-1: raise ValueError('32-bit kernel indexing limit')
    words=[]; scalars=[]
    codes=CODES
    for node in nodes:
        if not isinstance(node,GraphOp): raise TypeError('nodes must be GraphOp objects')
        if node.kind not in codes: raise ValueError(f'unsupported graph operator: {node.kind}')
        if not isinstance(node.output,str) or not node.output or node.output in ids:
            raise ValueError('each node requires a new output name; in-place graphs are not supported')
        if node.left not in ids: raise ValueError('node input is missing or refers to a later node')
        a=ids[node.left]; spec=specs[a]
        if isinstance(node.scalar,bool): raise TypeError('graph scalar must be a finite number')
        try: scalar=struct.unpack('<f',struct.pack('<f',float(node.scalar)))[0]
        except (TypeError,ValueError,OverflowError,struct.error) as exc: raise ValueError('invalid FP32 scalar') from exc
        if not math.isfinite(scalar): raise ValueError('graph scalar must be finite in FP32')
        if node.kind in UNARY_CODES:
            if node.right is not None or scalar != 0. or math.copysign(1., scalar) < 0:
                raise ValueError('unary operation requires no right input and canonical zero scalar')
            b = a
        elif node.kind in SCALAR_CODES:
            if node.right is not None:
                raise ValueError('scalar operation has no right tensor')
            b = a
        elif node.kind in BINARY_CODES:
            if node.right not in ids: raise ValueError('missing second graph input')
            b = ids[node.right]
            if specs[b] != spec: raise ValueError('no broadcasting or mixed-dtype promotion in static graph')
            if node.kind != 'add' and (scalar != 0. or math.copysign(1., scalar) < 0):
                raise ValueError('binary scalar must be canonical zero')
        elif node.kind in ('mm', 'bmm'):
            if node.right not in ids: raise ValueError('missing matrix input')
            b = ids[node.right]; right = specs[b]
            rank = 2 if node.kind == 'mm' else 3
            if len(spec.shape) != rank or len(right.shape) != rank or spec.dtype != right.dtype:
                raise ValueError('matmul requires equal-dtype rank-2/rank-3 inputs')
            if spec.shape[-1] != right.shape[-2] or spec.shape[:-2] != right.shape[:-2]:
                raise ValueError('matmul dimensions/batches do not match')
            if scalar != 0. or math.copysign(1., scalar) < 0:
                raise ValueError('matmul scalar must be canonical zero')
            spec = TensorSpec(spec.shape[:-1] + (right.shape[-1],), spec.dtype)
            if math.prod(spec.shape) > 2**32-1: raise ValueError('matmul indexing limit')
        elif node.kind in ('softmax', 'log_softmax', 'softmax_backward', 'log_softmax_backward'):
            if not scalar.is_integer() or not 0 <= scalar < len(spec.shape):
                raise ValueError('softmax axis must be canonical and in range')
            b = a
            if node.kind.endswith('_backward'):
                if node.right not in ids or specs[ids[node.right]] != spec:
                    raise ValueError('softmax backward output/gradient mismatch')
                b = ids[node.right]
            elif node.right is not None:
                raise ValueError('softmax forward has no right tensor')
        elif node.kind in ('sum_keepdim', 'mean_keepdim'):
            # Bit mask, not a shape guessed from the resulting singleton axes.
            mask = int(scalar)
            if scalar != mask or not 0 < mask < (1 << len(spec.shape)) or node.right is not None:
                raise ValueError('reduction requires a nonempty in-range dimension mask')
            b = a
            spec = TensorSpec(tuple(1 if mask & (1 << d) else size
                                    for d, size in enumerate(spec.shape)), spec.dtype)
        else:
            if scalar <= 0: raise ValueError('RMSNorm epsilon must remain positive after FP32 conversion')
            if math.prod(spec.shape[:-1]) > (2**32-1)//32: raise ValueError('RMSNorm row grid limit')
            b = NO_WEIGHT
            if node.right is not None:
                if node.right not in ids: raise ValueError('missing RMSNorm weight')
                b = ids[node.right]
                if specs[b] != TensorSpec((spec.shape[-1],), spec.dtype):
                    raise ValueError('RMSNorm needs same-dtype last-axis weight')
        words.extend((codes[node.kind],a,b)); scalars.append(scalar)
        ids[node.output]=len(names); names.append(node.output); specs.append(spec)
    if outputs is None: outputs=(nodes[-1].output,)
    if isinstance(outputs,str): raise TypeError('outputs must be a sequence of names, not a string')
    outputs=tuple(outputs)
    if not outputs or len(set(outputs))!=len(outputs) or any(x not in ids or ids[x]<len(inputs) for x in outputs):
        raise ValueError('outputs must be unique produced node names')
    return Layout(tuple(names),tuple(specs),tuple(words),tuple(scalars),tuple(ids[n] for n in outputs),len(inputs))
