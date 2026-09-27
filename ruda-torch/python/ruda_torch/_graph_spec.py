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
    codes={'copy':0,'add':1,'mul':2,'silu':14,'rms_norm':100,'silu_mul':101}
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
        if node.kind in ('copy','silu'):
            if node.right is not None or scalar!=0. or math.copysign(1.,scalar)<0:
                raise ValueError('unary operation requires no right input and canonical zero scalar')
            b=a
        elif node.kind in ('add','mul','silu_mul'):
            if node.right not in ids: raise ValueError('missing second graph input')
            b=ids[node.right]
            if specs[b]!=spec: raise ValueError('no broadcasting or mixed-dtype promotion in static graph')
            if node.kind in ('mul','silu_mul') and (scalar!=0. or math.copysign(1.,scalar)<0):
                raise ValueError('mul/silu_mul scalar must be canonical zero')
        else:
            if scalar<=0: raise ValueError('RMSNorm epsilon must remain positive after FP32 conversion')
            if math.prod(spec.shape[:-1])>(2**32-1)//32: raise ValueError('RMSNorm row grid limit')
            b=NO_WEIGHT
            if node.right is not None:
                if node.right not in ids: raise ValueError('missing RMSNorm weight')
                b=ids[node.right]
                if specs[b]!=TensorSpec((spec.shape[-1],),spec.dtype):
                    raise ValueError('RMSNorm needs same-dtype last-axis weight')
        words.extend((codes[node.kind],a,b)); scalars.append(scalar)
        ids[node.output]=len(names); names.append(node.output); specs.append(spec)
    if outputs is None: outputs=(nodes[-1].output,)
    if isinstance(outputs,str): raise TypeError('outputs must be a sequence of names, not a string')
    outputs=tuple(outputs)
    if not outputs or len(set(outputs))!=len(outputs) or any(x not in ids or ids[x]<len(inputs) for x in outputs):
        raise ValueError('outputs must be unique produced node names')
    return Layout(tuple(names),tuple(specs),tuple(words),tuple(scalars),tuple(ids[n] for n in outputs),len(inputs))
