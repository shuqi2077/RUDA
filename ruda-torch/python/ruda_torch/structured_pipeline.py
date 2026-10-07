"""Structured native pipeline boundaries, including undefined gradient transport."""
from __future__ import annotations

from dataclasses import dataclass
from collections import namedtuple
import struct
import torch
import torch.distributed as dist
from torch.utils._pytree import TreeSpec,tree_flatten,tree_unflatten,tree_map
from .pipeline_training import PipelineStage,PipelineTensorSpec


def _literal_key(value):
    if value is None:return ('none',)
    if type(value) is float:return ('float',struct.pack('>d',value))
    if type(value) in (bool,int,str):return (type(value).__name__,value)
    raise TypeError('pipeline constants must be None, bool, int, float or str')


@dataclass(frozen=True,eq=False)
class PipelineConstant:
    """An explicit immutable boundary value, not an inferred tensor or input."""
    value:object

    def __post_init__(self):_literal_key(self.value)

    def __eq__(self,other):
        return isinstance(other,PipelineConstant) and _literal_key(self.value)==_literal_key(other.value)

    def __hash__(self):return hash(_literal_key(self.value))


class PipelineLeafSpec(PipelineTensorSpec):
    """Explicit scalar/empty/dense tensor leaf, including integer control payloads.

    Empty axes describe actual empty data, never a padded substitute microbatch.
    Only the selected module's supported operations determine whether its empty
    forward/backward is executable. Zero-byte transfers carry no tensor payload.
    """
    def __init__(self,shape,dtype=torch.float32):
        object.__setattr__(self,'shape',tuple(shape))
        object.__setattr__(self,'dtype',dtype)
        self.validate()

    def validate(self):
        if not isinstance(self.shape,(tuple,list)) or any(type(size) is not int or size<0 for size in self.shape):
            raise ValueError('leaf shape must contain actual nonnegative integer dimensions')
        if self.dtype not in (torch.float32,torch.float16,torch.bfloat16,torch.int64,torch.int32,torch.int16,torch.int8,torch.uint8,torch.bool):
            raise ValueError('unsupported pipeline leaf dtype')

    def __eq__(self,other):
        return isinstance(other,PipelineTensorSpec) and tuple(self.shape)==tuple(other.shape) and self.dtype==other.dtype

    def __hash__(self):return hash((tuple(self.shape),self.dtype))


@dataclass(frozen=True,init=False)
class PipelineTreeSpec:
    """Caller-declared registered PyTree of tensor specs and explicit constants.

    Dictionaries, tuples/lists, named tuples and registered custom PyTrees retain
    their exact structure. Tensor leaves cross ranks as independent values; views
    and aliasing between different leaves are not part of a pipeline boundary.
    A repeated tensor's derivatives are still summed when backpropagating its
    distinct boundary slots. Constants must be declared with PipelineConstant,
    except plain None, which denotes an absent value.
    """
    leaves:tuple
    tree:object

    def __init__(self,structure):
        structure=tree_map(lambda leaf:None if isinstance(leaf,PipelineConstant) and leaf.value is None else leaf,structure)
        leaves,tree=tree_flatten(structure)
        object.__setattr__(self,'leaves',tuple(leaves))
        object.__setattr__(self,'tree',tree)
        self.validate()

    @classmethod
    def from_flat(cls,tree,leaves):
        """Use an actual registered TreeSpec whose context needs real tensor data.

        The caller supplies tensor contracts/constants in the tree's leaf order;
        no placeholder tensors or guessed custom-container context are created.
        """
        if not isinstance(tree,TreeSpec):raise TypeError('expected an actual registered PyTree TreeSpec')
        leaves=tuple(leaves)
        if len(leaves)!=tree.num_leaves:raise ValueError('tensor contracts do not cover the actual tree leaves')
        result=object.__new__(cls)
        object.__setattr__(result,'leaves',leaves)
        object.__setattr__(result,'tree',tree)
        result.validate()
        return result

    def validate(self):
        for leaf in self.leaves:
            if isinstance(leaf,PipelineTensorSpec):leaf.validate()
            elif leaf is not None and not isinstance(leaf,PipelineConstant):
                raise TypeError('tree leaves must be tensor specs, PipelineConstant or None')

    @property
    def has_floating(self):
        return any(isinstance(leaf,PipelineTensorSpec) and leaf.has_floating for leaf in self.leaves)

    def allocate(self,device):
        values=[leaf.allocate(device) if isinstance(leaf,PipelineTensorSpec) else
                leaf.value if isinstance(leaf,PipelineConstant) else None for leaf in self.leaves]
        return tree_unflatten(values,self.tree)

    def matches(self,value):
        leaves,tree=tree_flatten(value)
        if tree!=self.tree or len(leaves)!=len(self.leaves):return False
        for leaf,spec in zip(leaves,self.leaves,strict=True):
            if isinstance(spec,PipelineTensorSpec):
                if not isinstance(leaf,torch.Tensor) or tuple(leaf.shape)!=tuple(spec.shape) or leaf.dtype!=spec.dtype:return False
            elif isinstance(spec,PipelineConstant):
                try:
                    if _literal_key(leaf)!=_literal_key(spec.value):return False
                except TypeError:return False
            elif leaf is not None:return False
        return True


def _spec_leaves(spec):
    if isinstance(spec,PipelineTreeSpec):return spec.leaves,spec.tree
    leaves,tree=tree_flatten(spec)
    return tuple(leaves),tree


def _tensor_leaves(value):
    return [leaf for leaf in tree_flatten(value)[0] if isinstance(leaf,torch.Tensor)]


def _data_key(value):
    if isinstance(value,(list,tuple)):return tuple(_data_key(item) for item in value)
    return _literal_key(value)


@dataclass(frozen=True)
class _GradientSpec:
    source:object


@dataclass(frozen=True)
class _GradientPayload:
    values:tuple
    tree:object


class StructuredPipelineStage(PipelineStage):
    """1F1B on real Tensor/PyTree inputs and outputs using native rank storage.

    All stages of a pipeline must select this transfer protocol. input_mode is
    explicit: single passes the whole value, args expands a tuple/list, kwargs
    expands a string-key dictionary. The caller still owns the stage partition,
    loss_sum, targets and microbatch interfaces. Control tensors travel forward
    without autograd, alongside actual floating activations.

    Gradient presence is transmitted separately before tensor data. A boundary
    slot unused by the loss stays None upstream, rather than fabricating a zero
    parameter gradient that would enable otherwise unused optimizer weight decay.
    No activation or gradient payload is downloaded for host numerical execution.
    """
    transport_protocol=2

    def __init__(self,module,group,*,input_spec,output_spec,input_mode='single'):
        if input_mode not in ('single','args','kwargs'):raise ValueError('select single, args or kwargs invocation')
        self.input_mode=input_mode
        super().__init__(module,group,input_spec=input_spec,output_spec=output_spec)

    def _is_spec(self,spec):return isinstance(spec,(PipelineTensorSpec,PipelineTreeSpec))

    def _validate_input_spec(self,spec):
        if self.input_mode=='single':return
        if not isinstance(spec,PipelineTreeSpec):raise TypeError('expanded invocation requires a structured input contract')
        root=spec.tree.type
        if self.input_mode=='args' and root not in (tuple,list,namedtuple):
            raise TypeError('args invocation requires a tuple/list or named-tuple input contract')
        if self.input_mode=='kwargs' and (not isinstance(root,type) or not issubclass(root,dict)):
            raise TypeError('kwargs invocation requires a dictionary input contract')

    def _matches(self,value,spec):
        matched=spec.matches(value) if isinstance(spec,PipelineTreeSpec) else (
            isinstance(value,torch.Tensor) and tuple(value.shape)==tuple(spec.shape) and value.dtype==spec.dtype)
        return matched

    def _is_cpu_payload(self,value):
        for leaf in tree_flatten(value)[0]:
            if isinstance(leaf,torch.Tensor):
                if leaf.device.type!='cpu':return False
            else:
                try:_literal_key(leaf)
                except TypeError:return False
        return True

    def _payload_metadata(self,value,shapes_only=False):
        leaves,tree=tree_flatten(value)
        metadata=[]
        for leaf in leaves:
            if isinstance(leaf,torch.Tensor):
                metadata.append(('tensor',tuple(leaf.shape),str(leaf.dtype),
                    None if shapes_only else _data_key(leaf.tolist())))
            else:metadata.append(('constant',_literal_key(leaf)))
        return tree,tuple(metadata)

    def _to_device(self,value):
        return tree_map(lambda leaf:leaf.to(self.device) if isinstance(leaf,torch.Tensor) else leaf,value)

    def _received_input(self,value):
        for leaf in _tensor_leaves(value):
            if leaf.is_floating_point():leaf.requires_grad_(True)
        return value

    def _invoke(self,value):
        if self.input_mode=='single':result=self.module(value)
        elif self.input_mode=='args':
            if not isinstance(value,(tuple,list)):raise TypeError('args invocation requires a tuple/list boundary')
            result=self.module(*value)
        else:
            if not isinstance(value,dict) or any(not isinstance(key,str) for key in value):
                raise TypeError('kwargs invocation requires a string-key dictionary boundary')
            result=self.module(**value)
        if any(leaf.device!=self.device for leaf in _tensor_leaves(result)):
            raise ValueError('pipeline output tensor leaves must stay on the stage device')
        return result

    @property
    def invocation_contract(self):return self.transport_protocol,self.input_mode

    def _gradient_spec(self,spec):return _GradientSpec(spec)

    def _input_gradient(self,value):
        if self.rank==0:return None
        leaves,tree=tree_flatten(value)
        gradients=tuple(leaf.grad if isinstance(leaf,torch.Tensor) and leaf.is_floating_point() else None for leaf in leaves)
        return _GradientPayload(gradients,tree)

    def _backward_output(self,output,gradient):
        leaves,tree=tree_flatten(output)
        if gradient is None:gradients=(None,)*len(leaves)
        else:
            if not isinstance(gradient,_GradientPayload) or gradient.tree!=tree or len(gradient.values)!=len(leaves):
                raise ValueError('pipeline gradient structure differs from its actual output')
            gradients=gradient.values
        tensors,seeds=[],[]
        for leaf,seed in zip(leaves,gradients,strict=True):
            if isinstance(leaf,torch.Tensor) and leaf.requires_grad and (gradient is None or seed is not None):
                tensors.append(leaf)
                seeds.append(seed)
        if tensors:torch.autograd.backward(tensors,grad_tensors=seeds)

    def _exchange(self,*,send=None,send_rank=None,receive_spec=None,receive_rank=None):
        sending_gradient=isinstance(send,_GradientPayload)
        receiving_gradient=isinstance(receive_spec,_GradientSpec)
        output=None
        received_header=None
        output_specs,output_tree=(),None
        operations,owners,received=[],[],[]

        def add_send(value,rank):
            if not value.numel():return
            if value.device!=self.device:raise ValueError('pipeline sends must stay on the stage device')
            value=value.detach().contiguous()
            alias=self.group._alias(value)
            owners.extend((value,alias))
            operations.append(dist.P2POp(dist.isend,alias,self._peer(rank),self.group.group))

        def add_receive(value,rank):
            if not value.numel():return
            alias=self.group._alias(value)
            owners.append(alias)
            received.append(value)
            operations.append(dist.P2POp(dist.irecv,alias,self._peer(rank),self.group.group))

        def complete():
            if operations:
                for request in dist.batch_isend_irecv(operations):request.wait()
                self.group._complete(received)
            operations.clear()
            owners.clear()
            received.clear()

        # Header and opposite-direction forward values are one matched P2P
        # phase. Payload sends cannot precede their receiver's presence flags.
        if sending_gradient:
            header=torch.tensor([leaf is not None for leaf in send.values],dtype=torch.bool,device=self.device)
            add_send(header,send_rank)
        elif send is not None:
            for leaf in _tensor_leaves(send):add_send(leaf,send_rank)
        if receiving_gradient:
            output_specs,output_tree=_spec_leaves(receive_spec.source)
            received_header=torch.empty(len(output_specs),dtype=torch.bool,device=self.device)
            add_receive(received_header,receive_rank)
        elif receive_spec is not None:
            output=receive_spec.allocate(self.device)
            for leaf in _tensor_leaves(output):add_receive(leaf,receive_rank)
        complete()

        # Only the small boolean control header is read on the host. Tensor
        # gradients remain on native storage and transmit only present slots.
        if sending_gradient:
            for leaf in send.values:
                if leaf is not None:add_send(leaf,send_rank)
        if receiving_gradient:
            flags=received_header.cpu().tolist()
            values=[]
            for active,spec in zip(flags,output_specs,strict=True):
                if not active:values.append(None)
                else:
                    if not isinstance(spec,PipelineTensorSpec) or not spec.has_floating:
                        raise ValueError('a non-floating boundary slot received a gradient')
                    value=spec.allocate(self.device)
                    values.append(value)
                    add_receive(value,receive_rank)
            output=_GradientPayload(tuple(values),output_tree)
        complete()
        return output
