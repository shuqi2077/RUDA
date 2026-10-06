"""Explicit shared parameters across pipeline stages, including optimizer storage."""
from __future__ import annotations

import copy
import torch
import torch.distributed as dist
from .distributed_training import ReplicaGroup
from .distributed_checkpoint import model_shard_layout


class PipelineTiedParameters:
    """Bind the SAME logical weight by caller-supplied symbols on its owner stages.

    Every global rank constructs this object, including nonowners with {}.
    A symbol may span two or more stages; compatible TP shards remain local.
    Tied gradients sum over PP consumers BEFORE DP reduction. Optimizer policies
    must agree across consumers; post-step weights/moments come from the first
    owner, preserving one shared logical parameter rather than diverging copies.
    """
    def __init__(self,model,mesh,bindings):
        if not isinstance(bindings,dict) or any(not isinstance(key,str) or not key for key in bindings):
            raise ValueError('tie bindings must use explicit nonempty symbols')
        if len({id(p) for p in bindings.values()})!=len(bindings):raise ValueError('one parameter cannot name two different tie symbols')
        names={id(p):name for name,p in model.named_parameters()}
        layout=model_shard_layout(model)
        local={}
        for symbol,parameter in bindings.items():
            if id(parameter) not in names or parameter.device.type!=mesh.device_type or not parameter.is_contiguous():
                raise ValueError('bind contiguous parameters of this stage on the mesh device')
            spec=layout.get(names[id(parameter)],{})
            local[symbol]=(tuple(parameter.shape),str(parameter.dtype),parameter.requires_grad,
                           tuple(getattr(parameter,'_ruda_logical_shape',spec.get('shape',parameter.shape))),
                           getattr(parameter,'_ruda_tp_axis',None if spec.get('replicated') else spec.get('axis')),
                           getattr(parameter,'_ruda_full_shape',None))
        ranks=mesh.world.gather_metadata(local)
        data,pipeline,tensor=mesh.shape
        symbols=sorted(set().union(*(entry.keys() for entry in ranks)))
        self.model,self.mesh,self.bindings,self.groups=model,mesh,dict(bindings),{}
        self.consumer_counts={}
        self._owned=[]
        backend='nccl' if mesh.device_type=='ruda' else 'gloo'
        for symbol in symbols:
            owners=[p for p in range(pipeline) if symbol in ranks[p*tensor]]
            if len(owners)<2:raise ValueError('a pipeline tie needs at least two owner stages')
            reference=ranks[owners[0]*tensor][symbol]
            for d in range(data):
                for t in range(tensor):
                    if [p for p in range(pipeline) if symbol in ranks[(d*pipeline+p)*tensor+t]]!=owners:
                        raise ValueError('tie stage ownership differs between DP/TP coordinates')
                    if any(ranks[(d*pipeline+p)*tensor+t][symbol]!=reference for p in owners):
                        raise ValueError('tied parameter shape/dtype/gradient/TP axes differ')
            self.consumer_counts[symbol]=len(owners)
            for d in range(data):
                for t in range(tensor):
                    members=[(d*pipeline+p)*tensor+t for p in owners]
                    process_group=dist.new_group(members,backend=backend)
                    if mesh.rank in members:
                        self.groups[symbol]=ReplicaGroup(process_group,device_type=mesh.device_type)
                        self._owned.append(process_group)
        with torch.no_grad():
            for symbol,group in self.groups.items():group.broadcast_(self.bindings[symbol],0)

    def _entry(self,optimizer,parameter):
        from .sharded_training import Zero2Optimizer
        if optimizer is None:return None,None,None
        actual=optimizer.optimizer if isinstance(optimizer,Zero2Optimizer) else optimizer
        if isinstance(optimizer,Zero2Optimizer):
            parameter=next((shard for original,shard in optimizer.entries if original is parameter),None)
        for options in actual.param_groups:
            if any(value is parameter for value in options['params']):return actual,parameter,options
        return actual,None,None

    def validate_optimizer(self,optimizer):
        for symbol,group in self.groups.items():
            actual,parameter,options=self._entry(optimizer,self.bindings[symbol])
            policy=None if parameter is None else (type(actual).__module__,type(actual).__qualname__,
                {key:value for key,value in options.items() if key!='params'},getattr(actual,'max_grad_norm',None))
            values=group.gather_metadata(policy)
            if any(value!=policy for value in values):raise ValueError('tied pipeline consumers use different optimizer policies')
            if policy is None and self.bindings[symbol].requires_grad:raise ValueError('trainable tied weight is missing from its optimizer')

    @torch.no_grad()
    def synchronize_gradients(self):
        for symbol,group in self.groups.items():
            parameter=self.bindings[symbol]
            active=group.gather_metadata(parameter.grad is not None)
            if not any(active):continue
            value=torch.zeros_like(parameter,dtype=torch.float32) if parameter.grad is None else parameter.grad.detach().float().contiguous()
            group.sum_(value)
            parameter.grad=value.to(parameter.dtype)

    def replication_count(self,parameter):
        symbol=next((symbol for symbol,value in self.bindings.items() if value is parameter),None)
        return 1 if symbol is None else self.consumer_counts[symbol]

    @torch.no_grad()
    def synchronize_state(self,optimizer):
        for symbol,group in self.groups.items():
            original=self.bindings[symbol]
            group.broadcast_(original,0)
            actual,parameter,options=self._entry(optimizer,original)
            if parameter is None:continue
            if parameter is not original:group.broadcast_(parameter,0)
            state=actual.state.get(parameter,{})
            descriptor={}
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and value.device.type==group.device_type:
                    descriptor[key]=('tensor',tuple(value.shape),value.dtype)
                elif isinstance(value,torch.Tensor):
                    if value.device.type!='cpu' or value.numel()!=1:raise ValueError('only scalar host optimizer metadata can accompany tied device states')
                    descriptor[key]=('scalar',value.item(),value.dtype,tuple(value.shape))
                else:descriptor[key]=('metadata',copy.deepcopy(value))
            descriptors=group.gather_metadata(descriptor)
            if any(value!=descriptors[0] for value in descriptors):raise ValueError('tied optimizer state layouts/counters differ')
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and value.device.type==group.device_type:group.broadcast_(value,0)
            if hasattr(actual,'_versions'):actual._versions[id(parameter)]=parameter._version
            if hasattr(actual,'_expected_versions'):actual._expected_versions[id(parameter)]=parameter._version

    def close(self):
        for group in reversed(self._owned):dist.destroy_process_group(group)
        self._owned.clear()
