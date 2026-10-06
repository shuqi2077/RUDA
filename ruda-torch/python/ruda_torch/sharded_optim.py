"""Complete-matrix Muon over FSDP storage and global sharded-gradient norms."""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from .optim import Muon, _validate_group, _number


def distributed_grad_norm(parameters, group, *, loss_scale=1., replicated_ids=(),replication_factors=None):
    """Stable global L2 norm; count replicated parameters once, shards once each."""
    parameters=list(parameters)
    if not math.isfinite(loss_scale) or loss_scale<=0:
        raise ValueError('gradient norm requires parameters and a positive finite scale')
    replicated_ids=set(replicated_ids)
    replication_factors={} if replication_factors is None else dict(replication_factors)
    if any(type(factor) is not int or factor<1 or factor>group.world_size for factor in replication_factors.values()):
        raise ValueError('gradient replication factors must be positive group counts')
    device=parameters[0].device if parameters else torch.device('ruda:0' if group.device_type=='ruda' else 'cpu')
    magnitude=torch.zeros(1,device=device,dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            gradient=parameter.grad.detach().float().abs()
            if gradient.numel():magnitude=torch.maximum(magnitude,gradient.amax().reshape(1))
    maxima=group.gather_metadata(float(magnitude.item()))
    largest=max(maxima)
    if not math.isfinite(largest):return float('inf')
    if largest==0:return 0.
    squares=torch.zeros(1,device=device,dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            contribution=(parameter.grad.detach().float()/largest).square().sum()
            factor=replication_factors.get(id(parameter),group.world_size if id(parameter) in replicated_ids else 1)
            contribution=contribution/factor
            squares.add_(contribution)
    group.sum_(squares)
    return largest*math.sqrt(float(squares.item()))/loss_scale


def _local(value,parameter,rank,world):
    result=torch.zeros_like(parameter,dtype=value.dtype)
    begin=rank*result.numel()
    end=min(begin+result.numel(),value.numel())
    if end>begin:result[:end-begin].copy_(value.reshape(-1)[begin:end])
    return result


class ShardedMuon(torch.optim.Optimizer):
    """FSDP parameter/state slices with exact complete-matrix Muon updates.

    One rank orthogonalizes each COMPLETE matrix through the existing Muon
    implementation; moments/masters are gathered for that matrix and scattered
    after its update. Persistent state remains sharded. All proposed local
    slices are retained until every matrix preflight succeeds, before commit.
    Tensor-parallel matrix fragments require a distinct joint TP/DP algorithm.
    """
    complete_matrix_shards=True

    def __init__(self,params,group,lr=.02,*,momentum=.95,weight_decay=0.,nesterov=True,
                 momentum_mode='sgd',dampening=0.,ns_steps=5,ns_coefficients=(3.4445,-4.775,2.0315),
                 eps=1e-7,adjust_lr='original',matrix_layout='as_stored',stable_normalization=True,
                 flatten=False,max_grad_norm=None,owners=None):
        self.transport=group
        self.last_step_skipped=False
        self.last_grad_norm=None
        self.max_grad_norm=None if max_grad_norm is None else _number(max_grad_norm,'max_grad_norm',nonnegative=True)
        defaults=dict(lr=lr,momentum=momentum,weight_decay=weight_decay,nesterov=nesterov,momentum_mode=momentum_mode,
                      dampening=dampening,ns_steps=ns_steps,ns_coefficients=tuple(ns_coefficients),eps=eps,
                      adjust_lr=adjust_lr,matrix_layout=matrix_layout,stable_normalization=stable_normalization,
                      flatten=flatten,use_muon=True,betas=(.9,.999))
        super().__init__(params,defaults)
        parameters=[p for options in self.param_groups for p in options['params']]
        if not parameters:raise ValueError('ShardedMuon requires FSDP parameters')
        if len({id(p) for p in parameters})!=len(parameters):
            raise ValueError('ShardedMuon parameters must be unique')
        shapes=[]
        for options in self.param_groups:
            _validate_group(options)
            for parameter in options['params']:
                shape=getattr(parameter,'_ruda_full_shape',None)
                if shape is None or parameter.ndim!=1 or getattr(parameter,'_ruda_tp_sharded',False):
                    raise ValueError('ShardedMuon requires FSDP slices of complete, non-TP parameters')
                if options['use_muon'] and (len(shape)<2 or len(shape)>2 and not options['flatten']):
                    raise ValueError('Muon requires complete matrices; select AdamW groups for vectors')
                shapes.append(shape)
        self.owners=list(owners) if owners is not None else [i%group.world_size for i in range(len(parameters))]
        if len(self.owners)!=len(parameters) or any(type(owner) is not int or not 0<=owner<group.world_size for owner in self.owners):
            raise ValueError('provide one valid owner per FSDP parameter')
        group.validate_training_options((shapes,self.owners,self.max_grad_norm,
            [{k:v for k,v in options.items() if k!='params'} for options in self.param_groups]))

    @torch.no_grad()
    def step(self,closure=None,*,loss_scale=1.):
        if closure is not None:raise ValueError('sharded Muon does not implicitly rerun distributed backward')
        scale=_number(loss_scale,'loss_scale')
        if scale<=0:raise ValueError('loss_scale must be positive')
        entries=[(p,options) for options in self.param_groups for p in options['params']]
        flags=self.transport.gather_metadata(tuple(p.grad is not None for p,options in entries))
        active=[any(rank[index] for rank in flags) for index in range(len(entries))]
        self.last_step_skipped=False
        if not any(active):return None
        bad=torch.zeros(1,device=entries[0][0].device,dtype=torch.float32)
        for parameter,options in entries:
            _validate_group(options)
            if parameter.grad is not None:bad.add_((~torch.isfinite(parameter.grad)).any().float())
        self.transport.sum_(bad)
        if bad.item():
            self.last_step_skipped=True
            return None
        clip=1.
        if self.max_grad_norm is not None:
            norm=distributed_grad_norm([p for p,options in entries],self.transport,loss_scale=scale)
            self.last_grad_norm=norm
            clip=min(1.,self.max_grad_norm/(norm+1e-6))
        proposed=[]
        for index,((parameter,options),used) in enumerate(zip(entries,active)):
            if not used:continue
            shape=parameter._ruda_full_shape
            elements=math.prod(shape)
            value=self.transport.all_gather(parameter)[:elements].view(shape)
            gradient=parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            full_gradient=self.transport.all_gather(gradient)[:elements].view(shape)
            full=nn.Parameter(value,requires_grad=True)
            full.grad=full_gradient
            local_state=self.state.get(parameter,{})
            counters=self.transport.gather_metadata({key:item for key,item in local_state.items() if not isinstance(item,torch.Tensor)})
            if any(counter!=counters[0] for counter in counters):
                raise ValueError('Muon shard counters/algorithms differ')
            state=dict(counters[0])
            names=('master','momentum_buffer') if options['use_muon'] else ('master','exp_avg','exp_avg_sq')
            for name in names:
                if local_state:
                    item=local_state[name]
                    if item.shape!=parameter.shape or item.dtype!=torch.float32:
                        raise ValueError('Muon moment/master shard metadata differs')
                    state[name]=self.transport.all_gather(item)[:elements].view(shape)
            owner=self.owners[index]
            failed,error=False,None
            core=None
            if self.transport.rank==owner:
                try:
                    core=Muon([dict(options,params=[full])])
                    if state:core.state[full]=state
                    if clip==0:
                        full.grad=torch.zeros_like(full)
                        core.step(loss_scale=scale)
                    else:
                        core.step(loss_scale=scale/clip)
                    failed=core.last_step_skipped
                except Exception as failure:
                    error=str(failure)
            decisions=self.transport.gather_metadata((failed,error))
            if decisions[owner][1]:raise RuntimeError(decisions[owner][1])
            if decisions[owner][0]:
                self.last_step_skipped=True
                return None
            self.transport.broadcast_(full,owner)
            new_state=core.state[full] if core is not None else {}
            metadata=self.transport.gather_metadata({key:item for key,item in new_state.items() if not isinstance(item,torch.Tensor)}
                                                     if self.transport.rank==owner else None)[owner]
            local_new=dict(metadata)
            for name in names:
                tensor=new_state[name] if self.transport.rank==owner else torch.empty(shape,device=parameter.device,dtype=torch.float32)
                self.transport.broadcast_(tensor,owner)
                local_new[name]=_local(tensor,parameter,self.transport.rank,self.transport.world_size)
            proposed.append((parameter,_local(full,parameter,self.transport.rank,self.transport.world_size),local_new))
        for parameter,value,state in proposed:
            parameter.copy_(value)
            self.state[parameter]=state
        return None

    def state_dict(self):
        result=super().state_dict()
        result['ruda_sharded_muon']={'version':1,'max_grad_norm':self.max_grad_norm,
            'shapes':[p._ruda_full_shape for options in self.param_groups for p in options['params']]}
        return result

    def load_state_dict(self,record):
        expected=self.state_dict()['ruda_sharded_muon']
        if record.get('ruda_sharded_muon')!=expected:
            raise ValueError('sharded Muon shape/clipping configuration differs')
        if len(record['param_groups'])!=len(self.param_groups):raise ValueError('Muon parameter groups differ')
        replacement,groups={},[]
        for current,saved in zip(self.param_groups,record['param_groups'],strict=True):
            if len(current['params'])!=len(saved['params']):raise ValueError('Muon parameter counts differ')
            options=dict(saved,params=current['params'])
            _validate_group(options)
            groups.append(options)
            for parameter,identity in zip(current['params'],saved['params'],strict=True):
                entry=record['state'].get(identity)
                if entry is None:continue
                checked={}
                for name,value in entry.items():
                    if isinstance(value,torch.Tensor):
                        if value.shape!=parameter.shape or value.dtype!=torch.float32:
                            raise ValueError('Muon checkpoint state shard shape/dtype differs')
                        value=value.to(device=parameter.device,dtype=torch.float32).clone()
                    checked[name]=copy.deepcopy(value) if not isinstance(value,torch.Tensor) else value
                replacement[parameter]=checked
        self.param_groups=groups
        self.state.clear()
        self.state.update(replacement)
        self.last_step_skipped=False
