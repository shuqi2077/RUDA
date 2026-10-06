"""Exact complete-matrix Muon/AdamW updates with TP-local persistent storage."""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from .optim import Muon,_validate_group,_parameter,_overlap_check,_number
from .distributed_checkpoint import model_shard_layout
from .sharded_optim import distributed_grad_norm


class TensorParallelMuon(torch.optim.Optimizer):
    """Orthogonalize complete logical matrices, never individual TP fragments.

    One TP owner executes the existing Muon algorithm for each matrix. Master
    weights/moments remain partitioned like model parameters. Only one complete
    matrix and its state are gathered at a time; local proposed updates remain
    pending until every matrix succeeds, matching Muon's whole-step skip rule.
    Gradients must already represent the logical mean (DP synchronization is
    external). Explicit use_muon=False groups run the same core AdamW formula.
    """
    complete_tensor_parallel_matrices=True

    def __init__(self,params,model,group,lr=.02,*,momentum=.95,weight_decay=0.,nesterov=True,
                 momentum_mode='sgd',dampening=0.,ns_steps=5,ns_coefficients=(3.4445,-4.775,2.0315),
                 eps=1e-7,adjust_lr='original',matrix_layout='as_stored',stable_normalization=True,
                 flatten=False,max_grad_norm=None,owners=None,consensus_group=None):
        defaults=dict(lr=lr,momentum=momentum,weight_decay=weight_decay,nesterov=nesterov,
            momentum_mode=momentum_mode,dampening=dampening,ns_steps=ns_steps,
            ns_coefficients=tuple(ns_coefficients),eps=eps,adjust_lr=adjust_lr,matrix_layout=matrix_layout,
            stable_normalization=stable_normalization,flatten=flatten,use_muon=True,betas=(.9,.999))
        super().__init__(params,defaults)
        self.model,self.transport=model,group
        self.consensus_group=group if consensus_group is None else consensus_group
        self.max_grad_norm=None if max_grad_norm is None else _number(max_grad_norm,'max_grad_norm')
        if self.max_grad_norm is not None and self.max_grad_norm<=0:raise ValueError('max_grad_norm must be positive')
        self.last_step_skipped,self.last_grad_norm=False,None
        self._distributed_grad_norm=None
        self._versions={}
        parameters=[p for options in self.param_groups for p in options['params']]
        if not parameters or len({id(p) for p in parameters})!=len(parameters):
            raise ValueError('provide nonempty unique model parameters')
        names={id(p):name for name,p in model.named_parameters()}
        layout=model_shard_layout(model)
        self._layouts={}
        schema=[]
        for options in self.param_groups:
            _validate_group(options)
            for parameter in options['params']:
                _parameter(parameter,options)
                if id(parameter) not in names or hasattr(parameter,'_ruda_full_shape'):
                    raise ValueError('TensorParallelMuon requires model TP/replicated parameters, not flat FSDP slices')
                if parameter.device.type!=group.device_type or parameter.dtype not in (torch.float32,torch.float16,torch.bfloat16):
                    raise ValueError('select a contiguous FP32/FP16/BF16 process-group parameter')
                name=names[id(parameter)]
                spec=layout.get(name,{'shape':list(parameter.shape),'replicated':True,'axis':None})
                axis=None if spec.get('replicated') else spec.get('axis')
                if axis is None and getattr(parameter,'_ruda_tp_sharded',False):
                    raise ValueError('TP parameter lacks its full logical matrix layout')
                expected=list(parameter.shape)
                if axis is not None:expected[axis]*=group.world_size
                if expected!=list(spec['shape']):raise ValueError('parameter shard/global shape differs')
                self._layouts[id(parameter)]=(name,tuple(spec['shape']),axis)
                schema.append((name,tuple(spec['shape']),axis,str(parameter.dtype)))
        _overlap_check(parameters)
        self.owners=list(owners) if owners is not None else [index%group.world_size for index in range(len(parameters))]
        if len(self.owners)!=len(parameters) or any(type(rank) is not int or not 0<=rank<group.world_size for rank in self.owners):
            raise ValueError('provide one valid TP owner per parameter')
        group.validate_training_options((schema,self.owners,self.max_grad_norm,
            [{key:value for key,value in options.items() if key!='params'} for options in self.param_groups]))
        self._specs=[(id(p),tuple(p.shape),p.dtype,p.device,p.requires_grad) for p in parameters]

    def _full(self,value,axis):
        return value.detach().clone() if axis is None else self.transport.all_gather(value,axis=axis)

    def _local(self,value,axis):
        return value.detach().clone() if axis is None else value.chunk(self.transport.world_size,dim=axis)[self.transport.rank].contiguous().clone()

    def _full_parameter(self,value,parameter,axis):return self._full(value,axis)

    def _local_parameter(self,value,parameter,axis):return self._local(value,axis)

    def _gradient_norm(self,parameters,scale):
        return distributed_grad_norm(parameters,self.transport,loss_scale=scale,
            replicated_ids={id(p) for p in parameters if self._layouts[id(p)][2] is None})

    @torch.no_grad()
    def step(self,closure=None,*,loss_scale=1.):
        global_grad_norm=self._distributed_grad_norm
        self._distributed_grad_norm=None
        if closure is not None:raise ValueError('TP Muon does not implicitly rerun distributed backward')
        scale=_number(loss_scale,'loss_scale')
        if scale<=0:raise ValueError('loss_scale must be positive')
        entries=[(p,options) for options in self.param_groups for p in options['params']]
        if [(id(p),tuple(p.shape),p.dtype,p.device,p.requires_grad) for p,options in entries]!=self._specs:
            raise ValueError('TP optimizer parameter layout changed')
        self.consensus_group.validate_training_options((scale,[{key:value for key,value in options.items() if key!='params'}
                                                            for options in self.param_groups]))
        flags=self.transport.gather_metadata(tuple(p.grad is not None for p,options in entries))
        active=[any(rank[index] for rank in flags) for index in range(len(entries))]
        self.last_step_skipped=False
        if not any(active):return None
        error=None
        for index,(parameter,options) in enumerate(entries):
            _validate_group(options)
            if parameter.grad is not None and (parameter.grad.shape!=parameter.shape or parameter.grad.dtype!=parameter.dtype
                    or parameter.grad.device!=parameter.device or not parameter.grad.is_contiguous()):
                error='TP gradient layout differs from parameter'
            if self._layouts[id(parameter)][2] is None and active[index] and not all(rank[index] for rank in flags):
                error='replicated TP parameter has inconsistent gradient participation'
            if self.state.get(parameter) and id(parameter) in self._versions and parameter._version!=self._versions[id(parameter)]:
                error='parameter changed outside TP Muon; reload/reset its master state'
        for failure in self.consensus_group.gather_metadata(error):
            if failure:raise ValueError(failure)
        _overlap_check([p for p,options in entries],[p.grad for p,options in entries if p.grad is not None])
        bad=torch.zeros(1,device=entries[0][0].device,dtype=torch.float32)
        for parameter,options in entries:
            if parameter.grad is not None:bad.add_((~torch.isfinite(parameter.grad)).any().float())
        self.consensus_group.sum_(bad)
        if bad.item():
            self.last_step_skipped=True
            return None
        clip=1.
        if self.max_grad_norm is not None:
            norm=(global_grad_norm if global_grad_norm is not None else
                self._gradient_norm([p for p,options in entries],scale))
            self.last_grad_norm=norm
            clip=1. if norm==0 else min(1.,self.max_grad_norm/norm)
        self._distributed_grad_norm=None
        proposals=[]
        for index,((parameter,options),used) in enumerate(zip(entries,active,strict=True)):
            if not used:continue
            name,shape,axis=self._layouts[id(parameter)]
            full=nn.Parameter(self._full_parameter(parameter,parameter,axis),requires_grad=True)
            gradient=parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            full.grad=self._full_parameter(gradient,parameter,axis)
            local=self.state.get(parameter,{})
            counters=self.transport.gather_metadata({key:value for key,value in local.items() if not isinstance(value,torch.Tensor)})
            if any(counter!=counters[0] for counter in counters):raise ValueError('TP Muon state counters/algorithms differ')
            state=dict(counters[0])
            fields=('master','momentum_buffer') if options['use_muon'] else ('master','exp_avg','exp_avg_sq')
            for field in fields:
                if local:
                    value=local[field]
                    if value.shape!=parameter.shape or value.dtype!=torch.float32 or value.device!=parameter.device:
                        raise ValueError('TP Muon master/moment layout differs')
                    state[field]=self._full_parameter(value,parameter,axis)
            owner=self.owners[index]
            core=None
            failed,error=False,None
            if self.transport.rank==owner:
                try:
                    core=Muon([dict(options,params=[full])])
                    if state:core.state[full]=state
                    if clip==0:full.grad.zero_()
                    core.step(loss_scale=scale if clip==0 else scale/clip)
                    failed=core.last_step_skipped
                except Exception as failure:error=f'{type(failure).__name__}: {failure}'
            decisions=self.consensus_group.gather_metadata((failed,error))
            for skip,failure in decisions:
                if failure:raise RuntimeError(failure)
            if any(skip for skip,failure in decisions):
                self.last_step_skipped=True
                return None
            self.transport.broadcast_(full,owner)
            new=core.state[full] if core is not None else {}
            metadata=self.transport.gather_metadata({key:value for key,value in new.items() if not isinstance(value,torch.Tensor)}
                                                   if self.transport.rank==owner else None)[owner]
            result=dict(metadata)
            for field in fields:
                tensor=new[field] if self.transport.rank==owner else torch.empty(shape,device=parameter.device,dtype=torch.float32)
                self.transport.broadcast_(tensor,owner)
                result[field]=self._local_parameter(tensor,parameter,axis)
            proposals.append((parameter,self._local_parameter(full,parameter,axis),result))
        for parameter,value,state in proposals:
            parameter.copy_(value)
            self.state[parameter]=state
            self._versions[id(parameter)]=parameter._version
        return None

    def state_dict(self):
        result=super().state_dict()
        result['ruda_tensor_muon']={'version':1,'max_grad_norm':self.max_grad_norm,
            'shapes':[self._layouts[id(p)][1] for options in self.param_groups for p in options['params']]}
        return result

    def load_state_dict(self,record):
        if record.get('ruda_tensor_muon')!=self.state_dict()['ruda_tensor_muon']:
            raise ValueError('TP Muon global shapes/clipping configuration differs')
        if len(record['param_groups'])!=len(self.param_groups):raise ValueError('TP Muon parameter groups differ')
        states,groups={},[]
        for current,saved in zip(self.param_groups,record['param_groups'],strict=True):
            if len(current['params'])!=len(saved['params']):raise ValueError('TP Muon parameter counts differ')
            options=dict(saved,params=current['params'])
            _validate_group(options)
            groups.append(options)
            for parameter,identity in zip(current['params'],saved['params'],strict=True):
                state=record['state'].get(identity)
                if state is None:continue
                expected='muon:'+options['momentum_mode'] if options['use_muon'] else 'adamw'
                fields=('master','momentum_buffer') if options['use_muon'] else ('master','exp_avg','exp_avg_sq')
                if state.get('algorithm')!=expected or type(state.get('step')) is not int or state['step']<0 or any(field not in state for field in fields):
                    raise ValueError('TP Muon checkpoint algorithm/counters/fields differ')
                restored={}
                for key,value in state.items():
                    if isinstance(value,torch.Tensor):
                        if value.shape!=parameter.shape or value.dtype!=torch.float32:raise ValueError('TP Muon state shard shape/dtype differs')
                        value=value.to(device=parameter.device,dtype=torch.float32).clone()
                    restored[key]=copy.deepcopy(value) if not isinstance(value,torch.Tensor) else value
                states[parameter]=restored
        self.param_groups=groups
        self.state.clear()
        self.state.update(states)
        self._versions={id(p):p._version for p in states}
        self.last_step_skipped=False
        self._distributed_grad_norm=None


class MeshShardedMuon(TensorParallelMuon):
    """Complete-matrix Muon with nested FSDP data slices and TP matrix axes.

    Gather data slices into their TP-local matrix, then gather the tensor axis.
    A single stage-mesh owner runs the same core update; FP32 masters/moments
    return to both partition dimensions. The inherited proposal/commit boundary
    preserves whole-step nonfinite skipping. Ordinary unsharded parameters are
    supported alongside FSDP units; vectors require explicit AdamW groups.
    """
    complete_mesh_matrices=True
    complete_matrix_shards=True

    def __init__(self,params,model,mesh,lr=.02,*,momentum=.95,weight_decay=0.,nesterov=True,
                 momentum_mode='sgd',dampening=0.,ns_steps=5,ns_coefficients=(3.4445,-4.775,2.0315),
                 eps=1e-7,adjust_lr='original',matrix_layout='as_stored',stable_normalization=True,
                 flatten=False,max_grad_norm=None,owners=None):
        from .sharded_mesh import sharded_mesh_layout
        if mesh.shape[1]!=1:raise ValueError('construct the optimizer on the actual DP/TP stage mesh')
        defaults=dict(lr=lr,momentum=momentum,weight_decay=weight_decay,nesterov=nesterov,
            momentum_mode=momentum_mode,dampening=dampening,ns_steps=ns_steps,
            ns_coefficients=tuple(ns_coefficients),eps=eps,adjust_lr=adjust_lr,matrix_layout=matrix_layout,
            stable_normalization=stable_normalization,flatten=flatten,use_muon=True,betas=(.9,.999))
        torch.optim.Optimizer.__init__(self,params,defaults)
        self.model,self.mesh,self.transport,self.consensus_group=model,mesh,mesh.world,mesh.world
        self.max_grad_norm=None if max_grad_norm is None else _number(max_grad_norm,'max_grad_norm')
        if self.max_grad_norm is not None and self.max_grad_norm<=0:raise ValueError('max_grad_norm must be positive')
        self.last_step_skipped,self.last_grad_norm=False,None
        self._distributed_grad_norm=None
        self._versions={}
        names={id(p):name for name,p in model.named_parameters()}
        layout=sharded_mesh_layout(model,mesh)
        parameters=[p for options in self.param_groups for p in options['params']]
        if not parameters or len({id(p) for p in parameters})!=len(parameters):
            raise ValueError('provide nonempty unique mesh model parameters')
        self._layouts,self._geometry={},{}
        schema=[]
        for options in self.param_groups:
            _validate_group(options)
            for parameter in options['params']:
                if id(parameter) not in names:raise ValueError('optimizer parameter is outside the mesh model')
                spec=layout[names[id(parameter)]]
                shape=tuple(spec['shape'])
                axis=None if spec.get('replicated') else spec.get('axis')
                local_shape=tuple(spec['fsdp_shape']) if 'fsdp_shape' in spec else tuple(parameter.shape)
                expected=list(local_shape)
                if axis is not None:expected[axis]*=mesh.tensor.world_size
                if tuple(expected)!=shape:raise ValueError('FSDP/TP physical and logical matrix shapes differ')
                if not parameter.requires_grad or not parameter.numel() or not parameter.is_contiguous() or parameter.device.type!=mesh.device_type:
                    raise ValueError('provide contiguous trainable mesh-device parameters')
                if parameter.dtype not in (torch.float32,torch.float16,torch.bfloat16):raise ValueError('mesh Muon supports FP32/FP16/BF16 parameters')
                if 'fsdp_shape' in spec and (parameter.ndim!=1 or parameter.numel()!=math.ceil(math.prod(local_shape)/mesh.data.world_size)):
                    raise ValueError('FSDP element storage differs from its logical data slice')
                if options['use_muon'] and (len(shape)<2 or len(shape)>2 and not options['flatten']):
                    raise ValueError('Muon needs full matrices; select use_muon=False for vector groups')
                self._layouts[id(parameter)]=(spec['name'],shape,axis)
                self._geometry[id(parameter)]=(local_shape,'fsdp_shape' in spec)
                schema.append((spec['name'],shape,axis,str(parameter.dtype),'fsdp_shape' in spec))
        _overlap_check(parameters)
        self.owners=list(owners) if owners is not None else [i%self.transport.world_size for i in range(len(parameters))]
        if len(self.owners)!=len(parameters) or any(type(owner) is not int or not 0<=owner<self.transport.world_size for owner in self.owners):
            raise ValueError('provide one actual stage-mesh owner per logical parameter')
        self.transport.validate_training_options((schema,self.owners,self.max_grad_norm,
            [{key:value for key,value in options.items() if key!='params'} for options in self.param_groups]))
        self._specs=[(id(p),tuple(p.shape),p.dtype,p.device,p.requires_grad) for p in parameters]

    def _full_parameter(self,value,parameter,axis):
        shape,sharded=self._geometry[id(parameter)]
        local=self.mesh.data.all_gather(value)[:math.prod(shape)].view(shape) if sharded else value.detach().clone()
        return local if axis is None else self.mesh.tensor.all_gather(local,axis=axis)

    def _local_parameter(self,value,parameter,axis):
        local=value.detach() if axis is None else value.chunk(self.mesh.tensor.world_size,dim=axis)[self.mesh.tensor.rank]
        if not self._geometry[id(parameter)][1]:return local.contiguous().clone()
        result=local.new_zeros(parameter.shape)
        begin=self.mesh.data.rank*parameter.numel()
        end=min(begin+parameter.numel(),local.numel())
        if end>begin:result[:end-begin].copy_(local.reshape(-1)[begin:end])
        return result

    def _gradient_norm(self,parameters,scale):
        factors={id(p):(1 if self._geometry[id(p)][1] else self.mesh.data.world_size)*
            (self.mesh.tensor.world_size if self._layouts[id(p)][2] is None else 1) for p in parameters}
        return distributed_grad_norm(parameters,self.transport,loss_scale=scale,replication_factors=factors)
