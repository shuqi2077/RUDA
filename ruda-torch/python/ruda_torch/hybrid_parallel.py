"""Joint DP/TP loss accounting, gradient synchronization and checkpoint geometry."""
from __future__ import annotations

import torch
import math
from .parallel_mesh import TensorParallelGroup
from .sharded_optim import distributed_grad_norm


class DataTensorParallelGroup:
    """Train one TP model replica per data coordinate, with exact token weighting.

    Construct after model loading/adapter injection and before the optimizer.
    DP copies are initialized from data rank zero; TP-replicated parameters and
    buffers are initialized from tensor rank zero. A logical token is counted
    once per DP replica, not once per GPU. Pipeline stages use PipelineTrainer.
    """
    def __init__(self,model,mesh,*,data_root=0,tensor_root=0,broadcast_buffers=True):
        from .sharded_training import FullyShardedModule
        from .distributed_checkpoint import model_shard_layout
        if mesh.shape[1]!=1:raise ValueError('SFT DP/TP coordination requires pipeline_parallel=1')
        if any(isinstance(module,FullyShardedModule) for module in model.modules()):
            raise ValueError('FSDP units already reduce during backward; use their sharded coordinator')
        self.model,self.mesh=model,mesh
        self.data,self.tensor,self.transport=mesh.data,mesh.tensor,mesh.world
        self.rank,self.world_size=mesh.world.rank,mesh.world.world_size
        self.device_type,self.cuda_index,self.group=mesh.device_type,mesh.world.cuda_index,mesh.world.group
        mesh.world.validate_training_options((mesh.shape,data_root,tensor_root,broadcast_buffers))
        self.data.initialize(model,root=data_root,broadcast_buffers=broadcast_buffers)
        self.tensor_coordinator=TensorParallelGroup(model,self.tensor)
        with torch.no_grad():
            for parameter in model.parameters():
                if not getattr(parameter,'_ruda_tp_sharded',False):self.tensor.broadcast_(parameter,tensor_root)
            if broadcast_buffers:
                layout=model_shard_layout(model)
                for name,value in model.named_buffers():
                    spec=layout.get(name,{})
                    if not spec.get('transform') and (not spec or spec.get('replicated')):
                        self.tensor.broadcast_(value,tensor_root)

    def gather_metadata(self,value):return self.transport.gather_metadata(value)

    def validate_training_options(self,value):return self.transport.validate_training_options(value)

    def validate_model(self,model):
        self.data.validate_model(model)
        self.tensor_coordinator.validate_model(model)

    def total_weight(self,local_weight):
        weights=self.gather_metadata(local_weight)
        data,_,tensor=self.mesh.shape
        if any(type(weight) is not int or weight<0 for weight in weights):
            raise ValueError('logical token weights must be nonnegative integers')
        if any(len(set(weights[d*tensor:(d+1)*tensor]))!=1 for d in range(data)):
            raise ValueError('tensor ranks must agree on the replica logical token count')
        total=sum(weights[::tensor])
        if not total:raise ValueError('global effective weight must be positive')
        return total

    def validate_microbatches(self,batches):
        prepared=[{name:value.tolist() if isinstance(value,torch.Tensor) else value for name,value in batch.items()} for batch in batches]
        requests=self.tensor.gather_metadata(prepared)
        error=None if all(item==prepared for item in requests) else 'TP ranks must execute the same logical minibatches'
        for failure in self.gather_metadata(error):
            if failure:raise ValueError(failure)

    def sum_(self,value):
        # Loss logging sums distinct data replicas, not identical TP losses.
        return self.data.sum_(value)

    def synchronize_gradients(self,*,local_weight,normalized=False,missing='zero'):
        total=self.total_weight(local_weight)
        self.validate_training_options((normalized,missing))
        self.data.synchronize_gradients(local_weight=local_weight,normalized=normalized,missing=missing)
        self._complete_logical_gradients(list(self.model.parameters()))
        return total

    def _complete_logical_gradients(self,parameters,originals=None):
        originals=parameters if originals is None else originals
        requests=self.gather_metadata(tuple(p.grad is not None for p in parameters))
        tensor=self.mesh.shape[2]
        with torch.no_grad():
            for index,(parameter,original) in enumerate(zip(parameters,originals,strict=True)):
                if not original.requires_grad:continue
                active=any(flags[index] for flags in requests)
                if not getattr(original,'_ruda_tp_sharded',False) and len({flags[index] for flags in requests[:tensor]})>1:
                    raise ValueError('a TP-replicated parameter has inconsistent gradient participation')
                if active and parameter.grad is None:parameter.grad=torch.zeros_like(parameter)

    def begin_gradient_overlap(self,*,global_weight,normalized=False,bucket_bytes=25*1024*1024):
        return self.data.begin_gradient_overlap(global_weight=global_weight,normalized=normalized,bucket_bytes=bucket_bytes)

    def replication_factors(self,parameters):
        data,_,tensor=self.mesh.shape
        return {id(p):data*(1 if getattr(p,'_ruda_tp_sharded',False) else tensor) for p in parameters}

    def prepare_optimizer_step(self,optimizer,*,loss_scale=1.):
        from .sharded_training import Zero2Optimizer
        from .optim import Muon
        zero=isinstance(optimizer,Zero2Optimizer)
        actual=optimizer.optimizer if zero else optimizer
        if zero and optimizer.group.group is not self.data.group:
            raise ValueError('ZeRO-2 must shard states over the mesh data group')
        parameters=[p for options in actual.param_groups for p in options['params']]
        original_by_shard={id(shard):original for original,shard in optimizer.entries} if zero else {}
        originals=[original_by_shard[id(p)] for p in parameters] if zero else parameters
        names={id(p):name for name,p in self.model.named_parameters()}
        self.validate_training_options([names[id(p)] for p in originals])
        if isinstance(actual,Muon) and any(getattr(p,'_ruda_tp_sharded',False) for p in originals):
            raise ValueError('use TensorParallelMuon for complete-matrix TP updates')
        self._complete_logical_gradients(parameters,originals)
        if getattr(actual,'complete_tensor_parallel_matrices',False):
            actual.consensus_group=self.transport
            return
        if not hasattr(actual,'last_step_skipped'):return
        bad=torch.zeros(1,device=parameters[0].device,dtype=torch.float32)
        for parameter in parameters:
            if parameter.grad is not None:bad.add_((~torch.isfinite(parameter.grad.float()/loss_scale)).any().float())
        self.transport.sum_(bad)
        if bad.item():
            for parameter in parameters:
                if parameter.grad is not None:parameter.grad.fill_(float('inf'))
            return
        if getattr(actual,'max_grad_norm',None) is not None and hasattr(actual,'_distributed_grad_norm'):
            data,_,tensor=self.mesh.shape
            factors={id(p):(1 if zero else data)*(tensor if not getattr(original,'_ruda_tp_sharded',False) else 1)
                     for p,original in zip(parameters,originals,strict=True)}
            norm=distributed_grad_norm(parameters,self.transport,loss_scale=loss_scale,replication_factors=factors)
            if zero:optimizer._mesh_grad_norm=norm
            else:actual._distributed_grad_norm=norm

    def checkpoint_layout(self,layout):
        data,_,tensor=self.mesh.shape
        result={name:dict(spec,mesh={'data':data,'tensor':tensor}) for name,spec in layout.items()}
        for name,value in self.model.named_parameters():
            if name not in result:
                result[name]={'name':name,'aliases':[],'shape':list(value.shape),'axis':None,
                              'replicated':True,'mesh':{'data':data,'tensor':tensor}}
        return result


def join_mesh_field(values,spec):
    """Remove identical DP copies before reconstructing a TP logical tensor."""
    from .distributed_checkpoint import _equal,_join
    data,tensor=spec['mesh']['data'],spec['mesh']['tensor']
    if len(values)!=data*tensor:raise ValueError('checkpoint mesh rank count differs')
    plain={key:value for key,value in spec.items() if key!='mesh'}
    if 'fsdp_shape' in plain:
        shape=plain.pop('fsdp_shape')
        pieces=[torch.cat([values[d*tensor+t].reshape(-1) for d in range(data)])[:math.prod(shape)].view(shape)
                for t in range(tensor)]
        return _join(pieces,plain)
    if plain.get('transform')=='fsdp-template-extra':
        from .sharded_mesh import join_fsdp_extra
        for t in range(tensor):
            if any(not _equal(values[t],values[d*tensor+t]) for d in range(1,data)):
                raise ValueError('data replicas contain different FSDP template states')
        return join_fsdp_extra(values[:tensor],plain)
    if plain.get('transform')=='zero2-tp-flat':
        pieces=[]
        for rank in range(tensor):
            shape=plain['local_shape']
            pieces.append(torch.cat([values[d*tensor+rank].reshape(-1) for d in range(data)])[:math.prod(shape)].view(shape))
        plain.pop('transform')
        plain.pop('local_shape')
        return _join(pieces,plain)
    for rank in range(tensor):
        if any(not _equal(values[rank],values[d*tensor+rank]) for d in range(1,data)):
            raise ValueError('data replicas contain different checkpoint values')
    return _join(values[:tensor],plain)


def slice_mesh_field(value,spec,rank,world):
    from .distributed_checkpoint import _slice
    data,tensor=spec['mesh']['data'],spec['mesh']['tensor']
    if world!=data*tensor:raise ValueError('target checkpoint mesh rank count differs')
    plain={key:item for key,item in spec.items() if key!='mesh'}
    if plain.get('transform')=='fsdp-template-extra':
        from .sharded_mesh import slice_fsdp_extra
        return slice_fsdp_extra(value,plain,rank%tensor,tensor)
    fsdp=plain.pop('fsdp_shape',None)
    zero=plain.get('transform')=='zero2-tp-flat'
    if zero:
        plain.pop('transform')
        plain.pop('local_shape')
    local=_slice(value,plain,rank%tensor,tensor)
    if not zero and fsdp is None:return local
    size=math.ceil(local.numel()/data)
    result=local.new_zeros(size)
    begin=(rank//tensor)*size
    end=min(begin+size,local.numel())
    if end>begin:result[:end-begin].copy_(local.reshape(-1)[begin:end])
    return result
