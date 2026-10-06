"""Explicit Cartesian DP/TP/PP process groups and tensor-parallel SFT coordination."""
from __future__ import annotations

from datetime import timedelta
from itertools import product
import torch
import torch.distributed as dist
from .distributed_training import ReplicaGroup


class ParallelMesh:
    """Rank = (data_rank*pipeline_size + pipeline_rank)*tensor_size + tensor_rank.

    Every global rank constructs groups in the same order. Devices/processes are
    selected by the launcher before construction; the mesh does not spawn jobs or
    guess hardware ordinals. CPU/Gloo remains an explicit reference selection.
    """
    def __init__(self, *, data_parallel=1, tensor_parallel=1, pipeline_parallel=1,
                 device_type='ruda', timeout_seconds=1800):
        shape=(data_parallel,pipeline_parallel,tensor_parallel)
        if any(type(size) is not int or size<1 for size in shape) or timeout_seconds<=0:
            raise ValueError('parallel dimensions and timeout must be positive')
        if not dist.is_initialized() or data_parallel*tensor_parallel*pipeline_parallel!=dist.get_world_size():
            raise ValueError('parallel mesh must cover the initialized global process group')
        if device_type not in ('ruda','cpu'):
            raise ValueError('select native RUDA or the explicit CPU reference')
        self.rank=dist.get_rank()
        self.shape,self.device_type=shape,device_type
        self.coordinates=(self.rank//(pipeline_parallel*tensor_parallel),
                          (self.rank//tensor_parallel)%pipeline_parallel,self.rank%tensor_parallel)
        contracts=[None]*dist.get_world_size()
        dist.all_gather_object(contracts,(shape,device_type,timeout_seconds))
        if any(value!=(shape,device_type,timeout_seconds) for value in contracts):
            raise ValueError('global ranks supplied different mesh configurations')
        self.world=ReplicaGroup(device_type=device_type)
        self.groups={}
        self._owned=[]
        backend='nccl' if device_type=='ruda' else 'gloo'
        names=('data','pipeline','tensor')
        for axis,name in enumerate(names):
            other=[d for d in range(3) if d!=axis]
            for fixed in product(*(range(shape[d]) for d in other)):
                coordinates=[0,0,0]
                for d,value in zip(other,fixed):coordinates[d]=value
                ranks=[]
                for value in range(shape[axis]):
                    coordinates[axis]=value
                    d,p,t=coordinates
                    ranks.append((d*pipeline_parallel+p)*tensor_parallel+t)
                process_group=dist.new_group(ranks,backend=backend,timeout=timedelta(seconds=timeout_seconds))
                if self.rank in ranks:
                    self.groups[name]=ReplicaGroup(process_group,device_type=device_type)
                    self._owned.append(process_group)
        for pipeline_rank in range(pipeline_parallel):
            ranks=[(d*pipeline_parallel+pipeline_rank)*tensor_parallel+t
                   for d in range(data_parallel) for t in range(tensor_parallel)]
            process_group=dist.new_group(ranks,backend=backend,timeout=timedelta(seconds=timeout_seconds))
            if self.rank in ranks:
                self.groups['stage']=ReplicaGroup(process_group,device_type=device_type)
                self._owned.append(process_group)

    @property
    def data(self):return self.groups['data']

    @property
    def tensor(self):return self.groups['tensor']

    @property
    def pipeline(self):return self.groups['pipeline']

    @property
    def stage(self):return self.groups['stage']

    def stage_mesh(self):
        """DP/TP view backed by the REAL current pipeline-stage process groups."""
        return _StageMesh(self)

    def close(self):
        for process_group in reversed(self._owned):
            dist.destroy_process_group(process_group)
        self._owned.clear()


class _StageMesh:
    def __init__(self,parent):
        self.parent=parent
        self.shape=(parent.shape[0],1,parent.shape[2])
        self.device_type=parent.device_type
        self.world,self.data,self.tensor=parent.stage,parent.data,parent.tensor
        self.rank=self.world.rank
        self.coordinates=(parent.coordinates[0],0,parent.coordinates[2])


class TensorParallelGroup:
    """Logical-loss coordinator; TP ranks execute the SAME logical minibatch.

    Model-parallel regions already provide projection derivatives. Replicated
    norm/bias gradients are not summed again, and token counts are not multiplied
    by tensor-parallel world size. A separate DP mesh axis combines distinct data.
    """
    def __init__(self, model, group):
        self.model,self.transport=model,group
        self.rank,self.world_size,self.device_type=group.rank,group.world_size,group.device_type
        self.group,self.cuda_index=group.group,group.cuda_index
        self._specs=[(name,id(p),tuple(p.shape),p.dtype,p.device,p.requires_grad) for name,p in model.named_parameters()]
        group.validate_training_options([(name,tuple(p.shape),str(p.dtype),p.requires_grad) for name,p in model.named_parameters()])

    def gather_metadata(self,value):return self.transport.gather_metadata(value)

    def validate_training_options(self,value):return self.transport.validate_training_options(value)

    def validate_model(self,model):
        current=[(name,id(p),tuple(p.shape),p.dtype,p.device,p.requires_grad) for name,p in model.named_parameters()]
        if model is not self.model or current!=self._specs:
            raise ValueError('tensor-parallel model changed after coordinator creation')

    def total_weight(self,local_weight):
        weights=self.gather_metadata(local_weight)
        if type(local_weight) is not int or local_weight<=0 or any(w!=local_weight for w in weights):
            raise ValueError('TP ranks must use the same positive logical token weight')
        return local_weight

    def validate_microbatches(self,batches):
        # Batches are CPU inputs to SFTTrainer, so exact equality does not cause
        # a GPU payload download or change rank-local compute placement.
        prepared=[{name:value.tolist() if isinstance(value,torch.Tensor) else value for name,value in batch.items()} for batch in batches]
        self.validate_training_options(prepared)

    def sum_(self,tensor):
        # Used by SFTTrainer for replicated logical loss reporting only.
        return tensor

    def synchronize_gradients(self,*,local_weight,normalized=False,missing='zero'):
        weight=self.total_weight(local_weight)
        if missing not in ('zero','error') or type(normalized) is not bool:
            raise ValueError('invalid tensor-parallel reduction policy')
        flags=tuple(p.grad is not None for p in self.model.parameters() if p.requires_grad)
        requests=self.gather_metadata(flags)
        if missing=='error' and any(not all(value) for value in requests):
            raise ValueError('a tensor-parallel shard lacks a trainable gradient')
        if not normalized:
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.grad is not None:parameter.grad.div_(weight)
        return weight

    def prepare_optimizer_step(self,optimizer,*,loss_scale=1.):
        from .optim import Muon
        if isinstance(optimizer,Muon) and any(getattr(p,'_ruda_tp_sharded',False) for options in optimizer.param_groups for p in options['params']):
            raise ValueError('use TensorParallelMuon to orthogonalize complete TP matrices')
        if getattr(optimizer,'complete_tensor_parallel_matrices',False):
            return
        if not hasattr(optimizer,'last_step_skipped'):
            return
        parameters=[p for options in optimizer.param_groups for p in options['params']]
        flag=torch.zeros(1,dtype=torch.float32,device=parameters[0].device)
        for parameter in parameters:
            if parameter.grad is not None:flag.add_((~torch.isfinite(parameter.grad.float()/loss_scale)).any().float())
        self.transport.sum_(flag)
        if flag.item():
            for parameter in parameters:
                if parameter.grad is not None:parameter.grad.fill_(float('inf'))
        elif getattr(optimizer,'max_grad_norm',None) is not None and hasattr(optimizer,'_distributed_grad_norm'):
            from .sharded_optim import distributed_grad_norm
            optimizer._distributed_grad_norm=distributed_grad_norm(parameters,self.transport,loss_scale=loss_scale,
                replicated_ids={id(p) for p in parameters if not getattr(p,'_ruda_tp_sharded',False)})
