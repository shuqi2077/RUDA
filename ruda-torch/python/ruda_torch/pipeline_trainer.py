"""DP/TP/PP optimizer boundaries and all-stage, shared-filesystem recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import uuid
import torch
from .hybrid_parallel import DataTensorParallelGroup
from .distributed_checkpoint import DistributedCheckpoint,_write_json
from .sharded_optim import distributed_grad_norm


class PipelineTrainer:
    """Train caller-built stages with the existing 1F1B native allocation schedule.

    Stage names are explicit, unique filesystem namespaces. Inputs/targets are
    CPU microbatches on the first/last stage; the caller supplies their actual
    local token/sample weight on every rank. TP input shards are an explicit
    option. No architecture, loss, stage partition or sampler is inferred.
    Driver/update failures require checkpoint restore, not an implicit retry.
    """
    def __init__(self,stage,optimizer,mesh,*,stage_name,base_id,run_config,
                 scaler=None,scheduler=None,input_is_parallel=False,tied_parameters=None):
        if not isinstance(stage_name,str) or not stage_name or Path(stage_name).name!=stage_name or stage_name in ('.','..'):
            raise ValueError('stage_name must be an explicit single directory component')
        if stage.group.group is not mesh.pipeline.group or stage.rank!=mesh.coordinates[1]:
            raise ValueError('stage schedule and trainer must use the same pipeline group')
        if type(input_is_parallel) is not bool:raise ValueError('input_is_parallel must be bool')
        if optimizer is None and any(p.requires_grad for p in stage.module.parameters()):
            raise ValueError('a trainable pipeline stage needs an optimizer')
        if optimizer is None and scheduler is not None:raise ValueError('a scheduler needs an optimizer')
        if scaler is not None and optimizer is not None and not hasattr(optimizer,'last_step_skipped'):
            raise TypeError('scaled pipeline training requires an explicit RUDA loss-scale optimizer')
        self.stage,self.optimizer,self.mesh=stage,optimizer,mesh
        self.stage_name,self.base_id,self.run_config=stage_name,base_id,run_config
        self.scaler,self.scheduler=scaler,scheduler
        self.input_is_parallel=input_is_parallel
        self.tied_parameters=tied_parameters
        if tied_parameters is not None and (tied_parameters.model is not stage.module or tied_parameters.mesh is not mesh):
            raise ValueError('tie bindings must belong to this actual stage and mesh')
        from .sharded_training import FullyShardedModule
        from .sharded_mesh import ShardedDataTensorParallelGroup
        coordinator=ShardedDataTensorParallelGroup if any(isinstance(unit,FullyShardedModule) for unit in stage.module.modules()) else DataTensorParallelGroup
        self.coordinator=coordinator(stage.module,mesh.stage_mesh())
        self.coordinator.validate_training_options(stage.invocation_contract)
        self.sharded_pipeline=any(mesh.world.gather_metadata(isinstance(self.coordinator,ShardedDataTensorParallelGroup)))
        if tied_parameters is not None:tied_parameters.validate_optimizer(optimizer)
        names=mesh.world.gather_metadata(stage_name)
        data,pipeline,tensor=mesh.shape
        self.stage_names=[names[p*tensor] for p in range(pipeline)]
        if len(set(self.stage_names))!=pipeline or any(names[(d*pipeline+p)*tensor+t]!=self.stage_names[p]
                for d in range(data) for p in range(pipeline) for t in range(tensor)):
            raise ValueError('each pipeline stage needs one unique namespace shared by its DP/TP ranks')
        mesh.world.validate_training_options((base_id,run_config,input_is_parallel))
        self.step,self.tokens,self.cursor=0,0,0
        self.last_checkpoint=None
        self.started=time.monotonic()
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)

    def train_step(self,inputs=None,targets=None,*,local_weight,loss_sum,microbatch_specs=None):
        inputs=list(inputs or [])
        targets=list(targets or [])
        if not callable(loss_sum):raise TypeError('supply a loss_sum(output,target) callable')
        weights=self.mesh.world.gather_metadata(local_weight)
        data,pipeline,tensor=self.mesh.shape
        width=pipeline*tensor
        if any(type(value) is not int or value<0 for value in weights) or any(len(set(weights[d*width:(d+1)*width]))!=1 for d in range(data)):
            raise ValueError('each DP replica must agree on its logical weight across PP/TP ranks')
        total=sum(weights[::width])
        if not total:raise ValueError('global pipeline effective weight must be positive')
        error=None
        for values in (inputs,targets):
            if any(not self.stage._is_cpu_payload(value) for value in values):
                error='pipeline trainer input/target tensor leaves must be on CPU'
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise ValueError(failure)
        schedules=self.mesh.world.gather_metadata((len(inputs),len(targets)))
        counts=[schedules[d*width+t][0] for d in range(data) for t in range(tensor)]
        if any(schedules[d*width+t][0]!=schedules[d*width+(pipeline-1)*tensor+t][1]
               for d in range(data) for t in range(tensor)):
            raise ValueError('first-stage inputs and last-stage targets need identical microbatch counts')
        if self.sharded_pipeline and len(set(counts))!=1:
            raise ValueError('FSDP pipeline replicas must execute the same collective-bearing microbatch count')
        data_rank,_,tensor_rank=self.mesh.coordinates
        count=schedules[data_rank*width+tensor_rank][0]
        interfaces=None
        try:interfaces=self.stage.validate_interfaces(count,microbatch_specs)
        except (TypeError,ValueError) as failure:error=str(failure)
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise ValueError(failure)
        contracts=self.mesh.tensor.gather_metadata(interfaces)
        error=None if all(other==interfaces for other in contracts) else 'TP ranks supplied different pipeline boundary contracts'
        if self.stage.rank==0 and any(not self.stage._matches(value,spec[0]) for value,spec in zip(inputs,interfaces,strict=True)):
            error='first-stage input differs from its explicit pipeline interface'
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise ValueError(failure)
        payload=([self.stage._payload_metadata(value,self.input_is_parallel) for value in inputs],
                 [self.stage._payload_metadata(value) for value in targets])
        requests=self.mesh.tensor.gather_metadata(payload)
        error=None if all(item==payload for item in requests) else 'TP ranks supplied different pipeline logical microbatches'
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise ValueError(failure)
        self.coordinator.validate_model(self.stage.module)
        self.mesh.world.validate_training_options((self.step,self.base_id,self.run_config,
                                                  None if self.scaler is None else self.scaler.state_dict()))
        if self.optimizer is not None:self.optimizer.zero_grad(set_to_none=True)
        scale=1. if self.scaler is None else self.scaler.begin_backward()
        started=time.monotonic()
        self.stage.module.train()
        result=self.stage.run(inputs,targets,loss_sum=loss_sum,global_weight=total,loss_scale=scale,microbatch_specs=interfaces)
        if self.tied_parameters is not None:self.tied_parameters.synchronize_gradients()
        if self.optimizer is not None:
            if hasattr(self.optimizer,'synchronize_gradients'):
                self.optimizer.synchronize_gradients(local_weight=local_weight,normalized=True,missing='zero')
            else:self.coordinator.synchronize_gradients(local_weight=local_weight,normalized=True,missing='zero')
        bad=torch.zeros(1,device=self.stage.device,dtype=torch.float32)
        if self.optimizer is not None:
            for options in self.optimizer.param_groups:
                for parameter in options['params']:
                    if parameter.grad is not None:bad.add_((~torch.isfinite(parameter.grad.float()/scale)).any().float())
        self.mesh.world.sum_(bad)
        if bad.item() and self.optimizer is not None:
            for options in self.optimizer.param_groups:
                for parameter in options['params']:
                    if parameter.grad is not None:parameter.grad.fill_(float('inf'))
        if self.optimizer is not None:self.coordinator.prepare_optimizer_step(self.optimizer,loss_scale=scale)
        self._prepare_global_norm(scale)
        failed,skipped=None,False
        try:
            if self.optimizer is not None:
                if self.scaler is None:self.optimizer.step()
                else:self.optimizer.step(loss_scale=scale)
                skipped=bool(getattr(self.optimizer,'last_step_skipped',False))
        except Exception as failure:failed=f'{type(failure).__name__}: {failure}'
        decisions=self.mesh.world.gather_metadata((self.optimizer is not None,skipped,failed))
        if any(failure for active,skip,failure in decisions):
            raise RuntimeError('pipeline optimizer failed; restore the committed mesh checkpoint: '+
                               '; '.join(failure for active,skip,failure in decisions if failure))
        flags={skip for active,skip,failure in decisions if active}
        if len(flags)>1:raise RuntimeError('pipeline optimizers disagreed on update; restore the committed mesh checkpoint')
        skipped=next(iter(flags),False)
        if self.tied_parameters is not None and not skipped:self.tied_parameters.synchronize_state(self.optimizer)
        if self.scaler is not None:
            self.scaler.record_step(skipped)
            self.scaler.update()
        if self.scheduler is not None and not skipped:self.scheduler.step()
        if self.optimizer is not None:self.optimizer.zero_grad(set_to_none=True)
        loss=torch.tensor([result['loss_sum']],device=self.stage.device,dtype=torch.float32)
        self.mesh.data.sum_(loss)
        count=self.mesh.pipeline.gather_metadata(len(inputs))[0]
        self.step+=1
        self.tokens+=total
        self.cursor+=count
        elapsed=time.monotonic()-started
        return {'step':self.step,'loss':float(loss.item())/total,'supervised_tokens':total,
                'total_supervised_tokens':self.tokens,'microbatch_cursor':self.cursor,
                'step_seconds':elapsed,'tokens_per_second':total/elapsed,'optimizer_update_skipped':skipped}

    def _prepare_global_norm(self,scale):
        from .sharded_training import Zero2Optimizer
        zero=isinstance(self.optimizer,Zero2Optimizer)
        actual=self.optimizer.optimizer if zero else self.optimizer
        requested=actual is not None and getattr(actual,'max_grad_norm',None) is not None and hasattr(actual,'_distributed_grad_norm')
        if not any(self.mesh.world.gather_metadata(requested)):return
        parameters=[] if actual is None else [p for options in actual.param_groups for p in options['params']]
        originals={id(shard):original for original,shard in self.optimizer.entries} if zero else {}
        data,_,tensor=self.mesh.shape
        factors={} if zero else self.coordinator.replication_factors(parameters)
        for parameter in parameters:
            original=originals[id(parameter)] if zero else parameter
            consumers=1 if self.tied_parameters is None else self.tied_parameters.replication_count(original)
            factors[id(parameter)]=(tensor if not getattr(original,'_ruda_tp_sharded',False) else 1)*consumers if zero else factors[id(parameter)]*consumers
        norm=distributed_grad_norm(parameters,self.mesh.world,loss_scale=scale,replication_factors=factors)
        if requested:
            if zero:self.optimizer._mesh_grad_norm=norm if math.isfinite(norm) else None
            else:actual._distributed_grad_norm=norm if math.isfinite(norm) else None

    def save_distributed(self,directory,*,application_state=None):
        """Commit a global pointer only after EVERY pipeline stage has committed."""
        directory=Path(directory)
        self.mesh.world.validate_training_options((str(directory.resolve()),self.step,self.tokens,self.base_id,self.run_config))
        state={'base_id':self.base_id,'run_config':self.run_config,'tokens':self.tokens,
               'cursor':self.cursor,'application':application_state}
        result,error=None,None
        try:
            result=DistributedCheckpoint(self.coordinator,directory/self.stage_name).save(self.stage.module,
                self.optimizer,step=self.step,application_state=state,scheduler=self.scheduler,scaler=self.scaler)
        except Exception as failure:error=f'{type(failure).__name__}: {failure}'
        receipts=self.mesh.world.gather_metadata((self.stage_name,result,error))
        for name,saved,failure in receipts:
            if failure:raise RuntimeError('mesh checkpoint not committed: '+failure)
        generation=self.mesh.world.gather_metadata(uuid.uuid4().hex if self.mesh.rank==0 else None)[0]
        generation=f'step-{self.step}-{generation}'
        error=None
        if self.mesh.rank==0:
            try:
                stages={}
                for name,saved,failure in receipts:
                    entry={'name':name,'generation':Path(saved['directory']).name}
                    if name in stages and stages[name]!=entry:raise ValueError('stage checkpoint generations differ')
                    stages[name]=entry
                folder=directory/generation
                folder.mkdir(parents=True,exist_ok=False)
                _write_json(folder/'manifest.json',{'version':1,'step':self.step,'mesh':list(self.mesh.shape),
                                                  'stages':[stages[name] for name in self.stage_names]})
                _write_json(directory/'latest.json',{'generation':generation,'step':self.step})
            except Exception as failure:error=f'{type(failure).__name__}: {failure}'
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise RuntimeError('mesh checkpoint not committed: '+failure)
        self.last_checkpoint={'directory':str((directory/generation).resolve()),'step':self.step}
        return self.last_checkpoint

    @staticmethod
    def _manifest(directory,generation):
        directory=Path(directory)
        if generation is None:generation=json.loads((directory/'latest.json').read_text(encoding='utf-8'))['generation']
        if not isinstance(generation,str) or not generation or generation in ('.','..') or Path(generation).name!=generation:raise ValueError('invalid mesh checkpoint generation')
        manifest=json.loads((directory/generation/'manifest.json').read_text(encoding='utf-8'))
        if manifest['version']!=1:raise ValueError('unsupported mesh checkpoint format')
        for stage in manifest['stages']:
            if not stage['name'] or not stage['generation'] or stage['name'] in ('.','..') or stage['generation'] in ('.','..') or Path(stage['name']).name!=stage['name'] or Path(stage['generation']).name!=stage['generation']:
                raise ValueError('invalid stage checkpoint path')
        return directory,manifest

    def resume_distributed(self,directory,*,generation=None):
        manifest,error=None,None
        try:
            directory,manifest=self._manifest(directory,generation)
            if manifest['mesh']!=list(self.mesh.shape) or [stage['name'] for stage in manifest['stages']]!=self.stage_names:
                raise ValueError('pipeline mesh changed; consolidate and explicitly reshard')
        except Exception as failure:error=str(failure)
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise ValueError(failure)
        stage=manifest['stages'][self.mesh.coordinates[1]]
        def validate(state):
            if (state['base_id'],state['run_config'])!=(self.base_id,self.run_config):raise ValueError('checkpoint base/config differs')
        restored,error=None,None
        try:
            restored=DistributedCheckpoint(self.coordinator,directory/self.stage_name).load(self.stage.module,self.optimizer,
                generation=stage['generation'],scheduler=self.scheduler,scaler=self.scaler,validate_application=validate)
            if restored[0]!=manifest['step']:raise ValueError('stage checkpoint step differs from the mesh commit')
        except Exception as failure:error=f'{type(failure).__name__}: {failure}'
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise RuntimeError('mesh checkpoint restore failed: '+failure)
        step,state=restored
        self.step,self.tokens,self.cursor=step,state['tokens'],state['cursor']
        if self.optimizer is not None:self.optimizer.zero_grad(set_to_none=True)
        return state['application']

    def consolidate(self,directory,*,generation=None):
        """Explicit CPU consolidation; stage namespaces and optimizer classes stay separate."""
        directory,manifest=self._manifest(directory,generation)
        stages={entry['name']:DistributedCheckpoint(self.coordinator,directory/entry['name']).consolidate(generation=entry['generation'])
                for entry in manifest['stages']}
        return {'version':1,'step':manifest['step'],'mesh':manifest['mesh'],'stages':stages}

    def resume_consolidated(self,state,*,reshard_application,stage_transform=None):
        """Resize DP/TP, or explicitly remap PP models/optimizers with stage_transform.

        stage_transform(stages, stage_name, pipeline_rank) must return an ordinary
        consolidated DistributedCheckpoint record for this caller-built stage.
        Data/RNG repartitioning remains an explicit reshard_application callback.
        """
        if state.get('version')!=1:raise ValueError('invalid consolidated pipeline checkpoint')
        source,error=None,None
        try:
            source=state['stages'][self.stage_name] if stage_transform is None else stage_transform(state['stages'],self.stage_name,self.mesh.coordinates[1])
            if source['step']!=state['step']:raise ValueError('consolidated stage step differs from mesh commit')
            if any((rank['application']['base_id'],rank['application']['run_config'])!=(self.base_id,self.run_config)
                   for rank in source['rank_states']):raise ValueError('checkpoint base/config differs')
        except Exception as failure:error=f'{type(failure).__name__}: {failure}'
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise ValueError(failure)
        restored,error=None,None
        try:
            restored=DistributedCheckpoint(self.coordinator,'.').load_consolidated(self.stage.module,self.optimizer,
                source,reshard_application=reshard_application,scheduler=self.scheduler,scaler=self.scaler)
            step,application=restored
            if (application['base_id'],application['run_config'])!=(self.base_id,self.run_config):raise ValueError('checkpoint base/config differs')
        except Exception as failure:error=f'{type(failure).__name__}: {failure}'
        for failure in self.mesh.world.gather_metadata(error):
            if failure:raise RuntimeError('consolidated mesh restore failed: '+failure)
        step,application=restored
        self.step,self.tokens,self.cursor=step,application['tokens'],application['cursor']
        if self.optimizer is not None:self.optimizer.zero_grad(set_to_none=True)
        return application['application']
