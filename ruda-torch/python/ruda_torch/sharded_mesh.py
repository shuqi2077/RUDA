"""FSDP data slices nested inside tensor-parallel model replicas."""
from __future__ import annotations

import math
import torch
from .hybrid_parallel import DataTensorParallelGroup
from .parallel_mesh import TensorParallelGroup
from .sharded_training import FullyShardedModule,ShardedReplicaGroup
from .distributed_checkpoint import _join,_slice,_equal,model_shard_layout
from .sharded_optim import distributed_grad_norm


def _field(unit,name,shape):
    return dict(unit._template_layout.get(name,
        {'name':name,'aliases':[],'shape':list(shape),'axis':None,'replicated':True}))


def _prefix(spec,prefix):
    return dict(spec,name=prefix+spec['name'],aliases=[prefix+name for name in spec.get('aliases',[])])


def sharded_mesh_layout(model,mesh):
    """Physical storage -> logical tensor, retaining both partition dimensions."""
    data,_,tensor=mesh.shape
    geometry={'data':data,'tensor':tensor}
    layout={name:dict(spec,mesh=geometry) for name,spec in model_shard_layout(model).items()}
    for path,unit in model.named_modules(remove_duplicate=False):
        if not isinstance(unit,FullyShardedModule):continue
        prefix=path+'.' if path else ''
        for name,index,shape,dtype,trainable in unit._schema:
            local=prefix+unit._shard_names[index]
            spec=_prefix(_field(unit,name,shape),prefix)
            alias=spec['name']
            if local in layout and 'fsdp_shape' in layout[local]:
                if alias!=layout[local]['name'] and alias not in layout[local]['aliases']:
                    layout[local]['aliases'].append(alias)
            else:layout[local]=dict(spec,fsdp_shape=list(shape),mesh=geometry)
        buffer_fields={}
        for name,key in unit._buffer_names:
            shape=unit._buffer_shards.get(key,tuple(unit.get_buffer(key).shape))
            spec=_field(unit,name,shape)
            local=prefix+key
            if key in buffer_fields:
                alias=prefix+spec['name']
                if alias!=layout[local]['name'] and alias not in layout[local]['aliases']:
                    layout[local]['aliases'].append(alias)
                continue
            buffer_fields[key]=spec
            spec=_prefix(spec,prefix)
            if key in unit._buffer_shards:spec['fsdp_shape']=list(shape)
            layout[local]=dict(spec,mesh=geometry)
        layout[prefix+'_extra_state']={'name':prefix+'_extra_state','aliases':[],
            'transform':'fsdp-template-extra','template':unit.get_extra_state(),
            'fields':unit._template_layout,'buffers':buffer_fields,'mesh':geometry}
    for name,value in list(model.named_parameters())+list(model.named_buffers()):
        if name not in layout:
            layout[name]={'name':name,'aliases':[],'shape':list(value.shape),
                          'axis':None,'replicated':True,'mesh':geometry}
    return layout


def _extra_path(name):
    return name[:-len('._extra_state')] if name.endswith('._extra_state') else '' if name=='_extra_state' else name


def _logical_shape(spec):
    shape=spec['shape']
    if spec.get('transform')=='nf4-packed':return ((math.prod(shape)+1)//2,)
    if spec.get('transform')=='nf4-scales':return ((math.prod(shape)+spec['block_size']-1)//spec['block_size'],)
    return tuple(shape)


def _extra_schema(state,spec):
    fields=spec['fields']
    schema=[]
    for name,index,shape,dtype,trainable in state['schema']:
        field=fields.get(name,{'name':name,'shape':shape})
        schema.append((field['name'],index,_logical_shape(field),dtype,trainable))
    buffers={spec['buffers'][key]['name']:_logical_shape(spec['buffers'][key])
             for key in state['buffer_shards']}
    return tuple(schema),buffers


def join_fsdp_extra(values,spec):
    """Consolidate template geometry and NF4/LoRA state without copying weights."""
    schemas=[_extra_schema(value,spec) for value in values]
    if any(schema!=schemas[0] for schema in schemas):raise ValueError('FSDP logical template schemas differ')
    paths=values[0]['module_extra'].keys()
    if any(value['version']!=1 or value['module_extra'].keys()!=paths for value in values):
        raise ValueError('FSDP template extra-state structure differs')
    extras={}
    for path in paths:
        key=path+'._extra_state' if path else '_extra_state'
        field=spec['fields'].get(key)
        name=path if field is None else _extra_path(field['name'])
        value=_join([item['module_extra'][path] for item in values],field)
        if name in extras and not _equal(extras[name],value):raise ValueError('aliased template extra states differ')
        extras[name]=value
    schema,buffers=schemas[0]
    return {'version':2,'schema':schema,'buffers':buffers,'module_extra':extras}


def slice_fsdp_extra(value,spec,rank,world):
    template=spec['template']
    schema,buffers=_extra_schema(template,spec)
    if value.get('version')!=2 or (value['schema'],value['buffers'])!=(schema,buffers):
        raise ValueError('consolidated FSDP logical template differs')
    extras={}
    for path in template['module_extra']:
        key=path+'._extra_state' if path else '_extra_state'
        field=spec['fields'].get(key)
        logical=path if field is None else _extra_path(field['name'])
        if logical not in value['module_extra']:raise ValueError('FSDP template extra state is missing')
        extras[path]=_slice(value['module_extra'][logical],field,rank,world)
    return dict(template,module_extra=extras)


class ShardedDataTensorParallelGroup(DataTensorParallelGroup):
    """DP-reduce-scattered FSDP slices of actual TP layers; no second DP sum.

    Wrap units with fully_shard(model, mesh.data) AFTER TP partitioning and before
    construction. Unit all-gathers/reduce-scatters stay on the data axis; TP
    projection collectives stay on the tensor axis. Each DP replica executes the
    same collective-bearing FSDP microbatch shapes/order, with distinct data.
    """
    supports_gradient_overlap=False

    def __init__(self,model,mesh,*,data_root=0,tensor_root=0,broadcast_buffers=True):
        if mesh.shape[1]!=1:raise ValueError('use a pipeline stage mesh for FSDP/TP coordination')
        self.model,self.mesh=model,mesh
        self.data,self.tensor,self.transport=mesh.data,mesh.tensor,mesh.world
        self.rank,self.world_size=mesh.world.rank,mesh.world.world_size
        self.device_type,self.cuda_index,self.group=mesh.device_type,mesh.world.cuda_index,mesh.world.group
        units=[unit for unit in model.modules() if isinstance(unit,FullyShardedModule)]
        error=None
        if not units:error='fully_shard the TP model before constructing its coordinator'
        if any(unit.group.group is not mesh.data.group for unit in units):error='FSDP units must use the actual mesh data group'
        for unit in units:
            for path,layer in unit._template.named_modules():
                prefix=path+'.' if path else ''
                if not any(prefix+name in unit._template_layout for name in ('weight','lora_A','local.packed')):continue
                group=getattr(layer,'group',None)
                if group is not None and hasattr(group,'world_size') and group.group is not mesh.tensor.group:
                    error='template parallel layers must use the actual mesh tensor group'
        if type(broadcast_buffers) is not bool or type(data_root) is not int or not 0<=data_root<mesh.data.world_size:
            error='invalid data initialization policy'
        if type(tensor_root) is not int or not 0<=tensor_root<mesh.tensor.world_size:error='invalid tensor root'
        for failure in mesh.world.gather_metadata(error):
            if failure:raise ValueError(failure)
        mesh.world.validate_training_options((mesh.shape,data_root,tensor_root,broadcast_buffers))
        self.sharded_data=ShardedReplicaGroup(model,self.data)
        self.sharded_ids=self.sharded_data.sharded_ids
        self.tensor_coordinator=TensorParallelGroup(model,self.tensor)
        self._layout=sharded_mesh_layout(model,mesh)
        partitioned={name for name,spec in self._layout.items() if 'fsdp_shape' in spec}
        buffers=list(model.named_buffers()) if broadcast_buffers else []
        self.data.validate_training_options([(name,tuple(value.shape),str(value.dtype),name in partitioned)
                                            for name,value in buffers])
        error=None
        if any(value.device.type!=self.device_type or not value.is_contiguous()
               for name,value in list(model.named_parameters())+buffers):
            error='FSDP/TP storage must be contiguous on the mesh device'
        for failure in self.gather_metadata(error):
            if failure:raise ValueError(failure)
        with torch.no_grad():
            for name,value in list(model.named_parameters())+buffers:
                if name not in partitioned:self.data.broadcast_(value,data_root)
                spec=self._layout[name]
                if not spec.get('transform') and spec.get('replicated'):self.tensor.broadcast_(value,tensor_root)

    def validate_model(self,model):
        self.sharded_data.validate_model(model)
        self.tensor_coordinator.validate_model(model)

    def validate_microbatches(self,batches):
        super().validate_microbatches(batches)
        error=None
        try:self.sharded_data.validate_microbatches(batches)
        except ValueError as failure:error=str(failure)
        for failure in self.gather_metadata(error):
            if failure:raise ValueError(failure)

    def begin_gradient_overlap(self,**options):
        raise ValueError('FSDP backward already reduce-scatters its slices; replica bucket overlap is not this path')

    def synchronize_gradients(self,*,local_weight,normalized=False,missing='zero'):
        total=self.total_weight(local_weight)
        self.validate_training_options((normalized,missing))
        self.sharded_data.synchronize_gradients(local_weight=local_weight,normalized=normalized,missing=missing)
        self._complete_logical_gradients(list(self.model.parameters()))
        return total

    def replication_factors(self,parameters):
        data,_,tensor=self.mesh.shape
        return {id(p):(1 if id(p) in self.sharded_ids else data)*
                (1 if getattr(p,'_ruda_tp_sharded',False) else tensor) for p in parameters}

    def prepare_optimizer_step(self,optimizer,*,loss_scale=1.):
        from .sharded_training import Zero2Optimizer
        if isinstance(optimizer,Zero2Optimizer):raise ValueError('FSDP storage is already data-sharded; do not apply ZeRO-2 to its slices')
        parameters=[p for options in optimizer.param_groups for p in options['params']]
        names={id(p):name for name,p in self.model.named_parameters()}
        self.validate_training_options([names[id(p)] for p in parameters])
        if 'Muon' in type(optimizer).__name__ and not getattr(optimizer,'complete_mesh_matrices',False):
            raise ValueError('use MeshShardedMuon for complete FSDP/TP matrices')
        self._complete_logical_gradients(parameters)
        if not hasattr(optimizer,'last_step_skipped'):return
        bad=torch.zeros(1,device=parameters[0].device,dtype=torch.float32)
        for p in parameters:
            if p.grad is not None:bad.add_((~torch.isfinite(p.grad.float()/loss_scale)).any().float())
        self.transport.sum_(bad)
        if bad.item():
            for p in parameters:
                if p.grad is not None:p.grad.fill_(float('inf'))
        elif getattr(optimizer,'max_grad_norm',None) is not None and hasattr(optimizer,'_distributed_grad_norm'):
            optimizer._distributed_grad_norm=distributed_grad_norm(parameters,self.transport,loss_scale=loss_scale,
                replication_factors=self.replication_factors(parameters))

    def checkpoint_layout(self,layout):return sharded_mesh_layout(self.model,self.mesh)
