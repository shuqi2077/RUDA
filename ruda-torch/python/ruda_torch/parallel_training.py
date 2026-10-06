"""Model-parallel derivatives, model conversion and ordered gradient buckets."""
from __future__ import annotations

from collections import OrderedDict
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
import torch.distributed as dist


class _Region(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group, kind, axis):
        ctx.group, ctx.kind, ctx.axis = group, kind, axis
        if kind == 'copy':
            return value.clone()
        if kind == 'reduce':
            return group.sum_(value.detach().clone())
        if kind == 'gather':
            return group.all_gather(value, axis=axis)
        if value.shape[axis] % group.world_size:
            raise ValueError('tensor-parallel dimension must divide the group size')
        return value.chunk(group.world_size, dim=axis)[group.rank].contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        if ctx.kind == 'copy':
            result = ctx.group.sum_(gradient.contiguous().clone())
        elif ctx.kind == 'reduce':
            result = gradient
        elif ctx.kind == 'gather':
            result = gradient.chunk(ctx.group.world_size, dim=ctx.axis)[ctx.group.rank].contiguous()
        else:
            result = ctx.group.all_gather(gradient, axis=ctx.axis)
        return result, None, None, None


def copy_to_tensor_parallel(value, group):
    return _Region.apply(value, group, 'copy', -1)


def reduce_from_tensor_parallel(value, group):
    return _Region.apply(value, group, 'reduce', -1)


def gather_from_tensor_parallel(value, group, *, axis=-1):
    return _Region.apply(value, group, 'gather', axis)


def scatter_to_tensor_parallel(value, group, *, axis=-1):
    return _Region.apply(value, group, 'scatter', axis)


class ColumnParallelLinear(nn.Module):
    """Output-feature shard; optionally return a full replicated output."""
    def __init__(self, linear, group, *, gather_output=True):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.out_features % group.world_size:
            raise ValueError('column parallelism requires divisible nn.Linear output features')
        self.group, self.gather_output = group, bool(gather_output)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.local_out_features = linear.out_features // group.world_size
        start = group.rank * self.local_out_features
        self.weight = nn.Parameter(linear.weight.detach().narrow(0, start, self.local_out_features).contiguous().clone(),
                                   requires_grad=linear.weight.requires_grad)
        self.bias = None if linear.bias is None else nn.Parameter(
            linear.bias.detach().narrow(0, start, self.local_out_features).contiguous().clone(),
            requires_grad=linear.bias.requires_grad)
        self.weight._ruda_tp_sharded=True
        if self.bias is not None:self.bias._ruda_tp_sharded=True

    def forward(self, value):
        value = copy_to_tensor_parallel(value, self.group)
        output = F.linear(value, self.weight, self.bias)
        return gather_from_tensor_parallel(output, self.group) if self.gather_output else output


class RowParallelLinear(nn.Module):
    """Input-feature shard; the replicated bias is added once, after the sum."""
    def __init__(self, linear, group, *, input_is_parallel=False):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.in_features % group.world_size:
            raise ValueError('row parallelism requires divisible nn.Linear input features')
        self.group, self.input_is_parallel = group, bool(input_is_parallel)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.local_in_features = linear.in_features // group.world_size
        start = group.rank * self.local_in_features
        self.weight = nn.Parameter(linear.weight.detach().narrow(1, start, self.local_in_features).contiguous().clone(),
                                   requires_grad=linear.weight.requires_grad)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(),
                                                                  requires_grad=linear.bias.requires_grad)
        self.weight._ruda_tp_sharded=True

    def forward(self, value):
        if not self.input_is_parallel:
            value = scatter_to_tensor_parallel(value, self.group)
        output = reduce_from_tensor_parallel(F.linear(value, self.weight, None), self.group)
        return output if self.bias is None else output + self.bias


class VocabParallelEmbedding(nn.Module):
    """Vocabulary shards, including padding semantics and replicated token inputs."""
    def __init__(self, embedding, group):
        super().__init__()
        if not isinstance(embedding, nn.Embedding) or embedding.num_embeddings % group.world_size:
            raise ValueError('vocabulary parallelism requires divisible nn.Embedding vocabulary')
        if embedding.max_norm is not None or embedding.scale_grad_by_freq or embedding.sparse:
            raise ValueError('max_norm, frequency-scaled and sparse embeddings require separate shard semantics')
        self.group = group
        self.num_embeddings, self.embedding_dim = embedding.num_embeddings, embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.local_vocab = embedding.num_embeddings // group.world_size
        self.start = self.local_vocab * group.rank
        self.weight = nn.Parameter(embedding.weight.detach().narrow(0, self.start, self.local_vocab).contiguous().clone(),
                                   requires_grad=embedding.weight.requires_grad)
        self.weight._ruda_tp_sharded=True

    def forward(self, tokens):
        outside = (tokens < self.start) | (tokens >= self.start + self.local_vocab)
        local = (tokens - self.start).masked_fill(outside, 0)
        local = local.masked_fill((tokens < 0) | (tokens >= self.num_embeddings), self.local_vocab)
        padding = None if self.padding_idx is None or not self.start <= self.padding_idx < self.start+self.local_vocab else self.padding_idx-self.start
        output = F.embedding(local, self.weight, padding_idx=padding)
        output = output.masked_fill(outside.unsqueeze(-1), 0)
        return reduce_from_tensor_parallel(output, self.group)


class TensorParallelMLP(nn.Module):
    """Gated MLP with rank-local intermediate activations."""
    def __init__(self, gate, up, down, group, *, activation=None):
        super().__init__()
        from .parallel_adapters import parallel_linear
        if gate.in_features!=up.in_features or gate.out_features!=up.out_features or down.in_features!=up.out_features:
            raise ValueError('gated MLP projection dimensions differ')
        self.gate_proj = parallel_linear(gate,group,axis=0,gather_output=False)
        self.up_proj = parallel_linear(up,group,axis=0,gather_output=False)
        self.down_proj = parallel_linear(down,group,axis=1,input_is_parallel=True)
        self.activation = nn.SiLU() if activation is None else activation

    def forward(self, value):
        return self.down_proj(self.activation(self.gate_proj(value))*self.up_proj(value))


class TensorParallelAttention(nn.Module):
    """MHA/GQA head shards with explicit projection, position and mask semantics.

    Positional transforms are supplied as position_transform(q,k). No model
    naming convention or positional formula is guessed.
    """
    def __init__(self, query, key, value, output, group, *, num_heads, num_kv_heads=None,
                 head_dim=None, scale=None, dropout_p=0., attention_impl=None):
        super().__init__()
        from .parallel_adapters import parallel_linear
        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if type(num_heads) is not int or type(kv_heads) is not int or num_heads<=0 or kv_heads<=0 or num_heads%kv_heads:
            raise ValueError('attention head counts must be positive and Q heads divisible by KV heads')
        head_dim = query.out_features//num_heads if head_dim is None else head_dim
        if type(head_dim) is not int or head_dim<=0 or query.out_features!=num_heads*head_dim:
            raise ValueError('query projection does not match head geometry')
        if num_heads%group.world_size or kv_heads%group.world_size:
            raise ValueError('query and KV heads must divide the TP group size')
        if key.out_features!=kv_heads*head_dim or value.out_features!=key.out_features or output.in_features!=query.out_features:
            raise ValueError('attention projection dimensions differ')
        if any(p.in_features!=query.in_features for p in (key,value)) or not 0<=dropout_p<=1:
            raise ValueError('attention input/dropout configuration differs')
        self.q_proj = parallel_linear(query,group,axis=0,gather_output=False)
        self.k_proj = parallel_linear(key,group,axis=0,gather_output=False)
        self.v_proj = parallel_linear(value,group,axis=0,gather_output=False)
        self.o_proj = parallel_linear(output,group,axis=1,input_is_parallel=True)
        self.heads,self.kv_heads,self.head_dim = num_heads//group.world_size,kv_heads//group.world_size,head_dim
        self.scale,self.dropout_p = scale,float(dropout_p)
        if attention_impl is not None and not callable(attention_impl):
            raise TypeError('attention_impl must be a callable attention implementation')
        self.attention_impl=attention_impl

    def forward(self, hidden, *, attention_mask=None, is_causal=False, position_transform=None):
        if hidden.ndim!=3:
            raise ValueError('attention inputs must have [batch,sequence,hidden] shape')
        batch,length,_=hidden.shape
        query=self.q_proj(hidden).view(batch,length,self.heads,self.head_dim).transpose(1,2)
        key=self.k_proj(hidden).view(batch,length,self.kv_heads,self.head_dim).transpose(1,2)
        value=self.v_proj(hidden).view(batch,length,self.kv_heads,self.head_dim).transpose(1,2)
        if position_transform is not None:
            query,key=position_transform(query,key)
        if self.heads!=self.kv_heads:
            factor=self.heads//self.kv_heads
            key=key.repeat_interleave(factor,dim=1)
            value=value.repeat_interleave(factor,dim=1)
        if self.attention_impl is not None:
            attention=self.attention_impl
        elif hidden.device.type=='ruda':
            from .attention import scaled_dot_product_attention
            attention=scaled_dot_product_attention
        else:
            attention=F.scaled_dot_product_attention
        output=attention(query,key,value,attn_mask=attention_mask,is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.,scale=self.scale)
        output=output.transpose(1,2).contiguous().view(batch,length,self.heads*self.head_dim)
        return self.o_proj(output)


def tensor_parallelize(model, group, plan):
    """Replace exact module paths using an explicit, architecture-independent plan.

    Values are 'column', 'row', 'embedding', or dictionaries with 'kind' and
    gather_output/input_is_parallel. Default linear replacements preserve global
    input/output shapes; sharded attention/MLP wiring opts into local outputs.
    Shared module references and compatible tied weight shards stay shared.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    from .parallel_adapters import parallel_linear,parameter_shards
    replacements, weight_shards, signatures = {}, {}, {}
    error = None
    try:
        if not isinstance(plan, dict) or not plan or any(not name or name not in modules for name in plan):
            raise ValueError('supply exact, nonempty module paths')
        names = list(plan)
        if any(a != b and b.startswith(a+'.') for a in names for b in names):
            raise ValueError('tensor-parallel paths must not overlap')
        for name, options in plan.items():
            options = {'kind': options} if isinstance(options, str) else dict(options)
            kind = options.pop('kind')
            source = modules[name]
            if kind not in ('column','row','embedding'):raise ValueError('unsupported parallel plan kind')
            signature = (kind, tuple(sorted(options.items())))
            if id(source) in signatures and signatures[id(source)] != signature:
                raise ValueError('shared modules require identical parallel plans')
            signatures[id(source)] = signature
            if id(source) not in replacements:
                shard_axis = 1 if kind == 'row' else 0
                replacement = VocabParallelEmbedding(source,group,**options) if kind=='embedding' else parallel_linear(source,group,axis=shard_axis,**options)
                entries=[(source.weight,replacement,'weight',0)] if kind=='embedding' else parameter_shards(source,replacement,shard_axis)
                for original,owner,leaf,axis in entries:
                    key=id(original)
                    parameter=getattr(owner,leaf)
                    weight_signature=(axis,tuple(parameter.shape),parameter.dtype,parameter.requires_grad)
                    if key in weight_shards:
                        previous_signature,shared=weight_shards[key]
                        if previous_signature!=weight_signature:
                            raise ValueError('tied parameters require compatible shard axes, shapes and flags')
                        setattr(owner,leaf,shared)
                    else:weight_shards[key]=(weight_signature,parameter)
                replacement.train(source.training)
                replacements[id(source)] = replacement
        selected=[name for name,module in modules.items() if id(module) in replacements]
        selected_weights = set(weight_shards)
        for name, module in modules.items():
            for parameter in module.parameters(recurse=False):
                if id(parameter) in selected_weights and not any(name==path or name.startswith(path+'.') for path in selected):
                    raise ValueError('include every consumer of a tied weight in the parallel plan')
    except (KeyError, TypeError, ValueError) as failure:
        error = str(failure)
    schema=[(name,type(module).__name__,[(key,tuple(p.shape),str(p.dtype),p.requires_grad) for key,p in module.named_parameters()],
             [(key,tuple(b.shape),str(b.dtype)) for key,b in module.named_buffers()],
             [(key,child.get_extra_state()) for key,child in module.named_modules() if type(child).get_extra_state is not nn.Module.get_extra_state])
            for name,module in modules.items() if isinstance(plan,dict) and name in plan]
    contracts = group.gather_metadata((plan,schema,error))
    for other_plan,other_schema,failure in contracts:
        if failure:
            raise ValueError(failure)
        if other_plan != plan or other_schema!=schema:
            raise ValueError('tensor-parallel plans differ across ranks')
    for name, source in modules.items():
        if name and id(source) in replacements:
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent), leaf, replacements[id(source)])
    return model


def tensor_parallel_state_dict(model, group):
    """Collectively reconstruct an ordinary full state dictionary on every rank."""
    result = OrderedDict((name,value.detach().cpu() if isinstance(value,torch.Tensor) else copy.deepcopy(value))
                         for name,value in model.state_dict().items())
    from .parallel_adapters import _ParallelLoRA,_ParallelNF4,full_adapter_matrix,full_nf4_state
    with torch.no_grad():
        for path, module in model.named_modules(remove_duplicate=False):
            prefix = path+'.' if path else ''
            if isinstance(module,_ParallelLoRA):
                for name in ('lora_A','lora_B'):result[prefix+name]=full_adapter_matrix(module,name)
            elif isinstance(module,_ParallelNF4):
                for key in tuple(result):
                    if key.startswith(prefix+'local.'):del result[key]
                result.update((prefix+key,value) for key,value in full_nf4_state(module).items())
            elif isinstance(module, (ColumnParallelLinear, VocabParallelEmbedding)):
                result[prefix+'weight'] = group.all_gather(module.weight, axis=0).cpu()
                if isinstance(module, ColumnParallelLinear) and module.bias is not None:
                    result[prefix+'bias'] = group.all_gather(module.bias, axis=0).cpu()
            elif isinstance(module, RowParallelLinear):
                result[prefix+'weight'] = group.all_gather(module.weight, axis=1).cpu()
    return result


def load_tensor_parallel_state_dict(model, state, group, *, strict=True):
    """Partition full pretrained/checkpoint tensors for the current TP topology."""
    local = dict(state)
    from .parallel_adapters import _ParallelLoRA,_ParallelNF4,local_adapter_matrix,local_nf4_state
    for path, module in model.named_modules(remove_duplicate=False):
        prefix = path+'.' if path else ''
        if isinstance(module,_ParallelLoRA):
            for name in ('lora_A','lora_B'):
                key=prefix+name
                if key in local:local[key]=local_adapter_matrix(module,name,local[key])
        elif isinstance(module,_ParallelNF4):
            replacement=local_nf4_state(module,state,prefix)
            for key in ('packed','scales','codebook','bias','_extra_state'):
                local.pop(prefix+key,None)
            local.update(replacement)
        elif isinstance(module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)):
            axis = 1 if isinstance(module, RowParallelLinear) else 0
            key = prefix+'weight'
            if key in local:
                value = local[key]
                if value.shape[axis] != module.weight.shape[axis]*group.world_size:
                    raise ValueError(f'global checkpoint shape mismatch: {key}')
                local[key] = value.chunk(group.world_size, dim=axis)[group.rank].contiguous()
            if isinstance(module, ColumnParallelLinear) and module.bias is not None and prefix+'bias' in local:
                local[prefix+'bias'] = local[prefix+'bias'].chunk(group.world_size)[group.rank].contiguous()
    return model.load_state_dict(local, strict=strict)


class GradientOverlap:
    """Ordered asynchronous NCCL/Gloo buckets with post-accumulation hooks."""
    def __init__(self, group, *, global_weight, normalized, bucket_bytes):
        if group._schema is None:
            raise RuntimeError('initialize replicas before starting gradient overlap')
        if type(global_weight) is not int or global_weight <= 0 or type(normalized) is not bool:
            raise ValueError('supply the positive global effective weight and normalization policy')
        if type(bucket_bytes) is not int or bucket_bytes <= 0:
            raise ValueError('bucket_bytes must be positive')
        group.validate_training_options((global_weight, normalized, bucket_bytes))
        if getattr(group, '_overlap_active', False):
            raise RuntimeError('finish the preceding gradient overlap before starting another')
        group._overlap_active = True
        self.group, self.weight, self.normalized = group, global_weight, normalized
        self.parameters = [p for _, p in group._parameters if p.requires_grad]
        self.buckets, current, size = [], [], 0
        for parameter in reversed(self.parameters):
            count = parameter.numel()*4
            if current and size+count > bucket_bytes:
                self.buckets.append(current)
                current, size = [], 0
            current.append(parameter)
            size += count
        if current:
            self.buckets.append(current)
        self.ready, self.works, self.next_bucket = set(), [], 0
        self.handles = [p.register_post_accumulate_grad_hook(self._ready) for p in self.parameters]
        self.closed = False

    def _ready(self, parameter):
        self.ready.add(id(parameter))
        self._drain()

    def _drain(self):
        while self.next_bucket < len(self.buckets):
            bucket = self.buckets[self.next_bucket]
            if any(id(p) not in self.ready for p in bucket):
                break
            pieces = [torch.zeros(p.numel(), device=p.device, dtype=torch.float32)
                      if p.grad is None else p.grad.detach().float().reshape(-1) for p in bucket]
            work = torch.cat(pieces)
            alias = self.group._alias(work)
            request = dist.all_reduce(alias, group=self.group.group, async_op=True)
            self.works.append((request, work, alias, bucket))
            self.next_bucket += 1

    def finish(self, *, local_weight, missing='zero'):
        if self.closed:
            raise RuntimeError('gradient overlap is already finished')
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        flags = [p.grad is not None for p in self.parameters]
        requests = self.group.gather_metadata((local_weight, flags, missing))
        failure = None
        active = [False]*len(flags)
        total = 0
        for weight, other_flags, policy in requests:
            if type(weight) is not int or weight < 0 or len(other_flags) != len(flags) or policy != missing or policy not in ('zero', 'error'):
                failure = 'invalid rank gradient policy/weight'
                continue
            total += weight
            if weight:
                active = [a or b for a, b in zip(active, other_flags)]
                if missing == 'error' and not all(other_flags):
                    failure = 'positive-weight rank is missing a gradient'
        if total != self.weight:
            failure = 'global effective weight differs from overlap normalization'
        if not local_weight and any(flags):
            failure = 'a zero-weight rank must not contribute gradients to overlap'
        self.ready.update(id(p) for p in self.parameters)
        self._drain()
        with torch.no_grad():
            for request, work, alias, bucket in self.works:
                request.wait()
                self.group._complete([work])
                if not self.normalized:
                    work.div_(self.weight)
                start = 0
                for parameter in bucket:
                    end = start+parameter.numel()
                    parameter.grad = work[start:end].view_as(parameter).to(parameter.dtype).clone()
                    start = end
            for parameter, present in zip(self.parameters, active):
                if not present:
                    parameter.grad = None
        self.works.clear()
        self.closed = True
        self.group._overlap_active = False
        if failure:
            raise ValueError(failure)
        return total
