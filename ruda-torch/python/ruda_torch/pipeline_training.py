"""Explicit multi-process 1F1B pipeline execution on RUDA tensor allocations."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PipelineTensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32

    def validate(self):
        if not self.shape or any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError('pipeline activation shape must be explicit and nonempty')
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16,torch.int64,torch.int32,torch.bool):
            raise ValueError('unsupported pipeline tensor dtype')

    def allocate(self, device):
        self.validate()
        return torch.empty(self.shape, device=device, dtype=self.dtype)

    @property
    def has_floating(self):
        return self.dtype in (torch.float32,torch.float16,torch.bfloat16)


class PipelineStage:
    """One caller-constructed stage per rank, with a noninterleaved 1F1B schedule.

    Stage zero receives input tensors; the final stage computes the supplied
    loss_sum(output, target). Every rank receives matching microbatch counts and
    total effective weight. Input/output interfaces are explicit tensors, not a
    guessed transformer architecture. Optimizer steps occur after run() returns.
    """
    transport_protocol=1

    def __init__(self, module, group, *, input_spec, output_spec):
        self.module, self.group = module, group
        self.input_spec, self.output_spec = input_spec, output_spec
        self.rank, self.world_size = group.rank, group.world_size
        self.device = torch.device('ruda:0' if group.device_type == 'ruda' else 'cpu')
        group.validate_training_options(group.world_size)
        error=None
        try:
            for spec in (input_spec,output_spec):
                if not self._is_spec(spec):raise TypeError('unsupported pipeline boundary contract')
                spec.validate()
            self._validate_input_spec(input_spec)
        except (TypeError,ValueError) as failure:error=str(failure)
        specs = group.gather_metadata((input_spec, output_spec,self.transport_protocol,error))
        for _,_,protocol,failure in specs:
            if failure:raise ValueError(failure)
            if protocol!=self.transport_protocol:raise ValueError('pipeline stages must use the same tensor transfer protocol')
        for rank in range(group.world_size-1):
            if specs[rank][1] != specs[rank+1][0]:
                raise ValueError('adjacent pipeline activation interfaces differ')
            if not specs[rank][1].has_floating:
                raise ValueError('differentiable stage boundaries must carry floating activations')
        self._running = False

    def _is_spec(self,spec):return isinstance(spec,PipelineTensorSpec)

    def _validate_input_spec(self,spec):return None

    def _matches(self,value,spec):
        return isinstance(value,torch.Tensor) and tuple(value.shape)==spec.shape and value.dtype==spec.dtype

    def _is_cpu_payload(self,value):return isinstance(value,torch.Tensor) and value.device.type=='cpu'

    def _payload_metadata(self,value,shapes_only=False):return tuple(value.shape) if shapes_only else value.tolist()

    def _to_device(self,value):return value.to(self.device)

    def _received_input(self,value):return value.requires_grad_(True)

    def _invoke(self,value):return self.module(value)

    def _gradient_spec(self,spec):return spec

    def _backward_output(self,output,gradient):
        if output.requires_grad:output.backward(gradient)

    def _input_gradient(self,value):
        if self.rank>0 and value.grad is None:return torch.zeros_like(value)
        return value.grad

    @property
    def invocation_contract(self):return self.transport_protocol,'single'

    def validate_interfaces(self,count,microbatch_specs=None):
        """Collectively validate the actual per-microbatch boundary contracts.

        Each stage supplies (input_spec,output_spec) pairs for its own rank.
        Different microbatches may have different batch/sequence lengths. The
        default retains this stage's fixed interfaces; no shape is guessed from
        another rank's data or activation values.
        """
        interfaces,error=(),None
        try:
            if type(count) is not int or count<1:raise ValueError('pipeline requires actual microbatches')
            interfaces=tuple((self.input_spec,self.output_spec) for _ in range(count)) if microbatch_specs is None else tuple(microbatch_specs)
            if len(interfaces)!=count:raise ValueError('supply one boundary contract per actual microbatch')
            for pair in interfaces:
                if not isinstance(pair,(tuple,list)) or len(pair)!=2 or not all(self._is_spec(spec) for spec in pair):
                    raise TypeError('microbatch boundary contracts must be input/output PipelineTensorSpec pairs')
                for spec in pair:spec.validate()
                self._validate_input_spec(pair[0])
                if self.rank>0 and not pair[0].has_floating:
                    raise ValueError('received pipeline activations must be floating')
                if self.rank<self.world_size-1 and not pair[1].has_floating:
                    raise ValueError('sent pipeline activations must be floating')
            interfaces=tuple(tuple(pair) for pair in interfaces)
        except (TypeError,ValueError) as failure:error=str(failure)
        requests=self.group.gather_metadata((interfaces,error))
        for other,failure in requests:
            if failure:raise ValueError(failure)
            if len(other)!=count:raise ValueError('pipeline microbatch contract counts differ across stages')
        for index in range(count):
            for rank in range(self.world_size-1):
                if requests[rank][0][index][1]!=requests[rank+1][0][index][0]:
                    raise ValueError(f'microbatch {index}: adjacent pipeline interfaces differ')
        return interfaces

    def _peer(self, rank):
        return rank if self.group.group is None else dist.get_global_rank(self.group.group, rank)

    def _exchange(self, *, send=None, send_rank=None, receive_spec=None, receive_rank=None):
        output = None if receive_spec is None else receive_spec.allocate(self.device)
        operations, aliases = [], []
        if send is not None:
            send = send.detach().contiguous()
            alias = self.group._alias(send)
            aliases.append(alias)
            operations.append(dist.P2POp(dist.isend, alias, self._peer(send_rank), self.group.group))
        if output is not None:
            alias = self.group._alias(output)
            aliases.append(alias)
            operations.append(dist.P2POp(dist.irecv, alias, self._peer(receive_rank), self.group.group))
        if operations:
            for request in dist.batch_isend_irecv(operations):
                request.wait()
            self.group._complete([] if output is None else [output])
        return output

    def run(self, inputs=None, targets=None, *, loss_sum, global_weight, loss_scale=1.0,microbatch_specs=None,schedule='1f1b'):
        inputs, targets = list(inputs or []), list(targets or [])
        counts=self.group.gather_metadata((len(inputs),len(targets)))
        count=counts[0][0]
        if counts[-1][1]!=count:
            raise ValueError('first-stage input and last-stage target microbatch counts differ')
        self.group.validate_training_options((count, global_weight, loss_scale, schedule))
        if schedule not in ('1f1b', 'gpipe'):
            raise ValueError('pipeline schedule must be 1f1b or gpipe')
        if count < 1 or type(global_weight) is not int or global_weight <= 0 or not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError('pipeline needs microbatches and positive weighting/scaling')
        interfaces=self.validate_interfaces(count,microbatch_specs)
        if self._running:
            raise RuntimeError('a pipeline schedule is already active')
        self._running = True
        pending = deque()
        losses = []
        first, last = self.rank == 0, self.rank == self.world_size-1
        warmup = min(self.world_size-self.rank-1, count)

        def recv_input(index):
            input_spec=interfaces[index][0]
            if first:
                value = self._to_device(inputs[index])
                if not self._matches(value,input_spec):
                    raise ValueError('pipeline input differs from its explicit interface')
                return value
            return self._received_input(self._exchange(receive_spec=input_spec, receive_rank=self.rank-1))

        def forward(value, index):
            output = self._invoke(value)
            output_spec=interfaces[index][1]
            if not self._matches(output,output_spec):
                raise ValueError('pipeline output differs from its explicit interface')
            if last:
                loss = loss_sum(output,self._to_device(targets[index]))
                if loss.numel() != 1:
                    raise ValueError('pipeline loss_sum must return a scalar')
                losses.append(loss.detach())
                output = loss * (loss_scale/global_weight)
            pending.append((index, value, output))
            return output

        def backward(gradient):
            _, value, output = pending.popleft()
            self._backward_output(output,gradient)
            return self._input_gradient(value)

        try:
            if schedule == 'gpipe':
                # Fill/drain, preserving the existing FIFO gradient accumulation order.
                for index in range(count):
                    output = forward(recv_input(index), index)
                    if not last:
                        self._exchange(send=output, send_rank=self.rank+1)
                for _ in range(count):
                    backward_index = pending[0][0]
                    gradient = None if last else self._exchange(
                        receive_spec=self._gradient_spec(interfaces[backward_index][1]), receive_rank=self.rank+1)
                    input_gradient = backward(gradient)
                    if not first:
                        self._exchange(send=input_gradient, send_rank=self.rank-1)
                loss = torch.stack(losses).sum().float().reshape(1) if last else torch.zeros(1, dtype=torch.float32, device=self.device)
                self.group.broadcast_(loss, self.world_size-1)
                return {'loss_sum': float(loss.item()), 'loss': float(loss.item())/global_weight,
                        'microbatches': count, 'global_weight': global_weight}
            for index in range(warmup):
                output = forward(recv_input(index), index)
                if not last:
                    self._exchange(send=output, send_rank=self.rank+1)
            remaining = count-warmup
            value = recv_input(warmup) if remaining else None
            for step in range(remaining):
                index = warmup+step
                output = forward(value, index)
                backward_index=pending[0][0]
                gradient = None if last else self._exchange(send=output, send_rank=self.rank+1,
                    receive_spec=self._gradient_spec(interfaces[backward_index][1]), receive_rank=self.rank+1)
                input_gradient = backward(gradient)
                final = step == remaining-1
                if not first:
                    value = self._exchange(send=input_gradient, send_rank=self.rank-1,
                        receive_spec=None if final else interfaces[index+1][0],
                        receive_rank=None if final else self.rank-1)
                    if value is not None:
                        value=self._received_input(value)
                elif not final:
                    value = recv_input(index+1)
            for _ in range(warmup):
                backward_index=pending[0][0]
                gradient = None if last else self._exchange(receive_spec=self._gradient_spec(interfaces[backward_index][1]), receive_rank=self.rank+1)
                input_gradient = backward(gradient)
                if not first:
                    self._exchange(send=input_gradient, send_rank=self.rank-1)
            loss = torch.stack(losses).sum().float().reshape(1) if last else torch.zeros(1, dtype=torch.float32, device=self.device)
            self.group.broadcast_(loss, self.world_size-1)
            return {'loss_sum': float(loss.item()), 'loss': float(loss.item())/global_weight,
                    'microbatches': count, 'global_weight': global_weight}
        finally:
            self._running = False
