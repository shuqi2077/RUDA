"""Reusable causal-LM fine-tuning, independent of model family."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import time
import types
from datetime import datetime, timezone

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ._architecture_ops import positive_int, precision_context, work_dtype
from .finetuning import (NF4Linear, LoRALinear, inject_lora, load_nf4_safetensors,
                        finetune_state_dict, load_finetune_state_dict)


def chunked_lm_cross_entropy(hidden, head, labels, *, token_chunk_size=32,
                             ignore_index=-100, shift=True, reduction='mean',
                             recompute=True, label_smoothing=0.):
    """Compute loss without materializing [batch, sequence, vocabulary] logits.

    Each token chunk projects against the complete vocabulary: this is exact
    cross entropy, not sampled/truncated softmax. Backward recomputes each
    chunk's logits. Dense, NF4 and LoRA heads are supported, including head
    parameter gradients. Attention masks and supervision labels are separate.
    """
    positive_int(token_chunk_size, 'token_chunk_size')
    from .parallel_training import ColumnParallelLinear
    parallel=isinstance(head,ColumnParallelLinear)
    if not isinstance(head, (nn.Linear, NF4Linear, LoRALinear, ColumnParallelLinear)):
        raise TypeError('head must be Linear, NF4Linear, LoRALinear or ColumnParallelLinear')
    if parallel and head.gather_output:
        raise ValueError('a vocabulary-parallel loss requires gather_output=False')
    if not 0<=label_smoothing<=1:
        raise ValueError('label smoothing must be in [0,1]')
    if hidden.ndim != 3 or labels.shape != hidden.shape[:2] or labels.device != hidden.device:
        raise ValueError('hidden [B,T,D] and labels [B,T] must match on one device')
    if labels.dtype not in (torch.int32, torch.int64) or hidden.shape[-1] != head.in_features:
        raise ValueError('invalid labels dtype or head input width')
    if reduction not in ('mean', 'sum'):
        raise ValueError("reduction must be 'mean' or 'sum'")
    if shift:
        hidden, labels = hidden[:, :-1], labels[:, 1:]
    flat = hidden.reshape(-1, hidden.shape[-1])
    targets = labels.reshape(-1)
    dtype = work_dtype(hidden)
    def loss_chunk(x, target):
        logits = head(x)
        if parallel:
            from .parallel_loss import vocab_parallel_cross_entropy
            return vocab_parallel_cross_entropy(logits,target,head.group,ignore_index=ignore_index,
                                                 label_smoothing=label_smoothing,reduction='sum')
        with precision_context(logits):
            logp = F.log_softmax(logits.to(dtype), -1)
            valid = target != ignore_index
            safe = torch.where(valid, target, torch.zeros_like(target)).to(torch.int64)
            selected = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            loss=-selected
            if label_smoothing:
                loss=(1-label_smoothing)*loss-label_smoothing*logp.mean(-1)
            return torch.where(valid,loss,torch.zeros_like(loss)).sum()
    losses = []
    for start in range(0, flat.shape[0], token_chunk_size):
        args = flat[start:start + token_chunk_size], targets[start:start + token_chunk_size]
        if recompute and torch.is_grad_enabled():
            losses.append(checkpoint(loss_chunk, *args, use_reentrant=False, preserve_rng_state=True))
        else:
            losses.append(loss_chunk(*args))
    total = torch.stack(losses).sum() if losses else hidden.to(dtype).sum() * 0
    if reduction == 'sum':
        return total
    count = (targets != ignore_index).sum().to(dtype)
    return total / torch.where(count > 0, count, torch.ones_like(count))


class SFTCollator:
    """Right-pad explicit labels or tokenizer-declared assistant masks on CPU.

    Chat examples contain `messages`. Pretokenized examples contain
    `input_ids` and `labels`. No assistant spans, pad token, or truncation
    policy are guessed when the tokenizer does not supply them.
    """
    def __init__(self, tokenizer=None, *, max_length, pad_token_id=None,
                 train_on_prompt=False, truncate=False, template_kwargs=None):
        self.max_length = positive_int(max_length, 'max_length')
        self.tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', None) if pad_token_id is None else pad_token_id
        if type(self.pad_token_id) is not int or self.pad_token_id < 0:
            raise ValueError('supply an explicit nonnegative pad_token_id')
        self.train_on_prompt, self.truncate = bool(train_on_prompt), bool(truncate)
        self.template_kwargs = {} if template_kwargs is None else dict(template_kwargs)
        reserved = {'tokenize', 'return_dict', 'return_assistant_tokens_mask', 'add_generation_prompt'}
        if reserved.intersection(self.template_kwargs):
            raise ValueError('template kwargs cannot override supervision/encoding controls')

    def encode(self, sample):
        if 'messages' in sample:
            if self.tokenizer is None:
                raise ValueError('chat messages require a tokenizer')
            encoded = self.tokenizer.apply_chat_template(
                sample['messages'], tokenize=True, return_dict=True,
                return_assistant_tokens_mask=not self.train_on_prompt,
                add_generation_prompt=False, **self.template_kwargs)
            ids = list(encoded['input_ids'])
            if self.train_on_prompt:
                labels = ids.copy()
            else:
                mask = encoded.get('assistant_masks', encoded.get('assistant_tokens_mask'))
                if mask is None or len(mask) != len(ids) or not any(mask):
                    raise ValueError('chat template must declare assistant spans; alternatively supply explicit labels')
                labels = [token if active else -100 for token, active in zip(ids, mask)]
        else:
            ids, labels = list(sample['input_ids']), list(sample['labels'])
        if len(ids) != len(labels) or not ids:
            raise ValueError('input_ids and labels must be equally sized nonempty sequences')
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('input token IDs must be nonnegative integers')
        if any(type(token) is not int or token < 0 and token != -100 for token in labels):
            raise ValueError('labels must be token IDs or -100')
        if len(ids) > self.max_length:
            if not self.truncate:
                raise ValueError('sample exceeds max_length; explicit truncation or preprocessing required')
            ids, labels = ids[:self.max_length], labels[:self.max_length]
        return ids, labels

    def __call__(self, samples):
        encoded = [self.encode(sample) for sample in samples]
        if not encoded:
            raise ValueError('batch must not be empty')
        length = max(len(ids) for ids, _ in encoded)
        ids = torch.full((len(encoded), length), self.pad_token_id, dtype=torch.int64, device='cpu')
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, (tokens, targets) in enumerate(encoded):
            ids[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.int64, device='cpu')
            labels[row, :len(tokens)] = torch.tensor(targets, dtype=torch.int64, device='cpu')
            mask[row, :len(tokens)] = True
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def activation_checkpoint_modules(model, target_modules, *, preserve_rng_state=True):
    """Checkpoint explicitly named modules without changing state-dict paths.

    Non-reentrant recomputation supports frozen inputs and keyword arguments.
    RNG preservation is enabled by default; disabling it is an explicit caller
    decision suitable only for deterministic module forwards.
    """
    paths = list(target_modules)
    if not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError('supply nonempty exact checkpoint module paths')
    if len(set(paths)) != len(paths) or any(a != b and b.startswith(a + '.') for a in paths for b in paths):
        raise ValueError('checkpoint paths must be distinct and non-overlapping')
    modules = [model.get_submodule(path) for path in paths]
    if any(hasattr(module, '_ruda_checkpoint_forward') for module in modules):
        raise ValueError('module is already checkpointed')
    for module in dict.fromkeys(modules):
        original = module.forward
        module._ruda_checkpoint_forward = original
        def forward(self, *args, **kwargs):
            if not self.training or not torch.is_grad_enabled():
                return self._ruda_checkpoint_forward(*args, **kwargs)
            keys, count = tuple(kwargs), len(args)
            def invoke(*inputs):
                return self._ruda_checkpoint_forward(*inputs[:count],
                                                     **dict(zip(keys, inputs[count:])))
            return checkpoint(invoke, *args, *kwargs.values(), use_reentrant=False,
                              preserve_rng_state=preserve_rng_state)
        module.forward = types.MethodType(forward, module)
    return model


class CausalLMFinetuner(nn.Module):
    """An explicit backbone/head split; no model-specific module-path inference."""
    def __init__(self, backbone, head, *, token_chunk_size=32,
                 activation_checkpointing=True, checkpoint_modules=None,
                 preserve_rng_state=True):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.token_chunk_size = positive_int(token_chunk_size, 'token_chunk_size')
        self.activation_checkpointing = bool(activation_checkpointing)
        self.preserve_rng_state = bool(preserve_rng_state)
        self._checkpoint_backbone = False
        if checkpoint_modules is not None and not activation_checkpointing:
            raise ValueError('checkpoint_modules requires activation_checkpointing')
        if activation_checkpointing:
            enable = getattr(backbone, 'gradient_checkpointing_enable', None)
            if checkpoint_modules is not None:
                activation_checkpoint_modules(backbone, checkpoint_modules,
                                              preserve_rng_state=self.preserve_rng_state)
            elif enable is not None:
                enable(gradient_checkpointing_kwargs={'use_reentrant': False,
                                                      'preserve_rng_state': self.preserve_rng_state})
            else:
                self._checkpoint_backbone = True

    def hidden(self, input_ids, attention_mask):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if hasattr(self.backbone, 'config'):
            kwargs.update(use_cache=False, return_dict=True)
        if self._checkpoint_backbone and self.training and torch.is_grad_enabled():
            def invoke(ids, mask):
                return self.backbone(**dict(kwargs, input_ids=ids, attention_mask=mask))
            result = checkpoint(invoke, input_ids, attention_mask, use_reentrant=False,
                                preserve_rng_state=self.preserve_rng_state)
        else:
            result = self.backbone(**kwargs)
        if isinstance(result, torch.Tensor):
            return result
        if not hasattr(result, 'last_hidden_state'):
            raise TypeError('backbone must return hidden states, not vocabulary logits')
        return result.last_hidden_state

    def forward(self, input_ids, attention_mask, labels, *, reduction='mean'):
        hidden = self.hidden(input_ids, attention_mask)
        return chunked_lm_cross_entropy(hidden, self.head, labels,
                                       token_chunk_size=self.token_chunk_size, reduction=reduction)


def load_hf_nf4_model(directory, *, target_modules, device, dtype=torch.bfloat16,
                      auto_class=None, rank=16, alpha=16., block_size=64,
                      tile_rows=128, adapter_dtype=torch.float32, parameter_dtypes=None,
                      config_kwargs=None, model_kwargs=None):
    """Instantiate a local HF architecture on meta and stream its real weights.

    `auto_class` selects the caller's AutoModel class (causal LM by default).
    No model family, remote-code permission, checkpoint renaming, download,
    text/vision submodel extraction or LoRA target paths are inferred.
    accelerate's parameter-only meta context leaves nonpersistent buffers
    materialized, while checkpoint loading preserves their original dtypes.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    directory = Path(directory).resolve()
    ck = {} if config_kwargs is None else dict(config_kwargs)
    mk = {} if model_kwargs is None else dict(model_kwargs)
    if {'local_files_only', 'trust_remote_code'}.intersection(ck) or 'trust_remote_code' in mk:
        raise ValueError('loader uses local files and does not authorize remote code')
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False, **ck)
    factory = AutoModelForCausalLM if auto_class is None else auto_class
    with init_empty_weights(include_buffers=False):
        model = factory.from_config(config, trust_remote_code=False, **mk)
    # Parameter registration inside the meta context can break alias identity.
    # Reapply the architecture's declared ties before exact-name streaming.
    model.tie_weights()
    overrides = {}
    keep = getattr(model, '_keep_in_fp32_modules', None) or []
    if keep:
        pattern = re.compile(r'(^|\.)(?:' + '|'.join(keep) + r')(\.|$)')
        overrides = {name: torch.float32 for name, parameter in model.named_parameters()
                     if parameter.is_floating_point() and pattern.search(name)}
    if parameter_dtypes is not None:
        overrides.update(parameter_dtypes)
    load_nf4_safetensors(model, directory, target_modules=target_modules, device=device,
                        dtype=dtype, block_size=block_size, tile_rows=tile_rows,
                        parameter_dtypes=overrides)
    inject_lora(model, target_modules=target_modules, rank=rank, alpha=alpha,
                adapter_dtype=adapter_dtype)
    return model


class SFTTrainer:
    """Token-weighted accumulation with adapter-only step-boundary recovery.

    Batches are supplied on CPU; transfer is explicit. The caller identifies
    the exact frozen base and immutable input/config/source snapshot through
    `base_id` and `run_config`. A checkpoint never contains the frozen model.
    """
    def __init__(self, model, optimizer, *, base_id, run_config,
                 scaler=None, scheduler=None, replica_group=None, gradient_overlap=False,
                 bucket_bytes=25*1024*1024):
        from .compiler import CompiledModel
        from .sharded_training import FullyShardedModule
        original = model.original if isinstance(model, CompiledModel) else model
        template = original._template if isinstance(original,FullyShardedModule) else original
        if not isinstance(template, CausalLMFinetuner):
            raise TypeError('model must be a CausalLMFinetuner or its RUDA compiled wrapper')
        self.model, self.executable, self.optimizer = original, model, optimizer
        self.base_id, self.run_config = base_id, copy.deepcopy(run_config)
        self.scaler, self.scheduler = scaler, scheduler
        self.replica_group = replica_group
        self.gradient_overlap, self.bucket_bytes = gradient_overlap, bucket_bytes
        if type(gradient_overlap) is not bool or type(bucket_bytes) is not int or bucket_bytes <= 0:
            raise ValueError('gradient_overlap must be bool and bucket_bytes positive')
        if gradient_overlap and (replica_group is None or not hasattr(replica_group, 'begin_gradient_overlap')):
            raise ValueError('gradient overlap requires an initialized replicated group')
        if gradient_overlap and hasattr(optimizer, 'synchronize_gradients'):
            raise ValueError('ZeRO reduction is a separate path, not replica all-reduce overlap')
        if replica_group is not None:
            replica_group.validate_model(original)
        if scaler is not None and not hasattr(optimizer, 'last_step_skipped'):
            raise TypeError('scaled SFT requires an optimizer exposing last_step_skipped')
        self.step = self.cursor = self.tokens = 0
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._step_durations = []
        self.elapsed_before_resume = 0.
        self.last_checkpoint = None
        self.optimizer.zero_grad(set_to_none=True)

    def train_step(self, microbatches):
        microbatches = list(microbatches)
        counts = []
        error = None
        try:
            for batch in microbatches:
                if set(batch) != {'input_ids', 'attention_mask', 'labels'} or any(t.device.type != 'cpu' for t in batch.values()):
                    raise ValueError('supply collated CPU input_ids, attention_mask and labels')
                if batch['labels'].shape != batch['input_ids'].shape or batch['attention_mask'].shape != batch['labels'].shape:
                    raise ValueError('SFT batch shapes must match')
                if batch['attention_mask'].dtype != torch.bool or ((batch['labels'] != -100) & ~batch['attention_mask']).any():
                    raise ValueError('padding must not be supervised')
                counts.append(int((batch['labels'][:, 1:] != -100).sum()))
            if self.replica_group is not None:
                self.replica_group.validate_model(self.model)
        except (ValueError, TypeError, AttributeError) as failure:
            if self.replica_group is None:
                raise
            error = str(failure)
        if self.replica_group is not None:
            for error in self.replica_group.gather_metadata(error):
                if error:
                    raise ValueError(error)
            if hasattr(self.replica_group, 'validate_microbatches'):
                self.replica_group.validate_microbatches(microbatches)
            self.replica_group.validate_training_options((
                self.step, self.base_id,
                self.gradient_overlap, self.bucket_bytes,
                None if self.scaler is None else self.scaler.state_dict(),
                [(type(self.optimizer).__module__, type(self.optimizer).__qualname__),
                 [{k: v for k, v in group.items() if k != 'params'} for group in self.optimizer.param_groups]],
            ))
        local_count = sum(counts)
        count = local_count if self.replica_group is None else self.replica_group.total_weight(local_count)
        if self.replica_group is None and (not microbatches or count == 0):
            raise ValueError('optimizer step requires at least one supervised next-token target')
        device = next(self.model.parameters()).device
        started = time.monotonic()
        self.executable.train()
        self.optimizer.zero_grad(set_to_none=True)
        overlap = None
        loss_total = None
        for index, batch in enumerate(microbatches):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            loss = self.executable(**batch, reduction='sum')
            scaled = loss / count
            if self.scaler is not None:
                scaled = self.scaler.scale(scaled)
            if self.gradient_overlap and index == len(microbatches)-1:
                overlap = self.replica_group.begin_gradient_overlap(global_weight=count, normalized=True,
                                                                   bucket_bytes=self.bucket_bytes)
            scaled.backward()
            loss_total = loss.detach() if loss_total is None else loss_total + loss.detach()
        if self.replica_group is not None:
            if loss_total is None:
                loss_total = torch.zeros((), dtype=torch.float32, device=device)
                if self.scaler is not None:
                    self.scaler.scale(loss_total)
            if self.gradient_overlap:
                if overlap is None:
                    overlap = self.replica_group.begin_gradient_overlap(global_weight=count, normalized=True,
                                                                       bucket_bytes=self.bucket_bytes)
                overlap.finish(local_weight=local_count, missing='zero')
            elif hasattr(self.optimizer, 'synchronize_gradients'):
                self.optimizer.synchronize_gradients(local_weight=local_count, normalized=True, missing='zero')
            else:
                self.replica_group.synchronize_gradients(local_weight=local_count, normalized=True, missing='zero')
            self.replica_group.sum_(loss_total)
            if hasattr(self.replica_group, 'prepare_optimizer_step'):
                self.replica_group.prepare_optimizer_step(self.optimizer,
                    loss_scale=1. if self.scaler is None else self.scaler.get_scale())
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer); self.scaler.update()
        skipped = bool(getattr(self.optimizer, 'last_step_skipped', False))
        if self.scheduler is not None and not skipped:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1; self.cursor += len(microbatches); self.tokens += count
        loss_value = float(loss_total.cpu()) / count
        elapsed = time.monotonic() - started
        self._step_durations = (self._step_durations + [elapsed])[-20:]
        return {'step': self.step, 'microbatch_cursor': self.cursor,
                'supervised_tokens': count, 'total_supervised_tokens': self.tokens,
                'loss': loss_value, 'step_seconds': elapsed,
                'tokens_per_second': count / elapsed,
                'optimizer_update_skipped': skipped,
                'total_seconds': self.elapsed_before_resume + time.monotonic() - self.started,
                'updated_at': datetime.now(timezone.utc).isoformat()}

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        latest, previous, temporary = (directory / name for name in ('latest.pt', 'previous.pt', 'next.pt'))
        state = finetune_state_dict(self.model, self.optimizer, base_id=self.base_id,
                                   step=self.step, data_state={'cursor': self.cursor, 'tokens': self.tokens},
                                   scaler=self.scaler)
        state['run_config'] = self.run_config
        if self.replica_group is not None:
            state['replica'] = {'rank': self.replica_group.rank, 'world_size': self.replica_group.world_size}
        state['scheduler'] = None if self.scheduler is None else {
            'class': type(self.scheduler).__module__ + '.' + type(self.scheduler).__qualname__,
            'state': self.scheduler.state_dict()}
        state['saved_at'] = datetime.now(timezone.utc).isoformat()
        state['elapsed_seconds'] = self.elapsed_before_resume + time.monotonic() - self.started
        with temporary.open('wb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        checked = torch.load(temporary, map_location='cpu', weights_only=True)
        if checked['run_config'] != self.run_config or checked['step'] != self.step:
            raise RuntimeError('new checkpoint validation failed; previous version preserved')
        if latest.exists():
            os.replace(latest, previous)
        os.replace(temporary, latest)
        self.last_checkpoint = {'path': str(latest.resolve()), 'saved_at': state['saved_at']}
        return latest

    def resume(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        expected_replica = None if self.replica_group is None else {
            'rank': self.replica_group.rank, 'world_size': self.replica_group.world_size}
        if state.get('replica') != expected_replica:
            raise ValueError('checkpoint rank/world size differs; explicit repartitioning is required')
        if state.get('run_config') != self.run_config:
            raise ValueError('resume requires the exact input/source/config snapshot')
        saved = state.get('scheduler')
        name = None if self.scheduler is None else type(self.scheduler).__module__ + '.' + type(self.scheduler).__qualname__
        if (saved is None) != (name is None) or saved is not None and saved['class'] != name:
            raise ValueError('scheduler mismatch')
        self.step, data = load_finetune_state_dict(self.model, self.optimizer, state,
                                                  base_id=self.base_id, scaler=self.scaler)
        if saved is not None:
            self.scheduler.load_state_dict(saved['state'])
        self.cursor, self.tokens = data['cursor'], data['tokens']
        self.elapsed_before_resume = state.get('elapsed_seconds', 0.)
        self.started = time.monotonic()
        self.last_checkpoint = {'path': str(Path(path).resolve()), 'saved_at': state['saved_at']}
        return self.step

    def save_distributed(self, directory):
        """Collectively commit model/optimizer/data state for replica, TP or FSDP storage."""
        from .distributed_checkpoint import DistributedCheckpoint
        if self.replica_group is None:
            raise ValueError('save_distributed requires an explicit distributed coordinator')
        state = {'base_id': self.base_id, 'run_config': self.run_config, 'cursor': self.cursor,
                 'tokens': self.tokens, 'elapsed_seconds': self.elapsed_before_resume+time.monotonic()-self.started}
        result = DistributedCheckpoint(self.replica_group, directory).save(self.model, self.optimizer,
            step=self.step, application_state=state, scheduler=self.scheduler, scaler=self.scaler)
        self.last_checkpoint = result
        return result

    def resume_distributed(self, directory, *, generation=None):
        from .distributed_checkpoint import DistributedCheckpoint
        if self.replica_group is None:
            raise ValueError('resume_distributed requires an explicit distributed coordinator')
        def validate(state):
            if state['base_id'] != self.base_id or state['run_config'] != self.run_config:
                raise ValueError('distributed checkpoint base/source/config differs')
        step, state = DistributedCheckpoint(self.replica_group, directory).load(self.model, self.optimizer,
            generation=generation, scheduler=self.scheduler, scaler=self.scaler,validate_application=validate)
        self.step, self.cursor, self.tokens = step, state['cursor'], state['tokens']
        self.elapsed_before_resume = state['elapsed_seconds']
        self.started = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        return self.step

    def write_progress(self, directory, metrics, *, total_steps=None):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        status = dict(metrics, phase='training', total_steps=total_steps,
                      started_at=self.started_at,
                      checkpoint=self.last_checkpoint,
                      gpu_peak_bytes=(torch.cuda.max_memory_allocated() if next(self.model.parameters()).device.type == 'cuda' else None))
        if total_steps is not None:
            remaining = max(0, total_steps - self.step)
            status['estimated_remaining_seconds_range'] = (
                [remaining * min(self._step_durations), remaining * max(self._step_durations)]
                if len(self._step_durations) >= 2 else None)
            status['estimate_basis'] = 'range of recent measured steps; valid only for comparable batch shapes, not an extrapolation to larger inputs'
        temporary = directory / 'progress.next.json'
        temporary.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, directory / 'progress.json')
        return status
