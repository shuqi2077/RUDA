#!/usr/bin/env python3
"""Fine-tune a local HF checkpoint using explicit, model-independent paths.

JSONL rows contain input_ids/labels or messages. The frozen checkpoint is
identified by --base-id; --backbone, --head and --targets are exact module
paths supplied by the caller. No pretrained weights are downloaded.
Default execution is RUDA; --cpu-reference is an explicit reference run.
Start with a bounded short --steps trial on the intended input shapes before
any long run. Checkpoints/progress live only in the caller's output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.metadata import version
import itertools
import json
import math
import os
from pathlib import Path
import sys
import types

import torch


def components(cpu_reference):
    if not cpu_reference:
        import ruda_torch
        return ruda_torch.causal_finetuning, ruda_torch.AdamW
    name = 'ruda_sft_cpu_reference'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'ruda_torch')]
        package._graph_available = False
        sys.modules[name] = package
    return importlib.import_module(name + '.causal_finetuning'), torch.optim.AdamW


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'blank JSONL row at line {line_number}')
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'JSONL row {line_number} must be an object')
            yield row


def run(args):
    from transformers import AutoTokenizer
    import transformers
    cft, optimizer_class = components(args.cpu_reference)
    for name in ('steps', 'batch_size', 'accumulation', 'checkpoint_every'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('lr must be finite and positive')
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[3]
    if output == repo or output.is_relative_to(repo):
        raise ValueError('checkpoint/progress output must be outside the repository')
    output.mkdir(parents=True, exist_ok=True)
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)
    device = 'cpu' if args.cpu_reference else 'ruda:0'
    factory = getattr(transformers, args.auto_class)
    config = vars(args).copy()
    for name in ('resume', 'output', 'steps'):
        config.pop(name)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config['data_sha256'] = sha256(args.data)
    config['source_sha256'] = {str(path.relative_to(repo)): sha256(path) for path in
                               [Path(__file__).resolve(), *sorted((repo / 'ruda-torch/python/ruda_torch').glob('*.py'))]}
    config['versions'] = {name: version(name) for name in ('torch', 'transformers', 'accelerate', 'safetensors')}
    config['versions']['python'] = sys.version
    if not args.cpu_reference:
        runtime = sys.modules['ruda_torch']
        config['native_artifacts_sha256'] = {str(path.resolve()): sha256(path)
                                               for path in (Path(runtime._library), Path(runtime._C.__file__))}
        config['ruda_environment'] = {name: os.environ.get(name) for name in
                                       ('RUDA_TORCH_ASYNC', 'RUDA_TORCH_DLL_DIR', 'RUDA_TORCH_INDEX_CHECK',
                                        'RUDA_TORCH_LIBRARY', 'RUDA_TORCH_MATMUL', 'RUDA_TORCH_SOFTMAX',
                                        'RUDA_CUDA_COMPILER', 'CUDA_VISIBLE_DEVICES')}
    # Record exact local checkpoint file metadata alongside the caller's content
    # identity; metadata is not claimed to be a content hash of frozen weights.
    config['base_files'] = {str(path.relative_to(args.model)): [path.stat().st_size, path.stat().st_mtime_ns]
                             for path in sorted(args.model.rglob('*')) if path.is_file()}
    pretrained = cft.load_hf_nf4_model(args.model, target_modules=args.targets, device=device,
                                       dtype=dtype, auto_class=factory, rank=args.rank, alpha=args.alpha,
                                       model_kwargs={'attn_implementation': args.attention})
    model = cft.CausalLMFinetuner(pretrained.get_submodule(args.backbone), pretrained.get_submodule(args.head),
                                token_chunk_size=args.token_chunk_size,
                                checkpoint_modules=args.checkpoint_modules,
                                preserve_rng_state=not args.no_preserve_rng_state)
    optimizer = optimizer_class([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    compiler = importlib.import_module(cft.__package__ + '.compiler')
    executable = (compiler.compile(model, device_type='cpu' if args.cpu_reference else 'ruda',
                                   native='off' if args.cpu_reference else 'auto', fullgraph=True)
                  if args.compile else model)
    try:
        trainer = cft.SFTTrainer(executable, optimizer, base_id=args.base_id, run_config=config)
        if args.resume:
            trainer.resume(args.resume)
        config_path = output / 'run-config.json'
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('output directory belongs to a different input/source/config snapshot')
        config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        tokenizer = (AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
                     if args.chat else None)
        collate = cft.SFTCollator(tokenizer, max_length=args.max_length, pad_token_id=args.pad_token_id,
                                 train_on_prompt=args.train_on_prompt, truncate=args.truncate)
        source = iter(records(args.data))
        # This example traverses the file once, without implicit reshuffling/repeat.
        for _ in itertools.islice(source, trainer.cursor * args.batch_size):
            pass
        latest = None
        while trainer.step < args.steps:
            batches = []
            for _ in range(args.accumulation):
                rows = list(itertools.islice(source, args.batch_size))
                if len(rows) != args.batch_size:
                    raise ValueError('dataset exhausted: prepare enough full microbatches for requested steps; no implicit repeats')
                batches.append(collate(rows))
            latest = trainer.train_step(batches)
            if trainer.step % args.checkpoint_every == 0 or trainer.step == args.steps:
                trainer.save(output / 'checkpoints')
            trainer.write_progress(output, latest, total_steps=args.steps)
            print(json.dumps(latest), flush=True)
        return latest
    finally:
        if args.compile:
            executable.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('base-id', 'backbone', 'head'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--checkpoint-modules', nargs='+')
    parser.add_argument('--auto-class', default='AutoModelForCausalLM')
    parser.add_argument('--attention', default='eager')
    parser.add_argument('--cpu-reference', action='store_true')
    parser.add_argument('--compile', action='store_true', help='reuse RUDA AOT forward/backward capture')
    parser.add_argument('--no-preserve-rng-state', action='store_true',
                        help='explicitly permit checkpoint recomputation without RNG restoration; deterministic forwards only')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--train-on-prompt', action='store_true')
    parser.add_argument('--truncate', action='store_true')
    parser.add_argument('--pad-token-id', type=int)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=1)
    parser.add_argument('--token-chunk-size', type=int, default=32)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=16.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=51)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), default='bf16')
    parser.add_argument('--resume', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
