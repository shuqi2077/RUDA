# Model-independent LoRA and NF4 fine-tuning

[Documentation](README.md) · [Training](training.md) · [中文](../zh/finetuning.md)

## Choose the integration layer

| Model interface | Entry point |
| --- | --- |
| Rust `ruda-model` model | `ruda_nn::LoRALinearConfig`, with the existing tensor backend and optimizer |
| Ordinary PyTorch model | `ruda_torch.inject_lora`, optionally preceded by `quantize_nf4` |
| Local Hugging Face safetensors checkpoint | `load_hf_nf4_model` or the architecture-independent `load_nf4_safetensors` |
| Causal language-model training | `CausalLMFinetuner`, `SFTCollator`, `SFTTrainer` |

These APIs do not select a model family, infer adapter targets, download weights or install missing device operators. The model's actual forward and backward operations must be supported by its execution backend. Native PyTorch uses `ruda:0`, not `torch.cuda` storage. Prepare the matching [native components](../../ruda-torch/README.md#build-and-install), including the NF4 extensions when using packed weights.

For safetensors streaming, install `safetensors`. The Hugging Face helper and command-line example also require `transformers` and `accelerate`. The selected architecture must exist in the installed Transformers version without remote code. FP16 is the low-precision option for T4; do not select BF16 on hardware without the required support.

## Select projections explicitly

Inspect `model.named_modules()` before modifying the model. `target_modules` is either the string `'all-linear'` or a nonempty sequence of exact qualified module names, not suffix matching or a regular expression. A root `Linear` must first be placed in a container. Selecting one alias of a shared module also replaces its other aliases.

For dense LoRA, call `inject_lora(model, target_modules=targets, rank=16, alpha=16., adapter_dtype=torch.float32)`. It modifies and returns the model, freezes all original parameters, clears their gradients and leaves only adapter parameters trainable. Construct the optimizer **after** injection and pass only parameters whose `requires_grad` is true. Repeated injection is rejected.

Each `LoRALinear` computes `base(x) + (alpha / rank) * B(A(x))`. The base is `torch.nn.Linear` or `NF4Linear`; A has shape `[rank, in_features]` and B `[out_features, rank]`. B starts at zero. The Python adapter has no dropout parameter. `rank` is a positive integer, `alpha` is finite and positive, and `adapter_dtype` is FP32, FP16 or BF16. The update is converted to the base output dtype before addition.

### Rust LoRA

`LoRALinearConfig::new(rank, alpha).with_dropout(p).init(base)` consumes an existing Rust `Linear<B>` and freezes its base without changing base parameter IDs. A maps input width to rank, B maps rank to output width and starts at zero. Both use the base weight dtype. `forward` preserves supported leading dimensions.

Rust requires `rank > 0`, finite `alpha`, and `0 <= dropout < 1`; invalid configuration asserts. Unlike the Python API, Rust does not require positive alpha. `merge(self)` consumes the adapter and produces a frozen dense projection without adapter dropout. It is not an optimizer-state conversion or a checkpoint for resuming LoRA training. See [Rust implementation](../../ruda-nn/src/modules/lora.rs).

## Pack a frozen NF4 base

For a model already loaded on CPU, call `quantize_nf4(model, target_modules=targets, block_size=64, tile_rows=128)` **before** injecting LoRA and before moving the model to RUDA. Selected weights must be contiguous CPU FP32/FP16/BF16 matrices. Keep embedding/output ties outside the quantization targets: shared module aliases are preserved, but a weight tied to another module is rejected. Conversion commits one layer at a time; failure leaves completed layers converted.

`NF4Linear` keeps packed weight codes, FP32 block scales and an FP32 codebook, without a retained dense weight shadow. It supports first-order input gradients, not gradients for the frozen packed base. Moving or changing the module's floating dtype keeps scales and the codebook in FP32.

The RUDA NF4 format is row-major, two codes per byte, first value in the high nibble. Each flat block has one FP32 absolute maximum. Partial final blocks and odd element counts are supported. This is not a bitsandbytes or PEFT checkpoint format, and NF4 is not interchangeable with AWQ INT4 or learned fake quantization.

On RUDA, FP16/BF16 uses fused tile dequantization and GEMM when NF4 matmul API 1 is present. FP32, or absence of that optional matmul capability, uses bounded tiled decoding; decoding still requires NF4 API 1. A failed fused call is not retried through the tiled path. `tile_rows` controls the decoded output-row tile on that path, not the training batch size. CPU/CUDA execution is explicitly selected same-device reference execution, not a recovery path from a failed RUDA call.

## Stream a local checkpoint

`load_nf4_safetensors(model, directory, target_modules=targets, device='ruda:0', dtype=torch.float16)` expects an already constructed model with **all parameters on meta**. It reads `model.safetensors`, or `model.safetensors.index.json` and its listed shards, one tensor at a time. Selected dense weights are quantized on CPU and uploaded packed; remaining tensors are loaded separately. Tensor names and shapes must match the architecture exactly. There is no key-renaming rule or tokenizer handling.

- `parameter_dtypes` and `buffer_dtypes` are dictionaries of exact floating tensor names to FP32/FP16/BF16. Conflicting overrides on tied tensors are rejected.
- A dtype-preserved weight cannot also be an NF4 target. Non-target floating parameters use `dtype` unless overridden; buffers preserve their original dtype unless overridden.
- Nonpersistent buffers omitted from the checkpoint must already be materialized by the constructor, not left on meta.
- Shards must be existing files inside the checkpoint directory. Missing tensors, mismatched shapes/kinds, invalid ties or unsupported target types raise errors.
- Loading is not transactional: after a failure, retry with a freshly constructed meta model.

`load_hf_nf4_model(directory, target_modules=targets, device='ruda:0', dtype=torch.float16, rank=16, alpha=16.)` creates a local HF architecture on meta, reapplies its declared weight ties, streams the base and injects adapters. `auto_class` defaults to `AutoModelForCausalLM`; callers can supply another compatible AutoModel class. `config_kwargs` and `model_kwargs` are passed to the relevant constructors; local-file and remote-code controls cannot be overridden. `_keep_in_fp32_modules` declarations are respected, and `parameter_dtypes` can supply explicit overrides. The helper does not choose the text backbone, vocabulary head or multimodal submodel for the caller.

Both loaders default to `dtype=torch.bfloat16`; explicitly pass FP16 for T4. Their packing defaults are `block_size=64, tile_rows=128`. The HF helper additionally defaults to `rank=16, alpha=16., adapter_dtype=torch.float32`; `parameter_dtypes`, `config_kwargs` and `model_kwargs` default to None. The lower-level loader also accepts `buffer_dtypes=None`.

## Supervision, loss and accumulation

`SFTCollator(tokenizer=None, max_length=..., pad_token_id=..., train_on_prompt=False, truncate=False)` right-pads batches on CPU and returns exactly `input_ids`, `attention_mask` and `labels`, all shaped `[B,T]`. IDs/labels are int64; the mask is Bool. Padding labels are `-100`.

Supply one of these input forms:

- Pretokenized records with nonempty, equally sized `input_ids` and `labels`. Labels are vocabulary IDs or `-100`. IDs are nonnegative integers. Do not pre-shift labels.
- Chat records with `messages` and a tokenizer. By default, its chat template must declare assistant spans through `assistant_masks` or `assistant_tokens_mask`. Missing/empty spans are errors. `train_on_prompt=True` explicitly supervises all template tokens instead.

`pad_token_id` must be supplied or declared by the tokenizer; it is not guessed. Samples exceeding `max_length` fail unless `truncate=True`, which explicitly keeps the prefix. `template_kwargs` cannot override encoding/supervision controls. The attention mask and label mask have separate purposes.

`CausalLMFinetuner(backbone, head, token_chunk_size=32, activation_checkpointing=True, checkpoint_modules=None, preserve_rng_state=True)` requires an explicit backbone/head split. The backbone accepts `input_ids` and `attention_mask` and returns `[B,T,D]` hidden states directly or through `last_hidden_state`; vocabulary logits are not a backbone result. A backbone with `config` is called with `use_cache=False, return_dict=True`. The head is a dense, NF4 or LoRA linear layer with input width D.

`chunked_lm_cross_entropy(hidden, head, labels, token_chunk_size=32, ignore_index=-100, shift=True, reduction='mean', recompute=True)` projects each token chunk against the **complete vocabulary**. It never constructs the full `[B,T,V]` logits tensor. `shift=True` pairs hidden positions `[:-1]` with labels `[1:]`. `mean` divides by nonignored target count; `sum` returns their loss sum. No valid targets yields a differentiable zero. Labels must be int32/int64 on the hidden-state device; nonignored IDs must be valid vocabulary indices. Backward recomputes chunk logits when requested, retaining head and hidden-state gradients.

Activation checkpointing uses non-reentrant recomputation. Explicit `checkpoint_modules` are nonempty, distinct, non-overlapping paths relative to the backbone. Otherwise the wrapper calls its checkpointing interface, or checkpoints the whole backbone. `activation_checkpoint_modules(model, paths, preserve_rng_state=True)` is also available directly and preserves state-dict paths. Disable RNG preservation only deliberately for deterministic forwards; it is not equivalent to saving RNG for a later process restart.

| Method | Arguments and return |
| --- | --- |
| `SFTCollator.encode(sample)` | One chat/pretokenized record → two Python lists `(input_ids, labels)`, before padding. |
| `SFTCollator(samples)` | Nonempty batch → the three CPU tensors described above. `template_kwargs=None` is the default. |
| `CausalLMFinetuner.hidden(input_ids, attention_mask)` | Backbone inputs → `[B,T,D]` hidden states without the vocabulary projection. |
| `CausalLMFinetuner.forward(input_ids, attention_mask, labels, reduction='mean')` | Scalar shifted full-vocabulary loss; reduction is mean or sum. |
| `activation_checkpoint_modules(model, target_modules, preserve_rng_state=True)` | Modifies and returns the model; already checkpointed modules and overlapping paths are errors. |

`SFTTrainer(model, optimizer, base_id=..., run_config=..., scaler=None, scheduler=None, replica_group=None)` accepts that wrapper or its RUDA compiled wrapper. `train_step(microbatches)` requires collated CPU batches and at least one supervised shifted target across the window. It transfers each batch to the model device, backpropagates each loss sum divided by the **total token count across the window**, then performs one optimizer step and clears gradients. It does not average unequal microbatch means. An optional scaler requires an optimizer exposing `last_step_skipped`; the scheduler advances only when the update is not skipped.

With an explicitly initialized [ReplicaGroup](distributed-training.md#python-replicated-training-with-nccl), normalization/loss reporting use global tokens and accumulated gradients synchronize before the update. A local empty window is allowed when the group has supervision. Initialize the final model before creating the optimizer; save separate rank-local checkpoints and restore matching rank/world size. The default remains single-process.

Metrics include attempted step/window number, microbatch cursor, supervised token counts, mean loss, elapsed time, rate and whether the optimizer update was skipped. Step/cursor counters still advance after a skipped update. `write_progress(directory, metrics, total_steps=...)` writes local `progress.json`; its ETA uses recent comparable step durations. RUDA GPU peak memory is reported as unknown rather than inferred from CUDA allocations.

## Command-line training

The existing [example](../../ruda-torch/python/examples/finetune_causal_lm.py) uses local model files and JSONL input. Set `MODEL_DIR`, `DATA_JSONL`, `RUN_DIR`, `BASE_ID`, `BACKBONE`, `HEAD`, `PAD_TOKEN_ID`, and the Bash array `TARGETS` to your actual checkpoint, dataset, exact module paths and tokenizer padding ID. `RUN_DIR` must be outside the repository; `BASE_ID` identifies the exact frozen weights and quantization configuration. CLI `--targets` takes exact paths, not the API's `'all-linear'` string.

Start with a bounded trial on the intended batch/sequence shapes:

```bash
python ruda-torch/python/examples/finetune_causal_lm.py \
  --model "$MODEL_DIR" --data "$DATA_JSONL" --output "$RUN_DIR" \
  --base-id "$BASE_ID" --backbone "$BACKBONE" --head "$HEAD" \
  --targets "${TARGETS[@]}" --dtype fp16 --pad-token-id "$PAD_TOKEN_ID" \
  --max-length 512 --batch-size 1 --accumulation 2 --steps 2 \
  --rank 16 --alpha 16 --token-chunk-size 32 --checkpoint-every 1
```

Add `--chat` for messages, `--train-on-prompt` or `--truncate` only when those supervision policies are intended. `--checkpoint-modules` takes explicit backbone-relative paths. `--compile` enables the general AOT compiler; it does not imply all operations are native. `--cpu-reference` is a deliberate CPU reference mode. The file is traversed once without implicit repetition or shuffling and must contain enough full microbatches for the requested steps.

Keep the trial's checkpoints/progress outside Git. Assess actual step time and memory on comparable inputs before increasing the workload; a small-input trial is not a capacity estimate for a larger model or sequence. To resume, repeat the same configuration with `--resume "$RUN_DIR/checkpoints/latest.pt"`; `--steps` is the new total target, not an additional step count.

## Save adapters or resume training

| API | Contents / contract |
| --- | --- |
| `adapter_state_dict(model)` | CPU copies of A/B plus versioned target names, dimensions, rank, alpha and dense/NF4 base kind; no base weights |
| `load_adapter_state_dict(model, state)` | Requires an already prepared model with identical adapter layout; validates all entries before copying tensors |
| `finetune_state_dict(model, optimizer, base_id=..., step=..., data_state=..., scaler=None)` | Adapter, optimizer type/state/group ordering, optional scaler, CPU RNG, explicitly used CUDA RNG, completed-window counter and caller data state |
| `load_finetune_state_dict(model, optimizer, state, base_id=..., scaler=None)` | Restores those values, clears gradients and returns `(step, data_state)` |
| `SFTTrainer.save(directory)` | Also records run configuration, scheduler and cursor/token totals; writes and checks `next.pt`, rotates `latest.pt` to `previous.pt`, then promotes the new checkpoint |
| `SFTTrainer.resume(path)` | Requires exact `run_config`, base identity, optimizer layout/type and scheduler/scaler configuration; restores trainer counters |

Snapshot only at optimizer-step boundaries **after** `zero_grad(set_to_none=True)` and scaler `update()`. Partial accumulated gradients are not saved. The optimizer must contain every trainable A/B parameter exactly once and no frozen/non-adapter parameter. The frozen base is not duplicated: recreate the same base, quantization and adapters, then the same optimizer groups, before restoring. Restore the data source to the saved microbatch cursor; custom sampler state belongs in `data_state`.

For direct APIs, serialize the returned mapping with `torch.save` and load with `torch.load(..., map_location='cpu', weights_only=True)` using compatible tensor/basic-container state. `base_id` and `run_config` are supplied by the caller; the CLI additionally captures input/source/dependency/native-library identities. The low-level snapshot does not automatically capture Python/NumPy RNG, arbitrary external sampler state or an independent RUDA device RNG state.

`merge_lora(model)` is only for non-root, eval-mode **dense** LoRA layers whose base weights are not shared with another module. It permanently replaces them with frozen dense layers. NF4 adapters are not expanded/requantized by this API; keep the adapter representation for NF4 deployment. Save an adapter checkpoint before merging if later adapter training is needed.

## Constructor and packing reference

| API | Parameters and limits |
| --- | --- |
| `finetuning.pack_nf4(weight, block_size=64, chunk_blocks=1024)` | Nonempty contiguous CPU `[out,in]` FP32/FP16/BF16 finite weight; positive even block size and positive chunk count; returns uint8 bytes and FP32 scales |
| `NF4Linear(in_features, out_features, packed, scales, block_size=64, tile_rows=128, bias=None)` | Positive geometry/tile values; at most uint32 weight elements and block size; contiguous same-device packed/scales with lengths `ceil(out*in/2)` and `ceil(out*in/block_size)`; scales frozen; optional floating bias `[out]` |
| `NF4Linear.from_linear(linear, block_size=64, tile_rows=128)` | CPU dense layer preprocessing; retains its training mode and frozen bias |
| `NF4Linear.forward(x)` | FP32/FP16/BF16 `[...,in]` on the weight device → `[...,out]`; autocast uses the device's selected activation dtype |
| `LoRALinear(base, rank=16, alpha=16., adapter_dtype=torch.float32)` | Dense/NF4 base; freezes base parameters and creates trainable A/B on its device |
| `quantize_nf4`, `inject_lora` | In-place model conversion; exact target names; quantize before injection; optimizer created afterward |

NF4Linear and LoRALinear `get_extra_state()` / `set_extra_state(state)` store and strictly check their versioned geometry/configuration during ordinary module state-dict loading. They are separate from adapter-only export and do not load a missing frozen base.

Invalid metadata/configuration raises `ValueError`/`TypeError`; missing native capabilities raise errors rather than CPU fallback. Read [packing and checkpoint implementation](../../ruda-torch/python/ruda_torch/finetuning.py), [causal training implementation](../../ruda-torch/python/ruda_torch/causal_finetuning.py), and the [native API reference](native-pytorch-api.md) for adjacent training interfaces.
