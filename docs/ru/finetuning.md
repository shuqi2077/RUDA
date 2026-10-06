# Независимое от модели LoRA/NF4 fine-tuning

[Содержание](README.md) · [API](native-pytorch-api.md) · [Подробные контракты](../en/finetuning.md)

Для Rust — ruda_nn::LoRALinearConfig, PyTorch — inject_lora, локального HF safetensors — load_hf_nf4_model/load_nf4_safetensors. Семейство модели, targets, backbone/head не угадываются; веса не скачиваются и отсутствующие операторы не добавляются. Подготовьте [native-компоненты](../../ruda-torch/README.md). Нужен safetensors, для HF-helper также transformers/accelerate без remote code.

## Порядок подготовки

1. По named_modules выберите точные полные пути. target_modules API — непустой список или строка all-linear, не suffix/regex.
2. quantize_nf4 преобразует contiguous CPU floating-веса, либо stream-loader читает safetensors по одному тензору в meta-модель.
3. inject_lora, **потом** optimizer только для requires_grad=True. Вся base замораживается; повторная injection — ошибка.

LoRALinear вычисляет `base(x)+(alpha/rank)*B(A(x))`. Defaults rank16/alpha16/adapter_dtype FP32, A `[rank,in]`, B `[out,rank]` начиная с B=0. Rank положителен, alpha конечен положителен, dtype FP32/FP16/BF16. В Python нет dropout-аргумента. Shared module alias сохраняется; root Linear оберните container.

Rust LoRALinearConfig::new(rank,alpha).with_dropout(p).init(base) сохраняет ID base. Alpha только конечен, dropout `[0,1)`. merge потребляет adapter и создаёт frozen dense слой, не преобразует optimizer-state для продолжения обучения.

## NF4 и потоковое чтение

quantize_nf4 вызывается до injection/device move. Defaults block_size64, tile_rows128, положительные, block чётный. Tied embedding/head-веса исключите. При ошибке ранее преобразованные слои остаются изменёнными.

NF4Linear не хранит dense shadow: row-major два кода/byte, первый в старшем nibble, FP32 absmax на flat block. Scales/codebook остаются FP32 при dtype move. Есть только input-градиенты первого порядка, frozen base не обучается. Формат отличается от bitsandbytes/PEFT, AWQ INT4 и learned fake quantization.

pack_nf4 принимает непустые contiguous CPU `[out,in]` FP32/FP16/BF16, возвращает uint8 длины ceil(out*in/2) и scales ceil(out*in/block_size). NF4Linear input `[...,in]` на устройстве weight. Half/BF16 с matmul API1 используют fused tile dequantization, FP32/нет optional matmul — bounded decode. Decode API1 остаётся необходим; ошибка fused-пути не вызывает повтор через другой путь.

load_nf4_safetensors требует все parameters на meta, точные имена/shapes, model.safetensors либо index/shards. Отсутствующие nonpersistent buffers конструктор материализует. parameter_dtypes/buffer_dtypes — точные floating overrides, tied-конфликт запрещён, dtype-protected weight нельзя выбрать NF4-target. После нетранзакционного сбоя используйте fresh meta model.

load_hf_nf4_model строит local architecture, повторяет tie_weights, stream и injection. Default AutoModelForCausalLM, remote code не разрешён. Явные config_kwargs/model_kwargs и parameter_dtypes поддерживаются. Оба loader по умолчанию BF16; **на T4 явно задайте FP16**.

## Метки, loss и накопление

SFTCollator делает right padding на CPU, отдаёт int64 input_ids/labels `[B,T]`, Bool attention_mask. Записи содержат равноразмерные IDs/labels или messages. Labels — vocabulary ID/-100, заранее не сдвигайте. Assistant-only chat требует declared assistant mask шаблона, train_on_prompt=True включает все template tokens. Задайте реальные pad_token_id/max_length, truncate=False по умолчанию.

CausalLMFinetuner явно разделяет backbone/head. Backbone выдаёт `[B,T,D]` либо last_hidden_state, head — Dense/NF4/LoRA Linear. Defaults token_chunk_size32, activation_checkpointing=True, preserve_rng_state=True. checkpoint_modules — неперекрывающиеся backbone-relative paths. Сохранение RNG при recomputation не равно restart-checkpoint.

chunked_lm_cross_entropy проецирует token chunks на полный vocabulary без всего `[B,T,V]`. shift=True, ignore_index=-100, recompute=True, reduction mean/sum. Mean делит на допустимые targets, полностью ignored даёт дифференцируемый ноль. Labels int32/int64 на том же устройстве.

SFTTrainer.train_step принимает collated CPU microbatches, каждый local loss sum делит на **все эффективные tokens окна**, делает backward, один optimizer update и clear gradients. Это не среднее microbatch-средних. Scheduler только после не-skipped update, step/cursor продвигаются даже при skip.

## Checkpoint и запуск

adapter_state_dict/load_adapter_state_dict содержат CPU A/B и строгие имена/rank/alpha/размеры/base kind. finetune_state_dict/load_finetune_state_dict добавляют тип/порядок optimizer, optional scaler, CPU и использованный CUDA RNG, step/data_state; base не дублируется. Сохраняйте после завершённого step с .grad=None, не частичную аккумуляцию. Восстановите одинаковые base_id/квантизацию/adapters/groups и data cursor.

SFTTrainer.save пишет/проверяет next.pt, вращает latest→previous, resume требует точный run_config/scheduler. write_progress пишет локально, RUDA GPU peak остаётся неизвестным. Low-level API не сохраняет автоматически Python/NumPy RNG, внешний sampler или независимый RUDA RNG. merge_lora только eval-Dense без чужих tied base weights, не NF4.

[CLI](../../ruda-torch/python/examples/finetune_causal_lm.py): явно model/data/output/base-id/backbone/head, точные targets, max-length/steps. Output вне repo, targets — paths, не all-linear. Сначала ограниченные реальные input shapes, затем та же конфигурация с --resume; --steps новое общее число. Один проход файла без скрытого repeat/shuffle. --chat/--truncate/--train-on-prompt лишь при нужной supervision; --compile не означает полностью native.
