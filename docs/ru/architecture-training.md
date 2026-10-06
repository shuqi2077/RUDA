# mHC, сжатое внимание и Python Muon

[Содержание](README.md) · [API](native-pytorch-api.md) · [Подробные примеры](../en/architecture-training.md)

Компоненты объединяют дифференцируемые операции на одном устройстве. Это не официальный загрузчик весов DeepSeek, не полное воспроизведение модели и не FP4/FP8-ядра внимания. Случайную инициализацию выполняйте на CPU, затем явно переносите модули и входы. Для FP16/BF16 отображения и оценки вычисляются в FP32; возможность исполнения определяется доступными операторами устройства.

## mHC и ветви без собственного residual

MHC(width,streams=4,sinkhorn_iterations=20,eps=1e-6,gate_init=.01) принимает состояние `[...,streams,width]`. Размеры и число итераций положительные целые, eps положителен, gate_init конечен. Параметры и состояние на одном устройстве.

| Метод | Контракт |
| --- | --- |
| expand(x) | `[...,width]` → копия состояния потоков. |
| reduce(state) | Среднее по streams → `[...,width]`. |
| coefficients(state) | MHCCoefficients: pre/post `[...,streams]`, residual `[...,streams,streams]`. |
| pre(state) | Объединённый вход ветви и коэффициенты. |
| post(state,branch_output,coefficients) | Остаточное отображение и обновление; выход ветви не имеет только оси streams. |
| forward(state,branch,...) | pre → ветвь → post. |

sinkhorn(logits,iterations=20) требует непустые квадратные последние оси и нормализует в log-пространстве сначала столбцы, затем строки. Конечные итерации не дают точную проекцию на многообразие.

MHCResidual(branch,width,streams=4,checkpoint_branch=False,**options) оборачивает nn.Module, который **не прибавляет собственный residual**. Checkpoint ветви использует пересчитывание с use_reentrant=False при обучении. MHCSequential(width,branches,streams=4,**options) один раз расширяет состояние, выполняет хотя бы одну ветвь и усредняет потоки. Параметры участвуют в обычных autograd, optimizer и state_dict.

## DSA indexer

LightningIndexer и DSAIndexer — псевдонимы. По умолчанию num_heads4, head_dim16, query_dim=None (width), topk32, query_chunk_size32, key_chunk_size128, external_keys=False, detach_inputs=True, rope_dim0, eps1e-6. RoPE-размер неотрицательный чётный и не превышает head_dim.

Вход `[B,T,width]`, необязательный latent `[B,T,query_dim]`, подготовленные ключи `[B,S,head_dim]`. project_keys строит token keys, кроме режима external_keys. scores возвращает плотные `[B,T,S]` для диагностики/разогрева, select сканирует чанки. Ограничение промежуточной памяти не устраняет квадратичную арифметику оценок.

allowed `[B,T,S]`, key_valid `[B,S]`, query_valid `[B,T]` — Bool на устройстве входа. query_positions `[T]`, key_end_positions `[S]` — абсолютные int64. Для сжатых внешних ключей нужны реальные конечные позиции. forward(...,causal=True) возвращает IndexerOutput(indices,scores,valid), недействительные индексы -1; causal=False явно отключает причинность.

Дискретный top-k не имеет градиента. Входные признаки по умолчанию detached, но параметры обучаются через повторно вычисленные выбранные scores и явно добавленный indexer_kl_loss/distillation_loss. Teacher — неотрицательная масса `[B,T,S]` или `[B,T,H,S]`, detached; пустые строки дают ноль. Один LM-loss не обучает selector.

## CSA/HCA и кеш

CSA объединяет перекрывающуюся обучаемую компрессию, DSA и локальное окно. HCA использует неперекрывающиеся блоки и все причинно видимые сжатые элементы без indexer. Одна softmax включает локальные и сжатые элементы; видны только завершённые блоки.

| Опция | Значение по умолчанию / условие |
| --- | --- |
| head_dim | width/num_heads, при пропуске должно делиться нацело. |
| compress_ratio | CSA4, HCA128, положительное. |
| topk / window_size | 32 / 128. |
| query_rank / index_heads / index_dim | width / 4 / 16. |
| rope_dim / rope_base | 0 / 10000, чётный размер помещается в attention/indexer, не YaRN. |
| output_groups / output_rank | 1 / width/output_groups, группы делят число голов. |
| query_chunk_size / key_chunk_size | 32 / 128. |
| attention_sink / eps | True / 1e-6, sink добавляет массу знаменателя, а не значение KV. |

forward(x,valid_mask=None,return_aux=False,indexer_warmup=False) принимает непустые `[B,T,width]` и Bool `[B,T]`, True означает допустимую позицию. return_aux возвращает AttentionOutput(output,indexer_loss). Warm-up использует все видимые сжатые элементы и сохраняет плотную стоимость; для обучения только indexer делайте backward только вспомогательного loss. HCA возвращает ноль без связи с основными параметрами.

LearnedKVCompressor(width,head_dim,ratio,overlap=False,eps=1e-6) выдаёт floor(T/ratio) элементов и маску. Неполные tails не выдаются, полностью masked блоки нулевые. RotaryEmbedding работает с `[B,T,D]`/`[B,T,H,D]` и `[T]` позициями, частоты остаются FP32, inverse=True обращает вращение.

forward_cached(x,cache=None,valid_mask=None) и compressor.append требуют **eval и no_grad/inference_mode одновременно**. Передавайте возвращённый `(output,new_cache)` следующему чанку. Иной слой, изменённые параметры, batch/device/dtype вызывают ошибку, начните с None. Обновление функциональное. reorder принимает одномерные int64 beam-индексы на том же устройстве. tensor_bytes — логические байты тензоров, не пиковая VRAM; сжатая история продолжает расти с длиной.

## Модель и Python Muon

MHCTransformerBlock сохраняет `[B,T,streams,width]`, attention_kind csa/hca, FFN обычно4*width. HybridAttentionLanguageModel(vocab_size,width,num_heads,num_layers,streams=4,csa_ratio=4,hca_ratio=128,tie_embeddings=False) чередует CSA/HCA и выдаёт `[B,T,V]` logits из int32/int64 `[B,T]`. return_aux включает сумму indexer-loss. У всей модели нет объединённого forward_cached, он есть у отдельных слоёв внимания.

next_token_loss усредняет следующие токены лишь по парам, где обе позиции допустимы. Даже masked токены должны быть корректными ID словаря, не -100. Для игнорируемых SFT-labels используйте отдельный [контракт fine-tuning](finetuning.md).

Muon — eager-оптимизатор одного устройства для полных матриц. Defaults lr.02, momentum.95, nesterov=True, momentum_mode sgd, dampening0, ns_steps5, коэффициенты `(3.4445,-4.775,2.0315)`, eps1e-7, adjust_lr original, matrix_layout as_stored, stable_normalization=True, flatten=False. Итерации1–99, коэффициенты конечные. Лишь flatten=True превращает старшие размерности в `[первая,-1]`. Nesterov требует положительный momentum и нулевой dampening; EMA также требует нулевой dampening.

original умножает LR на `sqrt(max(1,rows/cols))`, match_rms_adamw на `.2*sqrt(max(rows,cols))`. input_output меняет смысл осей для масштаба, не транспонирует параметры. Необязательный положительный max_grad_norm ограничивает unscaled градиенты, сохраняя .grad. Half-master/moments FP32. Readback конечности исключает capture; нечисловые/бесконечные значения пропускают весь update, ошибка устройства при commit не транзакционна.

MuonAdamW.from_model(model,muon_modules=...,adamw_modules=...,lr=.02,adamw_lr=.001) принимает реальные объекты модулей. Embeddings и явные исключения имеют приоритет AdamW, shared параметр включается один раз, прочие обучаемые параметры также AdamW. Нет подходящей назначенной матрицы — ошибка. Ручные группы требуют use_muon=True/False. Синхронизируйте градиенты до ортогонализации полных матриц, не обрабатывайте локальные TP/FSDP-shards независимо. Python state не является Rust Muon Record.
