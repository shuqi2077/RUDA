# Контракты native PyTorch API

[Содержание](README.md) · [Подробная справка методов](../en/native-pytorch-api.md)

Импорт ruda_torch регистрирует PrivateUse1, другой backend в том же процессе несовместим. ruda:0 и torch.cuda — разные allocations. device_count1/current_device0. is_available означает инициализацию, не всю hardware capability. execution_stats — cumulative counters, не timer.

Rust/C++ требуют base ABI10, graph3, training4. Router/sequence/NF4 decode/NF4 matmul API1, paged backward совместимый1/2, ordered2. Совпадение base ABI не предоставляет все optional features. Unsupported operator — ошибка без generic CPU fallback.

## Обучение

Fused normalization принимает contiguous dense ruda:0 FP32/FP16/BF16 без unresolved views. Статистика FP32, результат dtype input, только первые производные.

| API | Контракт |
| --- | --- |
| rms_norm(x,weight=None,eps=None) | `[...,D]`, положительный последний размер. Weight `[D]`, тот же device, input dtype/FP32. eps None использует finfo.eps. |
| RMSNorm(width,eps=1e-5,elementwise_affine=True) | Width положителен, affine FP32 по умолчанию; CPU construction затем явный RUDA move. |
| layer_norm / LayerNorm | Только последняя ось, optional weight/bias `[D]` input dtype/FP32, конечный положительный eps. |
| silu_mul(gate,up) | Равные shape/dtype/device, без broadcasting, сохраняет storage-rounding boundary. |

AdamW defaults lr.001, betas(.9,.999), eps1e-8, decay.01, fused_step=False, hierarchical_stats=False, max_grad_norm=None. Неперекрывающиеся обычные leaf parameters и contiguous matching gradients, без capturable/amsgrad/maximize/differentiable. FP32 master/moments. Default меняет gradient при unscale и читает4 byte; fused оставляет gradient и читает12 byte, один kernel на активный параметр. Clipping/hierarchical требуют fused; нечисловые/бесконечные gradients пропускают весь step.

GradScaler — один RUDA AdamW/Muon/MuonAdamW на цикл, loss один FP32-элемент: scale/backward → step → update. Нет unscale_, closure и multi-optimizer cycle. Defaults65536/2/.5/2000, границы `2**-24..2**24`; state сохраняется между завершёнными циклами.

## Streams и внимание

Stream priority только0. current_stream/default_stream и stream-context; wait_event/wait_stream задают device-порядок, query не ждёт, synchronize ждёт на host. elapsed_time — ms после завершения timing-enabled событий. record_stream удерживает memory lifetime без dependency. RUDA_TORCH_ASYNC задайте до первого submission, значение кешируется процессом.

PagedAttentionPlan — immutable schedule. Q `[queries,q_heads,key_dim]`, K/V `[pages,page_size,kv_heads,dim]`, contiguous same floating dtype, q_heads кратно kv_heads, dim<=1024, scale конечен положителен. MLA queries `[queries,heads,rank]` плюс позиции, cache `[pages,page_size,1,rank]`, position_dim<=256. Caller обновляет cache/RoPE, использует исходный QK scale. Splits1 либо2–32, workspace_bytes лишь forward scratch. Первые градиенты, default atomic; ordered явно, не autotune.

selected_router_weights: logits `[tokens,experts]` floating, indices `[tokens,top_k]` int32/int64, top_k<=min(experts,64). FP32 weights/dlogits, без выбора экспертов. Softmax всех experts, sigmoid pointwise, renormalize выбранных slots. Duplicate indices суммируют gradients, invalid index даёт NaN строки forward/backward.

## Последовательности и квантизация

solve_triangular: same floating dtype/device, A `[...,N,N]`, left B `[...,N,K]` / right `[...,K,N]`, broadcast batch. Bool flags, первые градиенты, RUDA sequence API1.

gated_delta_rule: Q/K `[B,H,T,Dk]`, V `[B,H,T,Dv]`, beta/decay `[B,H,T]`, optional state `[B,H,Dk,Dv]`. Q/K/V dtype одинаков, chunk_size64; output/final_state имеют градиенты. Half state FP32. Native chunk forward и same-device recompute backward, не отдельный fused backward.

learned_fake_quantize: bits2/4/8, floating input и same-device FP32 scales. Без block_shape один scale, иначе положительные блоки на каждую ось и `product(ceil(shape/block))` scales. Ниже floor1e-8 scale-gradient ноль. Floating output первого порядка, не packed NF4.

Далее: [Compiler](model-compiler.md), [Graph](static-pytorch-graphs.md), [Fine-tuning](finetuning.md), [mHC/CSA/HCA/Muon](architecture-training.md), [Распределённое обучение](distributed-training.md). Асинхронная ошибка может проявиться в sync/readback.
