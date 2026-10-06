# Компиляция обычных PyTorch-моделей

[Содержание](README.md) · [Подробная справка](../en/model-compiler.md)

ruda_torch.compile оборачивает nn.Module/callable. AOTAutograd строит forward/backward, поддерживаемые области используют StaticGraph, остальные — исходный device dispatch. Все реально используемые operators всё ещё требуют kernels. StaticGraph.from_model возвращает тот же callable wrapper, не replay-object.

| Опция | Default / контракт |
| --- | --- |
| capture | aot; eager явно без компиляции. |
| native | auto, off сохраняет AOT без native regions, required отвергает unsupported captured operators/guards. |
| device_type | ruda, тип без :0, cpu — явная reference. |
| fullgraph / dynamic | False / None; отсутствие breaks не означает все operators native. |
| min_native_ops | 2, диапазон1–256; required минимум1. |
| cache_size | 4, диапазон1–64, **на область**. |
| decompositions | None, точные overload→callable математические преобразования. |

Eager несовместим с required/fullgraph/dynamic/непустыми decompositions. AOT wrapper отвергает suppress_errors=True, ошибка не превращается в скрытое eager-повторение.

Native: clone, unary из UNARY_CODES, add/sub/mul/div Tensor/Scalar, mm/bmm, keepdim sum/mean, softmax/log-softmax и backward, SiLU/sigmoid/tanh backward. Нет binary/batch broadcasting, dtype promotion, неявной half→FP32 softmax и dtype override редукции.

Native inputs dense strided ruda:0 FP32/FP16/BF16, непустой rank1–8, uint32 indexing. Contiguous staging/output metadata, предел256 nodes/512 tensors. Views/mutation/RNG/неизвестное в auto выполняется на исходном device, в required — ошибка. Нет CPU fallback или retry после dispatch.

Явно переносите model/inputs. Identity parameters и прямые state_dict keys сохраняются, wrapper-дочерний модуль имеет _original prefix. AOT только первые производные, eager — по поддержке operator. Bounded LRU различает shape/stride/dtype/device/stream/inference state. Outputs клонируются, поздний replay их не перезаписывает; staging/copy стоят памяти/времени, capture сам по себе не гарантирует acceleration.

info различает plan и native_replays/native_nodes_executed/reference_reasons. NativeCoverageError относится к required-контракту, GraphExecutionError к ошибке исходного device. Close после всех backwards. Make_backend интегрирует внешний torch.compile, его global-настройки отвечает caller. Полный capture optimizer/Python side effects не обещается.
