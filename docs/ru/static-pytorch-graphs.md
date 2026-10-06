# PyTorch-графы с фиксированными адресами

[Содержание](README.md) · [Compiler](model-compiler.md) · [Таблица nodes и примеры](../en/static-pytorch-graphs.md)

StaticGraph(inputs,nodes,outputs=None,infer_dependencies=False,track_completion=False,optimize=False,reuse_workspace=False,training=False) исполняет явные GraphOp через существующий RUDA CudaGraph, не произвольный model capture.

Inputs имена→fixed contiguous dense ruda:0 FP32/FP16/BF16, непустой rank1–8 и uint32 elements. Предел256 nodes/512 tensors, options bool. Outputs — уникальные произведённые имена, default последний node. Нельзя перезаписывать имя или читать будущий node.

GraphOp(kind,output,left,right=None,scalar=0.): copy/unary без right и canonical zero, add использует alpha, прочие binary zero. Scalar arithmetic — конечный FP32. Mm/bmm равные dtype, rank2/3, inner/batch matching. RMSNorm weight того же dtype/последней ширины, scalar положительный eps. Softmax axis canonical неотрицательная, keepdim reduction — непустая битовая маска осей.

Helpers copy/add/mul/silu/silu_mul/rms_norm. Нет broadcasting/implicit promotion. Геометрия/scalar fixed, можно обновлять contents inputs, не storage/shape/stride/device.

Optimize проверяет весь plan, убирает unused и объединяет single-use левый SiLU/mul. Некорректный unused node остаётся ошибкой. Fusion сохраняет half storage rounding. Reuse только same-spec scratch после последнего чтения, не input/requested output.

```python
import torch
import ruda_torch as r
x = torch.ones(2, 8).to('ruda:0').requires_grad_()
with r.StaticGraph({'x': x}, [r.GraphOp.silu('y', 'x')], training=True) as graph:
    graph.replay()['y'].float().mean().backward()
```

training=False запрещает requires_grad. True даёт native forward, same-device recomputation первых градиентов, snapshots и независимые results, не backward/optimizer capture. До backward исходные inputs не менять.

Inference output — buffer, перезаписываемый следующим replay. Клонируйте и упорядочивайте для сохранения. Replay на stream создания, run_eager тот же optimized plan. synchronize ждёт, query проверяет ready, track_completion включает query_completion/wait_completion. Workspace — node allocations, не peak VRAM. Close ждёт/освобождает, удержанные outputs действительны.

Совместимые base ABI10/graph API3 у обеих native частей. Нет attention/MoE nodes, второго native-device, AMD/Intel graph-adapter и полного optimizer capture.
