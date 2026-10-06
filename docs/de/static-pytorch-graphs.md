# PyTorch-Subgraphen mit festen Adressen

[Inhalt](README.md) · [Modellcompiler](model-compiler.md) · [Node-Tabelle und Beispiele](../en/static-pytorch-graphs.md)

StaticGraph(inputs,nodes,outputs=None,infer_dependencies=False,track_completion=False,optimize=False,reuse_workspace=False,training=False) nutzt vorhandene RUDA CudaGraph-Runtime für explizite GraphOps, kein beliebiges Modell-Capture.

Inputs sind Namen→feste kontiguierliche Dense-ruda:0 FP32/FP16/BF16, nichtleer Rank1–8 und uint32-Elementzahl. Maximal256 Nodes/512 Tensoren, Optionen bool. Outputs sind verschiedene erzeugte Namen, standardmäßig letzter Node. Kein Überschreiben oder zukünftige Referenz.

GraphOp(kind,output,left,right=None,scalar=0.): copy/unary ohne rechts und kanonische Null; add hat alpha, andere binäre Nodes Null. Scalar-Arithmetik nutzt endliches FP32; mm/bmm gleiche Dtypes, Rank2/3 und passende innere/Batchdimensionen. RMSNorm hat gleichartigen Last-Axis-Weight und positives eps; Softmax-Achse nichtnegativ kanonisch, keepdim-Reduktion nutzt nichtleere Achsen-Bitmaske.

Hilfskonstruktoren copy/add/mul/silu/silu_mul/rms_norm. Kein Broadcasting/implizite Promotion. Geometrie/Scalar fest, Eingabeinhalt darf aktualisiert werden, Storage/Shape/Stride/Device nicht wechseln.

Optimize validiert den ganzen Plan, entfernt Ungebrauchtes und fusioniert linksseitiges single-use SiLU/mul. Ungültige unbenutzte Nodes bleiben Fehler. Fusion erhält die Half-Speicherrundung. Reuse nur nach letzter Verwendung gleichartiger Scratch-Allokationen, niemals Inputs/angeforderte Outputs.

```python
import torch
import ruda_torch as r
x = torch.ones(2, 8).to('ruda:0').requires_grad_()
with r.StaticGraph({'x': x}, [r.GraphOp.silu('y', 'x')], training=True) as graph:
    graph.replay()['y'].float().mean().backward()
```

training=False lehnt requires_grad ab. True nutzt nativen Forward, gleichgerätige Backward-Neuberechnung erster Ordnung, Snapshots und unabhängige Outputs. Kein Backward-/Optimizer-Capture; Inputs vor Backward nicht verändern.

Inferenzoutputs aliasieren beim nächsten Replay überschreibbare Buffer. Bei Persistenz klonen und Reihenfolge sichern. Replay auf dem Erstellungsstream, run_eager bleibt derselbe optimierte Plan. synchronize wartet, query meldet Bereitschaft, track_completion aktiviert query_completion/wait_completion. Workspace-Zahlen zählen Node-Allokationen, nicht Peak-VRAM. Close wartet/freigibt, gehaltene Outputs bleiben gültig.

Basis-ABI10/Graph-API3 müssen zusammenpassen. Keine Attention/MoE-Nodes, zweites natives Gerät, AMD/Intel-Graphadapter oder vollständige Optimizer-Capture.
