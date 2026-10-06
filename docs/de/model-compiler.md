# Allgemeine PyTorch-Modellkompilierung

[Inhalt](README.md) · [Detaillierte Referenz](../en/model-compiler.md)

ruda_torch.compile umhüllt nn.Module/Callable; AOTAutograd erzeugt Forward-/Backward-Graphen. Geeignete Regionen nutzen StaticGraph, andere Operatoren ihr ursprüngliches Gerät. Für jeden ausgeführten Operator bleibt ein Gerätekernel nötig. StaticGraph.from_model liefert denselben aufrufbaren Wrapper, kein replay-Objekt.

| Option | Default / Vertrag |
| --- | --- |
| capture | aot; eager ist explizite unkompilierte Ausführung. |
| native | auto; off behält AOT ohne native Regionen; required lehnt nichtnative captured Operatoren/Guards ab. |
| device_type | ruda, ein Typ ohne :0; cpu ist explizite Referenz. |
| fullgraph / dynamic | False / None; keine Graph-Breaks bedeutet nicht vollständig native Operatoren. |
| min_native_ops | 2, Bereich1–256; required verwendet Minimum1. |
| cache_size | 4, Bereich1–64, **pro Region**. |
| decompositions | None; genaue Overload→Callable-Abbildung. |

Eager kann nicht mit required/fullgraph/dynamic oder nichtleeren Decompositions kombiniert werden. AOT-Wrapper lehnt suppress_errors=True ab; Fehler werden nicht in implizite Eager-Wiederholung umgewandelt.

Native Overloads: clone, UNARY_CODES-Unary, add/sub/mul/div Tensor/Scalar, mm/bmm, keepdim sum/mean, softmax/log-softmax und Backward sowie SiLU/sigmoid/tanh Backward. Keine binären/batch Broadcasts, Dtype-Promotion, implizite Half→FP32-Softmax oder Reduktions-Dtype-Overrides.

Native Eingaben sind Dense-Strided-ruda:0 FP32/FP16/BF16, nichtleer Rank1–8, uint32-Indizierung. Kontiguierliches Staging und zusammenhängende Output-Metadaten, maximal256 Operatoren/512 Tensoren. Views, Mutation, RNG und Unbekanntes bleiben in auto auf ihrem Gerät oder erzeugen required-Fehler. Kein CPU-Fallback oder Retry nach Dispatch.

Modell/Eingaben explizit verschieben. Parameteridentität und direkte state_dict-Schlüssel bleiben, als Kindmodul enthält der Wrapper _original. AOT erste Ableitungen, eager höhere nur nach Operatorfähigkeit. Cache ist bounded LRU nach Shape/Stride/Dtype/Device/Stream/Inference-State. Outputs werden vor Rückgabe geklont; Capture benötigt zusätzlichen Speicher/Copy und garantiert keinen Speedup.

info trennt Plan von tatsächlichem native_replays/native_nodes_executed/reference_reasons. NativeCoverageError betrifft required, GraphExecutionError Fehler auf dem Originalgerät. close nach allen Backwards. make_backend integriert externes torch.compile, dessen Global-Konfiguration bleibt Caller-Verantwortung. Kein Versprechen vollständiger Optimizer-/Python-Sideeffect-Capture.
