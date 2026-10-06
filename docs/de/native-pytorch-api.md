# Native PyTorch-API-Verträge

[Inhalt](README.md) · [Ausführliche Methodenreferenz](../en/native-pytorch-api.md)

Import ruda_torch registriert PrivateUse1; kein anderer Backend im selben Prozess. ruda:0 und torch.cuda verwenden verschiedene Allokationen. device_count1/current_device0. is_available zeigt Initialisierung, nicht vollständige Hardwarefähigkeit. execution_stats sind kumulative Zähler, keine Timer.

Rust/C++: Basis-ABI10, Graph3, Training4. Router/Sequence/NF4-Decode/NF4-Matmul jeweils1, Paged-Backward kompatibel1/2 und Ordered2. Basis-ABI allein stellt keine optionalen APIs bereit. Nichtunterstützte Operatoren schlagen fehl; kein generischer CPU-Fallback.

## Training

Fused-Normalisierung benötigt kontiguierliche Dense-ruda:0 FP32/FP16/BF16 ohne unresolved Views. Statistik FP32, Ausgabe Eingabe-Dtype, nur erste Ableitungen.

| API | Vertrag |
| --- | --- |
| rms_norm(x,weight=None,eps=None) | `[...,D]`, positive letzte Dimension. Weight `[D]` auf demselben Gerät, Eingabe-Dtype/FP32. eps=None nutzt finfo.eps. |
| RMSNorm(width,eps=1e-5,elementwise_affine=True) | Positive Breite, affine Parameter standardmäßig FP32; CPU-Konstruktion vor explizitem RUDA-Move. |
| layer_norm / LayerNorm | Nur letzte Achse; optionale Weight/Bias `[D]` im Eingabetyp/FP32, endliches positives eps. |
| silu_mul(gate,up) | Gleiche Shape/Dtype/Device, kein Broadcasting, erhält Speicherrundungsgrenze. |

AdamW: lr.001, betas(.9,.999), eps1e-8, decay.01, fused_step=False, hierarchical_stats=False, max_grad_norm=None. Nichtüberlappende normale Leaf-Parameter, kontiguierliche passende Gradienten; kein capturable/amsgrad/maximize/differentiable. FP32-Master/Momente. Standard verändert Gradienten beim Unscale und liest4 Byte; fused erhält sie und liest12 Byte, weiterhin ein Kernel pro aktivem Parameter. Clipping/hierarchical benötigen fused; nichtendliche Gradienten skippen den ganzen Step.

GradScaler unterstützt ein RUDA AdamW/Muon/MuonAdamW pro Zyklus mit FP32-Skalar-Loss: scale/backward → step → update. Kein unscale_, keine Closure oder Multi-Optimizer-Zyklen. Defaults65536/2/.5/2000 und Grenzen `2**-24..2**24`; Zustand nur zwischen abgeschlossenen Zyklen speichern.

## Streams und Attention

Stream-Priorität nur0. current_stream/default_stream/stream-Kontext; wait_event/wait_stream etablieren Gerätereihenfolge, query wartet nicht, synchronize wartet hostseitig. elapsed_time benötigt abgeschlossene timingfähige Events und liefert ms. record_stream hält Speicher am Leben, erzeugt keine Abhängigkeit. RUDA_TORCH_ASYNC vor erster Submission setzen; prozessweit gecacht.

PagedAttentionPlan hält unveränderlichen Schedule. Q `[queries,q_heads,key_dim]`, K/V `[pages,page_size,kv_heads,dim]`, kontiguierlicher gleicher Floating-Typ; q_heads durch kv_heads teilbar, dim<=1024, Scale endlich positiv. MLA-Queries `[queries,heads,rank]` plus Positionsanteil, Caches `[pages,page_size,1,rank]`, position_dim<=256. Caller hält Cache-Updates/RoPE und originale QK-Skalierung. Splits1 oder2–32, workspace_bytes nur Forward-Scratch. Erste Ableitungen, atomic Standard, ordered explizit und kein Autotuning.

selected_router_weights: logits `[tokens,experts]` Floating, indices `[tokens,top_k]` int32/int64, top_k<=min(experts,64). FP32-Gewichte/dlogits, keine Auswahl. Softmax über alle Experten, Sigmoid pointwise, renormalize über gewählte Slots. Doppelte Indizes akkumulieren; ungültiger Index erzeugt NaN der betroffenen Forward-/Backward-Zeile.

## Sequenzen und Quantisierung

solve_triangular: gleiches Floating-Dtype/Device, A `[...,N,N]`, links B `[...,N,K]` oder rechts `[...,K,N]`, broadcastbare Batchdimensionen; bool-Flags, erste Ableitungen, RUDA-Sequence-API1.

gated_delta_rule: Q/K `[B,H,T,Dk]`, V `[B,H,T,Dv]`, beta/decay `[B,H,T]`, optional State `[B,H,Dk,Dv]`. Q/K/V-Dtype gleich, chunk_size64, Ergebnis output/final_state mit beiden Gradienten. Half-State FP32. Native Chunk-Forward und gleichgerätige Backward-Neuberechnung, kein eigenständiger fusionierter Backward.

learned_fake_quantize: bits2/4/8, Floating-Eingabe und FP32-Scales auf demselben Gerät. Ohne block_shape ein Scale, sonst positive Blöcke pro Achse und `product(ceil(shape/block))` Scales. Unter floor1e-8 kein Scale-Gradient. Floating-Ergebnis erster Ordnung, kein gepacktes NF4.

Weitere Referenzen: [Compiler](model-compiler.md), [Graph](static-pytorch-graphs.md), [Fine-tuning](finetuning.md), [mHC/CSA/HCA/Muon](architecture-training.md), [Verteiltes Training](distributed-training.md). Asynchrone Fehler können bei Sync/Readback sichtbar werden.
