# mHC, komprimierte Attention und Python Muon

[Inhalt](README.md) · [API](native-pytorch-api.md) · [Ausführliche Beispiele](../en/architecture-training.md)

Die Komponenten kombinieren differenzierbare Tensoroperationen auf demselben Gerät. Sie sind weder offizielle DeepSeek-Checkpoint-Loader noch FP4/FP8-Attention-Kernel oder eine vollständige Modellreproduktion. Zufällige Initialisierung auf CPU durchführen, anschließend Module und Eingaben explizit verschieben. FP16/BF16-Mappings und Scores verwenden FP32-Arithmetik; verfügbare Geräteoperatoren begrenzen die Ausführung.

## mHC und residualfreie Zweige

`MHC(width,streams=4,sinkhorn_iterations=20,eps=1e-6,gate_init=.01)` erwartet Zustand `[...,streams,width]`. Breite, Streamanzahl und Iterationen sind positive Ganzzahlen, eps positiv, gate_init endlich. Zustand und Parameter liegen auf demselben Gerät.

| Methode | Vertrag |
| --- | --- |
| expand(x) | `[...,width]` → geklonter Streamzustand. |
| reduce(state) | Mittelwert über Streams → `[...,width]`. |
| coefficients(state) | MHCCoefficients mit pre/post `[...,streams]`, residual `[...,streams,streams]`. |
| pre(state) | Zusammengeführte Zweigeingabe und Koeffizienten. |
| post(state,branch_output,coefficients) | Residualmapping und Update; nur die Streamdimension entfällt im Zweigergebnis. |
| forward(state,branch,...) | pre → residualfreier Zweig → post. |

sinkhorn(logits,iterations=20) normalisiert nichtleere quadratische Enddimensionen im Lograum, zuerst Spalten, dann Zeilen. Endliche Iterationen sind keine exakte Mannigfaltigkeitsprojektion.

MHCResidual(branch,width,streams=4,checkpoint_branch=False,**options) verlangt einen nn.Module-Zweig, der **keine eigene Residualaddition** macht. Checkpointing verwendet nichtreentrante Neuberechnung. MHCSequential(width,branches,streams=4,**options) expandiert einmal, verlangt mindestens einen Zweig und mittelt am Ende. Parameter nehmen regulär an Autograd, Optimizer und state_dict teil.

## DSA-Indexer

LightningIndexer und DSAIndexer sind Aliase. Defaults: num_heads4, head_dim16, query_dim=None (width), topk32, query_chunk_size32, key_chunk_size128, external_keys=False, detach_inputs=True, rope_dim0, eps1e-6. RoPE-Dimension ist nichtnegativ, gerade und höchstens head_dim.

Eingaben `[B,T,width]`, optionales Query-Latent `[B,T,query_dim]`, vorbereitete Keys `[B,S,head_dim]`. project_keys erstellt Token-Keys außer im external_keys-Modus. scores erzeugt dichte `[B,T,S]` für Diagnose/Warm-up, select scannt Query/Key-Chunks. Speicher wird begrenzt, die quadratische Score-Arithmetik bleibt.

allowed `[B,T,S]`, key_valid `[B,S]`, query_valid `[B,T]` sind Bool auf demselben Gerät. query_positions `[T]` und key_end_positions `[S]` sind absolute int64-Positionen. Externe komprimierte Keys benötigen tatsächliche Endpositionen. forward(...,causal=True) liefert IndexerOutput(indices,scores,valid), ungültige Slots haben Index -1. causal=False wählt explizit nichtkausale Indizierung.

Top-k-Entscheidungen sind diskret und ohne Gradienten. Eingabefeatures sind standardmäßig detached; Indexerparameter trainieren über neu berechnete ausgewählte Scores und explizite indexer_kl_loss/distillation_loss. Teacher ist nichtnegative Masse `[B,T,S]` oder `[B,T,H,S]`, wird detached, leere Zeilen tragen null bei. Der LM-Loss allein trainiert den Selector nicht.

## CSA/HCA und Caches

CSA kombiniert überlappende gelernte Kompression, DSA-Selektion und lokales Fenster. HCA komprimiert ohne Überlappung und verwendet alle kausal sichtbaren komprimierten Einträge ohne Indexer. Beide verwenden eine gemeinsame Softmax und nur abgeschlossene Blöcke.

| Option | Default / Grenze |
| --- | --- |
| head_dim | width/num_heads; bei Auslassung muss die Division aufgehen. |
| compress_ratio | CSA4, HCA128; positiv. |
| topk / window_size | 32 / 128. |
| query_rank / index_heads / index_dim | width / 4 / 16. |
| rope_dim / rope_base | 0 / 10000; gerade Dimension, passend für Attention und Indexer; kein YaRN. |
| output_groups / output_rank | 1 / width/output_groups; Gruppen teilen die Headanzahl. |
| query_chunk_size / key_chunk_size | 32 / 128. |
| attention_sink / eps | True / 1e-6; Sink trägt zum Nenner bei, ist kein KV-Wert. |

forward(x,valid_mask=None,return_aux=False,indexer_warmup=False) nimmt nichtleere `[B,T,width]` und Bool `[B,T]`, True bedeutet gültig. return_aux liefert AttentionOutput(output,indexer_loss). Warm-up nutzt alle sichtbaren komprimierten Einträge und behält die dichten Kosten. Für reines Indexer-Warm-up nur den Hilfsloss ableiten; HCA liefert eine vom Hauptmodell getrennte Null.

LearnedKVCompressor(width,head_dim,ratio,overlap=False,eps=1e-6) liefert floor(T/ratio) Einträge und Gültigkeit. Unvollständige Tails werden nicht ausgegeben; vollständig ungültige Blöcke sind null. RotaryEmbedding wirkt auf `[B,T,D]`/`[B,T,H,D]` mit `[T]` Positionen; Frequenzen bleiben FP32, inverse=True kehrt die Rotation um.

forward_cached(x,cache=None,valid_mask=None) und compressor.append benötigen **eval und no_grad/inference_mode**. `(output,new_cache)` funktional weiterreichen. Anderer Layer, geänderte Parameter oder Batch/Device/Dtype werden abgewiesen; neu mit None beginnen. reorder erwartet einen gleichgerätigen eindimensionalen int64-Beamindex. tensor_bytes zählt logische Tensorbytes, nicht Peak-VRAM; komprimierte Historie wächst weiterhin mit Sequenzlänge.

## Modell und Python Muon

MHCTransformerBlock verwendet `[B,T,streams,width]`, attention_kind csa/hca, FFN standardmäßig 4*width. HybridAttentionLanguageModel(vocab_size,width,num_heads,num_layers,streams=4,csa_ratio=4,hca_ratio=128,tie_embeddings=False) alterniert CSA/HCA und liefert `[B,T,V]` aus int32/int64 `[B,T]`. return_aux enthält die Summe der Indexer-Losses. Das Gesamtmodell hat kein kombiniertes forward_cached.

next_token_loss berücksichtigt nur gültige Quell/Zielpaare. Auch maskierte Token brauchen gültige Vokabular-IDs, nicht -100. Für ignorierte SFT-Labels gilt der separate [Fine-tuning-Vertrag](finetuning.md).

Muon ist ein Single-Device-Eager-Optimizer für vollständige Matrizen. Defaults: lr.02, momentum.95, nesterov=True, momentum_mode sgd, dampening0, ns_steps5, Koeffizienten `(3.4445,-4.775,2.0315)`, eps1e-7, adjust_lr original, matrix_layout as_stored, stable_normalization=True, flatten=False. Iterationen1–99, endliche Koeffizienten. Nur flatten=True formt höhere Dimensionen zu `[erste_Dimension,-1]`. Nesterov benötigt positives Momentum und null Dampening; EMA ebenfalls null Dampening.

original skaliert LR mit `sqrt(max(1,rows/cols))`, match_rms_adamw mit `.2*sqrt(max(rows,cols))`. input_output tauscht die Achsensemantik für Skalierung, nicht die Speicherung. Optional positives max_grad_norm clippt unskalierte Gradienten ohne Änderung von .grad. Half-Master/Momente sind FP32. Finite-Readbacks verhindern Graph-Capture; nichtendliche Werte überspringen den ganzen Update, ein Gerätefehler beim Commit ist nicht transaktional.

MuonAdamW.from_model(model,muon_modules=...,adamw_modules=...,lr=.02,adamw_lr=.001) nominiert echte Modulobjekte. Embeddings und explizite Ausschlüsse erhalten AdamW-Priorität, geteilte Parameter werden einmal berücksichtigt; andere trainierbare Parameter verwenden ebenfalls AdamW. Ohne nominierte geeignete Matrix entsteht ein Fehler. Manuelle Gruppen brauchen use_muon=True/False. Gradienten vor Orthogonalisierung vollständiger Matrizen synchronisieren, keine lokalen TP/FSDP-Shards unabhängig behandeln. Python-State ist kein Rust-Muon-Record.
