# Modellunabhängiges LoRA- und NF4-Fine-tuning

[Inhalt](README.md) · [API](native-pytorch-api.md) · [Detaillierte Verträge](../en/finetuning.md)

Rust nutzt ruda_nn::LoRALinearConfig, PyTorch inject_lora, lokale HF-safetensors load_hf_nf4_model/load_nf4_safetensors. Modellfamilie, Targets und Backbone/Head werden nicht geraten; keine Downloads oder fehlenden Geräteoperatoren werden ergänzt. Passende [native Komponenten](../../ruda-torch/README.md) vorbereiten. Streaming benötigt safetensors, der HF-Helper außerdem transformers/accelerate ohne Remote-Code.

## Reihenfolge

1. Exakte qualifizierte Modulpfade über named_modules bestimmen. API target_modules ist eine nichtleere Namensliste oder der String all-linear, kein Suffix/Regex.
2. Kontiguierliche CPU-Floating-Weights mit quantize_nf4 konvertieren oder ein Meta-Modell tensorweise aus safetensors laden.
3. inject_lora ausführen, **danach** nur requires_grad=True-Parameter an den Optimizer geben. Die gesamte Base wird eingefroren; erneute Injection ist ein Fehler.

LoRALinear berechnet `base(x)+(alpha/rank)*B(A(x))`. Defaults rank16/alpha16/adapter_dtype FP32, A `[rank,in]`, B `[out,rank]` mit B=0. Rang positiv, Alpha endlich positiv, Adapter FP32/FP16/BF16. Python hat keinen Dropout-Parameter. Modulalias bleibt erhalten, Root-Linear zuerst in Container legen.

Rust LoRALinearConfig::new(rank,alpha).with_dropout(p).init(base) erhält Base-IDs. Alpha muss nur endlich sein, Dropout `[0,1)`. merge konsumiert den Adapter und erzeugt eine gefrorene dichte Schicht, keine Umwandlung des Optimizer-Fortsetzungszustands.

## NF4 und Streaming

quantize_nf4 vor Injection/Geräteverschiebung. Defaults block_size64, tile_rows128, positive Werte und gerader Block. Gebundene Embedding/Head-Gewichte ausschließen. Bei Fehlern bleiben bereits konvertierte Schichten konvertiert.

NF4Linear speichert keinen Dense-Shadow: zeilenweise zwei Codes/Byte, erster Code im hohen Nibble, FP32-Absmax pro flachem Block. Scales/Codebook bleiben bei Dtype-Moves FP32. Nur Input-Gradienten erster Ordnung, keine eingefrorenen Base-Gradienten. Das Format ist weder bitsandbytes/PEFT noch AWQ-INT4 oder Learned-Fake-Quantization.

pack_nf4 nimmt nichtleere kontiguierliche CPU `[out,in]` FP32/FP16/BF16; uint8-Länge ceil(out*in/2), Scale-Länge ceil(out*in/block_size). NF4Linear-Eingabe `[...,in]` auf dem Gewichtsgerät. Half/BF16 nutzen NF4-matmul-API1 für fusionierte Tile-Dequantisierung, FP32/fehlende optionale Matmul-Capability bounded Decode. Decode-API1 bleibt nötig; kein erneuter Pfad nach fehlgeschlagener Fusion.

load_nf4_safetensors benötigt alle Parameter auf Meta und exakte Namen/Shapes. Liest model.safetensors oder Index plus Shards. Nichtpersistente fehlende Buffer müssen im Constructor materialisiert sein. parameter_dtypes/buffer_dtypes sind exakte Floating-Namens-Overrides, keine widersprüchlichen Ties; geschützte Gewichte nicht als NF4-Target. Fehler sind nicht transaktional, mit frischem Meta-Modell erneut starten.

load_hf_nf4_model baut die lokale Architektur, erneuert tie_weights, streamt und injiziert. AutoModelForCausalLM ist Standard, Remote-Code nicht autorisiert. config_kwargs/model_kwargs und parameter_dtypes sind explizit. Beide Loader verwenden standardmäßig BF16; **auf T4 FP16 ausdrücklich setzen**.

## Supervision und Akkumulation

SFTCollator paddet rechts auf CPU und liefert int64 input_ids/labels `[B,T]` und Bool attention_mask. Datensätze liefern gleichlange IDs/Labels oder messages. Labels sind Vokabular-IDs/-100, nicht vorher verschoben. Assistant-only-Chat braucht eine vom Template deklarierte Assistant-Maske; train_on_prompt=True überwacht alle Template-Tokens. Tatsächliche pad_token_id/max_length angeben; truncate=False ist Standard.

CausalLMFinetuner trennt Backbone und Head explizit. Backbone liefert `[B,T,D]` direkt oder last_hidden_state, Head ist Dense/NF4/LoRA-Linear. Defaults token_chunk_size32, activation_checkpointing=True, preserve_rng_state=True. checkpoint_modules sind nichtüberlappende relative Backbone-Pfade. RNG-Erhaltung bei Neuberechnung ist nicht gleich Neustart-Checkpointing.

chunked_lm_cross_entropy projiziert jeden Tokenchunk auf das vollständige Vokabular ohne komplettes `[B,T,V]`. Defaults shift=True, ignore_index=-100, recompute=True; reduction mean/sum. Mean teilt durch gültige Targets, vollständig ignorierte Eingabe ergibt differenzierbare Null. Labels gleichgerätig int32/int64.

SFTTrainer.train_step akzeptiert collatierte CPU-Mikrobatches, teilt jede Loss-Summe durch **alle effektiven Tokens des Fensters**, akkumuliert, aktualisiert einmal und löscht Gradienten. Kein Mittel unterschiedlich gewichteter Mikrobatch-Mittelwerte. Scheduler nur nach nichtübersprungenem Update; Step/Cursor laufen auch bei Skip weiter.

## Speichern und Ausführen

adapter_state_dict/load_adapter_state_dict speichern CPU A/B und strikte Namen/Rank/Alpha/Geometrie/Base-Typ. finetune_state_dict/load_finetune_state_dict ergänzen Optimizer-Typ/Parameterreihenfolge, optionalen Scaler, CPU und verwendeten CUDA-RNG, Step/data_state, nicht die Base. Nur an Grenzen mit .grad=None speichern, keine Teilakkumulation. Dieselbe Base-ID, Quantisierung, Adapter-/Optimizer-Konfiguration und Datenposition wiederherstellen.

SFTTrainer.save schreibt/geprüft next.pt und rotiert latest nach previous; resume verlangt exakte run_config und Scheduler. write_progress schreibt lokal, RUDA GPU-Peak bleibt unbekannt. Low-level API erfasst nicht automatisch Python/NumPy-RNG, externe Sampler oder unabhängigen RUDA-RNG. merge_lora nur für eval-Dense ohne geteilte Base-Gewichte, nicht NF4.

[CLI](../../ruda-torch/python/examples/finetune_causal_lm.py): model/data/output/base-id/backbone/head, exakte targets, max-length/steps angeben. Output außerhalb Repo; targets nimmt Pfade, nicht all-linear. Zuerst begrenzte reale Input-Shapes, dann dieselbe Konfiguration mit --resume. --steps ist das neue Gesamtziel. Ein Dateidurchlauf, kein implizites Repeat/Shuffle. --chat/--truncate/--train-on-prompt nur mit beabsichtigter Supervision; --compile bedeutet nicht vollständig native Ausführung.
