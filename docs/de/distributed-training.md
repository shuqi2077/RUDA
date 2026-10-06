# Explizite Ranks, Geräte und repliziertes Training

[Inhalt](README.md) · [Training](training.md) · [ruCCL](libraries/ruccl.md) · [Verbindungscode](../en/distributed-training.md)

Rankanzahl ist nicht GPU-Anzahl. Das registrierte collective_training-Beispiel verwendet zwei logische Ranks auf demselben Default-Gerät. in_process::Communicator besitzt lokale Gerätekontexte; RankCommunicator verwendet explizite Ranks/Weltgröße und TCP-Sessions; DataParallel implementiert Replikatsemantik auf einem vom Aufrufer bereitgestellten Communicator. PyTorch ruda:0 ist Single-Device und kein Ersatz für einen torch.distributed-Launcher. Hier geht es um Rust-Tensor/Autodiff-Training, nicht automatisches TP/PP/FSDP/ZeRO.

## Platzierung und Start

CudaDevice { index } wählt eine prozesssichtbare CUDA-Ordinalzahl, unabhängig vom globalen Rank. Modell und TensorDevice müssen dasselbe Gerät verwenden. Zwei GPUs im selben Prozess explizit als0/1 auswählen; zweimal Default wählt zweimal0. Bei veränderter Sichtbarkeit zählt die tatsächlich sichtbare Reihenfolge des jeweiligen Prozesses.

Allen Ranks dieselbe erreichbare Adresse, eine gemeinsame UniqueId, verschiedene `0..world_size-1` Ranks, Weltgröße und lokale Gerätezuordnung übergeben. UniqueId einmal mit new erzeugen, as_bytes verteilen und mit from_bytes rekonstruieren. Unabhängige new-Aufrufe erzeugen verschiedene Sessions.

- Koordinator: TcpRendezvousServer::bind(address,id,world_size)?.run(), parallel zu allen Workern.
- Worker: RankCommunicator::connect(initialize,address,id,rank,world_size,timeout,queue_name); Initializer erzeugt sein TensorDevice.
- TCP-/Heartbeat-Timeouts, Rails und Transport konsistent konfigurieren. Automatische torchrun/NCCL-Umgebungsinterpretation wird nicht bereitgestellt.
- Standard-TCP-Tensortransport ist host-staged; TCP-Peers sind nicht automatisch GPU-P2P/NVLink/RDMA.

Cargo-Abhängigkeiten: ruCCL, ruda-autodiff, ruda-nn, ruda-optim mit collective, ruda-tensor-device mit cuda-default. [Vollständige Initialisierung](../en/distributed-training.md#rendezvous-and-rank-connection) und [Verbindungs-API](../../ruCCL/src/rank/communicator/connect.rs) zeigen die Fehlerkonvertierung. Eigener nativer Transport implementiert DataParallelCommunicator inklusive geordneter Metadaten und Broadcast/Reduce-Verträge.

## Trainingsreihenfolge

DataParallel::initialize(communicator,model,root) prüft Pfade, Shapes, Dtypes, Frozen-Flags und geteilte Aliase, dann Broadcast der Floating-Parameter. Lokale IDs dürfen sich unterscheiden und bleiben erhalten. Vor Optimizer-Erstellung beziehungsweise nach passender Rank-Wiederherstellung aufrufen. initialize_with_buffers synchronisiert I32/I64/Bool einmal, nicht vor jedem Forward.

Alle Ranks benötigen gleiche Collective-Reihenfolge, Shape/Dtype, Gradiententracking, Root und Gather/Scatter-Achse. Ein Rank darf nicht aussetzen, während Peers kommunizieren. Differenzierbare Collectives brauchen gleiche Backward-Reihenfolge; Checkpoint-Neuberechnung wiederholt nicht die Kommunikation.

Lokale Loss-**Summen** ableiten und akkumulieren. An der Grenze reduce(&model,gradients,local_weight,policy) aufrufen, danach mit zurückgegebenen gradients aktualisieren. Die globale Gradientensumme wird durch die Gesamtzahl effektiver Tokens/Samples global_weight geteilt, nicht durch Rankanzahl oder ein Mittel lokaler Mittelwerte. reduce_fp32 erhält FP32 für Half-Parameter; normales reduce castet am Ende zum Speicher-Dtype. Alle Ranks wählen dieselbe Variante.

MissingGradientPolicy::Error verlangt Gradienten bei lokaler Gewichtung>0; Zero trägt für fehlende Gradienten null bei. Global ungenutzte Parameter bleiben außen vor. Scheduler, Clipping, Optimizer und Reset sind nicht implizit. Ungültige Modell-/ID-/Frozen-Konfigurationen und abweichende Reduktionsoptionen sind Vertragsfehler.

## Checkpoint und Neustart

An derselben abgeschlossenen Grenze pro Rank Modell/Optimizer/Scheduler, verwendete Akkumulation, Daten/Sampler-Position und RNG speichern. TrainingRecord wählt keinen konsistenten Mehr-Rank-Snapshot automatisch. Alle Ranks aus derselben Grenze wiederherstellen, lokale Modell-IDs mit ihrem Optimizer-Record paaren, keinen neuen Base über alte Momente broadcasten. World/Device/Source/Transport gehören zur externen Laufkonfiguration.

```bash
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- run ../ruda-collective-state
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- resume ../ruda-collective-state
```

run verlangt ein neues Verzeichnis und speichert nach Step1, resume führt von dort Step2 aus. Unverändert verwenden beide Ranks dasselbe Gerät; das ist kein Mehr-GPU-Benchmark. Fehler nicht zu stiller Wiederholung nur eines Ranks umwandeln. Vor langen Läufen reale Ressourcen, Arbeitsmenge, Checkpoint-Wiederaufnahme und sichtbaren persistenten Fortschritt vorbereiten. Ergebnisse/Checkpoints bleiben außerhalb Git.
