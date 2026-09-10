# ruCCL

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

**Englisch** | [简体中文](../zh/README.md)

Dieses Repository ist ein Quellspiegel. Führen Sie die folgenden Befehle im Stammverzeichnis [RUDA monorepo](https://github.com/shuqi2077/RUDA) aus.

Rudas kollektive Kommunikationsbibliothek. Die öffentliche Tensorschnittstelle verwendet Ruda-Tensor-Backends und Rechenbibliotheken wieder. `rank` und `in_process` bieten geräteunabhängige Kommunikationsprotokolle, Zeitplanung und Geräteadapterverträge.

## CUDA Tensorbeispiel

```sh
cargo run -p ruCCL --features cuda --example all_reduce
```

Erfordert einen funktionierenden NVIDIA-Treiber und ein CUDA-Toolkit. Das Beispiel erstellt vier logische Ränge auf GPU 0, führt Tensorberechnungen und Ring AllReduce durch, liest Ergebnisse zurück und schließt die Sitzung. Es überprüft die Summe/den Mittelwert über 257 FP32-Elemente und stellt sicher, dass die ursprünglichen Eingaben unverändert bleiben.

## Features

| Feature |Geltungsbereich|
| --- | --- |
|Standard|Allgemeiner Kommunikationskern und Tensor-Backend-Schnittstellen; aktiviert nicht automatisch ein GPU-Backend|
|`cuda`|CUDA Tensor-Backend, behält seine Standard-Fusion- und Tuning-Konfiguration bei|
|`test-cuda`|Wählt zusätzlich zu `cuda` das vorhandene Test-Backend CUDA aus|
|`test-wgpu` / `test-metal` / `test-vulkan`|Vorhandene WGPU-Testeinstiegspunkte; werden separat von den CUDA-Testfunktionen ausgeführt|
|`tracing`|Bestehende schichtübergreifende Tracing-Integration|

Der öffentliche Tensor API umfasst `register`, `all_reduce`, `reduce`, `broadcast` und `finish_collective`. Alle Ränge müssen übereinstimmende Sammeloperationen in derselben Reihenfolge aufrufen. Autodiff-Aufrufer verwenden das innere Backend; Die Optimierungsschicht übernimmt die Gradientensynchronisierung.

## ruCCL Benutzerhandbuch

[Computerbibliotheken](../../../docs/de/libraries/README.md) · [Tensoren und Frameworks](../../../docs/de/tensor-framework.md) · [中文](../zh/README.md)

### 1. Ebenen und Einstiegspunkte

Das Cargo-Paket ist `ruCCL` und die Rust-Crate ist `ruccl`.

ruCCL umfasst Tensor-Backend-Kollektive, einen Rank-Kern und In-Process-Implementierungen. `ruda-communication` stellt Kommunikationsinfrastruktur bereit. Die `orchestrator`-Funktion ermöglicht Orchestrierungseinstiegspunkte.

### 2. Tensorkollektiv API

|Funktion|Verhalten|
| --- | --- |
|`register<B>`|Registriert einen Peer, ein Gerät und CollectiveConfig|
|`all_reduce<B>`|Gibt das reduzierte Ergebnis an die Teilnehmer zurück|
|`broadcast<B>`|Absender übergibt Some(tensor); Empfänger passieren Keine|
|`reduce<B>`|Reduziert auf eine angegebene Wurzel; Nicht-Root-Teilnehmer erhalten „Keine“.|
|`finish_collective<B>`|Beendet die gemeinsame Sitzung des Peers|
|`reset_collective<B>`|Setzt den lokalen Sammeldienst zurück und verwirft Registrierungen und den Status des laufenden Vorgangs|

-Schnittstellen verwenden `B: ruda_tensor::Backend` und `B::FloatTensorPrimitive`. Registrieren Sie bei der Integration mit automatischer Differenzierung das innere Backend; ein Sammelaufruf definiert selbst keine automatische Rückwärtsregel.

### 3. Registrierung und Anrufverträge

Konfiguration mit `CollectiveConfig::default()` erstellen. Verwenden Sie `with_num_devices` für die Anzahl der lokal teilnehmenden Geräte. Konfigurieren Sie Strategien und Multinode-Adressen über ihre Konfigurationsmethoden.

Die Teilnehmer müssen sich auf die Anzahl der Geräte einigen, eindeutige Peer-IDs verwenden und übereinstimmende Kollektive in derselben Reihenfolge aufrufen. Form, Reduktionsoperation, Wurzel und andere Parameter müssen übereinstimmen. Jede Sendung muss genau einen Absender haben.

Konfigurieren Sie für die Ausführung mit mehreren Knoten die Anzahl der Knoten, globale und lokale Adressen sowie Datendienst-Ports gemeinsam.

### 4. Fehler und Lebenszyklus

`CollectiveError` deckt doppelte oder fehlende Registrierungen, Formkonflikte, inkonsistente Reduktionsoperationen oder Roots sowie ungültige Broadcast-Absenderzahlen ab.

Verwenden Sie `finish_collective` für den regulären Abschluss. `reset_collective` verwirft laufenden Zustand; es schließt keine Operation ab, erstellt keinen Checkpoint einer Geräteaufgabe und bietet keine verlustfreie Wiederherstellung.

### 5. Beispiel CUDA

Die Funktion `cuda` aktiviert das Tensor-Backend CUDA. Führen Sie `cargo run --locked -p ruCCL --features cuda --example all_reduce` aus, um Ring AllReduce mit vier logischen Rängen auf GPU 0 auszuführen. Es überprüft die Summe/den Mittelwert über 257 FP32-Elemente, die Beibehaltung der Eingaben und den Sitzungsausgang.

Geräteadapter befinden sich in [tensor_device](../../src/tensor_device). Informationen zur Optimierungsschnittstelle finden Sie unter [explizite Ranggradientenreduzierung](../../../ruda-optim/src/optim/grads/collective.rs). Übertragungen umfassen einen vom Host bereitgestellten Pfad, keine Nullkopie P2P.

Quelle: [kollektiv API](../../src/api.rs), [Konfiguration](../../src/config.rs), [Rang](../../src/rank/mod.rs) und [In-Prozess-Implementierung](../../src/in_process/mod.rs).

### 6. Kollektive Schulung

Aktivieren Sie `collective` in `ruda-optim`. Konvertieren Sie mit einem explizit besessenen Rangkommunikator Rückwärtsverläufe in `GradientsParams`, rufen Sie `grads.all_reduce_with::<InnerBackend>(&communicator, ReduceOperation::Mean)?` auf und übergeben Sie dann die zurückgegebenen Verläufe an `optimizer.step`. Parameter-IDs, Verlaufsformen, D-Typen und Aufrufreihenfolge müssen über alle Ränge hinweg übereinstimmen. Für das Autodiff-Training ist `InnerBackend` das Backend ohne den `Autodiff`-Wrapper.

Führen Sie das zweistufige Trainingsbeispiel aus dem Quellbaum aus:

```powershell
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- run ./collective-training-state
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- resume ./collective-training-state
```

`run` erfordert ein Verzeichnis, das noch nicht existiert. Es speichert das Modell und den Optimierer jedes Rangs nach der ersten Aktualisierung und führt dann die zweite Aktualisierung aus. `resume` stellt dieses Verzeichnis wieder her und führt das zweite Update aus. Wenn CUDA aktiviert ist, verwenden beide logischen Ränge in diesem Beispiel dasselbe Standardgerät.

Die vollständige Anrufsequenz finden Sie im [Beispiel für ein gemeinsames Training](../../../ruda-optim/examples/collective_training.rs). Um auch den Planerstatus und die ausstehenden akkumulierten Verläufe zu speichern, verwenden Sie `TrainingRecord` aus [Trainings- und Speicherstatus](../../../docs/de/training.md).
