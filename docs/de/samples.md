# Beispiele und Tutorials

[English](../en/samples.md) | [简体中文](../zh/samples.md) | [日本語](../ja/samples.md) | **Deutsch** | [Русский](../ru/samples.md)

[Dokumentation](README.md) · [Schnellstart](getting-started.md) · [中文](../zh/samples.md)

## 1. NVIDIA Laufzeitbeispiel

Einstiegspunkt: [ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs).

Im Basisfall wird die FP32-Addition zweimal bei jeder Länge durchgeführt: 1, 63, 64, 65 und 257. Dabei werden jedes Ergebnis und 16 Tail-Sentinel-Werte überprüft. Es demonstriert Geräteauswahl, Upload, Argumentbindung, Tail-Bounds, Readback und wiederholte Ausführung.

Informationen zum Erstellen und Auswählen eines Backends finden Sie im [Schnellstart](getting-started.md).

## 2. Gezielte Fälle

Platzieren Sie diese Argumente nach `--` des Befehls. Führen Sie den Tensorfall beispielsweise aus mit:

```powershell
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --tensor
```

|Argument|Verhalten|Beispielquelle|
| --- | --- | --- |
|`--tensor`|Tensor-Metadaten- und Layoutprüfungen|[tensor.rs](../../ruda-driver-cuda/examples/ptx_runtime/tensor.rs)|
|`--shared`|Prüfungen des gemeinsam genutzten Speichers|[shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs)|
|`--half`|FP16/BF16 prüft|[half_precision.rs](../../ruda-driver-cuda/examples/ptx_runtime/half_precision.rs)|
|`--bitwise`|Bitweise Operationsprüfungen|[bitwise.rs](../../ruda-driver-cuda/examples/ptx_runtime/bitwise.rs)|
|`--shared-over-limit`|Diagnose der Grenze des gemeinsam genutzten Speichers|[shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs)|
|`--expect-cold`|Die Kompilierung der Asserts erfolgte ohne einen Festplatten-Cache-Treffer|[Haupteinstiegspunkt](../../ruda-driver-cuda/examples/ptx_runtime.rs)|
|`--expect-warm`|Stellt einen Festplatten-Cache-Treffer ohne Neukompilierung fest|[Haupteinstiegspunkt](../../ruda-driver-cuda/examples/ptx_runtime.rs)|

`--shared-over-limit` verwendet einen separaten Zweig mit vorzeitiger Rückkehr. Kombinieren Sie es nicht mit Kalt-/Warm-Cache-Prüfungen.

## 3. Kalte und warme Caches

Verwenden Sie für jeden Kompilierungspfad ein separates neues Cache-Verzeichnis. Verwenden Sie dieses Verzeichnis für die Kalt- und Warmläufe desselben Pfads erneut.

Nachdem Sie einen Kompilierungspfad ausgewählt haben, wie in [Erste Schritte](getting-started.md) beschrieben, führen Sie diese Befehle in derselben PowerShell-Sitzung aus:

```powershell
$env:RUDA_PTX_TEST_CACHE = 'target/ptx-example-cache-' + [guid]::NewGuid().ToString('N')
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-cold
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-warm
```

Lassen Sie den Compiler, die Eingabeargumente und den Cache-Pfad zwischen den Läufen unverändert.

## 4. Beispiele für Compute-Bibliotheken

Bereiten Sie die [Build-Umgebung](getting-started.md) vor, bevor Sie Beispiele ausführen.

|Aufgabe|Befehl|Ausgabeprüfungen|
| --- | --- | --- |
|FP32 CSR Matrix-Vektor-Multiplikation|`cargo run --locked -p ruSPARSE --features cuda --example csrmv`|Liest zurück und prüft `[7.0, 2.0, 18.5]`|
|CUDA Ring AllReduce|`cargo run --locked -p ruCCL --features cuda --example all_reduce`|Vier logische Ränge für GPU 0, 257 Elemente, Summe/Mittelwert und Eingabeerhaltung|

## 5. Training und Modellinferenz

- [Trainings- und Speicherstatus](training.md): Vorwärts-, Rückwärts-, Gradientenakkumulation und Trainingsaufzeichnungen.
- [Modellladen und Inferenz](model-inference.md): Qwen2/Qwen3.5-Beispielbefehle, Chat, Sampling, AWQ und Bildeingabe.
