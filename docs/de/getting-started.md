# Installation und Schnellstart

[English](../en/getting-started.md) | [简体中文](../zh/getting-started.md) | [日本語](../ja/getting-started.md) | **Deutsch** | [Русский](../ru/getting-started.md)

[Dokumentation](README.md) · [Weiter: Programmieranleitung](programming-guide.md) · [中文](../zh/getting-started.md)

## 1. Wählen Sie Ihren Einstiegspunkt

|Aufgabe|Einstiegspunkt|
| --- | --- |
|GPU-Kernel schreiben|`ruda-kernel::dsl` und eine Gerätelaufzeit|
|Verwenden Sie Matrixmultiplikation, FFTs oder Reduktionen|[Computerbibliotheken](libraries/README.md)|
|Arbeiten Sie mit Tensoren und Frameworks|[Tensoren und Frameworks](tensor-framework.md)|
|Modelle trainieren und Status speichern|[Schulungsleitfaden](training.md)|
|Laden Sie Modelle und generieren Sie Texte oder Prozessbilder|[Modellinferenzleitfaden](model-inference.md)|
|Integrieren Sie ein Geräte-Backend|[Treiber API](driver-api.md)|

Verwenden Sie Ruda aus dem Quellarbeitsbereich.

## 2. Bereiten Sie eine NVIDIA-Umgebung vor

Sie benötigen Rust/Cargo, eine Linker-Toolchain für Ihre Plattform, einen NVIDIA GPU-Treiber und das CUDA Toolkit. Das CUDA-Backend umfasst NVRTC- und Toolkit-Build-Abhängigkeiten; Durch die Aktivierung des direkten PTX werden sie nicht entfernt.

Überprüfen Sie Ihre Umgebung vom Quellstammverzeichnis aus:

```powershell
rustc --version --verbose
cargo --version
nvidia-smi
nvcc --version
cargo metadata --no-deps --format-version 1 --offline --locked
```

Diese Befehle kompilieren Ruda nicht. `--offline` erfordert, dass die für die Auflösung erforderlichen Abhängigkeiten lokal zwischengespeichert werden.

Legen Sie `CUDA_PATH` fest, um das CUDA Toolkit-Stammverzeichnis auszuwählen. Verweisen Sie unter Windows auf ein installiertes Versionsverzeichnis und nicht auf das übergeordnete Verzeichnis, das mehrere Versionen enthält. Siehe die [CUDA-Installationspfadschnittstelle](../../ruda-driver-cuda/src/lib.rs).

## 3. Erstellen Sie das Beispiel

Wenn die Build-Umgebung bereit ist, führen Sie Folgendes aus:

```powershell
cargo build --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

Das Beispiel `ptx-runtime` erfordert `direct-ptx`. Durch die alleinige Aktivierung dieser Funktion wird der Standardcompiler nicht geändert.

## 4. Wählen Sie einen Kompilierungspfad und führen Sie ihn aus

Wählen Sie einen Pfad in einer separaten PowerShell-Sitzung. Beheben Sie alle Befehlsfehler, bevor Sie fortfahren.

Standard-CUDA C++/NVRTC-Pfad:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

Direkter PTX-Pfad:

```powershell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

Die PTX-Version muss mit Ihrem Ziel-GPU und Treiber übereinstimmen. Siehe die [PTX Backend-Referenz](ptx.md).

Das Beispiel führt die FP32-Addition in mehreren Längen durch, prüft Ergebnisse und Tail-Sentinels über wiederholte Ausführungen hinweg und gibt Cache-Zähler aus. Siehe [Beispiele und Tutorials](samples.md) für optionale Fälle.

## 5. Entwickeln Sie weiter

Folgen Sie der Geräteauswahl, dem Daten-Upload und dem Kernel-Start im Beispiel und lesen Sie dann die [Programmieranleitung](programming-guide.md). Bei Build-, Treiberlade- oder Ausführungsfehlern verwenden Sie [Debugging und Diagnose](debugging.md), um die fehlerhafte Phase zu lokalisieren.
