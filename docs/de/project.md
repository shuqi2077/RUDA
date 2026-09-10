# Ruda — Hochleistungsrechnen mit Rust

![Rust](https://img.shields.io/badge/Rust-2024_Edition-orange?logo=rust&logoColor=white)
![Sprache](https://img.shields.io/github/languages/top/shuqi2077/RUDA)
![Forks](https://img.shields.io/github/forks/shuqi2077/RUDA?style=flat)
![Issues](https://img.shields.io/github/issues/shuqi2077/RUDA)
![Letzter Commit](https://img.shields.io/github/last-commit/shuqi2077/RUDA?display_timestamp=committer)

[English](../../README.md) | [简体中文](../zh/project.md) | [日本語](../ja/project.md) | **Deutsch** | [Русский](../ru/project.md)

Ruda ist eine Rust-Bibliothek für Hochleistungsrechnen. Sie baut einen vollständigen Software-Stack auf, von GPU-Kernels, Compilern und Laufzeitumgebungen bis hin zu mathematischen Berechnungen, Tensoren und Modellen.

Ruda entwickelt Rust-Kompilierungs- und Ausführungspfade für PTX, HIP und eigene ISAs und behält zugleich den CUDA-C++-Kompilierungspfad bei. Kontrollierte `unsafe`-Kapselung auf niedriger Ebene verbindet sich mit Rusts Typsystem, Ownership und Borrowing auf höheren Ebenen und ermöglicht sowohl hardwarenahe Leistungskontrolle als auch Speichersicherheit auf höherer Ebene.

## Schnellstart

Erforderlich sind Git, Rust/Cargo, eine Linker-Toolchain, eine NVIDIA-GPU samt Treiber und das CUDA Toolkit. Installationsdetails stehen unter [Umgebung einrichten](getting-started.md).

### Klonen

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
```

### Einen GPU-Kernel ausführen

Wählen Sie in Ihrer Shell den direkten PTX-Compiler:

```sh
# Bash
export RUDA_CUDA_COMPILER=ptx
export RUDA_PTX_VERSION=8.0
```

```powershell
# PowerShell
$env:RUDA_CUDA_COMPILER = 'ptx'
$env:RUDA_PTX_VERSION = '8.0'
```

Erstellen und starten Sie anschließend das Beispiel:

```sh
cargo run --release --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime
```

Das Beispiel führt FP32-Addition auf der GPU aus und gibt `PASS`-Zeilen sowie Zähler des Kompilierungscaches aus. Wählen Sie eine von GPU und Treiber unterstützte [PTX-Version](ptx.md).

### Text mit ruLLM erzeugen

Legen Sie ein lokales Qwen3.5-0.8B-Modell unter `./models/qwen35` ab oder ersetzen Sie den folgenden Pfad durch Ihr Modellverzeichnis. Modelldateien sind nicht enthalten; siehe [Modell vorbereiten](model-inference.md#bereiten-sie-ein-lokales-modell-vor).

```sh
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

Das Beispiel gibt den erzeugten Text und die Token-IDs aus. Um stattdessen den CUDA-C++-/NVRTC-Pfad zu verwenden, setzen Sie `RUDA_CUDA_COMPILER` vor dem Start des jeweiligen Beispiels auf `nvrtc`.

## Aufbau des Stacks

Ein Repository, mehrere Crates mit klar abgegrenzten Zuständigkeiten. Von Fachbibliotheken bis zu übergeordneten Frameworks ist der Stack in Schichten gegliedert und wird gemeinsam entwickelt.

| Schicht | Komponenten |
| --- | --- |
| Gemeinsame Schnittstellenverträge | `ruda-core` |
| Kompilierung und Kernels | `ruda-compiler`, `ruda-kernel`, Makrokomponenten |
| Laufzeit und Treiber-Backends | `ruda`, `ruda-driver-cuda/cpu/wgpu/hip` |
| Fachbibliotheken | ruBLAS, ruDNN, ruPRIM, ruFFT, ruRAND, ruSPARSE |
| Kollektive Kommunikation | ruCCL, `ruda-communication` |
| Tensoren und Frameworks | `ruda-tensor*`, `ruda-autodiff`, `ruda-fusion` |
| Modelle und Daten | `ruda-model`, `ruda-nn`, `ruda-optim`, `ruda-store`, `ruda-dataset` |

## Wege zur Hardware

- **NVIDIA-GPUs:** CUDA C++ → NVRTC → PTX ist der standardmäßige Kompilierungspfad. Auch die direkte Erzeugung IR → PTX kann ausdrücklich gewählt werden. Beide Pfade werden über den NVIDIA-Treiber ausgeführt.
- **Weitere Ausführungs-Backends:** Backend-Quellcode für CPU, WGPU und HIP ist vorhanden. Den Unterstützungsumfang beschreibt [Kompatibilität](compatibility.md).

## Erkunden und beitragen

- [Ruda-Dokumentation](README.md): Schnellstart, Programmierleitfäden, Compiler, API-Referenzen und Handbücher der Rechenbibliotheken.
- [NVIDIA-Demo](getting-started.md): Das Beispiel und seine Voraussetzungen kennenlernen.
- [Beitragsleitfaden](CONTRIBUTING.md): Zu Operatoren, Compilern, Laufzeitumgebungen und Frameworks beitragen.

Wenn Sie sich für Rust, GPU-Kernels, Compiler oder Hochleistungsrechnen interessieren, helfen Sie mit, diesen Stack weiterzuentwickeln und schneller zu machen.

## Herkunft und Lizenzierung

[Hinweise zu Drittanbietern](../../THIRD_PARTY_NOTICES.md)

Originärer Ruda-Softwarecode, für den das Projekt das Recht zur Lizenzvergabe besitzt, steht unter der [Apache License 2.0](../../LICENSE). Dateien Dritter behalten ihre ursprünglichen Lizenzen; die Lizenz im Stammverzeichnis setzt die `MIT OR Apache-2.0`-Deklarationen übernommener Komponenten nicht außer Kraft.
