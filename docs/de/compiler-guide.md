# Compiler-Handbuch

[English](../en/compiler-guide.md) | [简体中文](../zh/compiler-guide.md) | [日本語](../ja/compiler-guide.md) | **Deutsch** | [Русский](../ru/compiler-guide.md)

[Dokumentation](README.md) · [PTX-Referenz](ptx.md) · [Programmierhandbuch](programming-guide.md) · [中文](../zh/compiler-guide.md)

## 1. Kompilierungspipeline

Der allgemeine Kernel-Pfad beginnt mit den Makros und Typen in `ruda-kernel::dsl`, erzeugt Kernel-IR und nutzt ein Backend für Lowering und Codegenerierung. `ruda-compiler` enthält die Compilerimplementierungen; Gerätetreiber übergeben deren Ausgabe an die Ausführungsumgebung.

NVIDIA bietet zwei Kompilierungspfade:

|Auswahl|Kernel-Kompilierungspipeline|
| --- | --- |
|`nvrtc` (Standard)|Rust-Kernel-Frontend → IR → CUDA C++ → NVRTC → PTX|
|`ptx`|Rust-Kernel-Frontend → IR → PTX|

Beide werden über den NVIDIA-Treiber ausgeführt. Der C++-Kompilierungspfad CUDA ist keine Schnittstelle zum Importieren beliebiger C++-Projekte oder zur Bereitstellung vollständiger CUDA-Quellkompatibilität.

## 2. Cargo-Features

|Komponente/Funktion|Zweck|
| --- | --- |
|`ruda-kernel/frontend`|Kernel DSL-Frontend|
|`ruda-kernel/lowering-cpp`| Integration des C++-Lowerings |
|`ruda-compiler/cpp`|C++-Backend-Implementierung|
|`ruda-compiler/ptx`|Direkter PTX-Compiler|
|`ruda-driver-cuda/direct-ptx`|Ermöglicht die direkte PTX-Auswahl im CUDA-Backend|

Funktionen bestimmen, welcher Code erstellt wird; Umgebungsvariablen wählen zur Laufzeit einen Pfad aus. Dies sind separate Steuerelemente. Siehe das [CUDA-Manifest](../../ruda-driver-cuda/Cargo.toml) und das [Compiler-Manifest](../../ruda-compiler/Cargo.toml).

## 3. Umgebungsvariablen

|Variable|Verhalten|
| --- | --- |
|`RUDA_CUDA_COMPILER`|Akzeptiert `nvrtc` oder `ptx`; Der Standardwert ist nvrtc, wenn dieser Wert nicht festgelegt ist|
|`RUDA_PTX_VERSION`|Direct PTX erfordert eine explizite `major.minor`-Version|
|`CUDA_PATH`|CUDA Toolkit-Installationsstammverzeichnis|
|`RUDA_PTX_TEST_CACHE`| Cache-Verzeichnis-Override, das nur das Beispiel ptx-runtime ausliest |

Unbekannte Compilerwerte führen zu einem Fehler. Auch die Auswahl von `ptx` ohne Aktivierung von `direct-ptx` führt zu einem Fehler, anstatt zu NVRTC zu wechseln. Siehe [compiler_backend.rs](../../ruda-driver-cuda/src/compiler_backend.rs).

## 4. Ziele und Caches

Die eigenständige direkte PTX-Kompilierung erfordert eine explizite PTX-Version und ein explizites SM-Ziel. PTX identifiziert die Befehlssatzversion; SM identifiziert die Zielarchitektur. Sie sind nicht austauschbar.

Der direkte PTX-Cache-Namespace des CUDA-Treibers umfasst die Backend-ID SM und die PTX-Version und ist vom NVRTC-Cache getrennt.

## 5. Kompilierungsfehler

Nicht unterstütztes IR, Argumentmetadaten oder Zielbedingungen führen zu Fehlern. Direct PTX meldet nicht unterstützte Vorgänge, anstatt den Kompilierungspfad automatisch zu wechseln. Siehe [Debugging](debugging.md).

`ruda-compiler` enthält außerdem die Module WGSL, SPIR-V und MLIR.
