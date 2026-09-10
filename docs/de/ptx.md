# PTX Backend-Referenz

[English](../en/ptx.md) | [简体中文](../zh/ptx.md) | [日本語](../ja/ptx.md) | **Deutsch** | [Русский](../ru/ptx.md)

[Dokumentation](README.md) · [Compiler-Anleitung](compiler-guide.md) · [Beispiele](samples.md) · [中文](../zh/ptx.md)

## 1. Geltungsbereich

`ruda_compiler::ptx` generiert PTX-Text aus dem Ruda-Kernel IR. Es ist weder ein Interpreter für beliebige PTX-Programme noch ein Generator des endgültigen NVIDIA-Maschinencodes.

Aktivieren Sie `ruda-compiler/ptx` für die eigenständige Kompilierung. Um diesen Pfad über die CUDA-Laufzeit auszuwählen, aktivieren Sie `ruda-driver-cuda/direct-ptx`.

## 2. Zieltypen

|Typ/Feld|Bedeutung|
| --- | --- |
|`PtxTarget::version: (u32, u32)`|PTX Haupt-/Nebenversion|
|`PtxTarget::sm: u32`|SM Ziel|
|`PtxCompilationOptions::target: Option<PtxTarget>`|Explizite Zielkonfiguration|
|`PtxCompiler`|Direktes Backend, das das öffentliche Compiler-Merkmal implementiert|

Die Kompilierung gibt einen Validierungsfehler zurück, wenn `target` nicht vorhanden ist. Der eigenständige Compiler leitet die GPU-Architektur nicht von seinem Hostcomputer ab.

Das Parsen von Umgebungsvariablen akzeptiert `major.minor` mit einer Hauptversion von mindestens 6 und einer Nebenversion von nicht mehr als 9. Syntaktische Gültigkeit allein stellt keine Treiber- oder Generatorunterstützung für diese Version dar.

## 3. Kompilierungsausgabe

`PtxKernel` enthält:

- `source`: PTX Text.
- `entrypoint`: Name des Einstiegspunkts.
- `ruda_dim`: die Arbeitsgruppendimensionen des ursprünglichen Kernels.
- `shared_memory_bytes`: Erforderlicher gemeinsamer Speicher.
- `dynamic_metadata_index`: die Argumentposition des dynamischen Metadatenzeigers, falls erforderlich.

Behalten Sie beim Aufruf der Ausführungsschicht das Argumentlayout, den Einstiegspunkt und die Anforderungen an den gemeinsamen Speicher bei. Der Text allein enthält nicht den vollständigen Startvertrag.

## 4. Fehlerbehandlung

Nicht unterstützt. IR gibt `CompilationError::UnsupportedInstruction` zurück. Eine ungültige Konfiguration oder Struktur gibt `CompilationError::Validation` zurück. Die Diagnose umfasst das Präfix `Direct PTX:`. Das Backend greift nicht automatisch auf NVRTC zurück.

Definitionen: [PTX-Modul](../../ruda-compiler/src/ptx/mod.rs), [Compiler-Tests](../../ruda-compiler/src/ptx/tests.rs) und [Laufzeit-Backend-Auswahl](../../ruda-driver-cuda/src/compiler_backend.rs).
