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

Definitionen: [PTX-Modul](../../ruda-compiler/src/ptx/mod.rs), [Compiler-Tests](../../ruda-compiler/src/ptx/tests.rs) und [Laufzeit-Backend-Auswahl](../../ruda-driver/cuda/src/compiler_backend.rs).

## 5. Packed-FP4-Transport und NVFP4-Scaled-MMA

`StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)` speichert zwei FP4-Werte pro Byte. Direktes PTX unterstützt bit-erhaltendes Kopieren, Indexieren, Select, Reinterpret und Speichertransport. Ein gepacktes Byte ist kein skalarer FP4-Rechenwert; nicht unterstützte Arithmetik oder Konvertierung führt zum Fehler. Reiner Transport benötigt kein `sm_120a`-Ziel.

`CoopMma::ExecuteScaled` akzeptiert nur SM 120 mit PTX ab 8.7, ein `16×8×64`-Tile, gepacktes E2M1x2 für A/B, F32 für C/D, E4M3-Skalen und `scales_factor = 4`. Es erzeugt `mxf4nvf4.block_scale.scale_vec::4X` und setzt das Ausgabeziel auf `sm_120a`. Andere Architekturen, Tile-Formen, Skalenformate oder Faktoren werden abgelehnt; ausreichende Fragmentregister sind ebenfalls erforderlich.

[Packed-Verarbeitung](../../ruda-compiler/src/ptx/packed.rs), [Scaled-MMA](../../ruda-compiler/src/ptx/matrix/scaled.rs), [Transport-/Layoutbeispiel](../../ruda-driver/cuda/examples/ptx_packed.rs). Das Beispiel liest exportierte PTX-Dateien und prüft Bytetransport sowie Fragmentkoordinaten, führt aber keine NVFP4-MMA-Arithmetik aus.

## 6. Automatisches Lowering und explizite Layoutwahl

U32/U64-Division und -Rest mit einer zur Kompilierzeit bekannten, von null verschiedenen Konstanten nutzen exakte ganzzahlige Stärkereduktion über den gesamten vorzeichenlosen Eingabebereich. Zweierpotenzen verwenden Shift oder Maske; geeignete andere Divisoren das obere Produktwort und Shift; große Divisoren mit Quotient null oder eins einen Vergleich. Die übrigen verwenden reziproke Multiplikation und eine ganzzahlige Korrektur. Der Rest kann durch Multiply-Add mit dem unteren Produktwort entstehen. Eingabe- und Ausgaberegister dürfen identisch sein. Dies geschieht automatisch bei IR-Konstanten, nicht anhand von Operatornamen oder festen Formen, ohne zusätzliche Einschränkung des Zählers oder Gleitkommakonvertierung. Dynamische Divisoren, Nullteiler und vorzeichenbehaftete Division behalten ihr bisheriges Lowering. Siehe [Ganzzahl-Lowering](../../ruda-compiler/src/ptx/constant_division.rs).

Weitere automatische direkte PTX-Pfade:

- Paarweise FP16/BF16-Arithmetik und Konvertierungen, wenn Typ und Ziel dies unterstützen; paarweise FP32-Addition, -Subtraktion, -Multiplikation und explizites FMA auf SM100+ mit PTX 8.6+. Getrennte Multiplikation und Addition behalten getrennte Rundung; reduzierte Genauigkeit wird nicht implizit aktiviert.
- Packed-/Vektorbefehle für zusammenhängende Vektorzugriffe, mit Ausrichtungsprüfung und skalarem Fallback bei nicht bewiesener Ausrichtung. Globale Zugriffe erlauben bis zu 32 Byte pro Befehl auf SM100+ mit PTX 8.8+, sonst bis zu 16 Byte auf unterstützten Zielen. Beliebige Slices sind dadurch nicht automatisch ausgerichtet.
- Warp-Reduktionen lassen dynamische Teilnahmemasken nur weg, wenn Blockgeometrie und Kontrollfluss die Teilnahme beweisen. Partielle oder divergente Warps behalten den Active-Lane-Pfad.

Die [Zeilenlayout-Helfer](../../ruda-compiler/src/ptx/row.rs) `compile_row_variants` und `select_row_variant` kompilieren vom Aufrufer bereitgestelltes IR und wählen Varianten über dessen Korrektheits- und Zeitmessungs-Callback. Sie erkennen keinen Operator anhand seines Kernelnamens und führen beim normalen Kompilieren keine Benchmarks aus. Layout und Genauigkeit sind getrennte Entscheidungen; `FastMath::ReducedPrecision` muss explizit angefordert werden.

PTX ist kein finaler Maschinencode. NVIDIA-Assemblierung kann Befehlsanzahl und Registerbelegung ändern; weniger Befehle garantieren keine kürzere Laufzeit. Vergleichen Sie assemblierten Code und Gerätemessungen für den Zielworkload. Kompilierung oder Assemblierung allein ist kein Leistungsnachweis. Die standardmäßige NVRTC-Auswahl sowie Start- und Argumentverträge bleiben gleich.
