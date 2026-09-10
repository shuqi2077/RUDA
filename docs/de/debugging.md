# Debugging und Diagnose

[English](../en/debugging.md) | [简体中文](../zh/debugging.md) | [日本語](../ja/debugging.md) | **Deutsch** | [Русский](../ru/debugging.md)

[Dokumentation](README.md) · [Compiler](compiler-guide.md) · [Laufzeit API](runtime-api.md) · [中文](../zh/debugging.md)

## 1. Suchen Sie die fehlerhafte Phase

|Stadium oder Symptom|Zuerst prüfen|
| --- | --- |
|Cargo-Manifest oder Fehler bei der Abhängigkeitsauflösung|Pfade, Funktionen, Sperrdatei und Abhängigkeitscache|
|Rust-Build-Fehler|Erster Compilerfehler, Toolchain- und Funktionskombination|
|CUDA Toolkit-Suchfehler|CUDA_PATH, Installationsverzeichnis und Header|
|Treiber- oder Geräteinitialisierungsfehler|Treiberverfügbarkeit, Geräteindex und dynamisches Laden der Bibliothek|
|Direkter PTX-Konfigurationsfehler|Direct-PTX-Funktion, Compiler-Auswahl und PTX-Version|
|Direkter PTX-Kompilierungsfehler|Im Fehler genannter nicht unterstützter Vorgang, nicht unterstütztes Ziel oder Argumentlayout|
|Rücklese- oder Synchronisierungsfehler|Frühere asynchrone Übermittlungen, Eingabebindungen und Gerätefehler|
|Falsche numerische Ergebnisse|Form, Schritte, dtype, Grenzen, Synchronisation und Algorithmusverträge|

Behalten Sie den ersten Fehler und seinen Kontext bei, nicht nur die endgültige Zusammenfassung der Buildfehler.

## 2. Kompilierungs- und Cache-Protokolle

Die Laufzeitkonfiguration liegt in `ruda::runtime::config`. `CompilationConfig` stellt logger, cache und check_mode bereit. `CompilationLogLevel` wird als disabled, basic oder full serialisiert; full enthält Kompilierungsinformationen auf Quelltextebene.

Das Beispiel `ptx-runtime` zählt Kompilierungen und PTX Festplatten-Cache-Treffer. `RUDA_PTX_TEST_CACHE` wird nur von diesem Beispiel gelesen, nicht automatisch von jeder Anwendung.

Siehe [compilation.rs](../../ruda/src/runtime/config/compilation.rs) und [ptx_runtime.rs](../../ruda-driver-cuda/examples/ptx_runtime.rs).

## 3. Grenzenprüfung

`BoundsCheckMode` hat drei Konfigurationen:

|Konfiguration|Laufzeitverhalten|
| --- | --- |
| auto |Regelmäßige Starts verwenden Überprüfungen; Bei expliziten, ungeprüften Starts werden sie möglicherweise übersprungen|
| enforce |Erzwingt Prüfungen für Starts|
| validate |Regelmäßige Starts behalten Schecks bei; Bei ungeprüften Pfaden wird der Validierungsmodus ausgewählt|

Die Erkennungsmöglichkeiten hängen von Compiler und Backend ab. Diese Modi erkennen nicht automatisch jeden Zugriff außerhalb der Grenzen, jede Race Condition oder jeden Lebensdauerfehler. Deaktivieren Sie Prüfungen nicht allein, um einen fehlerhaften Fall weiterlaufen zu lassen.

## 4. Asynchrone Fehler

`ComputeClient::launch` gibt keine berechneten Ergebnisse zurück. Beobachten Sie den Abschluss durch Rücklesen, das einen `Result` zurückgibt, oder durch Warten auf die Synchronisierung. Unterscheiden Sie zwischen Argumentvalidierungs-, Kernel-Kompilierungs- und Geräteausführungsfehlern.

## 5. Reproduzieren Sie ein Problem

Die [Beispielseite](samples.md) listet Shared-Memory-Grenzwerte, Tensor-, bitweise und Fälle mit halber Genauigkeit auf.

Geben Sie beim Melden eines Problems die Quellversion, Funktionen, den vollständigen Befehl, das Backend, die PTX-Version, GPU/Treiber/Toolkit, Eingaben und tatsächliche Ausgaben an. Entfernen Sie vertrauliche Informationen und befolgen Sie die [Mitwirkende Anleitung](CONTRIBUTING.md).
