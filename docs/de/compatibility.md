# Kompatibilitätsleitfaden

[English](../en/compatibility.md) | [简体中文](../zh/compatibility.md) | [日本語](../ja/compatibility.md) | **Deutsch** | [Русский](../ru/compatibility.md)

[Dokumentation](README.md) · [Programmierleitfaden](programming-guide.md) · [中文](../zh/compatibility.md)

## 1. CUDA-Konzepte

| Vertrautes CUDA-Konzept | Ruda-Einstiegspunkt |
| --- | --- |
| Zuständigkeiten von Host und Gerät | Rust-Hostcode und Kernel-DSL |
| Grid/block | RudaCount/RudaDim |
| Eindimensionale globale Threadposition | ABSOLUTE_POS |
| Speicherzuweisung und Transfers | Speicher- und Rückleseschnittstellen von ComputeClient |
| Kernel-Kompilierung und -Start | ruda-kernel, ruda-compiler und Gerätelaufzeit |
| BLAS-, DNN-, FFT- und Sparse-Bibliotheken | ruBLAS, ruDNN, ruFFT und ruSPARSE |
| Kollektive Kommunikation | ruCCL |

Dies ordnet Konzepte zu, nicht direkt austauschbare Funktionen. Beachten Sie Rudas API-Verträge zu Ownership, Argumenten, Synchronisierung und Fehlern.

## 2. CUDA C++ und PTX

Ruda bietet den CUDA-C++-/NVRTC-Pfad und einen ausdrücklich wählbaren direkten PTX-Pfad. Beide werden über den NVIDIA-Treiber ausgeführt.

Der CUDA-C++-Kompilierungspfad ermöglicht weder die unveränderte Kompilierung beliebiger CUDA-Projekte noch einen vollständigen Ersatz der CUDA-Runtime-/Driver-ABI oder das direkte Neuverlinken bestehender Bibliotheksbinärdateien. Dieser Leitfaden enthält keinen Befehl zur automatischen Konvertierung aller CUDA-Anwendungen.

Direktes PTX verarbeitet die vom Generator implementierte Kernel-IR, nicht beliebige PTX-Eingabeprogramme. Nicht unterstützte Operationen fallen nicht automatisch auf einen anderen Compiler zurück.

## 3. Backends und Datentypen

Backends unterscheiden sich hinsichtlich skalarer Typen, atomarer Operationen, Matrixinstruktionen, Speicherlayouts und Synchronisierung. Fragen Sie die Gerätefähigkeiten ab und prüfen Sie dann die Typ- und Layoutanforderungen jeder Operation.

Ein Typ der gemeinsamen DType-Enumeration ist nicht zwingend für jede Operation auf jedem Backend verfügbar. Dieselbe generische Rust-Schnittstelle garantiert weder identische Rundung noch identische Leistung.

HIP verwendet eine separate Ausführungsschnittstelle; PTX-Befehlssatzversionsnummern gelten dafür nicht.

## 4. Cargo und Benennung

Anzeigenamen von Bibliotheken, Cargo-Paketnamen und Rust-Importnamen können abweichen. Siehe [Rechenbibliotheksindex](libraries/README.md). Features steuern die verfügbaren Kombinationen; ein Standard-Build aktiviert nicht jeden Pfad.

Das Kernel-Frontend verwendet `#[ruda]`, RudaCount und RudaDim.

## 5. Versionen und numerische Validierung

Quellversion, Lockdatei, Features, Compiler-Backend, PTX/SM, Treiber und GPU definieren zusammen eine Validierungskonfiguration.

Beim Ersetzen eines Rechenbibliotheksaufrufs müssen Layout, Transposition, Indexbasis, Eingabe- und Akkumulationsdatentypen, Normalisierung, Sonderwerte und Synchronisierung übereinstimmen. ruFFT füllt beispielsweise Längen auf, die keine Zweierpotenzen sind; es ist kein semantikgleicher Ersatz für eine FFT beliebiger Länge.
