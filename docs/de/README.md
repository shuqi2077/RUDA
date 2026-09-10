# Ruda Dokumentation

[English](../en/README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

[Dokumentationsindex](../README.md) · [中文](../zh/README.md)

Von Ihrem ersten GPU-Kernel bis hin zu Rechenbibliotheken, Tensortraining und lokaler Modellinferenz. Beginnen Sie mit Ihrer Aufgabe und erkunden Sie dann die Programmierhandbücher und API-Referenzen.

## Beginnen Sie hier

|Ihre Aufgabe|Lesepfad|
| --- | --- |
|Führen Sie Ihren ersten GPU-Kernel aus|[Installation und Schnellstart](getting-started.md) → [Programmieranleitung](programming-guide.md)|
|Verwenden Sie Matrix-, Sparse- oder neuronale Netzwerkoperationen|[Computerbibliotheken](libraries/README.md) → [Tensoren und Frameworks](tensor-framework.md)|
|Modelle trainieren, Steigungen akkumulieren und Status speichern|[Trainings- und Speicherstatus](training.md)|
|Laden Sie lokale Modelle, generieren Sie Text oder verarbeiten Sie Bilder|[Modellladen und Inferenz](model-inference.md)|

## Erste Schritte

- [Installation und Schnellstart](getting-started.md): Quelleneinrichtung, Backend-Auswahl und Ihr erstes Beispiel.
- [Beispiele und Tutorials](samples.md): Vektoraddition, Tensoren, gemeinsam genutzter Speicher und halbe Präzision.

## Programmierhandbücher

- [Ruda Programmierhandbuch](programming-guide.md): Host- und Gerätecode, Ausführungshierarchie, Speicher, Synchronisierung und Sicherheit.
- [Tensoren und Frameworks](tensor-framework.md): Gerätetensoren, Bibliotheksversand, Fusion und automatische Differenzierung.
- [Trainings- und Speicherstatus](training.md): Trainingsschritte, Gradientenakkumulation, Lernratenplanung, Speichern und Wiederherstellen.
- [Modellladen und Inferenz](model-inference.md): ruLLM, Text- und Bildeingaben, Sampling, AWQ und kontinuierliche Stapelverarbeitung.

## Kompilierung und Ausführung

- [Compiler-Anleitung](compiler-guide.md): das Rust-Kernel-Frontend, IR, CUDA C++/NVRTC und direktes PTX.
- [PTX Backend-Referenz](ptx.md): Zielkonfiguration, Kompilierungsausgabe, Einschränkungen und Fehler.

## API Referenzen

- [Laufzeit API](runtime-api.md): Geräte-Clients, Speicher, Übermittlung, Rücklesen und Synchronisierung.
- [Treiber API und Backends](driver-api.md): Backend-Typen, Geräteauswahl und Laufzeitintegration.
- [Referenz zur Compute-Bibliothek](libraries/README.md): Bibliotheksauswahl, Cargo-Funktionen und Einstiegspunkte.

## Compute-Bibliotheken

|Bibliothek|Anleitung|
| --- | --- |
|ruBLAS|[Lineare Algebra und gruppierte Matrixmultiplikation](libraries/rublas.md)|
|ruDNN|[Neuronale Netzwerkoperationen und MoE](libraries/rudnn.md)|
|ruPRIM|[Reduktionen, Scans und Indizierung](libraries/ruprim.md)|
|ruFFT|[Schnelle Fourier-Transformationen](libraries/rufft.md)|
|ruRAND|[Zufallszahlengenerierung](libraries/rurand.md)|
|ruSPARSE|[Sparse-Berechnung](libraries/rusparse.md)|
|ruCCL|[Sammelkommunikation](libraries/ruccl.md)|

## Debugging und Kompatibilität

- [Debugging und Diagnose](debugging.md): Kompilierungsfehler, asynchrone Fehler, Caching und numerische Prüfungen.
- [Kompatibilitätsleitfaden](compatibility.md): CUDA-Konzepte, Backend-Unterschiede sowie API und Kompilierungsgrenzen.
- [Beitrag leisten](CONTRIBUTING.md): Probleme und Entwicklungskonventionen melden.

Befolgen Sie für ein erstes Projekt die Schnellstartanleitung, die Programmieranleitung und die entsprechende Bibliotheksanleitung. Konsultieren Sie bei der Entwicklung von Backends oder Kernels die Compiler- und API-Referenzen.

- [Muon und explizite Muon + AdamW-Gruppen (experimentell)](muon.md)
