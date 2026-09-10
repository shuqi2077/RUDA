# Compute-Bibliotheksreferenz

[English](../../en/libraries/README.md) | [简体中文](../../zh/libraries/README.md) | [日本語](../../ja/libraries/README.md) | **Deutsch** | [Русский](../../ru/libraries/README.md)

[Dokumentation](../README.md) · [Tensoren und Frameworks](../tensor-framework.md) · [中文](../../zh/libraries/README.md)

Compute-Bibliotheken implementieren Vorgänge, die Laufzeit führt sie auf Geräten aus und das Tensor-Framework setzt sie zusammen. Bibliotheksnamen beschreiben Verantwortlichkeiten, nicht identische APIs oder vollständige Funktionsparität mit ähnlich benannten CUDA-Bibliotheken.

## Wählen Sie eine Bibliothek

|Bibliothek|Cargo Paket| Rust-Crate |Hauptoperationen|
| --- | --- | --- | --- |
|[ruBLAS](rublas.md)|`rublas`|`rublas`|Matrixmultiplikation, Vektoroperationen, gruppierte Matrixmultiplikation und INT4|
|[ruDNN](rudnn.md)|`ruDNN`|`rudnn`|Achtung, Faltung, Pooling und MoE|
|[ruTENSOR](rutensor.md)|`ruTENSOR`|`rutensor`|Allgemeine Tensorkontraktionen, einsum, Reduktionen, Permutationen und elementweise Operationen|
|[ruPRIM](ruprim.md)|`ruPRIM`|`ruprim`|Reduzierungen, Scans, elementweise Operationen und Indizierung|
|[ruFFT](rufft.md)|`ruFFT`|`rufft`|Real FFT und inverse Transformationen|
|[ruRAND](rurand.md)|`ruRAND`|`rurand`|Gleichmäßige, normale und Bernoulli-Verteilungen|
|[ruSPARSE](rusparse.md)|`ruSPARSE`|`rusparse`|Sparse-Matrix-Formate und -Operationen|
|[ruCCL](ruccl.md)|`ruCCL`|`ruccl`|Kollektive Kommunikation und Orchestrierung|

## Schnittstellenschichten

- Kernel-/Startschnittstellen akzeptieren Gerätebindungen und Ausführungskonfigurationen für die Kernel- und Bibliotheksentwicklung.
- Tensorschnittstellen verarbeiten Zuweisungen, Layouts und Aufrufe und verwenden dabei üblicherweise `RudaTensor<R>`.
- Framework-Schnittstellen werden über Komponenten wie `ruda-tensor-device` an Bibliotheken weitergeleitet.

Operationen mit demselben Namen auf verschiedenen Ebenen können unterschiedliche Argumente, Rückgabetypen und Fehlerbehandlung haben. Überprüfen Sie Paket, Modulpfad und Funktion gemeinsam. Kernel-Eintrittspunktsignaturen sind keine Tensor-Eintrittspunktsignaturen.

## Verwendung der Anleitungen

Jedes Bibliothekshandbuch behandelt seinen Zweck, seine Funktionen, Schnittstellen, Datenverträge und Einschränkungen. Wählen Sie explizit die von einem Allzweckpfad benötigten Funktionen aus, anstatt davon auszugehen, dass Standardeinstellungen für jedes Gerät geeignet sind.
