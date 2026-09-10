# Tensoren und Frameworks

[English](../en/tensor-framework.md) | [简体中文](../zh/tensor-framework.md) | [日本語](../ja/tensor-framework.md) | **Deutsch** | [Русский](../ru/tensor-framework.md)

[Dokumentation](README.md) · [Computerbibliotheken](libraries/README.md) · [Programmierhandbuch](programming-guide.md) · [中文](../zh/tensor-framework.md)

## 1. Ebenen

|Ebene|Komponente|Verantwortung|
| --- | --- | --- |
|Gemeinsame Daten und Verträge|Ruda-Core|D-Typen, Formen, Geräte und Kompilierungsverträge|
|Gerätetensoren|ruda-kernel::tensor|Speicherung, Metadaten, Zuordnung und Layouts|
|Geräte-Backend|Ruda-Tensor-Gerät|Sendet Tensoroperationen an Rechenbibliotheken|
|Tensor API|Ruda-Tensor|Backend-generische Tensorschnittstellen|
|Fusion|Ruda-Fusion|Operation Fusion|
|Automatische Differenzierung|ruda-autodiff|Automatische Differenzierung|
|Modell- und Trainingskomponenten|Ruda-Modell, Ruda-nn, Ruda-Optim, Ruda-Store, Ruda-Datensatz|Modelle, Netzwerkmodule, Optimierer, Speicher und Daten|

## 2. Gerätetensoren

`RudaTensor<R>` enthält Client, Handle, Meta, Gerät, dtype und qparams. Speichergriffe sind von Form/Schritten getrennt und Quantisierungsparameter werden separat gespeichert.

Low-Level-Operationen müssen prüfen, ob die Eingaben ein gemeinsames Gerät haben, die D-Typen mit der Berechnung übereinstimmen und die quantisierten Daten die richtigen Parameter enthalten. Angrenzende Speicherung, transponierte Ansichten und materialisierte Kopien sind unterschiedlich.

Informationen zu Zuordnung, zusammenhängender Konvertierung, Umformung, Permutation, Übertragung und Rücklesung finden Sie im [Gerätetensormodul](../../ruda-kernel/src/tensor/mod.rs).

## 3. NVIDIA Backend

`ruda-tensor-device/cuda` aktiviert `ruda_tensor_device::cuda`.

Ohne `cuda-fusion` verwendet `Cuda<F, I>` den Alias `DeviceBackend<CudaRuntime, F, I, u8>`. Wenn diese Funktion aktiviert ist, wird ein Fusion-Wrapper verwendet. F ist standardmäßig auf f32 und I auf i32 eingestellt. Siehe [cuda.rs](../../ruda-tensor-device/src/cuda.rs).

Dies ist ein Tensor-Backend, kein CUDA-Treiber-API-Handle. Überprüfen Sie bei der Auswahl eines Backends die dtype- und Funktionsanforderungen jedes Vorgangs.

## 4. Computing-Bibliotheksversand

Matrixoperationen werden an ruBLAS, neuronale Netzwerkoperationen an ruDNN, Reduktionen und Indizierung an ruPRIM, FFTs an ruFFT und Zufallszahlenerzeugung an ruRAND weitergeleitet. Siehe die [Dispatch-Module](../../ruda-tensor-device/src/dispatch) des Geräte-Backends.

## 5. Sparse-Tensoren, Quantisierung und Batch-Readback

`ruda_tensor::api::CsrTensor<B>` kombiniert spärliche Struktur und Gleitkommawerttensoren durch `SparseOps`. Es bietet Sparse/Dense-Multiplikations-, Transponierungs-, Additions-, Gather-, Scatter-Add- und Sampling-Operationen. Es unterscheidet sich von `rusparse::tensor::CsrTensor<R>`: Ersteres ist generisch über Backend, letzteres über Runtime. Siehe den [Sparse-Leitfaden](libraries/rusparse.md).

Die Quantisierung umfasst mehrdimensionale Blöcke scales, Packen entlang nicht endgültiger Achsen und Teilpakete, Layouttransformationen, ausgewählte Indizierungsoperationen und fusioniertes quantisiertes Zurücklesen. Die logische Form unterscheidet sich von der gepackten Speicherform. FP8/FP4-Kodierungen dürfen nicht als Ganzzahlwerte konvertiert werden. Siehe [Kernel-Quantisierung](../../ruda-kernel/src/quantization), [quantisierte Tensor-Layouts](../../ruda-kernel/src/tensor/contiguous.rs) und [Fusionstransaktionen](../../ruda-fusion/src/ops/transaction.rs). Die Betriebsabdeckung variiert je nach System.

Batch-Readback organisiert Deskriptoren nach tatsächlichem Gerät und Stream. Siehe [Tensortransaktionen](../../ruda-kernel/src/tensor/transaction.rs).

## 6. Training und Modellinferenz

- [Trainings- und Speicherstatus](training.md): Konfigurieren Sie ein Autodiff-Backend, aktualisieren Sie Parameter, sammeln Sie Gradienten und speichern Sie den Trainingsstatus oder stellen Sie ihn wieder her.
- [Modellladen und Inferenz](model-inference.md): Laden Sie lokale Gewichte, erstellen Sie Chat-Eingabeaufforderungen, generieren Sie mit Stichproben und verarbeiten Sie Bildeingaben.
