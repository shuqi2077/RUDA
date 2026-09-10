# ruBLAS

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

**Englisch** | [简体中文](../zh/README.md)

Lineare Algebra und Matrixmultiplikation für Ruda.

- Cargo Paket: `rublas`
- Rostkiste: `rublas`

## Features

| Feature |Operationen|
| --- | --- |
|`tensor-matmul`|Matrixmultiplikation|
|`tensor-matmul-autotune`|Autotuning der Matrixmultiplikation|
|`tensor-vector`|Vektoroperationen|
|`tensor-grouped`|Gruppierte Matrixmultiplikation|
|`tensor-int4`|INT4 Gewichtsoperationen|

## Schnellstart

Build aus dem RUDA-Arbeitsbereich:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p rublas --features tensor-matmul
```

## Dokumentation

- [Benutzerhandbuch](../../../docs/de/libraries/rublas.md)
- [Umgebungseinrichtung](../../../docs/de/getting-started.md)
- [Cargo-Funktionen](../../Cargo.toml) · [Modulexporte](../../src/lib.rs)

## ruBLAS Benutzerhandbuch

[Compute-Bibliotheken](../../../docs/de/libraries/README.md) · [Laufzeit API](../../../docs/de/runtime-api.md) · [中文](../zh/README.md)

### 1. Übersicht und Funktionen

ruBLAS bietet lineare Algebraoperationen und Gerätetensorschnittstellen.

| Feature |Modul|
| --- | --- |
|`tensor-vector`|`rublas::tensor_vector`|
|`tensor-matmul`|`rublas::tensor_matmul`|
|`tensor-matmul-autotune`|Automatische Optimierung der Tensormatrixmultiplikation|
|`tensor-int4`|`rublas::tensor_int4`|
|`tensor-grouped`|`rublas::tensor_grouped`|

Das Cargo-Paket ist `rublas`. Wählen Sie die erforderlichen Funktionen aus und deaktivieren Sie unnötige Standardeinstellungen für allgemeine Pfade. Siehe [Cargo.toml](../../Cargo.toml).

### 2. Matrixmultiplikation, Vektoren und INT4

#### Tensormatrixmultiplikation

`rublas::tensor_matmul::matmul` nimmt lhs, rhs, die optionale Ausgabe out, MatmulStrategy und den Ausgabe-DType entgegen. Zurückgegeben wird ein Gerätetensor oder MatmulSetupError. Übergeben Sie eine vorhandene Ausgabe über out oder lassen Sie out weg, um eine neue zuzuweisen.

Zu den Strategien gehören Ruda, CmmaResidueFirst, Naive und bei aktiviertem `tensor-matmul-autotune` auch Autotune. Ohne dieses Feature ist Ruda der Standard, mit ihm Autotune. Bei quantisierten Eingaben kann Naive nach dem Scheitern seines ersten Pfads dequantisieren und rechnen; dies ist keine native INT4-Berechnung.

`CmmaResidueFirst` wählt explizit den Tensor-Core-Pfad aus, der zuerst eine teilweise K32-Kachel verarbeitet, ohne dass eine materialisierte gepolsterte Matrix erforderlich ist. lhs und rhs müssen übereinstimmende unquantisierte BF16 oder F16 dtype und kompatible Matrixdimensionen haben. Das Gerät muss die Matrixanweisungen des Pfades unterstützen. Diese Funktion gibt die F32-Ausgabe zurück:

```rust
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rublas::{
    kernel_ir::definition::MatmulSetupError,
    tensor_matmul::{MatmulStrategy, matmul},
};

fn residue_matmul<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    matmul(lhs, rhs, None, MatmulStrategy::CmmaResidueFirst, DType::F32)
}
```

Die Eingaben `[M, K]` und `[K, N]` erzeugen `[M, N]`. Ungültiges Setup oder fehlende Gerätefunktionen geben `MatmulSetupError` zurück, anstatt zu Naive zu wechseln.

Siehe [Einstiegspunkt für die Matrixmultiplikation](../../src/tensor_matmul/base.rs).

#### Vektorkreuzprodukt

`rublas::tensor_vector::cross(lhs, rhs, dim)` setzt die Länge 3 der gewählten Dimension voraus und gibt einen Gerätetensor zurück. Eine Berechnung entlang einer anderen als der letzten Dimension erfordert Permutation und Umwandlung in zusammenhängenden Speicher. Siehe [cross.rs](../../src/tensor_vector/cross.rs).

#### AWQ INT4

`rublas::tensor_int4::AwqGemm::new` verwendet qweight, qzeros, scales, optional bias und group_size, um gepackte Gewichte zu erstellen. `forward` akzeptiert die Eingabe F16 und gibt die Ausgabe F16 oder Int4Error zurück.

Für Eingabedimension K, Ausgabedimension N und Gruppengröße G:

|Daten|Dtype|Form|
| --- | --- | --- |
|qweight| Gepacktes I32 |[K, N/8]|
|qzeros| Gepacktes I32 |[K/G, N/8]|
|scales|F16|[K/G, N]|
|bias (optional)|F16|[N]|

K, N und G müssen ungleich Null sein; K muss durch G und N durch 8 teilbar sein. Die endgültige Dimension der Eingabe ist K, die in der Ausgabe durch N ersetzt wird. Alle Operanden müssen sich ein Gerät teilen. Die gepackte Bitreihenfolge muss mit dem [AWQ-Kernel](../../src/tensor_int4/kernel.rs) übereinstimmen. Eine beliebige INT4-Datei kann nicht direkt als qweight verwendet werden.

Informationen zum Objekt und seinen Prüfungen finden Sie in der [INT4-Schnittstelle](../../src/tensor_int4/mod.rs).

### 3. Gruppierte Matrixmultiplikation

`rublas::tensor_grouped::grouped_matmul_nt<R: Runtime>` übernimmt Eingaben, Gewichtungen und row_experts als `RudaTensor<R>`-Werte. Es wird `Result<RudaTensor<R>, GroupedMatmulError>` zurückgegeben.

|Argument|Form|Anforderungen|
| --- | --- | --- |
|Eingabe|[M, K]|Nicht quantisierter F32, F16 oder BF16|
|Gewichte|[E, N, K]| Derselbe dtype und dasselbe Gerät wie die Eingabe |
|row_experts|[M]|Nicht quantisierter U32 auf demselben Gerät|
|Ausgabe|[M, N]|Gleiches dtype wie Eingabe|

Zeile m wählt die durch row_experts[m] indizierte Gewichtsmatrix aus und berechnet Skalarprodukte zwischen der Eingabezeile und jeder Zeile dieser Matrix. Die letzten beiden Gewichtsdimensionen nehmen in transponierter Form teil, ohne dass eine materialisierte Transponierung erforderlich ist.

### 4. Gruppierte Multiplikationssemantik

- K, N und E müssen positiv sein; M kann Null sein.
- Expertenindizes außerhalb des gültigen Bereichs kennzeichnen Padding und erzeugen mit Nullen gefüllte Ausgabezeilen.
- Der Kernel akkumuliert in FP32 und wandelt ihn dann in die Eingabe dtype um.
- Der Einstiegspunkt macht Eingaben und Gewichtungen bei Bedarf zusammenhängend, wodurch möglicherweise Daten kopiert werden.
- Die Anzahl der relevanten Elemente muss zur U32-Indizierung passen.
- Ungültige Argumente geben `GroupedMatmulError` zurück. Behandeln Sie asynchrone Ausführungsfehler während des Rücklesens oder der Synchronisierung.

Quelle: [gruppierte Schnittstelle](../../src/tensor_grouped/mod.rs) und [Kernel](../../src/tensor_grouped/kernel.rs).

### 5. Integration mit anderen Bibliotheken

[ruDNN MoE](../../../docs/de/libraries/rudnn.md) verwendet gruppierte Multiplikation für Expertenprojektionen. Quantisierte Gewichte verwenden das separate Modul `tensor_int4`; Die gewöhnliche gruppierte Gleitkommamultiplikation ist keine INT4-Expertenberechnung.

Der gruppierte Kernel verwendet Skalarakkumulation. Messen Sie Matrixmultiplikationsstrategien für Ihr dtype, Ihre Form und Ihr Backend.
