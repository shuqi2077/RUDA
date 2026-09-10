# ruSPARSE

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

**Englisch** | [简体中文](../zh/README.md)

Dieses Repository ist ein Quellspiegel. Führen Sie die folgenden Befehle im Stammverzeichnis [RUDA monorepo](https://github.com/shuqi2077/RUDA) aus.

Rudas Sparse-Computing-Bibliothek, die gemeinsam genutzte Tensoren, Laufzeiten und Geräte-Backends wiederverwendet.

## Funktionen und Einstiegspunkte

| Feature |Einstiegspunkte und Abhängigkeiten|
| --- | --- |
|Standard (leer)|Sparse-Formate, Validierung und Host-Konvertierungen; umfasst keine Geräteimplementierungen|
|`tensor`|`CsrTensor<R>`, SpMV, SpMM, SpVV, SDDMM, SpGEAM und SpGEMM; Anrufer wählen die freigegebene Laufzeit aus|
|`cuda`|`tensor` plus der NVIDIA-Treiber für CUDA-Programme und Beispiele|
|`cuda-tests`|Kompatibilitätseinstiegspunkt für `cuda`-Tests|

## CUDA Beispiel

```sh
cargo run -p ruSPARSE --features cuda --example csrmv
```

Das Beispiel berechnet `alpha * A * x + beta * y`, liest das Ergebnis synchron zurück und prüft es gegen `[7.0, 2.0, 18.5]`.

## Aktueller Berechnungsvertrag

- Der gemeinsam genutzte Gerätepfad verwendet FP32-Werte und U32-Indizes. Numerische Kernel erfordern 32-spurige Untergruppen.
- Laden Sie validierte Strukturen mit `CsrTensor::from_csr` hoch; `with_values` aktualisiert Werte und `transpose` ordnet die aktuellen Gerätewerte um. Operationen geben neue Ergebnisse zurück, statt Eingaben zu überschreiben.
- SpGEAM/SpGEMM teilen hostseitige symbolische Algorithmen und bewahren die explizite Nullstruktur; Die numerische Stufe läuft auf dem Gerät. Strukturindizes behalten Wirtskopien; Werte werden für die hostseitige Berechnung nicht zurückgelesen.

## ruSPARSE Benutzerhandbuch

[Computerbibliotheken](../../../docs/de/libraries/README.md) · [中文](../zh/README.md)

### 1. Übersicht

ruSPARSE bietet eine spärliche Berechnung.

### 2. Datendeskriptoren

|Typ|Zweck|
| --- | --- |
|`CsrMatrix`, `CsrMatrixOwned`|Geliehene CSR-Ansichten und eigener Speicher|
|`CooMatrix`, `CooMatrixOwned`|COO Darstellung|
|`CscMatrix`, `CscMatrixOwned`|CSC Darstellung|
|`BsrMatrix`, `EllMatrix`|Blockiert spärliche und ELL-Darstellungen|
|`DenseMatrix`, `DenseMatrixOwned`|Dichte Matrixdeskriptoren|
|`SparseVector`|Sparse-Vektor|
|`IndexBase`| Indexbasis Zero oder One |
|`DenseOrder`| RowMajor oder ColumnMajor |
|`Operation`| None, Transpose oder ConjugateTranspose |

Indexbasis, Layout und Operationstyp sind explizite Verträge und können nicht allein aus Array-Längen abgeleitet werden. „Eigentümer“ unterscheidet eigene Daten von geliehenen Ansichten; Der allgemeine GPU-Speicher wird nicht automatisch zugewiesen.

### 3. Operationen

Operationen sind in Matrix-Vektor-Multiplikation, Matrix-Multiplikation, dünn besetzte Matrix-Multiplikation/Addition, dünn besetzte Vektorpunktprodukte, SDDMM und Formatkonvertierung unterteilt. Die Ausführungsschicht enthält `SparsePlan`, `SparseError` und vorgangsspezifische Ergebnistypen.

CSR Matrix-Vektor-Multiplikation verwendet den Alpha-, Op(A)-, X-, Beta- und Y-Vertrag. Passen Sie beim Portieren eines Anrufs die Indexbasis, den Transpositionsmodus und das Datenlayout an, anstatt nur das Bibliothekspräfix zu ändern.

### 4. CUDA CSR Matrix-Vektor-Multiplikation

Das Cargo-Paket ist `ruSPARSE`; Der Rust-Importname lautet `rusparse`. Funktion `tensor` ermöglicht `rusparse::tensor`; `cuda` ermöglicht sowohl Gerätetensoren als auch CUDA-Abhängigkeiten. Vom Quellstammverzeichnis ausführen:

```powershell
cargo run --locked -p ruSPARSE --features cuda --example csrmv
```

Dieses Beispiel berechnet `0.5 × A × x + 2 × y`:

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, IndexBase, Operation,
    tensor::{CsrTensor, csrmv},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let host = CsrMatrix::new(
        3, 3,
        &[0, 2, 3, 4],
        &[0, 1, 1, 2],
        &[2., 1., -1., 5.],
        IndexBase::Zero,
    )?;
    let matrix = CsrTensor::<CudaRuntime>::from_csr(host, Operation::None, &device)?;
    let x = from_data(TensorData::new(vec![3f32, 4., 5.], [3]), &device);
    let y = from_data(TensorData::new(vec![1f32, 2., 3.], [3]), &device);
    let output = into_data_sync(csrmv(&matrix, 0.5, x, 2., y)?).to_vec::<f32>()?;
    assert_eq!(output, [7., 2., 18.5]);
    println!("alpha * A * x + beta * y = {output:?}");
    Ok(())
}
```

Die Argumente für `CsrMatrix::new` sind Zeilen, Spalten, Zeilenoffsets, Spaltenindizes, Werte ungleich Null und Indexbasis. In der ersten Zeile werden zwei Einträge gespeichert. der zweite und der dritte speichern jeweils einen. Das Ergebnis ist `[7.0, 2.0, 18.5]`.

`CsrTensor::from_csr` lädt Struktur und Werte hoch. Geben Sie hier `Operation::Transpose` an, um die Matrix zu transponieren. Für `csrmv(&matrix, alpha, x, beta, y)` entspricht die x-Länge der Spaltenanzahl und die y-Länge der Zeilenanzahl. Dichte Eingaben müssen unquantisierte F32-Tensoren auf dem Matrixgerät sein. Der Aufruf gibt einen neuen Gerätetensor zurück; Für die Wiederverwendung der Matrix ist kein erneutes Hochladen der CSR-Struktur erforderlich.

Für die Multiplikation einer dünn besetzten/dichten Matrix verwenden Sie `csrmm(&matrix, operation_b, alpha, b, beta, c, output_order)`. Hier ist c optional und `DenseOrder` gibt das Ausgabelayout an. Um nur Werte ungleich Null zu ändern, rufen Sie `matrix.with_values(values)` mit einem F32-Gerätetensor der Länge nnz auf; Die Struktur bleibt erhalten.

### 5. CSR im Tensor-Framework

`ruda_tensor::api::CsrTensor<B>` zielt auf Backends ab, die `SparseOps` implementieren. Es handelt sich um einen anderen Typ als den oben genannten `rusparse::tensor::CsrTensor<R>`. Erstellen Sie es mit `CsrTensor::<B>::from_data(&data, &device)`, wobei die Daten den Typ `B::CsrData` des Backends haben.

|Methode|Verwendung|
| --- | --- |
|`matmul(rhs)`, `transpose_matmul(rhs)`|Multiplizieren Sie die dünn besetzte Matrix oder ihre Transponierte mit einem zweidimensionalen dichten Tensor|
|`sparse_matmul(&rhs)`|Multiplizieren Sie dünn besetzte Matrizen und erstellen Sie das Ausgabemuster|
|`add(&rhs)`, `add_scaled(&rhs, alpha, beta)`|Sparse-Matrizen hinzufügen, optional mit Koeffizienten|
|`gather(dense)`|Sammeln Sie dichte Werte an den CSR-Positionen in einem eindimensionalen Tensor|
|`scatter_add()`|Gespeicherte Werte zu ihren dichten Positionen hinzufügen|
|`mul_dense(rhs)`|Mit dichten Werten an den CSR-Positionen multiplizieren|
|`sampled_matmul(lhs, rhs)`|Bewerten Sie ein dichtes Matrixprodukt nur an den aktuellen CSR-Positionen|
|`sampled_sparse_matmul(&lhs, &rhs)`|Bewerten Sie ein Sparse-Matrix-Produkt nur an den aktuellen CSR-Positionen|
|`transpose()`, `to_dense()`|Transponiert oder konvertiert in einen dichten Tensor|
|`with_values(values)`|Ersetzen Sie den eindimensionalen Wertetensor unter Beibehaltung der Struktur|
|`to_data().await`|`B::CsrData` asynchron zurücklesen|

Diese Berechnungsmethoden geben `Result` zurück. Werttensoren, die in Framework-Sparse-Operationen verwendet werden, müssen unquantisiert sein.

API-Referenz: [Gerät CSR](../../src/tensor/mod.rs), [Framework CSR](../../../ruda-tensor/src/api/sparse.rs).
