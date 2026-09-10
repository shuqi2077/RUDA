# ruPRIM

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

**Englisch** | [简体中文](../zh/README.md)

Parallele Grundelemente, Reduzierungen, Scans und Indizierung für Ruda.

- Cargo Paket: `ruPRIM`
- Rostkiste: `ruprim`

## Features

| Feature |Operationen|
| --- | --- |
|`tensor-reduce`|Gesamttensor- und Achsenreduktionen|
|`tensor-reduce-autotune`|Reduzierung des Autotunings|
|`tensor-scan`|Kumulierte Summe, Produkt, Minimum und Maximum|
|`elementwise`|Elementweise Operationen|
|`indexing`|Auswählen, Schneiden, Sammeln und Streuen|

## Schnellstart

Build aus dem RUDA-Arbeitsbereich:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruPRIM --features tensor-reduce,tensor-scan,indexing
```

## Dokumentation

- [Benutzerhandbuch](../../../docs/de/libraries/ruprim.md)
- [Umgebungseinrichtung](../../../docs/de/getting-started.md)
- [Cargo-Funktionen](../../Cargo.toml) · [Modulexporte](../../src/lib.rs)

## ruPRIM Benutzerhandbuch

[Computerbibliotheken](../../../docs/de/libraries/README.md) · [Tensor-Framework](../../../docs/de/tensor-framework.md) · [中文](../zh/README.md)

ruPRIM bietet Gerätetensorreduzierungen, kumulative Scans, elementweise Operationen und Indizierung. Diese Seite verwendet `RudaTensor<R>`, wobei R eine Gerätelaufzeit ist.

### 1. Abhängigkeiten konfigurieren

Das Cargo-Paket ist `ruPRIM`; Sein Rust-Importname ist `ruprim`. Aktivieren Sie Funktionen für die von Ihnen benötigten Vorgänge:

| Feature |Schnittstelle|
| --- | --- |
|`tensor-reduce`|`ruprim::reduce::tensor`: Ganztensor- und Achsenreduktionen|
|`tensor-reduce-autotune`| Autotuning für Reduktionen; aktiviert auch tensor-reduce |
|`tensor-scan`|`ruprim::scan`: kumulative Summe, Produkt, Minimum, Maximum|
|`elementwise`|`ruprim::elementwise`: elementweise Berechnung|
|`indexing`| `ruprim::indexing`: Auswahl, Slicing, Gather, Scatter; aktiviert auch elementwise |

Diese Konfiguration platziert das Anwendungsverzeichnis neben dem `RUDA`-Quellverzeichnis. Informationen zum NVIDIA-Setup finden Sie unter [Erste Schritte](../../../docs/de/getting-started.md).

```toml
[dependencies]
ruprim = { package = "ruPRIM", path = "../RUDA/ruPRIM", default-features = false, features = ["std", "tensor-reduce", "tensor-scan", "indexing"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Summe, Zeilenreduzierungen und Scans

Dieser vollständige `src/main.rs` verwendet die F32-Matrix `[[1, 2, 3], [4, 5, 6]]`, um seine Gesamtsumme, Zeilenmittelwerte, Zeilen-Argmax-Indizes und Zeilenpräfixsummen zu berechnen. Führen Sie `cargo run` aus dem Anwendungsverzeichnis aus:

```rust
use ruda_core::tensor::{DType, TensorMetadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use ruprim::reduce::{
    components::instructions::ReduceOperationConfig,
    tensor::{KernelReduceStrategy, SumStrategy, reduce_dim, sum},
};
use ruprim::scan::cumsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        &device,
    );
    let total = sum(
        input.clone(),
        SumStrategy::Chained(KernelReduceStrategy::Unspecified),
    )?;
    assert_eq!(total.shape().as_slice(), &[1]);
    assert_eq!(into_data_sync(total).to_vec::<f32>()?, [21.0]);

    let means = reduce_dim(
        input.clone(), None, 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::Mean,
    )?;
    assert_eq!(means.shape().as_slice(), &[2, 1]);
    assert_eq!(into_data_sync(means).to_vec::<f32>()?, [2.0, 5.0]);

    let indices = reduce_dim(
        input.clone(), Some(DType::I32), 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::ArgMax,
    )?;
    assert_eq!(into_data_sync(indices).to_vec::<i32>()?, [2, 2]);

    let prefix = cumsum(input, 1);
    assert_eq!(into_data_sync(prefix).to_vec::<f32>()?, [1., 3., 6., 4., 9., 15.]);
    Ok(())
}
```

`sum` reduziert alle Elemente und gibt die Form `[1]` zurück. `reduce_dim` verarbeitet nur die ausgewählte Achse, behält normalerweise den Rang bei und setzt die Achsenlänge auf 1. Zeile bedeutet daher, dass sie die Form `[2, 1]` und nicht `[2]` hat.

### 3. Reduktionsoperationen und -strategien

Vollständige Parameterreihenfolge für Tensorschnittstellen:

|Funktion|Zweck|
| --- | --- |
|`sum(tensor, strategy)`|Ganze Tensorsumme|
|`sum_fallback(tensor, strategy)`|Gesamttensorsumme; schaltet OneShot auf „Chained“ um, wenn die erforderliche atomare Addition nicht verfügbar ist|
|`reduce(tensor, output_dtype, strategy, config)`|Reduziert nacheinander jede Achse und gibt die Form zurück [1]|
|`reduce_dim(tensor, output_dtype, dim, strategy, config)`|Reduziert eine ausgewählte Achse|

Wählen Sie `config` mit `ReduceOperationConfig` aus:

|Vorgang|`reduce_dim`-Ausgabe|
| --- | --- |
|`Sum`, `Prod`, `Mean`|Summe, Produkt, Mittelwert; gewählte Achslänge 1|
|`Min`, `Max`, `MaxAbs`|Minimaler, maximaler, maximaler Absolutwert; gewählte Achslänge 1|
|`ArgMin`, `ArgMax`|Nullbasierte Indizes innerhalb der ausgewählten Achse; Achslänge 1|
|`TopK(k)`|Top-k-Werte entlang der ausgewählten Achse; Achslänge k|
|`ArgTopK(k)`|Entsprechende Indizes innerhalb der Achse; Achslänge k|

Indexreduzierungen erfordern eine explizite Ganzzahlausgabe dtype wie `Some(DType::I32)`. Für Wertminderungen übergeben Sie `None`; Der Ausgang behält den Eingang dtype. Verwenden Sie `output_dtype` nicht als allgemeinen Umwandlungsparameter. F16/BF16 Summe, Prod und Mittelwert verwenden die FP32-Akkumulation in diesem Pfad, bevor die Eingabe dtype geschrieben wird. Die Achse muss gültig sein und TopK k sollte in `1..=axis length` sein. Verwenden Sie `reduce_dim` für TopK, nicht den Ganztensor `reduce`, der die endgültige Form auf `[1]` zurücksetzt.

Wählen Sie `SumStrategy` wie folgt:

- `OneShot(ruda_count)`: Legt explizit eine positive Arbeitsgruppenanzahl fest und erfordert eine atomare Addition für die Eingabe dtype.
- `Chained(KernelReduceStrategy::Unspecified)`: verwendet abgestufte Reduktionen, ohne dass die globale atomare Addition der Gesamttensorsumme erforderlich ist; im Beispiel verwendet.
- `Autotune`: verfügbar mit `tensor-reduce-autotune`. Ohne diese Funktion ist der Standardwert `OneShot(4)`; Damit ist der Standardwert `Autotune`.

`KernelReduceStrategy` bietet `Unspecified`, `Specific(ReduceStrategy)` und funktionsgesteuertes `Autotune`. Verwenden Sie „Spezifisch“, um eine Low-Level-Strategie festzulegen. Ohne Autotuning ist die Standardeinstellung Nicht angegeben.

Reduktionsschnittstellen geben `Result<RudaTensor<R>, ReduceError>` zurück. Eine Achse außerhalb des gültigen Bereichs ergibt `InvalidAxis`; fehlende atomare Addition für OneShot ergibt `MissingAtomicAdd`. `sum_fallback` ersetzt nur den Fall der nicht unterstützten atomaren OneShot-Addition und wechselt nicht zur CPU-Ausführung.

### 4. Kumulative Scans

Alle vier Funktionen benötigen `(tensor, dim)`:

|Funktion|Ergebnis für eine Eingabezeile [3, 1, 2]|
| --- | --- |
|`cumsum`|[3, 4, 6]|
|`cumprod`|[3, 3, 6]|
|`cummin`|[3, 1, 1]|
|`cummax`|[3, 3, 3]|

Dies sind inklusive Präfixoperationen. Ausgabeform und dtype stimmen mit der Eingabe überein; Stapelzeilen werden unabhängig voneinander verarbeitet. dim ist nullbasiert und muss innerhalb des Eingaberangs liegen. Scans geben Tensoren direkt zurück, nicht `Result`.

Der aktuelle Algorithmus liest das entsprechende Präfix für jede Ausgabeposition und ergibt O(n²) Gesamtlesevorgänge für eine Achse der Länge n. Berücksichtigen Sie diese Kosten für lange Scanachsen, anstatt die Arbeit allein anhand der Anzahl der Eingabeelemente abzuschätzen.

### 5. Auswahl, Sammeln, Schneiden und ganzzahlige Potenzen

Fügen Sie diese Funktion derselben Datei hinzu und rufen Sie `indexing_example(&device)?;` auf, bevor main zurückkehrt. Es verwendet die früheren Importe wieder:

```rust
fn indexing_example(device: &CudaDevice) -> Result<(), Box<dyn std::error::Error>> {
    use ruda_core::tensor::element::Scalar;
    use ruprim::{
        elementwise::binary::integer_power,
        indexing::{gather, select, slice},
    };

    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        device,
    );
    let columns = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0], [2]), device,
    );
    let selected = select(input.clone(), 1, columns);
    assert_eq!(into_data_sync(selected).to_vec::<f32>()?, [3., 1., 6., 4.]);

    let positions = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0, 1, 1], [2, 2]), device,
    );
    let gathered = gather(1, input.clone(), positions);
    assert_eq!(into_data_sync(gathered).to_vec::<f32>()?, [3., 1., 5., 5.]);

    let sliced = slice(input.clone(), &[0..2, 1..3]);
    assert_eq!(into_data_sync(sliced).to_vec::<f32>()?, [2., 3., 5., 6.]);

    let squared = integer_power::scalar(input, Scalar::Int(2));
    assert_eq!(into_data_sync(squared).to_vec::<f32>()?, [1., 4., 9., 16., 25., 36.]);
    Ok(())
}
```

|Funktion|Indizes und Ausgabe|
| --- | --- |
|`select(tensor, dim, indices)`|Eindimensionale Ganzzahlindizes; Wendet die gleichen Positionen auf andere Dimensionen an und ersetzt die Länge der ausgewählten Achse durch die Indexanzahl|
|`gather(dim, tensor, indices)`|dim ist das erste Argument; Wählt einen Eingabewert pro Ausgabeposition, wobei die Ausgabeform der Form der Indizes entspricht|
|`slice(tensor, ranges)`|Halboffene `start..end`-Bereiche pro Achse; Im Beispiel werden die Spalten 1 und 2 aus allen Zeilen ausgewählt|
|`slice_assign(tensor, slices, value)`|Verwendet `ruda_core::tensor::Slice`; schreibt einen Wert in den ausgewählten Bereich und gibt den aktualisierten Tensor zurück|

Indizes und Daten müssen sich ein Gerät teilen; Indizes müssen gültige, auf Null basierende Ganzzahlen sein. Verwenden Sie zum Sammeln Indextensoren mit demselben Rang und passenden nicht ausgewählten Dimensionen. Jede Beispielzeile kann verschiedene Spalten auswählen. Die Slice-Bereiche müssen innerhalb ihrer Achsen liegen und die Form des Zuweisungswerts muss mit der ausgewählten Region übereinstimmen.

Ganzzahlige Potenzen sind in `ruprim::elementwise::binary::integer_power`:

- `scalar(input, Scalar::Int(exponent))`: Verwendet einen ganzzahligen Exponenten für alle Elemente.
- `tensor(base, exponents)`: verwendet Gleitkommabasen und ganzzahlige Exponententensoren mit Broadcast-kompatiblen Formen; Ausgang behält Basis dtype.

Verwenden Sie `Scalar::Int(-2)` für einen negativen Exponenten, anstatt den Exponenten in einen Gleitkommawert umzuwandeln. Indizierung und elementweise Schnittstellen geben Gerätetensoren direkt zurück. Um eine Eingabe für andere Operationen beizubehalten, übergeben Sie geklonte Handles wie im Beispiel und behalten Sie den von jeder Operation zurückgegebenen Tensor bei.

API Referenz: [Reduktionen](../../src/reduce/tensor/base.rs), [Scans](../../src/scan/tensor.rs), [Indizierung](../../src/indexing/mod.rs), [Ganzzahlpotenzen](../../src/elementwise/binary/integer_power.rs).
