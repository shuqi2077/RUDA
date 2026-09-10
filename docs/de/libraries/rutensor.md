# ruTENSOR Benutzerhandbuch

[English](../../en/libraries/rutensor.md) | [简体中文](../../zh/libraries/rutensor.md) | [日本語](../../ja/libraries/rutensor.md) | **Deutsch** | [Русский](../../ru/libraries/rutensor.md)

[Computerbibliotheken](README.md) · [Tensor-Framework](../tensor-framework.md) · [中文](../../zh/libraries/rutensor.md)

ruTENSOR bietet Tensorkontraktionen, Reduktionen, physikalische Permutationen und elementweise Operationen mit benannten Achsen. Eingaben verwenden `RudaTensor<R>`; Die Anwendung wählt eine Gerätelaufzeit aus. Diese Bibliothek unterscheidet sich vom übergeordneten `ruda-tensor`-Framework.

## 1. Abhängigkeiten konfigurieren

Das Cargo-Paket ist `ruTENSOR`; Der Rust-Importname lautet `rutensor`. Die Standardeinstellungen ermöglichen `std` und die Berechnung des Gerätetensors, ohne dass ein Treiber ausgewählt werden muss. Die folgende Konfiguration platziert das Anwendungsverzeichnis neben dem `RUDA`-Quellverzeichnis:

```toml
[dependencies]
rutensor = { package = "ruTENSOR", path = "../RUDA/ruTENSOR" }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

## 2. Kontraktion, Reduktion und Permutation

Speichern Sie Folgendes als `src/main.rs` der Anwendung:

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rutensor::{einsum, permute, reduce, ReductionOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let a = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]), &device,
    );
    let b = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 0., 0., 1., 1., 1.], [3, 2]), &device,
    );
    let product = einsum("ik,kj->ij", &[&a, &b])?;
    let row_sums = reduce(&a, &[0, 1], &[0], ReductionOp::Sum)?;
    let transposed = permute(&a, &[1, 0])?;

    println!("product: {:?}", into_data_sync(product).to_vec::<f32>()?);
    println!("row sums: {:?}", into_data_sync(row_sums).to_vec::<f32>()?);
    println!("transpose: {:?}", into_data_sync(transposed).to_vec::<f32>()?);
    Ok(())
}
```

`einsum("ik,kj->ij", ...)` summiert über k und gibt die Form `[2, 2]` zurück. `reduce` behält Modus 0 bei und reduziert Modus 1, wodurch die Form `[2]` zurückgegeben wird. `permute` gibt einen neu zugewiesenen `[3, 2]`-Tensor zurück, keine Ansicht, die den Eingabespeicher gemeinsam nutzt.

## 3. Einsum-Ausdrücke

`einsum(expression, inputs)` akzeptiert eine oder mehrere Eingaben. Groß-/Kleinschreibung beachtende Buchstaben kennzeichnen Achsen; Auf der rechten Seite des Pfeils werden Ausgabeachsen ausgewählt und angeordnet.

|Ausdruck|Vorgang|
| --- | --- |
|`ik,kj->ij`|Matrixmultiplikation|
|`...ik,...kj->...ij`|Matrixmultiplikation mit Broadcast-Batch-Achsen|
|`abc,cde->abde`| Mehrdimensionale Tensorkontraktion |
|`ij,jk,kl->il`| Kontraktion mit drei Eingaben |
|`i,j->ij`| Äußeres Produkt |
|`ii->i`| Diagonale |
|`ii->`|Trace, der einen Skalar mit Rang Null zurückgibt|
|`ijk->ki`| j reduzieren und übrige Achsen umordnen |
|`...i->i`| Alle durch die Ellipse bezeichneten Achsen reduzieren |

- Übereinstimmungsmodi zwischen Eingaben müssen gleiche Ausdehnungen oder eine Ausdehnung von 1 haben. Ellipsenachsen werden mit rechter Ausrichtung übertragen.
- Wiederholte Beschriftungen innerhalb einer Eingabe wählen eine Diagonale aus; Diese Achsen müssen genau die gleichen Ausmaße haben.
- Ausgabebezeichnungen müssen eindeutig und in den Eingaben vorhanden sein.
- Ohne `->` beginnen die Ausgabeachsen mit den Ellipsenachsen, gefolgt von alphabetisch sortierten Beschriftungen, die genau einmal vorkommen.
- Eine Skalareingabe verwendet ein leeres Beschriftungssegment. Beispielsweise multipliziert `,ij->ij` die erste Skalareingabe in die Matrix.
- Eingaben können nicht zusammenhängend sein. Die Berechnung liest keine Tensorwerte zurück zum Host.

Für feste Ausdrücke, Formen, Schritte und dtypes konstruieren Sie `EinsumPlan::new(expression, descriptors)` und verwenden Sie `execute(inputs)` wieder. Um die Ausgabe- und arithmetische Genauigkeit anzugeben, verwenden Sie `EinsumPlan::with_options` oder `einsum_with_options`.

## 4. Deskriptoren und Ausführungspläne

Ein `Mode` ist ein `i32`-Label. Die gleiche Bezeichnung identifiziert die gleiche logische Achse über alle Tensoren hinweg, unabhängig von ihrer physischen Position. `TensorDescriptor` speichert Ausmaße, Elementschritte und Speicher dtype. `OperandDescriptor` fügt Beschriftungen und eine unäre Eingabetransformation hinzu.

Diese Funktionen erstellen und führen einen Plan für `D = alpha * A @ B + beta * C` aus:

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rutensor::{
    ComputeType, DType, OperandDescriptor, OperationDescriptor,
    Plan, Result, TensorDescriptor,
};

fn make_plan<R: Runtime>(
    a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    m: usize, n: usize,
) -> Result<Plan> {
    let operation = OperationDescriptor::contraction(
        OperandDescriptor::from_tensor(a, &[0, 2])?,
        OperandDescriptor::from_tensor(b, &[2, 1])?,
        Some(OperandDescriptor::from_tensor(c, &[0, 1])?),
        TensorDescriptor::contiguous(&[m, n], DType::F32)?,
        &[0, 1],
        ComputeType::F32,
    )?;
    Plan::new(operation)
}

fn execute<R: Runtime>(
    plan: &Plan, a: &RudaTensor<R>, b: &RudaTensor<R>, c: &RudaTensor<R>,
    alpha: f64, beta: f64,
) -> Result<RudaTensor<R>> {
    plan.execute(&[a, b, c], &[alpha, beta])
}
```

Pläne behalten keine Eingabepuffer bei. Nachfolgende Ausführungen können unterschiedliche Tensoren mit passenden Formen, Schritten und D-Typen verwenden. Alle Eingaben müssen sich auf demselben Gerät befinden.

|Konstruktor|Ausführungseingabereihenfolge|Skalare Reihenfolge|
| --- | --- | --- |
|`contraction`| A, B; optional C | alpha; beta bei vorhandenem C |
|`sum_product`| Alle Produkteingaben; optional C | alpha; beta bei vorhandenem C |
|`reduction`| A; optional C | alpha; beta bei vorhandenem C |
|`permutation`| A | alpha |
|`elementwise_binary`| A, B | alpha, beta |
|`elementwise_trinary`| A, B, C | alpha, beta, gamma |

`Plan::execute` ordnet die Ausgabe zu. `Plan::execute_into` akzeptiert einen Ausgabetensor und gibt ihn zurück, dessen Form, Schritte und dtype mit dem Ausgabedeskriptor übereinstimmen. Sein Puffer muss exklusiver Eigentümer sein und darf nicht mit Eingaben oder anderen Ansichten geteilt werden.

Verwenden Sie `TensorDescriptor::new(extents, strides, dtype)` für aufgefüllte oder neu angeordnete Ausgabelayouts. Abtriebsachsen dürfen sich nicht überlappen. Explizite Operationsdeskriptoren können über die Ausgabeform zusätzliche Broadcast-Achsen einführen.

## 5. Reduktion und elementweise Operationen

`reduce(input, input_modes, output_modes, operation)` reduziert Etiketten, die in der Ausgabe fehlen; reduzierte Achsen werden entfernt:

|ReductionOp|Vorgang| Leere Reduktion |
| --- | --- | --- |
|`Sum`|Summe|0|
|`Product`| Produkt |1|
|`Min`|Mindestens|Positive Unendlichkeit|
|`Max`|Maximal|Negativ unendlich|

`elementwise_binary` und `elementwise_trinary` richten benannte Achsen aus und übertragen sie unter Verwendung von `BinaryOp::{Add, Mul, Min, Max}`. Trinäre Operationen werten `(alpha * op(A) op_ab beta * op(B)) op_abc gamma * op(C)` aus.

Wählen Sie `Identity`, `Negate`, `Abs`, `Sqrt`, `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Relu`, `Reciprocal` oder `Conjugate` bis `OperandDescriptor::with_unary`. `Log` ist der natürliche Logarithmus; `Conjugate` entspricht `Identity` für reale Werte. Min und Max verbreiten NaNs.

## 6. Präzision, Speicher und Fehler

- Speichertypen sind F16, BF16, F32 und F64. Quantisierte, ganzzahlige und komplexe Speicherung werden nicht akzeptiert.
- `ComputeType::F32` oder `F64` steuert die Eingabekonvertierung, unäre Transformationen, Produkte, Reduktionen und Skalararithmetik. F64-Eingaben erfordern eine F64-Berechnung.
- Standardmäßig wird die F64-Arithmetik verwendet, wenn eine Eingabe F64 ist, andernfalls F32. Identische Eingabespeichertypen behalten diesen Ausgabetyp bei; Gemischte Eingaben erzeugen F64, wenn eine Eingabe F64 ist, andernfalls F32.
- Fordern Sie eine explizite Ausgabe dtype für eine Speicherung mit höherer Genauigkeit an. Eine Erhöhung der Speichergenauigkeit kann die bereits in einer Eingabe verlorene Präzision nicht wiederherstellen.
- Die Ausgabekonvertierung erfolgt beim letzten Schreibvorgang. Eingaben mit einem anderen dtype als dem Rechentyp erfordern Gerätekonvertierungspuffer; übereinstimmende Eingaben werden in ihrem ursprünglichen Layout gelesen.
- Reduzierungen nutzen die parallele Zusammenführung von Arbeitsgruppenbäumen. Die Reihenfolge der Gleitkomma-Addition und -Multiplikation kann von der seriellen Auswertung abweichen.
- Leere Ausgaben übermitteln keine Berechnung; Ausgänge mit Rang Null enthalten einen Skalar. Form-, Schritt- und Adressberechnungen verwenden die Plattform `usize`; Dispatches unterliegen auch Backend-Ressourcenlimits.
- Zurückgegebene `Result`-Werte melden Ausdrucks-, Deskriptor-, Form-, Geräte- und Versandbereichsfehler. Die Geräteübermittlung folgt der Fehlerbehandlung der Laufzeit; Asynchrone Ausführungsfehler treten bei der Synchronisierung oder dem Rücklesen auf.
- Allgemeine Kontraktionen durchlaufen direkt Reduktionskoordinaten. Sie suchen nicht nach Kontraktionspfaden mit mehreren Eingaben und senken sich nicht automatisch auf die Tensor-Core-Matrixmultiplikation ab. Die Speicherkonvertierung erfordert zusätzlichen Gerätespeicher.

ruTENSOR macht einen Ruda Rust API sichtbar, nicht den NVIDIA cuTENSOR C ABI. Das Gerät muss die ausgewählten Speicher- und Rechentypen unterstützen.
