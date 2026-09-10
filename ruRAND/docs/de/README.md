# ruRAND

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

**Englisch** | [简体中文](../zh/README.md)

Zufallszahlengenerierung für Ruda.

- Cargo Paket: `ruRAND`
- Rostkiste: `rurand`

## Features

| Feature |Operationen|
| --- | --- |
|`tensor`|Gleichmäßige, normale und Bernoulli-Verteilungen auf Gerätetensoren|

Verwenden Sie `rurand::tensor::{random_uniform, random_normal, random_bernoulli}`, um Tensoren oder die Funktionen am Crate-Stamm zuzuweisen, um den vorhandenen Gerätespeicher zu füllen. `rurand::seed` legt den gemeinsamen Zufallsstartwert fest.

## Schnellstart

Build aus dem RUDA-Arbeitsbereich:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruRAND --no-default-features --features std,tensor
```

## Dokumentation

- [Benutzerhandbuch](../../../docs/de/libraries/rurand.md)
- [Umgebungseinrichtung](../../../docs/de/getting-started.md)
- [Cargo-Funktionen](../../Cargo.toml) · [Modulexporte](../../src/lib.rs)

## ruRAND Benutzerhandbuch

[Compute-Bibliotheken](../../../docs/de/libraries/README.md) · [Laufzeit API](../../../docs/de/runtime-api.md) · [中文](../zh/README.md)

ruRAND generiert Gerätetensoren mit gleichmäßiger, normaler und Bernoulli-Verteilung. Verwenden Sie `rurand::tensor`, um neue Tensoren zuzuweisen, oder die gleichnamigen Funktionen im Crate-Stamm, um den vorhandenen Gerätespeicher zu füllen.

### 1. Abhängigkeiten konfigurieren

Das Cargo-Paket ist `ruRAND`; Sein Rust-Importname ist `rurand`. Die Funktion `tensor` ermöglicht Geräte-Tensor-Schnittstellen. Bei dieser Konfiguration wird das Anwendungsverzeichnis neben dem Quellverzeichnis `RUDA` platziert. Informationen zum NVIDIA-Setup finden Sie unter [Erste Schritte](../../../docs/de/getting-started.md).

```toml
[dependencies]
rurand = { package = "ruRAND", path = "../RUDA/ruRAND", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Generieren Sie zufällige Tensoren

Dieser vollständige `src/main.rs` generiert F32-Tensoren der Form `[2, 3]` und spielt einen Aufruf durch Zurücksetzen des Startwerts ab. Führen Sie `cargo run` aus dem Anwendungsverzeichnis aus:

```rust
use ruda_core::tensor::DType;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::readback::into_data_sync;
use rurand::tensor::{
    random_bernoulli, random_like_uniform, random_normal, random_uniform,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    rurand::seed(42);
    let uniform = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let first = into_data_sync(uniform.clone()).to_vec::<f32>()?;

    rurand::seed(42);
    let repeated = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let repeated = into_data_sync(repeated).to_vec::<f32>()?;
    assert_eq!(first, repeated);
    assert!(first.iter().all(|&x| (0.0..1.0).contains(&x)));

    let normal = random_normal::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let mask = random_bernoulli::<CudaRuntime>([2, 3].into(), &device, 0.25, DType::F32);
    let noise = random_like_uniform(&uniform, -0.1, 0.1, DType::F32);

    let mask = into_data_sync(mask).to_vec::<f32>()?;
    assert!(mask.iter().all(|&x| x == 0.0 || x == 1.0));
    println!("uniform={first:?}");
    println!("normal={:?}", into_data_sync(normal).to_vec::<f32>()?);
    println!("mask={mask:?}");
    println!("noise={:?}", into_data_sync(noise).to_vec::<f32>()?);
    Ok(())
}
```

`random_like_uniform` übernimmt nur Form und Gerät aus der Referenz. Es weist einen neuen Tensor zu, ohne die Referenzwerte zu ändern oder zu kopieren. Sein letztes Argument wählt explizit die Ausgabe dtype aus.

`random_bernoulli` gibt Nullen und Einsen im ausgewählten dtype zurück, keine automatisch erstellte boolesche Maske. In diesem Beispiel werden die F32-Werte 0.0 und 1.0 zurückgegeben.

### 3. Parameter und Verteilungen

Die ersten drei Tensorfunktionen teilen sich `shape: Shape`, `device: &R::Device` und einen letzten `dtype: DType`-Parameter. Der zurückgegebene `RudaTensor<R>` hat die angegebene Form, dtype und das angegebene Gerät.

|Funktions- und Parameterreihenfolge|Verteilungsparameter|
| --- | --- |
|`random_uniform(shape, device, lower_bound, upper_bound, dtype)`|f32 untere und obere Grenzen|
|`random_normal(shape, device, mean, std, dtype)`|f32 Mittelwert und Standardabweichung; Standard ist keine Varianz|
|`random_bernoulli(shape, device, probability, dtype)`|f32 Rückgabewahrscheinlichkeit 1|
|`random_like_uniform(&reference, lower_bound, upper_bound, dtype)`|Referenztensor, f32-Grenzen, Ausgabe dtype|

Geben Sie endliche Parameter an, die mit der Verteilung übereinstimmen: Untergrenze unter Obergrenze, nichtnegative normale Standardabweichung und Bernoulli-Wahrscheinlichkeit in `[0, 1]`. Diese Schnittstellen validieren diese Bereiche nicht über `Result`; Verwenden Sie keine ungültigen Parameter, um eine Fehlerrückgabe anzufordern.

Die einheitliche Generierung erzeugt zunächst F32 `u ∈ [0, 1)`, berechnet `lower_bound + (upper_bound - lower_bound) × u` und wandelt sie dann in die Ausgabe dtype um. Der Fall F32 `[0, 1)` schließt 1 aus. Die Skalierung auf andere Intervalle oder die Konvertierung in eine niedrigere Genauigkeit kann einen Wert in der Nähe der Obergrenze auf diese Grenze runden.

Die normale Generierung verwendet eine FP32 Box-Muller-Transformation vor der Umwandlung in die Ausgabe dtype. Durch die Auswahl des F64-Speichers wird die interne Zufalls-Gleitkommagenauigkeit nicht erhöht. Bernoulli vergleicht `u < probability`, sodass Wahrscheinlichkeit 0 alle Nullen und Wahrscheinlichkeit 1 alle Einsen erzeugt.

### 4. Seeds und Aufrufreihenfolge

`rurand::seed(seed: u64)` setzt den zufälligen Status des gemeinsam genutzten Hosts zurück. Jeder Generationsaufruf zieht neue Samen aus diesem Zustand und treibt ihn voran:

- Legen Sie den Startwert einmal fest, um eine Sequenz zu initialisieren und dann aufeinanderfolgende Stapel zu generieren.
- Um eine Sequenz wiederzugeben, setzen Sie denselben Startwert zurück und behalten Sie die Aufrufreihenfolge, Form, dtype und Gerätekonfiguration bei.
- Setzen Sie den gleichen Seed nicht vor jedem Trainingsbatch zurück, es sei denn, wiederholte Zufallseingaben sind beabsichtigt.
- Gleichzeitige Threads teilen diesen Status; Die Verschachtelung beeinflusst, welche Seeds jeder Aufruf erhält. Es handelt sich nicht um ein separates Generatorobjekt pro Thread oder Gerät.

Der obige Wiedergabevergleich verwendet einen Prozess, ein Gerät und passende Argumente. Übereinstimmende Seeds allein bedeuten keine elementare Gleichheit mit einer anderen zufälligen Bibliothek oder einem anderen Backend.

### 5. Füllen Sie einen vorhandenen Puffer

Verwenden Sie Crate-Root-Startfunktionen, um eine Ausgabezuordnung wiederzuverwenden. Diese Funktion nimmt einen bereits zugewiesenen F32-Gerätetensor, schreibt Zufallswerte in seinen Speicher und gibt ihn zurück:

```rust
use ruda_kernel::{
    dsl::{Runtime, prelude::LaunchError},
    tensor::RudaTensor,
};

fn fill_uniform<R: Runtime>(
    output: RudaTensor<R>,
    lower: f32,
    upper: f32,
) -> Result<RudaTensor<R>, LaunchError> {
    rurand::random_uniform(
        &output.client,
        lower,
        upper,
        output.clone().binding(),
        output.dtype.into(),
    )?;
    Ok(output)
}
```

Die Funktion ruft keinen Tensor-Allokator auf. Andere Handles, die diesen Speicher gemeinsam nutzen, beobachten die Schreibvorgänge ebenfalls. Entsprechende Funktionen für Normal- und Bernoulli-Werte sind `rurand::random_normal(&client, mean, std, binding, dtype)` und `rurand::random_bernoulli(&client, probability, binding, dtype)`.

Funktionen auf Tensorebene geraten in Panik, wenn beim Starten ein Fehler zurückgegeben wird. Der zugrunde liegende Einstiegspunkt oben gibt `Result<(), LaunchError>` zurück. Eine erfolgreiche Übermittlung bedeutet nicht, dass die Ausführung von GPU abgeschlossen ist. Lesen Sie den Tensor zurück oder warten Sie auf die Client-Synchronisierung. `into_data_sync` gerät bei Rücklesefehler in Panik; Verwenden Sie asynchrones `into_data(tensor).await`, um Rücklesefehler weiterzugeben.

API Referenz: [Tensorschnittstellen](../../src/tensor/mod.rs), [Seed-Status](../../src/state.rs), [Verteilungen](../../src/distributions/mod.rs).
