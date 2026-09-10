# ruFFT

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

**Englisch** | [简体中文](../zh/README.md)

Schnelle Fourier-Transformationen für Ruda.

- Cargo Paket: `ruFFT`
- Rostkiste: `rufft`

## Features

| Feature |Operationen|
| --- | --- |
|`tensor`|Realsignal-FFTs und inverse Transformationen auf Gerätetensoren|

Verwenden Sie `rufft::tensor::{rfft, irfft}`, um Transformationsausgänge zuzuweisen, oder `rfft_launch` und `irfft_launch`, um direkt mit Gerätebindungen zu arbeiten.

## Schnellstart

Aus dem RUDA-Arbeitsbereich erstellen:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruFFT --no-default-features --features std,tensor
```

## Dokumentation

- [Benutzerhandbuch](../../../docs/de/libraries/rufft.md)
- [Umgebungseinrichtung](../../../docs/de/getting-started.md)
- [Cargo-Funktionen](../../Cargo.toml) · [Modulexporte](../../src/lib.rs)

## ruFFT Benutzerhandbuch

[Computerbibliotheken](../../../docs/de/libraries/README.md) · [Tensor-Framework](../../../docs/de/tensor-framework.md) · [中文](../zh/README.md)

ruFFT berechnet Realsignal-FFTs und inverse Transformationen auf dem Gerät. Die `rufft::tensor`-Schnittstelle weist Ausgänge zu; Verwenden Sie `rfft_launch` und `irfft_launch`, wenn Sie Gerätebindungen direkt verwalten.

### 1. Abhängigkeiten konfigurieren

Das Cargo-Paket ist `ruFFT`; Sein Rust-Importname ist `rufft`. Die Funktion `tensor` ermöglicht Geräte-Tensor-Schnittstellen. Diese Konfiguration platziert das Anwendungsverzeichnis neben dem `RUDA`-Quellverzeichnis. Informationen zum NVIDIA-Setup finden Sie unter [Erste Schritte](../../../docs/de/getting-started.md).

```toml
[dependencies]
rufft = { package = "ruFFT", path = "../RUDA/ruFFT", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Vorwärts- und Rücktransformationen

Dieser vollständige `src/main.rs` lädt ein Vierpunktsignal F32 hoch, berechnet sein Spektrum und rekonstruiert die Eingabe. Führen Sie `cargo run` aus dem Anwendungsverzeichnis aus:

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rufft::tensor::{irfft, rfft};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = vec![1.0f32, 0.0, -1.0, 0.0];
    let signal = from_data::<CudaRuntime>(TensorData::new(input.clone(), [4]), &device);
    let (real, imag) = rfft(signal, 0, None);

    let re = into_data_sync(real.clone()).to_vec::<f32>()?;
    let im = into_data_sync(imag.clone()).to_vec::<f32>()?;
    for (actual, expected) in re.iter().zip([0.0f32, 2.0, 0.0]) {
        assert!((actual - expected).abs() < 1e-5);
    }
    assert!(im.iter().all(|value| value.abs() < 1e-5));

    let restored = irfft(real, imag, 0, Some(input.len()));
    let restored = into_data_sync(restored).to_vec::<f32>()?;
    for (actual, expected) in restored.iter().zip(input) {
        assert!((actual - expected).abs() < 1e-5);
    }
    println!("real={re:?}, imag={im:?}, restored={restored:?}");
    Ok(())
}
```

Ein Vierpunktsignal erzeugt drei Bins: DC, ein Bin mit positiver Frequenz und Nyquist. Real- und Imaginärkomponenten sind separate Tensoren, keine verschachtelten komplexen Werte. Die Vorwärtstransformation ist nicht normalisiert; Die Umkehrung beinhaltet die Normalisierung für die tatsächliche FFT-Länge, daher nicht erneut durch die Länge dividieren.

Das Beispiel verwendet `clone()`, um Tensor-Handles für die inverse Transformation beizubehalten. `into_data_sync` wartet auf das Zurücklesen und gibt Host-Daten zurück; Es gerät in Panik, wenn der Rücklesefehler fehlschlägt.

### 3. Parameter und Ausgabeformen

|Funktion|Parameter|Gibt zurück|
| --- | --- | --- |
|`rfft(signal, dim, n)`|Signal, nullbasierte Achse, optional angeforderte Länge|Zwei Gerätetensoren `(real, imag)`|
|`irfft(real, imag, dim, n)`|Realteil, Imaginärteil, Achse, optionale Ausgabelänge|Ein reeller Gerätetensor|

Geräteberechnung verwendet F32; Bereitstellung von F32-Signalen und -Spektren. `dim` muss kleiner als der Eingaberang sein. Inverse Transformationskomponenten müssen eine passende Form, dtype und ein Gerät haben. Diese Schnittstellen geben Tensoren anstelle von `Result` zurück; Argumentationsbehauptungen scheitern oder Panik auslöst.

Für die angeforderte Länge n ist die tatsächliche FFT-Länge N die kleinste Zweierpotenz größer oder gleich n:

- Mit `n = None` verwendet `rfft` die Eingabelänge entlang des Dims. Andernfalls wird zuerst n gekürzt oder mit Nullen aufgefüllt, dann wird N mit Nullen aufgefüllt.
- Die Vorwärtsausgabelänge entlang des Dims beträgt `N / 2 + 1`; andere Abmessungen bleiben unverändert.
- Mit `n = None` verwendet `irfft` `2 × (bin count - 1)`. Ein explizites n gibt die zurückgegebene Länge an.
- Die Umkehrung schneidet zunächst beide Spektrumkomponenten ab oder füllt sie mit Nullen auf `N / 2 + 1`-Bins auf, berechnet die N-Punkt-Umkehrung und schneidet sie dann auf n zu.
- Die angeforderte Länge muss mindestens 2 betragen; Eine Ein-Punkt-Transformation besteht die Längenprüfung des Gerätekernels nicht.

|Eingabe/Aufruf|Tatsächliche Transformationslänge|Ausgabelänge entlang Bemaßung|
| --- | --- | --- |
|Länge 8, `rfft(..., None)`|8|5|
|Länge 8, `rfft(..., Some(6))`|8; Behalten Sie nur die ersten 6 Eingabepunkte und den Nullpunkt bei|5|
|Länge 6, `rfft(..., None)`|8, mit Nullen aufgefüllt|5|
| 5 Frequenzbins, `irfft(..., None)` |8|8|
| 5 Frequenzbins, `irfft(..., Some(6))` |8|6|

Somit fordert `Some(6)` keinen Sechspunkt-DFT an. Für einen Roundtrip eines ursprünglichen Sechspunktsignals übergeben Sie `Some(6)` an die Umkehrung, um die beiden gepolsterten Endpunkte zu entfernen.

### 4. Stapel und mehrdimensionale Eingabe

Andere Achsen als dim sind Chargenmaße. Beispielsweise erzeugt die Transformation von `[batch, channels, 1024]` entlang dim=2 reale und imaginäre Tensoren der Form `[batch, channels, 513]`.

Diese Funktion verwendet den vorherigen `rfft`-Import wieder und akzeptiert einen Gerätetensor, ohne ihn zurückzulesen:

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

fn batched_rfft<R: Runtime>(
    signals: RudaTensor<R>,
    dim: usize,
    length: usize,
) -> (RudaTensor<R>, RudaTensor<R>) {
    rfft(signals, dim, Some(length))
}
```

Ein einzelner `rfft`-Aufruf transformiert nur die ausgewählte Achse, nicht einen gesamten mehrdimensionalen FFT. Behandeln Sie die getrennten Real- und Imaginärkomponenten nicht als vollständige komplexe Signale, indem Sie sie jeweils unabhängig an `rfft` übergeben.

### 5. Puffer und Ausführung

Tensorschnittstellen weisen Ausgänge zu und führen bei Bedarf Auffüllen und Zuschneiden durch. Tatsächliche Längen über 4096 verwenden automatisch den bereitgestellten Pfad, ohne den Aufruf zu ändern. Um Ausgabepuffer selbst zu verwalten, akzeptieren die Startschnittstellen einen Client, geben `TensorBinding`-Werte, Dim und `StorageType` ein/aus und geben `Result<(), LaunchError>` zurück. Ordnen Sie passende Ausgabelayouts für das tatsächliche N zu.

Um zu vermeiden, dass beim Verwalten von Puffern Eingaben mit Nullen aufgefüllt werden, verwenden Sie diese Einstiegspunkte:

|Funktion|Zusätzlicher Längenparameter|
| --- | --- |
|`rfft_launch_padded(client, signal, real, imag, dim, signal_len, dtype)`|Liest nur die ersten signal_len-Signalelemente und behandelt den Rest als Null; N wird aus der Länge des Ausgangsspektrums abgeleitet|
|`irfft_launch_padded(client, real, imag, signal, dim, spec_bins, dtype)`|Liest nur die ersten spec_bins-Frequenzintervalle und behandelt den Rest als Null; N ergibt sich aus der Länge des Ausgangssignals|

N muss eine Zweierpotenz von mindestens 2 sein und reale/imaginäre Formen müssen übereinstimmen. `signal_len` darf die Länge der Eingabeachse oder N nicht überschreiten. `spec_bins` muss mindestens 1 sein und darf die Länge der Eingabespektrumsachse oder `N / 2 + 1` nicht überschreiten. Ausgabepuffer erfordern weiterhin eine vollständige Zuweisung; Diese Schnittstellen vermeiden die Materialisierung des mit Nullen aufgefüllten Eingabeendes.

API Referenz: [Tensor-Schnittstelle](../../src/tensor.rs), [Vorwärtsstart](../../src/fft/rfft.rs), [Inverser Start](../../src/fft/irfft.rs).
