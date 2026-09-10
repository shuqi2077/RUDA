# ruda-test-utils

[English](../../README.md) | [日本語](../ja/README.md) | **Deutsch** | [Русский](../ru/README.md)

Gemeinsame Bausteine für Kernel-Tests in Ruda: Test-Tensor-Builder,
hostseitige Referenzvergleiche und ein einheitlicher Renderer, der Tensoren
unter einer einzigen Konfiguration hübsch ausgibt (oder unterscheidet).

---

## Konfiguration: `ruda-test.toml`

Konfigurieren Sie Testrichtlinien und Tensordruck mit einer `ruda-test.toml` -Datei im Stammverzeichnis
des Arbeitsbereichs. Beim ersten Zugriff geht der Loader vom aktuellen Arbeitsverzeichnis nach
oben, bis er die Datei findet, und speichert dann die Konfiguration zwischen.

Die Konfiguration besteht aus zwei Abschnitten:

```toml
[test]
policy = "correct"   # "correct" | "strict" | "fail-if-run"

[print]
enabled = false       # toggle all printing
view = "table"        # "table" | "lines"
force-fail = true     # reject passing outcomes in correct/strict print mode
fail-only = false     # diff: only render cells where Δ > ε
show-expected = false # diff: render `got/expected` per cell (else just `got`)
filter = ""           # per-axis filter, same DSL as the slice helper
```

**Die gesamte Pipeline folgt einer Regel:** Bei `enabled = false` wird nichts gedruckt. Stellen Sie es auf `true`
ein, führen Sie einen Test durch und beobachten Sie, wie Ihre Tensoren gerendert werden. Das ist es.

### `[test] policy`

|Richtlinie|Kein Fehler|Numerischer Fehler|Kompilierungsfehler|
| ------------- | -------- | --------------- | ----------------- |
|`correct`|akzeptieren|fehlgeschlagen|akzeptieren|
|`strict`|akzeptieren|fehlgeschlagen|fehlgeschlagen|
|`fail-if-run`|fehlgeschlagen|akzeptieren|akzeptieren|

Bei aktiviertem Drucken und `force-fail = true` lehnen die Richtlinien `correct` und
`strict` bestandene Ergebnisse und Kompilierungsfehler ab, selbst wenn das Rendern
übersprungen wurde. Die Richtlinie `fail-if-run` bleibt durch diese Einstellung unverändert.

---

## Rendering: ein Pfad für alles

Sowohl `assert_equals_approx(actual, expected, ε)` als auch das kostenlose `print_tensors(label, &[&a, &b], Some(ε))`
durchlaufen denselben Renderer. Es gibt keinen „Diff-Pfad“ vs.
„Pretty-Print-Pfad“; Der Vergleich von tatsächlich und erwartet und
das hübsche Drucken zweier unabhängiger Tensoren gleicher Form
sind im wahrsten Sinne des Wortes derselbe Aufruf.

Regeln:

- Ein Tensor → nur Werte, keine Farbe.
- Zwei Tensoren mit demselben Rang und derselben Form** → endliche Werte sind grün gefärbt
  bei `Δ ≤ max(ε, ε × |expected|)` und ansonsten rot. Bei `show-expected = true` wird die Zelle angezeigt
  `got/expected`; sonst nur `got`.
- Zwei Tensoren von **unterschiedlichem Rang oder unterschiedlicher Form** → stillschweigend übersprungen. Der
  -Renderer gerät bei fehlerhaften Eingaben nie in Panik.
- Druckfilterrang ≠ Tensorrang → Rendering wird übersprungen. Eine Nichtübereinstimmung
  gibt `ValidationResult::Error` zurück.

```rust
use ruda_test_utils::print_tensors;

// Single tensor — table or lines per [print] view, no color.
print_tensors("input", &[&host], None);

// Two tensors — colored diff. Same path used by assert_equals_approx.
print_tensors("a vs b", &[&a, &b], Some(1e-3));
```

In der Tabellenansicht werden niemals Δ/ε-Zahlen angezeigt (die Zellenfarbe
trägt die Informationen). Die Linienansicht zeigt sie immer an.

### Beispiel für eine Tabellenansicht (mit `show-expected = true`)

```
=== diff  shape=[2, 3] ===
    |                 0                 1                 2
----+------------------------------------------------------
  0 | 0.000000/0.000000 1.000000/1.000000 2.000000/2.000000   ← green
  1 | 4.000000/3.000000 5.000000/4.000000 6.000000/5.000000   ← red
```

### Tabellenansicht + `fail-only = true`

```
=== diff  shape=[2, 3] ===
    |        0        1        2
----+---------------------------
  0 |                            ← matching cells blanked out
  1 | 4.000000 5.000000 6.000000 ← red
```

### Linienansicht + `fail-only = true`

```
=== diff  shape=[2, 3] ===
 index |      got | expected |        Δ |        ε | status
-----------------------------------------------------------
[1, 0] | 4.000000 | 3.000000 | 1.000000 | 0.003000 | FAIL    ← red
[1, 1] | 5.000000 | 4.000000 | 1.000000 | 0.004000 | FAIL    ← red
[1, 2] | 6.000000 | 5.000000 | 1.000000 | 0.005000 | FAIL    ← red
```

---

## Filtersyntax

Wird sowohl von `[print] filter` als auch von
`assert_equals_approx_in_slice` verwendet. Eine durch Kommas getrennte Liste dim-Einträge:

- `.` – Platzhalter (beliebiger Index entlang dieser Dim)
- `N` – ein einzelner Index
- `M-K` – inklusive Bereich

Beispiel für einen 4-D-Tensor: `.,.,10-20,30` wählt alle Elemente aus,
bei denen Dim 2 in `10..=20` und Dim 3
genau `30` ist. Der Filterrang muss dem Tensorrang entsprechen.

Von Rust:

```rust
use ruda_test_utils::{DimFilter, assert_equals_approx_in_slice};

// Vec<Range<usize>> works (half-open, like Rust slices).
assert_equals_approx_in_slice(&actual, &expected, 0.001, vec![0..1, 0..3]);

// Or build the canonical TensorFilter explicitly.
let filter = vec![
    DimFilter::Exact(0),
    DimFilter::Range { start: 0, end: 2 }, // inclusive: 0..=2
];
assert_equals_approx_in_slice(&actual, &expected, 0.001, filter);
```

`parse_tensor_filter("0,0-2")` analysiert die Zeichenfolge DSL in einen `TensorFilter`.

---

## Fehlermeldungen

`assert_equals_approx` gibt einen `ValidationResult` zurück und erfasst bis zu **8**
Nichtübereinstimmungen plus aggregierte Statistiken. Durch den Aufruf von `.as_test_outcome().enforce()` wird
die Testrichtlinie angewendet und bei einem abgelehnten Ergebnis in Panik versetzt:

```
Test failed: Got incorrect results: 17/4096 elements mismatched
  (max |Δ|=0.014648, mean |Δ|=0.004112, worst at [3, 12]) — shape=[16, 256]
First mismatches:
  [0, 5]: got 1.234, expected 1.220, |Δ|=0.014 > ε=0.001
  ...
  ... and 9 more
```

Wenn das Drucken aktiviert ist, wird die Ausgabe pro Element an stdout gesendet.
Die Paniknachricht behält nur den aggregierten Header, sodass der Dump nicht dupliziert wird.

---

## Tests werden ausgeführt

Vom RUDA-Arbeitsbereichsstammverzeichnis ausführen und eine Testlaufzeit auswählen:

```sh
cargo test --locked -p ruda-test-utils --test lib --features ruda-test-runtime/cuda
```

Die Laufzeitfunktion kann `cpu`, `cuda`, `hip` oder `wgpu` sein.

---

## Testeingaben werden erstellt

Zwei gleichwertige Möglichkeiten zum Aufbau eines Testtensors:

```rust
use ruda_kernel::dsl::prelude::RudaPrimitive;
use ruda_test_utils::{TestInput, StrideSpec, DataKind, Distribution};

// Long-form constructor.
let (handle, host) = TestInput::new(
    client.clone(),
    [4, 4],
    f32::as_type_native_unchecked().storage_type(),
    StrideSpec::RowMajor,
    DataKind::Random {
        seed: 0,
        distribution: Distribution::Uniform(-1.0, 1.0),
    },
)
.generate_with_f32_host_data();

// Fluent builder — `dtype` defaults to f32, `stride` defaults to RowMajor.
let (handle, host) = TestInput::builder(client.clone(), [4, 4])
    .uniform( 0, -1.0, 1.0)
    .generate_with_f32_host_data();
```

Builder-Setter (alle optional):

|Setter|Standard|Effekt|
| --------------- | ---------------------- | --------------------------- |
|`.dtype(d)`|`f32`|Überschreibt die Eingabe dtype.|
|`.stride(spec)`|`StrideSpec::RowMajor`|Überschreiben Sie das Schrittlayout.|

Builder-Finalisierer (jeder gibt einen `TestInput` zurück, der zur Generierung bereit ist):

|Finalizer|Äquivalent `DataKind`|
| -------------------------- | ---------------------------------------------------------------- |
|`.arange()`|`Arange { scale: None }`|
|`.arange_scaled(s)`|`Arange { scale: Some(s) }`|
|`.eye()`|`Eye`|
|`.zeros()`|`Zeros`|
|`.uniform(seed, lo, hi)`|`Random { Uniform(lo, hi) }`|
|`.bernoulli(seed, p)`|`Random { Bernoulli(p) }`|
|`.normal(seed, mean, std)`|`Random { Normal { mean, std } }`|
|`.random(seed, dist)`|`Random { dist }`|
|`.linspace(start, end)`|`Custom { data }` mit N gleichmäßig verteilten Werten von `start..=end`|
|`.custom(data)`|`Custom { data }`|

Rufen Sie nach einem Finalizer Folgendes
auf: `.generate()` , `.generate_with_f32_host_data()` , `.generate_with_bool_host_data()`
, `.generate_test_tensor()` , `.f32_host_data()` , `.bool_host_data()` .
