# ruDNN Benutzerhandbuch

[English](../../en/libraries/rudnn.md) | [简体中文](../../zh/libraries/rudnn.md) | [日本語](../../ja/libraries/rudnn.md) | **Deutsch** | [Русский](../../ru/libraries/rudnn.md)

[Computerbibliotheken](README.md) · [ruBLAS](rublas.md) · [Tensoren und Frameworks](../tensor-framework.md) · [中文](../../zh/libraries/rudnn.md)

## 1. Übersicht und Funktionen

ruDNN bietet neuronale Netzwerkoperationen. Das Cargo-Paket ist `ruDNN` und die Rust-Crate ist `rudnn`.

| Feature |Operationen|
| --- | --- |
|`tensor-attention`|Tensor-Aufmerksamkeit|
|`tensor-convolution`|Tensorfaltung|
|`pooling`, `interpolation`|Pooling und Interpolation|
|`grid-sample`, `ctc`|Rasterstichprobe und CTC|
|`tensor-moe`|Gerät MoE Routing und Expertenberechnung|
|`tensor-normalization`|Universelle Gerätenormalisierung für Modellebenen|
|`tensor-gated-delta`|Gated-Delta-Berechnung für Hybridarchitekturen wie Qwen3.5|

Aufmerksamkeit und Faltung verfügen jeweils über entsprechende Autotune-Funktionen. Siehe [Cargo.toml](../../../ruDNN/Cargo.toml) und [Modulexporte](../../../ruDNN/src/lib.rs).

## 2. Attention und Faltung

### Attention

`rudnn::attention::tensor::attention` akzeptiert Abfrage, Schlüssel, Wert, optionale Maske, optionales attn_bias, AttentionModuleOptions und AttentionStrategy. Es gibt einen Gerätetensor oder AttentionSetupError zurück.

Die Strategien umfassen FlashBlackboxAccelerated, FlashUnit, Fallback und bei aktiviertem Feature Autotune. Ohne Autotuning ist Fallback der Standard, mit Autotuning Autotune. Fallback verwendet mehrere Kernels auf demselben Gerät, kein CPU-Backend.

Passen Sie Layout, Maske und Präzision an die ausgewählte Strategie an. Siehe die [Attention-Schnittstelle](../../../ruDNN/src/attention/tensor/base.rs).

### Faltung

`rudnn::convolution::tensor::conv_forward` übernimmt Eingabe, Gewicht, optional bias, ConvOptions<N> und ConvStrategy. Es gibt einen Gerätetensor oder ConvSetupError zurück. Es konvertiert Eingaben zur Ausführung in das Channel-Last-Layout und konvertiert die Ausgabe zurück. `conv_forward_nhwc` verwendet direkt das Channels-Last-Layout.

Zu den Strategien gehören Direct, ImplicitGemm und optional Autotune. Der Einstiegspunkt verwendet Direct für die dreidimensionale F32-Faltung. Die gruppierte Faltung verwendet auch Direct, wenn ImplicitGemm ausgewählt ist. Der Strategieparameter ist daher nicht immer eine strikte Anforderung, einen Algorithmus beizubehalten.

Das gleiche Modul stellt conv_data_backward und conv_weight_backward bereit. Überprüfen Sie Form, Optionen und Ausführungsanforderungen für jede Richtung. Siehe die [Faltungsschnittstelle](../../../ruDNN/src/convolution/tensor/base.rs).

## 3. MoE-Workflow

`rudnn::moe` stellt diese lokale Berechnungssequenz bereit:

Eingabe-Logits → softmax/top-k → kompakte Verteilung → SwiGLU-Experten → gewichtete Zusammenführung.

|API|Zweck|
| --- | --- |
|`route(logits, RoutingOptions)`|Erstellt ein RoutingPlan|
|`RoutingPlan::expert_indices()`, `weights()`|Greift auf ausgewählte Experten und Gewichtungen zu|
|`RoutingPlan::dispatch(input)`|Versendet Token durch Experten|
|`SwiGluExperts::new(gate, up, down)`|Erstellt bias-freie Expertengewichte|
|`SwiGluExperts::forward_dispatched`|Berechnet versendete Token|
|`DispatchedTokens::combine`|Stellt die Token-Reihenfolge wieder her und kombiniert gewichtete Ergebnisse|
|`SwiGluExperts::forward(input, logits, options)`|Führt die komplette lokale Vorwärtssequenz aus|

Diese Schnittstellen verwenden `RudaTensor<R>`. Berechnungseinstiegspunkte geben `Result` mit `MoeError` für ungültige Formen, D-Typen oder andere Argumente zurück.

## 4. Routing-Vertrag

Logits haben die Form [T, E] und verwenden unquantisiertes F32, F16 oder BF16. `RoutingOptions` enthält `top_k: usize` und `renormalize: bool`, wobei 1 ≤ top_k ≤ E gilt.

Softmax berechnet vor der Top-K-Auswahl alle Experten in FP32. Unentschieden begünstigen niedrigere Experten-IDs. Wenn die Renormierung aktiviert ist, werden ausgewählte Gewichtungen neu normalisiert und dann in die Logits dtype umgewandelt. Zeilen, die NaN, positive Unendlichkeit oder nur negative Unendlichkeit enthalten, behalten die Gewichtungen von NaN bei, anstatt eine gleichmäßige Verteilung zu verwenden.

Ausgewählte Expertenindizes und -gewichte haben beide die Form [T, top_k]. Indizes verwenden U32.

## 5. Expertengewichte und Verteilung

Eingabetokens haben die Form [T, H]; gate und up haben [E, I, H], down hat [E, H, I]. Gewichte müssen denselben Gleitkomma-dtype und dasselbe Gerät verwenden. Expertenanzahl und Dimensionen müssen zu Routing und Eingabe passen.

Die Verteilung verwirft keine Tokens zur Durchsetzung einer Kapazitätsgrenze. Die atomare Zuweisungsreihenfolge innerhalb eines Experten ist nicht festgelegt; eine gespeicherte Zuordnung stellt die Token-Reihenfolge wieder her. Der Vorwärtseinstiegspunkt nimmt vorberechnete Logits entgegen, statt die Gate-Projektion des Modells auszuführen oder Gewichtsdateien zu laden.

Quelle: [Routing](../../../ruDNN/src/moe/routing.rs), [Dispatch and Combine](../../../ruDNN/src/moe/dispatch.rs), [Experten](../../../ruDNN/src/moe/experts.rs) und [Tests](../../../ruDNN/src/moe/tests.rs).

## 6. MoE aufrufen

Aktivieren Sie `tensor-moe` für Ihre `ruDNN`-Abhängigkeit. Bereiten Sie Gerätetensoren mit den oben genannten Layouts vor, erstellen Sie dann Expertengewichte und führen Sie die Vorwärtsoperation aus:

```rust
use rudnn::moe::{RoutingOptions, SwiGluExperts};

let experts = SwiGluExperts::new(gate, up, down)?;
let options = RoutingOptions {
    top_k: 2,
    renormalize: true,
};
let output = experts.forward(input, logits, options)?;
```

Dadurch werden zwei Experten pro Token ausgewählt, daher muss `E` mindestens 2 sein. `input` hat die Form `[T, H]`, `logits` hat die Form `[T, E]` und `output` hat die Form `[T, H]`. Verwenden Sie `experts` in nachfolgenden Eingabestapeln wieder, ohne das Gewichtsobjekt neu zu erstellen.

Um das Routing zu überprüfen oder Ihre eigene Verarbeitung zwischen den Phasen einzufügen, rufen Sie diese separat auf:

```rust
use rudnn::moe::route;

let routing = route(logits, options)?;
let selected_experts = routing.expert_indices();
let selected_weights = routing.weights();
let dispatched = routing.dispatch(input)?;
let expert_output = experts.forward_dispatched(&dispatched)?;
let output = dispatched.combine(expert_output)?;
```

Dies sind alternative Formen; Die zweite beginnt mit einer neuen Charge `input` und `logits`. `combine` verwendet die Dispatch-Zuordnung, um die Token-Reihenfolge wiederherzustellen und führt Expertenausgaben mithilfe von Routing-Gewichten zusammen.

Form, dtype oder Gerätekonflikte geben `MoeError` zurück. Um Ergebnisse auf dem Host zu lesen, verwenden Sie `ruda_kernel::tensor::readback::into_data_sync(output)`, das auf Geräteergebnisse wartet und `TensorData` zurückgibt.

## 7. LayerNorm, RMSNorm und Softmax

Aktivieren Sie `tensor-normalization` und importieren Sie diese Funktionen aus `rudnn::normalization`. Alle arbeiten auf der endgültigen Eingabeachse und behalten die Form bei:

|Funktion|Eingabe|Parameter|
| --- | --- | --- |
|`layer_norm(input, gamma, beta, epsilon)`|F32/F16/BF16|F32 Vektor `gamma`, optionaler F32 Vektor `beta`|
|`rms_norm(input, gamma, epsilon)`|F32/F16/BF16|F32 Vektor `gamma`|
|`softmax_last_axis(input)`|F32|Keine zusätzlichen Parameter|

Die Eingabe muss unquantisiert sein und eine nicht leere Endachse haben. Für die Endachsenlänge H müssen `gamma` und `beta` die Form `[H]` haben, unquantisiert sein und sich das Eingabegerät teilen. `epsilon` muss endlich und positiv sein. Affine Parameter bleiben auch für BF16/F16-Eingänge F32. Statistik und affine Arithmetik verwenden FP32, wobei die Ausgabe in die Eingabe dtype umgewandelt wird.

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rudnn::normalization::{NormalizationError, layer_norm, rms_norm, softmax_last_axis};

fn normalize<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    beta: Option<RudaTensor<R>>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    layer_norm(input, gamma, beta, epsilon)
}

fn normalize_rms<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    rms_norm(input, gamma, epsilon)
}

fn probabilities<R: Runtime>(
    logits: RudaTensor<R>,
) -> Result<RudaTensor<R>, NormalizationError> {
    softmax_last_axis(logits)
}
```

## 8. Gated-Delta-Prefill und Rekurrenz

Aktivieren Sie `tensor-gated-delta`. Verwenden Sie `chunk_gated_delta_rule(input, chunk_size)` für das Vorfüllen von Chunked-Sequenzen und `gated_delta_rule(input)` für die Token-für-Token-Wiederholung. Beide nehmen `GatedDeltaInput<R>`:

|Feld| Form/Typ |
| --- | --- |
|`query`, `key`|`[B, H, T, K]`, passend zu F32/F16/BF16|
|`value`|`[B, H, T, V]`, gleicher dtype wie Abfrage|
|`beta`|`[B, H, T]`, gleicher dtype wie Abfrage|
|`log_decay`|`[B, H, T]`, F32|
|`initial_state`|`[B, H, K, V]`, F32|
|`query_scale`| Endlicher f32-Wert gemäß Modellkonfiguration |

Alle Tensoren müssen unquantisiert sein und sich auf demselben Gerät befinden. Angebot Q/K nach modellspezifischer Normalisierung; Der Einstiegspunkt übernimmt dies nicht für Sie. Diese Funktion verwendet die vorherigen `Runtime`- und `RudaTensor`-Importe wieder:

```rust
use rudnn::gated_delta::{
    GatedDeltaError, GatedDeltaInput, GatedDeltaOutput, chunk_gated_delta_rule,
};

fn delta_prefill<R: Runtime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    beta: RudaTensor<R>,
    log_decay: RudaTensor<R>,
    initial_state: RudaTensor<R>,
    query_scale: f32,
    chunk_size: usize,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    chunk_gated_delta_rule(
        GatedDeltaInput {
            query, key, value, beta, log_decay, initial_state, query_scale,
        },
        chunk_size,
    )
}
```

Das zurückgegebene `output` hat die Form `[B, H, T, V]` und denselben dtype wie query. `final_state` ist F32 mit der Form `[B, H, K, V]`. Übergeben Sie für den nächsten Abschnitt derselben Sequenz diesen `final_state` als `initial_state`. Halten Sie für verschiedene Sequenzen getrennte Zustände vor. Der Anfangszustand wird nicht in-place überschrieben.

`chunk_size` muss positiv sein. Der dreieckige Arbeitsbereich von `4 × (chunk_size² + chunk_size)` Bytes darf das Shared-Memory-Limit des Geräts pro Arbeitsgruppe nicht überschreiten. Der Einstiegspunkt füllt den letzten Chunk auf; das Padding wird aus der Ausgabe ausgeschlossen. Ungültige Argumente liefern `GatedDeltaError`.

Informationen zu Text- und Bildaufrufen auf Modellebene finden Sie im [ruLLM-Inferenzleitfaden](../model-inference.md).
