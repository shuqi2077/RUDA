# Experimentelles fusioniertes AdamW / AMSGrad auf dem Gerät

[English](../en/fused-adamw.md) | [简体中文](../zh/fused-adamw.md) | [日本語](../ja/fused-adamw.md) | **Deutsch** | [Русский](../ru/fused-adamw.md)

Dies ist eine **Opt-in-Implementierung, die auf die Ausführungsvalidierung von RUDA Rust/GPU
wartet**. Es erweitert `ruda-optim` ; Es wird keine weitere Optimierungsbibliothek erstellt oder
das vorhandene `AdamW` , der Modelloptimierungsadapter, das Autograd-Diagramm oder das Prüfpunktformat geändert.

## Warum dieser Operator?

Der vorhandene generische Optimierer erstellt das Update aus Tensoroperationen.
Diese Ergänzung kombiniert explizit Gradientenunskalierung, Momente, optionales AMSGrad-Maximum, bias-Korrektur, entkoppelten
Gewichtsabfall und Parameteraktualisierung in einem Gerätekernel. Es wird nicht
behauptet, dass das bestehende Fusion-Backend notwendigerweise viele Kernel verwendet oder
dass diese Implementierung schneller ist als der Fused-Optimierer von PyTorch.

Für jedes endliche Eingabeelement und einsbasierte Aktualisierung `t`:

```text
g = stored_gradient / gradient_scale      # negate if maximize
m = beta1 * m_old + (1 - beta1) * g
v = beta2 * v_old + (1 - beta2) * g * g
v_used = max(v_max_old, v)                 # AMSGrad only; save uncorrected max
p_new = p * (1 - lr * weight_decay)
        - lr * (m / (1 - beta1^t)) / (sqrt(v_used / (1 - beta2^t)) + epsilon)
```

Ohne AMSGrad gilt `v_used = v` . Epsilon steht außerhalb der Quadratwurzel. Gewichtszerfall wird
weder zum Gradienten noch zu den Momenten addiert. Der Host berechnet die
Bias-Koeffizienten einmal pro Aufruf mit FP64-Potenzierung für ganzzahlige Exponenten und anschließendem Casting der
Koeffizienten nach FP32. Gerätearithmetik verwendet FP32; der Compiler kann Operationen zusammenfassen oder
neu klammern. Numerische Vergleiche verwenden Toleranzen und garantieren keine Bitidentität mit anderen Optimierern.

Formelreferenz: [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) . Die Standardwerte entsprechen absichtlich dem
vorhandenen `AdamWConfig` von RUDA für Beta, Epsilon und
Zerfall: Beta=(0.9,0.999), Epsilon=1e-5, weight_decay=1e-4. Legen Sie beim Vergleich
mit einem anderen Framework alle Optionen explizit fest.

## Unterstützter Vertrag

|Artikel|Diese Implementierung|
|---|---|
|Parameter und Momente|FP32 Master/Status|
|Gespeicherter Gradient|FP32, FP16 oder BF16; im Update zu FP32 hochgestuft|
|Form|Exakte Übereinstimmung, dicht zusammenhängend, keine Übertragung, konservativ <= u32-Byte-Bereich|
|Modus|AdamW, AMSGrad, maximieren, skalare Lernrate, positive Verlustskala|
|Leerer Tensor|Kein Start, keine Zuordnung oder kein Schrittvorlauf|
|Vom Anrufer erkannter Überlauf|`skip_update=true`: kein Start, keine Zuordnung oder Schrittweiterschaltung|
|Eingaben/Aliase|Schreibgeschützt; Ausgänge sind neue Puffer|
|Ausgangszustand|Berechnet im ersten Update; kein Zero-Fill-Start erforderlich|
|Warteschlange|Gleiches Gerät und gleiche Übermittlungswarteschlange für alle Eingaben; Nichtübereinstimmung ist ein Fehler|
|Abschluss|Asynchron, gesteuert durch die vorhandene Laufzeit|
|Modellgewichte mit halber Genauigkeit|Nicht automatisch aktualisiert/umgewandelt; Der Aufrufer wandelt die Master-Ausgabe explizit um|

Dies ist **nicht** der FP8/FP4-Optimiererstatus, ein GradScaler, eine automatische Prüfung
auf endliche Gradienten, ein vorsichtiger Gewichtsabfall, ein differenzierbarer Optimierer, ein allgemeiner
Strided-Kernel, ein Multi-Tensor-Optimierer mit variablen Hyperparametern, eine automatische FSDP-Integration oder
ein CUDA-Graph-Replay-Optimierer mit einem geräteseitigen Schritt Zähler. Die Wiedergabe erfasster Host-Koeffizienten
bias ohne Aktualisierung wird nicht unterstützt. Ein vorab abgeflachter Eimer
funktioniert nur, wenn alle Elemente dieselben Optionen und eine gemeinsame Schrittanzahl
haben. Sparse-Verläufe, beliebige Hostzeiger und ein stiller CPU-Fallback werden nicht hinzugefügt.

Der Masterstatus FP32 bedeutet nicht, dass ein
End-to-End-Training mit gemischter Präzision validiert wurde. Der Aufrufer
verwaltet Modellkopien, Skalierungsentscheidungen und akkumulierte oder verteilte Gradienten.

## Features

- `fused-adamw`: Optionen und explizit aufgerufene CPU-Referenz.
- `fused-adamw-device`: generischer RudaTensor-Gerätestarter und Kernel, keine spezifischen
  -Hardwaretreiber, der durch diese Funktion aktiviert wird.
- `fused-adamw-cuda`: CUDA-Laufzeit, direkte PTX-Fähigkeit und explizite CUDA-Tests/Beispiele.
  Es wird immer noch NVRTC oder PTX unter Verwendung von `RUDA_CUDA_COMPILER` ausgewählt.

Standardmäßig ist keine Funktion aktiviert. Es wird keine neue Abhängigkeitsversion
eingeführt. `Cargo.lock` erhält nur den vorhandenen CUDA-Treiber als optionale Ruda-Optim-Abhängigkeit.

## Low-Level-Nutzung

```rust
use ruda_optim::fused_adamw::{AdamWOptions, StepControl, adamw_step};

// master: dense FP32 RudaTensor<R>, gradient: same-shape F32/F16/BF16 tensor.
// state: Option<AdamWState<R>>, initially None.
let options = AdamWOptions {
    learning_rate: 1e-3,
    weight_decay: 0.01,
    amsgrad: true,
    ..Default::default()
};
let result = adamw_step(
    &master, &gradient, state.as_ref(), &options,
    StepControl { gradient_scale: 128.0, skip_update: found_inf },
)?;
master = result.parameters;
state = result.state;
```

`found_inf` wird vom Aufrufer geliefert und hier nicht berechnet. Das
bestehende `AdamW::step` nutzt weiterhin seine ursprüngliche Implementierung. Das Low-Level-Primitiv ersetzt
einen Modul- `Parameter` nicht automatisch und erstellt keinen differenzierbaren Update-Graphen.

Die Eingabevalidierung lässt alle Eingaben unverändert. Eine erfolgreiche Rückgabe bedeutet,
dass der Kernel eingereicht wurde; sie beweist keinen Ausführungsabschluss. Prüfen Sie
vor dem Checkpointing das Synchronisierungsergebnis der Laufzeit. Bei einem Gerätefehler verwerfen
Sie ausstehende Ausgaben und stellen einen extern festgeschriebenen Checkpoint wieder her.
Erhöhen Sie den Trainingsdaten-Cursor nicht allein deshalb, weil `updated` true ist.

`AdamWState::into_parts/from_parts` stellt die Schritt- und Momentpuffer für die explizite
Checkpoint-Integration bereit. Sie serialisieren oder übertragen nicht selbst. Behalten Sie
die Hauptparameter, alle Momente und Hyperparameter bei und führen Sie
sie zusammen. Dieses Modul rüstet das High-Level-Format `TrainingRecord` nicht nach.

## Zuordnungs- und Leistungsmodell

Diese erste Implementierung wählt out-of-Place-Updates, um Aliase beizubehalten und das Hinzufügen neuer
unsicherer Eigentumsregeln zu vermeiden. Jeder aktive Schritt weist drei FP32-Ausgänge zu (vier
mit AMSGrad). Es gibt keinen dazwischenliegenden Delta-Tensor, aber **keinen Anspruch auf einen
Null-Zuteilungsschritt**. Alte und neue Puffer können sich im Laufe ihrer Lebensdauer überschneiden.
Große Modelle erfordern daher ein Speicherbudget; Die Wiederverwendung vor Ort oder in
der Arena/im Eimer ist zukünftige Arbeit und wird hier nicht stillschweigend aktiviert.

Die nur auf Benchmarks basierende gestaffelte Baseline übermittelt explizit 4 Kernel oder
5 mit AMSGrad. Der verschmolzene Pfad übermittelt 1. Für einen *stationären* FP32-Gradientenschritt:

|Modell|Logische Bytes pro Element|Explizites Update wird gestartet|
|---|---:|---:|
|Inszeniert AdamW|48|4|
|mit Sicherung AdamW|28|1|
|Inszeniertes AMSGrad|60|5|
|Fused AMSGrad|36|1|

Die 28-Byte-Anzahl beträgt vier FP32-Lesevorgänge (Parameter, Gradient, zwei Momente) plus drei
FP32-Schreibvorgänge. Die 48-Byte-Basislinie umfasst wiederholte Gradientenlesevorgänge und den Delta-Puffer. Dabei handelt
es sich um eine Abrechnung auf Quellenebene, **nicht um den gemessenen DRAM-Verkehr
oder die Beschleunigung**. Caching, Zuordnung, Arithmetik und Startaufwand können das beobachtete
Ergebnis verändern. Ein Backend, das bereits generisches AdamW fusioniert, profitiert möglicherweise nicht.

## Validierungs- und Benchmark-Befehle

```sh
# Python/NumPy/PyTorch formula oracle ONLY, does not execute RUDA.
python tools/run_adamw_regressions.py --suite oracle

# Standalone Rust config/reference tests, without Cargo registry resolution.
python tools/run_adamw_regressions.py --suite reference

# Cargo unit tests, including a comparison with RUDA's existing Host AdamW.
python tools/run_adamw_regressions.py --suite host

# Type-check the opt-in generic device implementation.
python tools/run_adamw_regressions.py --suite build

# Explicit hardware regression, separately under both compilers.
python tools/run_adamw_regressions.py --suite cuda --compiler both

# Small controlled A/B run, then increase elements after checking resources.
python tools/run_adamw_regressions.py --suite bench --compiler both \
    --elements 65536 --dtype bf16 --iterations 20 --samples 7 --amsgrad
```

Verwenden Sie einen `RUDA_PTX_VERSION` , der vom tatsächlichen Treiber/GPU unterstützt wird. Für `--offline`
sind zwischengespeicherte Cargo-Abhängigkeiten erforderlich. `--timeout` begrenzt jeden Befehl; Es handelt sich nicht um eine
vorhergesagte Laufzeit. Der Runner behält Befehlszeilen, Quell-Hashes, Protokolle und Status bei. Fehlender Rust/Cargo
ist `blocked` ; Bei einem nicht verfügbaren GPU schlägt die explizite CUDA-Suite fehl. Kein
Test wechselt stillschweigend das Backend oder gibt ein gemessenes Ergebnis auf einem Simulator an.

Das Beispiel schließt die Erstellung/Rücklesung von Eingaben aus zeitgesteuerten Abschnitten
aus, erwärmt beide Varianten, startet beide mit identischem Materialisierungszustand, wechselt die
Ausführungsreihenfolge und synchronisiert vor/nach jeder gemessenen Charge. Es meldet Median/Min/Max
**Wall-Millisekunden pro Schritt**, einschließlich Zuweisungen und Host-Übermittlungen. Es handelt sich nicht
um einen reinen CUDA-Ereignis-GPU-Timer. Jede gemessene Charge vergleicht Parameter und
Momente. Speichern Sie GPU, Treiber, Takt-/Energieeinstellungen und Hardwarelast zusammen mit JSON.

Der festgeschriebene `pytorch_fixtures.json` wird mit echtem **CPU** PyTorch AdamW generiert; 12
Fälle decken 3 Gradienten-D-Typen x 2 AMSGrad-Modi x 2 Maximierungsmodi mit
jeweils 5 Schritten ab. `oracle.py --write-fixtures` generiert es explizit neu. Die CUDA-Tests
vergleichen die tatsächlichen RUDA-Kernel-Ausgaben mit diesen Daten sowie der separaten FP64-Referenz.
Das Bestehen des Python-Fixture-Generators bedeutet nicht, dass diese CUDA-Tests bestanden werden.

## Verbleibende Tore

Rosttyp/Makroerweiterung, CUDA-Ausführung und gemessene Leistung bleiben obligatorisch. Neue Tests müssen
vor der Veröffentlichung neben den vorherigen Sicherheitsregressionssuiten bestanden werden. Weitere
Optimierungen sollten von gemessenen Profiler-Ergebnissen ausgehen, nicht von diesem Verkehrsmodell.
Aktivieren Sie keinen neuen Standard-Dispatcher, bis der Vergleich abgeschlossen ist.


Eine Opt-in-Erweiterung finden Sie unter [experimentelle Verlaufsprüfungen und Gruppenbeschneidung](gradient-guard.md).
