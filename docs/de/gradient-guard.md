# Gradientenprüfung und Clipping für fusioniertes AdamW (experimentell)

[English](../en/gradient-guard.md) | [简体中文](../zh/gradient-guard.md) | [日本語](../ja/gradient-guard.md) | **Deutsch** | [Русский](../ru/gradient-guard.md)

Aktivieren Sie die Funktion ausdrücklich. Die öffentliche Signatur
von `adamw_step` und der standardmäßige Modelloptimierer bleiben unverändert.

## Was es tut

`gradient_stats_sync` durchsucht akkumulierte Gradienten, prüft Rohwerte und in FP32 entskalierte
Werte auf NaN/Inf und bestimmt eine gemeinsame L2-Norm der übergebenen lokalen
Gruppe. `guarded_adamw_step` verwendet diese Norm zum Clipping innerhalb des bestehenden fusionierten
AdamW-/AMSGrad-Kernels, ohne vollständige Tensoren geclippter Gradienten zu allozieren oder zu schreiben.

Die Reihenfolge ist: Konvertierung des Speichertyps -> FP32-Multiplikation mit dem Kehrwert der
Loss Scale -> Endlichkeitsprüfung/Norm -> gemeinsamer FP32-Clipping-Koeffizient -> Vorzeichen für maximize ->
AdamW. Dies ist ausdrücklich kein In-place-Clipping in FP16/BF16: Der effektive Gradient bleibt
FP32, statt vor dem Update wieder auf die halbe Speicherpräzision gerundet zu werden.

Für endliche Gradienten:

```
clip = min(max_norm / (L2_norm + epsilon), 1)
g_effective = (float32(g_stored) * reciprocal(loss_scale)) * clip
```

`max_norm=None` deaktiviert Clipping, nicht die Endlichkeitsprüfungen. `max_norm=0` setzt den effektiven Gradienten
auf null, führt aber die Aktualisierung von Momenten, Gewichtszerfall und Schrittzählern
weiter aus. Bei NaN/Inf überspringt die Richtlinie `Skip` die GESAMTE ausgewählte Gruppe
einschließlich Gewichtszerfall und Schrittzählern. `Error` meldet einen Fehler, bevor ein Optimierer-Update-Kernel
gestartet wird. Sämtliche Metadaten- und Schrittüberlaufprüfungen erfolgen vor dem Einreichen der Statistikberechnung.

## Funktionen und Verwendung

- `gradient-guard`: Hostkonfiguration, Statistiken/Entscheidungstypen und CPU Oracle.
- `gradient-guard-device`: generische Gerätereduzierung und geschützter Optimierer.
- `gradient-guard-cuda`: CUDA Integrationstests und Benchmark.

```rust,ignore
use ruda_optim::fused_adamw::{
    AdamWEntry, AdamWOptions, StepControl, guarded_adamw_step,
    gradient_norm::GradientGuardOptions,
};

// master1/2: contiguous FP32; grad1/2: F32/F16/BF16, same device and queue.
// Accumulation must already be finished, using ONE loss scale for this step.
let entries = [
    AdamWEntry { parameters: &master1, gradients: &grad1, state: state1.as_ref() },
    AdamWEntry { parameters: &master2, gradients: &grad2, state: state2.as_ref() },
];
let pending = guarded_adamw_step(
    &entries, &AdamWOptions::default(),
    StepControl { gradient_scale: 128.0, skip_update: false },
    GradientGuardOptions { max_norm: Some(1.0), ..Default::default() },
)?;
// Compact stats readback has completed, but the UPDATE is still asynchronous.
// Await/synchronize the runtime and check completion before replacing committed
// model/state or saving a checkpoint. Keep the existing committed state on failure.
```

Es werden weder ein automatisch aktualisierter GradScaler noch Modelladapter, FSDP-Reduktion, Tensor-Flattening,
Deduplizierung geteilter Gewichte oder erneutes Casting des Modells in niedrige Präzision hinzugefügt.
Gezählt werden nur ausdrücklich übergebene Parameter. Eine lokale Shard-Norm darf NICHT
als globale FSDP-/TP-Norm verwendet werden. Es erfolgt keine rankübergreifende Abstimmung der Skip-Entscheidung.

## Reduzierung der Implementierungs- und Ressourcenkosten

Ein Grid-Stride-Kernel behält ein `(scale, sumsq, bad)` -Triple pro Spur und reduziert sich
mit einem festen Shared-Memory-Baum. Es quadriert niemals direkt einen großen FP32-Rohwert, also
z. Endliche Werte in der Nähe von `1e30` erzeugen keinen falschen
FP32-Überlauf bei der Normberechnung. Bis zu 1024 Teiltripel werden um einen weiteren
Block reduziert. Es gibt keine schwebenden Atome; Dies ist kein automatischer Geräte-Autotuner.

Die Konfiguration erfordert 256 X-Threads und 3072 Bytes gemeinsam genutzten
Speicher pro Block. Nicht unterstützte Konfigurationen werden abgelehnt und nicht stillschweigend
an CPU gesendet. Leere Tensoren liefern keine Arbeit. Jeder nicht leere
Tensor übermittelt einen oder zwei Reduktionskerne und gibt eine 12-Byte-Zusammenfassung
zurück. Alle Tensorreduktionen werden vor einem gestapelten Host-Readback-Aufruf API in die
Warteschlange gestellt. Die Laufzeit kann mehrere DMA-Kopien implementieren. Der Host summiert
explizit kompakte Zusammenfassungen in FP64 und legt den Koeffizienten fest.
Ein Tensor benötigt höchstens 12 KiB Teilspeicher plus eine 12-Byte-Abschlusszusammenfassung, ohne
Allokatorausrichtung/Metadaten. `scratch_bytes` summiert diese Beträge pro Tensor und ist keine Spitzenzuteilungsmessung.

Neuzuordnung, FMA und fehlerhafte Behandlung hängen vom Backend ab. Normen werden
mit Toleranzen getestet, nicht versprochen, bitidentisch mit PyTorch oder geräteübergreifend. Dies ist
nicht die gesamte LAPACK LASSQ-Implementierung und wird nicht als numerische Kompatibilität beworben.
Große endliche Gradienten können den zweiten Moment von AdamW immer noch überlaufen,
wenn das Beschneiden deaktiviert ist. Alte Parameter/Momente werden nicht auf Endlichkeit geprüft.

## Leistungsanspruch und explizite Einschränkungen

Im Vergleich zur enthaltenen Basislinie (gleiche Normberechnung, dann ein separater Unscale+Clip-Kernel,
der FP32-Verläufe erzeugt, dann AdamW fusioniert) werden dadurch ein Clip-Start und ein
`4*N` -Byte temporär pro nicht leerem Tensor entfernt und `8*N` -Bytes logischer
temporärer Schreib-/Lesevorgänge vermieden. Hierbei handelt es sich um Zählwerte auf Quellenebene, nicht
um den gemessenen Gerätespeicherverkehr oder die Beschleunigung. Der alte unbewachte Optimierer zahlt
die neuen Statistikkosten nicht; Das Hinzufügen von Diagnosefunktionen kann einen Schritt verlangsamen.

Diese erste Version blockiert den Host beim kompakten Rücklesen bei
jedem Optimierungsschritt. Es ist nicht sicher bei der Grapherfassung, verspricht keine
Rechen-/Kommunikationsüberlappung und kann bei vielen kleinen Tensoren eine schlechte Leistung erbringen.
Die stabile Reduktion verfügt über zusätzliche Arithmetik. Benchmark vor der Aktivierung;
Es besteht kein Anspruch gegen PyTorch Fused/foreach AdamW. Die ursprünglichen
AdamW-Ausgangszuordnungen bleiben unverändert (out-Ort-FP32-Master und -Momente). Gradienten dürfen nicht über einen
anderen Alias/eine andere Warteschlange geändert werden, während Statistiken oder abhängige Aktualisierungen
ausgeführt werden. Laufzeitfehler sind kein transaktionales Gruppen-Commit; Vor dem Update-Start wird
nur die Entscheidung „Nicht endlich/Validierung überspringen“ getroffen. Laufzeitzuweisungs-/Start-APIs behalten ihren Fehlervertrag.

## Validierung und Benchmark

```bash
python tools/run_gradient_guard_regressions.py --suite oracle
python tools/run_gradient_guard_regressions.py --suite reference
python tools/run_gradient_guard_regressions.py --suite host
python tools/run_gradient_guard_regressions.py --suite build
python tools/run_gradient_guard_regressions.py --suite cuda --compiler both
python tools/run_gradient_guard_regressions.py --suite bench --compiler both --elements 65536 --tensors 4 --dtype bf16 --amsgrad
```

`oracle` benötigt NumPy und PyTorch CPU. Es führt nur ein
numerisches Python-Modell aus, keinen RUDA-Code. `reference` kompiliert eigenständige Rust-Tests ohne Cargo-Registrierungszugriff.
`build` prüft sowohl alte als auch neue Gerätefunktionen. `cuda` führt
sowohl alte AdamW-Regressionen als auch die neuen Hardwaretests aus. Fehlende Werkzeuge
werden gesperrt; Timeout und Fehler werden aufgezeichnet. `--dry-run` druckt nur
geplante Befehle. `--offline` erfordert zwischengespeicherte Cargo-Abhängigkeiten. Es wird keine Toolinstallation durchgeführt.

Der Benchmark wechselt die Pfade nach dem Aufwärmen, umfasst Zuweisungen, Norm,
Host-Rücklesung/-Entscheidung, Übermittlung und endgültige Gerätesynchronisierung und überprüft Parameter und alle Momente
nach jedem gemessenen Stapel. Das nur für die Vergleiche verwendete Readback
liegt außerhalb des Timings. Behalten Sie Umgebungs-/Quellversionen mit den rohen JSON-Beispielen bei.

## Numerische Referenzen

Konzeptuelle Referenzen, keine kopierten Implementierungen:
- PyTorch AMP Beispiele (unskaliert vor dem Abschneiden des Gradienten):
  https://docs.pytorch.org/docs/main/notes/amp_examples.html
- PyTorch Lokaler Vertrag mit verketteter Gradientennorm:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html
- Skalierte Quadratsummendarstellung:
  https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq.html
