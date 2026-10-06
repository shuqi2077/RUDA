# Gemeinsames Stack-Autotuning

[Inhalt](README.md) · [Runtime](runtime-api.md) · [Policy und Beispiele](../en/stack-autotuning.md)

Der Controller validiert semantisch gleiche Kandidaten, misst abgeschlossene Arbeit und cached pro Workload/Gerät. Angeschlossen sind Matmul, Attention, Forward-Convolution, fusioniertes Matmul und gepackte Greedy-Generation. FFT/Sparse/Solver/Kommunikation, GPU-Platzierung oder globale Modellsuche sind nicht automatisch enthalten.

enable_stack_autotune(policy,cache_directory) einmal vor Modell/Workern/erster Rechnung. None bedeutet Memory-only. Device stack-autotune, Fusion device-stack-autotune oder LLM stack-autotune aktivieren. Standardbibliothek/native Systeme, nicht no-std/WASM.

| Mode | Verhalten |
| --- | --- |
| Explore | Gültigen Cache verwenden, sonst validieren und messen. |
| CacheOnly | Kein neues Timing; bei Miss Referenz. Kalter Disk-Hit wird validiert. |
| Disabled | Referenz ohne diesen Cache/Trials, nicht Wiederherstellung des alten LocalTuner-Pfads. |

Nicht installieren erhält den alten Pfad. Defaults EndToEnd, Validierung erforderlich,2 Warmups/7 Paare,32 Kandidaten,30s Softbudget, min_speedup1.05, relative MAD.15. Kapazität1024, TTL7 Tage, parallel1, Regression7 Paare/Ratio1.15, workspace_limit=None. Abs1e-4/rel1e-3, gemeinsames Readbacklimit64MiB.

Frische Referenz/Kandidat-Paare mit wechselnder Reihenfolge, Median der Quotienten statt Einzelbestzeit. EndToEnd umfasst interne Allokation, Layout, Submission und Completion; isolierte Eingabevorbereitung liegt außerhalb. Softbudget nur zwischen abgeschlossenen Trials, keine Kernelunterbrechung.1.05 ist Auswahlregel, kein gemessener Speedup.

Writable State ist isoliert, Matmul-Strides/Offsets bleiben, Generation hat eigene KV-Caches. NaN/Inf/falsche Ergebnisse werden abgewiesen; fehlender Validator verwendet Referenz. require_validation=False gestattet ausdrücklich unverified Auswahl. Übereinstimmende Tokens/Ende beim Kalibrierprompt garantiert nicht alle Prompts.

Schlüssel enthalten Revisionen, Gerät/Treiber/Build, genaue Shape/Stride/Dtype/Präzision/Optionen/Context. Unbekannter Treiber begrenzt Diskreuse. RUDA_AUTOTUNE_DRIVER_TAG/BUILD_TAG/CONTEXT_TAG vor Initialisierung setzen, immutable Deploymentdaten; Default-Context erkennt tatsächliche Isolation nicht.

Diskverzeichnis stack-autotune-v1 prüft vollständigen Schlüssel, Version/Checksum/TTL. Digest ist nicht kryptografisch und keine Signatur. Memory-Hit erzwingt keinen Sync/Readback, Diskprüfung/Explore erhöhen Latenz: offline/Startup kalibrieren.

Konkurrenten/Nesting-Misses verwenden Referenz, keine Cross-Process-Sperre. Unbestätigte Completion faultet die Tuning-Lane, kein GPU-Reset oder aktueller Request-Retry. Hard-Workspace-Limit lehnt unbekannte Schätzungen auch für Referenz ab. record_comparison nimmt korrekte vergleichbare Paare vom Caller entgegen, führt keine Schattenmodelle automatisch aus.

StackTuner new/policy/stats/reports/select/invalidate/lower_level_fingerprint sind Controller-APIs. Stats sind keine GPU-Auslastung; Reports sind begrenzte lokale Diagnostik. Invalidate wirkt künftig, gehaltene GenerationPlan müssen explizit neu kalibriert werden. DiskCache ist eine interne Implementierung, kein öffentlich exportierter Runtime-Typ.
