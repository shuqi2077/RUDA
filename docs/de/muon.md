# Muon und explizite Muon + AdamW-Gruppen

[English](../en/muon.md) | [简体中文](../zh/muon.md) | [日本語](../ja/muon.md) | **Deutsch** | [Русский](../ru/muon.md)

Die vorhandene Implementierung `MuonConfig` , `Muon` , `MuonState` wird erweitert
und nicht dupliziert. Den detaillierten Umfang finden Sie unter [中文完整契约](../zh/muon.md) .

## Verwendung

```rust,ignore
use ruda_optim::{AdamWConfig, MuonAdamWConfig, MuonConfig, MuonMatrixLayout, MuonMomentumMode};
let mut optimizer = MuonAdamWConfig::new()
    .with_muon(MuonConfig::new()
        .with_momentum_mode(MuonMomentumMode::Ema)
        .with_stable_normalization(true)
        .with_matrix_layout(MuonMatrixLayout::InputOutput))
    .with_adamw(AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01))
    .init(&model, &[model.hidden.weight.id])?;
model = optimizer.try_step_with_lrs(0.02, 0.0003, model, gradients)?;
```

Wählen Sie explizit versteckte, nicht leere **vollständige 2D-Matrizen** aus. Einbettungen, Klassifikatorköpfe,
Bias und Normalisierungsparameter sollten normalerweise AdamW verwenden, auch wenn sie zweidimensional sind.
Alle nicht ausgewählten Parameter verwenden die vorhandene High-Level-Implementierung AdamW, nicht die
optionale Fused-AdamW-Schnittstelle. Bei den beispielhaften Lernraten handelt es sich nicht um Tuning-Empfehlungen.

`Optimizer::step(lr, ...)` verwendet `lr` für Muon und `lr * adamw_lr_ratio` für AdamW (Standardverhältnis 0.015). `try_step_with_lrs`
erlaubt unabhängige Zeitpläne. Bei fehlenden Gradienten werden Gewichtszerfall und Momentum übersprungen. `try_step_or_skip(..., true)`
überspringt beide Gruppen ohne Zustandsänderung. Zuerst werden die Metadaten beider Gruppen geprüft,
doch Gerätefehler bleiben asynchron: Dies ist keine Gerätetransaktion. Geteilte Parameter werden pro
ParamId einmal zugeordnet; überlappende Gewichte dürfen nicht durch voneinander unabhängige IDs dargestellt werden.

## Numerische Auswahl und Migration

Vorhandene Konstruktorstandards bewahren den SGD-Impuls, die Tensor-dtype-Normalisierung und die AsStored-Skalierung. Der
EMA-Modus ist optional: `m = beta*m + (1-beta)*g` , Null initialisiert; Nesterov verwendet `(1-beta)*g + beta*m` . Tauschen
Sie EMA- und Legacy-Kontrollpunktpuffer SGD nicht ohne eine bewusste Konvertierung aus. Das
endliche quintische Newton-Schulz-Polynom ist keine exakte Polarzerlegung; Eine exakte Identitätsmatrix-Behauptung ist ungültig.

Die stabile Normalisierung skaliert zunächst anhand des maximalen Betrags, bevor die Quadratsumme gebildet wird.
Sie ist wegen geänderter Rundung ausdrücklich zu aktivieren und verlangt FP32-Eingaben und -Zustand. Sie
schützt nicht vor Überlauf in sämtlichen anderen Phasen und untersucht keine nicht endlichen Werte. Es
gibt kein implizites BF16-Casting. Das native PyTorch Muon verwendet BF16 für NS; die FP32-Variante
ist nicht bitäquivalent. Automatische FP32-Mastergewichte oder Updates einer Modellkopie mit niedriger Präzision werden nicht bereitgestellt.

RUDA Linear speichert `[input, output]` ; verwenden Sie daher gegebenenfalls InputOutput für
die LR-Skalierung Original. AsStored interpretiert Zeilen als Ausgaben. MatchRmsAdamW ist unter Transposition
symmetrisch. Der Gewichtszerfall verwendet die ursprüngliche, nicht die formangepasste Lernrate. Eine Konfiguration
gilt für die gewählte Muon-Gruppe; gemischte logische Layouts benötigen separat konfigurierte Optimierer.

Das Konfigurationsmakro des Projekts stellt keine Serde-Standardwerte für neue Felder bereit.
In alten JSON-Konfigurationen müssen `"momentum_mode":"Sgd"` , `"stable_normalization":false` , `"matrix_layout":"AsStored"` hinzugefügt werden, um
die alten Auswahlmöglichkeiten beizubehalten. Programmatische Standardeinstellungen bleiben verfügbar. Das einfache Layout der
Muon-Tensoraufzeichnung bleibt unverändert. Gemischte Datensätze umfassen eine Version, eine Konfiguration, ein
Parameteridentität/-form/dtype-Manifest und beide Optimiererstatus. Stellen Sie zunächst das Modell mit den Original-IDs
wieder her. Für präzise Fortsetzungsvergleiche wird FullPrecisionSettings benötigt. Geänderte Gruppierung/Konfiguration wird abgelehnt.

Entskalieren und überprüfen Sie Gradienten extern vor der Aktualisierung und synchronisieren Sie
Übersprungsentscheidungen über Replikate hinweg. Dieser Patch integriert nicht automatisch den vorherigen Low-Level-Gradientenschutz API.
Implizite `step_multi` -, Ruda-Tensoren mit verteilter Markierung, FSDP/TP-Shards, spärliche Gradienten und 4D-Faltungsumformung sind
nicht implementiert. Orthogonalisieren Sie niemals beliebige Shards, als wären sie die vollständige Matrix.

## Validierung

```bash
cargo run --release --locked -p ruda-optim --example muon-training -- 20
python tools/run_muon_regressions.py --suite oracle
python tools/run_muon_regressions.py --suite reference
python tools/run_muon_regressions.py --suite host
python tools/run_muon_regressions.py --suite build
python tools/run_muon_regressions.py --suite cuda --compiler both
```

Das Beispiel verwendet das Host-Tensor-Backend, sofern es nicht mit test-cuda erstellt wurde. Es handelt sich nicht um einen
Leistungsmaßstab. Die Referenzsuite kompiliert ein unabhängiges skalares Referenzmodell nur mit rustc; Die Ausführung von RUDA wird nicht validiert. Die
Host- und CUDA-Suiten kompilieren und führen echte RUDA-Tensor-/Gruppentests aus. Build beinhaltet eine Prüfung auf keine Standardfunktionen. Fehlende Werkzeuge
werden gesperrt; Es wird kein Installations-/Fallbackversuch unternommen. Jeder Befehl verfügt über ein explizites Timeout und ein separates Protokoll.

Referenzen: [Muon-Autoren](https://github.com/KellerJordan/Muon)
, [PyTorch offizielle Schnittstelle](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
, [korrigierte v2.9-Quelle](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py) .
