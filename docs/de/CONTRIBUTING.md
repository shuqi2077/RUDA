# Zu Ruda beitragen

[English](../en/CONTRIBUTING.md) | [简体中文](../zh/CONTRIBUTING.md) | [日本語](../ja/CONTRIBUTING.md) | **Deutsch** | [Русский](../ru/CONTRIBUTING.md)

[Dokumentation](README.md) · [中文](../zh/CONTRIBUTING.md)

## Umfang

Beiträge betreffen den Rechen-Software-Stack: Compiler, Laufzeitumgebungen, Kernels, Rechenbibliotheken, Tensoren, Modellintegration, Dokumentation und Tests.

Halten Sie Bibliothekszuständigkeiten getrennt, statt Operationen der Rechenbibliotheken im Tensor-Framework zu duplizieren. Refaktorierungen dürfen weder Funktionen entfernen noch Datentypen, Operationsreihenfolge, Fehlerbehandlung oder Ressourcenlebensdauer verändern. Melden Sie nicht unterstützte Pfade korrekt, statt stillschweigend auf CPU, ein anderes Compiler-Backend oder geringere Präzision umzuschalten.

## Probleme melden

Geben Sie Quellversion, Betriebssystem, Rust-Toolchain, GPU-/Treiber-/Toolkit-Versionen, aktivierte Features, Reproduktionsbefehle, minimale Eingaben und tatsächliche sowie erwartete Ergebnisse an. Entfernen Sie vor dem Anhängen von Logs Tokens, persönliche Pfade sowie nicht zur Offenlegung freigegebene Modelle oder Daten. Reichen Sie keine Modellgewichte oder Build-Caches ein.

## Änderungen einreichen

- Prüfen Sie vor der Bearbeitung Komponentenzuständigkeiten und bestehende Tests. Erklären Sie bei schichtübergreifenden Änderungen Abhängigkeiten und Aufrufverträge.
- Bewahren Sie Urheberschaft, Copyright, Lizenzen und Herkunft Dritter. Überschreiben Sie nicht die Lizenzdeklarationen sämtlicher Dateien mit einer einzigen Lizenz.
- Fügen Sie Verhaltenstests hinzu. Unterscheiden Sie Testquellcode, Syntaxprüfungen, erfolgreiche Kompilierung und Ausführung auf echten Geräten.
- Nennen Sie nicht ausgeführte Prüfungen. Cargo-Metadaten- und Formatprüfungen sind keine Kompilierungstests.
- Berichten Sie bei Leistungsänderungen über Korrektheit und Vorher-nachher-Messungen mit gleichen Eingaben, Datentypen, Geräten und Konfigurationen. Ersetzen Sie die Validierung auf Zielhardware nicht durch einen Simulator oder eine verkleinerte Konfiguration.

## Lokale Prüfungen

Führen Sie die Befehle im Workspace-Stammverzeichnis aus. Sie lesen Manifeste, ohne Kernels zu kompilieren oder auszuführen:

```powershell
cargo metadata --no-deps --format-version 1 --offline --locked
cargo tree -p ruda-driver-cuda --no-default-features --features direct-ptx --edges normal,build --offline --locked
```

`--offline` setzt voraus, dass die zur Auflösung benötigten Abhängigkeiten lokal zwischengespeichert sind.

Den Einstieg zu Build und Demonstration finden Sie unter [Erste Schritte](getting-started.md).

Beitragende müssen berechtigt sein, ihren Code einzureichen. Originäre Ruda-Beiträge verwenden Apache-2.0; Drittanbieter-Code behält seine ursprüngliche Lizenz und die geltenden Bedingungen.
