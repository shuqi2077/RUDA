# Ruda Programmierhandbuch

[English](../en/programming-guide.md) | [简体中文](../zh/programming-guide.md) | [日本語](../ja/programming-guide.md) | **Deutsch** | [Русский](../ru/programming-guide.md)

[Dokumentation](README.md) · [Laufzeit API](runtime-api.md) · [Compute-Bibliotheken](libraries/README.md) · [中文](../zh/programming-guide.md)

## 1. Host und Gerät

Der Host-Rust-Code wählt Geräte aus, bereitet Eingaben vor, erstellt Argumente und liest Ergebnisse. Gerätekerne beschreiben parallele Berechnungen. Das Frontend erweitert sie zu IR für die Backend-Kompilierung und -Ausführung.

`ruda-kernel::dsl` ist das Allzweck-Kernel-Frontend. Kernel verwenden die Rust-Syntax mit den Typen, Makros und Operationen des Frontends; Beliebige Rust-Programme und Standardbibliothekscode können nicht einfach zu einem GPU kompiliert werden.

Das Tensor-Framework verteilt Vorgänge an Rechenbibliotheken. Anwendungen müssen keine Kernel auf Thread-Ebene schreiben, um die Matrixmultiplikation zu verwenden.

## 2. Ausführungshierarchie

|Ruda Konzept|Zweck|
| --- | --- |
|`RudaCount`|Anzahl der Arbeitsgruppen in einem Start|
|`RudaDim`|Ausführungsdimensionen jeder Arbeitsgruppe|
|`ABSOLUTE_POS`|Globale Position in einem eindimensionalen elementweisen Kernel|
|`Array<T>`|Eindimensionaler Array-Zugriff in Kerneln|
|`Tensor<T>`|Kernel-Tensor-Zugriff mit Form- und Schritt-Metadaten|
|`Runtime`|Ordnet Compiler-, Rechenserver- und Gerätetypen zu|

Die entsprechenden CUDA-Konzepte finden Sie im [Kompatibilitätsleitfaden](compatibility.md). Öffentliche Exporte finden Sie im [DSL-Vorspiel](../../ruda-kernel/src/dsl/prelude.rs).

## 3. Ihr erster Kernel

Dieser Kernel stammt aus dem [ptx-runtime-Beispiel](../../ruda-driver-cuda/examples/ptx_runtime.rs), das das vollständige Hostprogramm und die Ausführungsprüfungen enthält:

```rust
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn add(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}
```

Importieren Sie das Makro und die Typen mit `ruda_kernel::dsl::prelude::*`. Die Ausgabelänge schließt überschüssige Endfäden aus; Beide Eingabearrays müssen mindestens so viele Elemente enthalten wie die Ausgabe.

Das Beispiel startet 64 Ausführungseinheiten pro Arbeitsgruppe und rundet die Anzahl der Arbeitsgruppen auf. Dies ist die Konfiguration des Beispiels, keine allgemein optimale Kernelgröße.

## 4. Gedächtnis und Argumente

Verwenden Sie `ComputeClient`, um Gerätepuffer zu erstellen oder zuzuweisen, und erstellen Sie dann Kernel-Argumente aus ihren Handles. Unterscheiden Sie die Anzahl der Bytes von der Anzahl der Elemente:

- `client.empty(size)` nimmt eine Größe in Bytes an.
- Im `ArrayArg::from_raw_parts(handle, count)` des Beispiels ist `count` die Anzahl der Array-Elemente.
- `RudaTensor<R>` trägt ein Speicherhandle, Form, Schritte, dtype, Gerät und Quantisierungsparameter.

Durch das Klonen eines Handles oder Tensors werden die zugrunde liegenden Gerätedaten nicht kopiert. Um ein Layout zu ändern, verwenden Sie den entsprechenden zusammenhängenden Konvertierungs-, Kopier- oder Transformationsvorgang. Durch das Bearbeiten von Metadaten allein wird der Speicher nicht neu angeordnet.

## 5. Übermittlung, Rücklesung und Synchronisierung

Kernel-Übermittlung und Ergebnisverfügbarkeit sind separate Phasen. Der Abschluss der Host-Übermittlung stellt weder die Ausführungszeit noch den Erfolg des Geräts dar.

`read_one` wartet auf Rücklese und gibt ein `Result` zurück; `read_async` liefert asynchrone Ergebnisse. `sync()` gibt eine Zukunft zurück, die abgewartet werden muss. `flush()` übermittelt Befehle in der Warteschlange; es ersetzt nicht das Lesen des Ergebnisses.

Mehrere Streams, die auf dieselben Daten zugreifen, müssen Produzenten-Konsumenten-Abhängigkeiten berücksichtigen. `set_stream` ist unsicher und die Lebensdauer der Hostvariablen allein stellt keinen Abschluss der Geräteaufgabe dar.

## 6. Sicherheitsgrenzen

Typen, Besitz und Ausleihen schränken Hostressourcen und Schnittstellen ein. Low-Level-Wrapper müssen auch die Anforderungen an die Geräteausführung erfüllen:

- Argumentspeicherbereiche, dtype, Ausrichtung und Layout stimmen mit Kernelzugriffen überein.
- Von asynchronen Aufgaben verwendete Daten bleiben bis zum Abschluss gültig.
- Gemeinsame Schreibvorgänge über Threads und Streams hinweg werden korrekt synchronisiert.
- Anrufer, die Rohargumente konstruieren oder ungeprüfte Starts verwenden, erfüllen ihre Sicherheitsverträge.

Ein geprüfter Start ist kein vollständiger Sicherheitsbeweis für einen beliebigen Kernel. Das Beispiel verwendet explizite `unsafe`-Blöcke mit Sicherheitserklärungen für die Rohargumentkonstruktion und den Start. Siehe [Runtime API](runtime-api.md).

## 7. Von Kerneln zu Rechenbibliotheken

Verwenden Sie [ruBLAS](libraries/rublas.md), [ruDNN](libraries/rudnn.md) und [ruPRIM](libraries/ruprim.md) für allgemeine Vorgänge. Geben Sie beim Arbeiten auf Kernel-Ebene das Eingabelayout, die Akkumulationsgenauigkeit und die Ausführungskonfiguration an.
