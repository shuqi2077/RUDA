# Runtime API Referenz

[English](../en/runtime-api.md) | [简体中文](../zh/runtime-api.md) | [日本語](../ja/runtime-api.md) | **Deutsch** | [Русский](../ru/runtime-api.md)

[Dokumentation](README.md) · [Programmieranleitung](programming-guide.md) · [Treiber API](driver-api.md) · [中文](../zh/runtime-api.md)

Die allgemeine Laufzeit befindet sich in `ruda::runtime` und wird durch die Funktion `ruda/runtime` aktiviert. Jedes Backend benötigt außerdem seine entsprechende Treiberkiste.

## 1. Kerntypen

|Typ|Verantwortung|Definition|
| --- | --- | --- |
|`Runtime`|verbindet Compiler, Server und Gerät; stellt Geräte-Clients bereit|[backend.rs](../../ruda/src/runtime/backend.rs)|
|`ComputeClient<R>`|Zuweisung, Kernel-Übermittlung, Rücklesung, Synchronisierung und Funktionsabfragen|[client.rs](../../ruda/src/runtime/client.rs)|
|`ComputeServer`|Backend-Ausführungsvertrag|[Servermodul](../../ruda/src/runtime/server/mod.rs)|
|`RudaTensor<R>`|Gerätespeicher und Tensor-Metadaten|[Tensordefinition](../../ruda-kernel/src/tensor/base.rs)|

`R::client(&device)` ruft den Client der Laufzeit ab. `R::Device` bestimmt den Gerätetyp; Durch die gemeinsame Nutzung eines Laufzeittyps wird der Speicher auf verschiedenen physischen Geräten nicht austauschbar.

## 2. Speicher und Übertragungen

Diese Methoden gehören zu `ComputeClient<R>`:

|Methode|Verhalten|
| --- | --- |
|`create_from_slice(&[u8])`|Erstellt Gerätedaten aus Hostbytes und gibt ein Handle zurück|
|`empty(usize)`|Weist Speicher in Bytes zu, ohne eine Null-Initialisierung zu garantieren|
|`create_tensor_from_slice`, `empty_tensor`|Erstellt Tensorspeicherlayouts anhand von Form und Elementgröße|
|`read_one(Handle)`|Liest synchron ein Handle; gibt `Result<Bytes, ServerError>` zurück|
|`read_async(Vec<Handle>)`|Gibt asynchrone Leseergebnisse zurück|
|`read(Vec<Handle>)`|Liest mehrere Handles synchron; Panik bei Fehler|
|`memory_usage()`|Fragt die von der Laufzeit verfolgte Speichernutzung ab|

`read_one_unchecked` gerät in Panik, wenn das Rücklesen fehlschlägt; sein Name bedeutet nicht, dass es Kernel-Grenzprüfungen deaktiviert. Verwenden Sie für nicht zusammenhängende Tensoren Tensor-Readback-Schnittstellen, anstatt Rohbytes als zusammenhängende Elemente zu interpretieren.

## 3. Ausführungskontrolle

|Methode|Verhalten|
| --- | --- |
|`launch`|Sendet einen Kernel im geprüften Modus|
|`launch_unchecked`|Unsichere Schnittstelle; BoundsCheckMode steuert den eigentlichen Prüfmodus|
|`flush`|Sendet Befehle in der Warteschlange und gibt einen `Result` zurück|
|`sync`|Gibt einen Future zurück, der auf den Abschluss der Ausführung wartet|
|`set_stream`|Setzt den StreamId des Clients unsicher|

`launch` gibt keine berechneten Geräteergebnisse zurück. Beim Zurücklesen oder Synchronisieren können asynchrone Kompilierungs- oder Ausführungsfehler auftreten. Makrogenerierte Startschnittstellen erstellen auch Argumente. Ihre vollständigen Verträge sind nicht mit Client-Methodensignaturen austauschbar.

## 4. Fähigkeiten und Profilierung

`properties()` gibt Geräteeigenschaften zurück und `features()` gibt den Funktionssatz zurück. Fragen Sie die entsprechenden Funktionen ab, bevor Sie Dtypes, Atomics oder Matrixanweisungen auswählen. Verwenden Sie `enumerate_devices`, `enumerate_all_devices` und die Zählmethoden für die Aufzählung. `profile` bietet Laufzeitprofilierung; Unterscheiden Sie bei Messungen Einreichung, Ausführung und Übertragung.

## 5. Fehler und Sicherheit

Bei ungeprüften Low-Level-Starts muss der Aufrufer Zugriffe außerhalb der Grenzen und nicht terminierende Schleifen ausschließen. Layouts, Bindungslängen und streamübergreifende Lebensdauern müssen zum Kernel passen. Siehe [Debugging](debugging.md) zur Prüfkonfiguration und [Driver-API](driver-api.md) zur Geräteintegration.
