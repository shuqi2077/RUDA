# Treiber API und Backends

[English](../en/driver-api.md) | [简体中文](../zh/driver-api.md) | [日本語](../ja/driver-api.md) | **Deutsch** | [Русский](../ru/driver-api.md)

[Dokumentation](README.md) · [Laufzeit API](runtime-api.md) · [Kompatibilität](compatibility.md) · [中文](../zh/driver-api.md)

Rudas Treiber-Crates verbinden den allgemeinen Laufzeitvertrag mit den Ausführungs-Backends. Dieser Leitfaden beschreibt Rust-Backend-Einstiegspunkte, keine direkt austauschbaren Ersatzfunktionen für die CUDA Driver API.

## 1. Backend-Crates

|Kiste|Ausführungs-Backend|
| --- | --- |
|`ruda-driver-cuda`|NVIDIA CUDA Treiber- und Kompilierungspfade|
|`ruda-driver-cpu`|CPU|
|`ruda-driver-wgpu`|WGPU|
|`ruda-driver-hip`|HIP|

## 2. Wählen Sie ein NVIDIA-Gerät aus

`ruda_driver_cuda::CudaDevice` wählt ein Gerät über sein öffentliches `index: usize`-Feld aus, der Standardwert ist 0. `CudaRuntime` implementiert das Merkmal `Runtime`.

Rufen Sie den Client des Standardgeräts ab mit:

```rust
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::Runtime;

let client = CudaRuntime::client(&CudaDevice::default());
```

Ein vollständiges ausführbares Beispiel finden Sie unter [ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs). Ein Geräteindex identifiziert die Aufzählungsposition der aktuellen Maschine, nicht eine stabile Identität zwischen Maschinen oder Änderungen in der Aufzählungsreihenfolge.

## 3. Konfiguration

`RuntimeOptions` enthält `memory_config` für die Speicherverwaltung. `CudaCompiler` und `CudaComputeKernel` sind Typaliase für die C++-Kompilierungskette CUDA; Durch die Aktivierung des direkten PTX ändert sich ihre Bedeutung nicht.

Wählen Sie den Kompilierungspfad mit `RUDA_CUDA_COMPILER` aus, wie im [Compiler-Handbuch](compiler-guide.md) beschrieben. `install::cuda_path()`, `install::include_path()` und `install::cccl_include_path()` suchen Toolkit-Pfade.

## 4. Externe Abhängigkeiten

Die direkte PTX-Generierung umgeht den CUDA C++/NVRTC-Kompilierungsschritt des Kernels. Für die Ausführung ist weiterhin der Treiber NVIDIA erforderlich, und die Kiste behält NVRTC-Abhängigkeiten bei.

Die Schnittstellen garantieren nicht die Übernahme beliebiger externer CUDA-Kontexte, Streams oder Rohgerätezeiger. Bei der sprachübergreifenden Integration müssen Eigentum, Ausführungsabhängigkeiten und Fehlerausbreitung berücksichtigt werden. Ein CUDA-Backend allein bietet keine vollständige ABI-Kompatibilität.

## 5. Integrieren Sie ein weiteres Backend

Ein Backend verwendet `Runtime`, um ein Gerät, einen Compiler und einen Rechenserver zuzuordnen. Höhere Schichten greifen über `ComputeClient` auf den Vertrag zu. Legen Sie die Speicher-, Kompilierungsfehler-, Synchronisierungs- und Fähigkeitsabfragesemantik fest, bevor Sie Rechenbibliotheken integrieren.

Quelleneinstiegspunkte: [CUDA-Exporte](../../ruda-driver-cuda/src/lib.rs), [Gerätetyp](../../ruda-driver-cuda/src/device.rs), [Laufzeitimplementierung](../../ruda-driver-cuda/src/runtime.rs) und [Laufzeitmerkmal](../../ruda/src/runtime/backend.rs).

## 6. CUDA-Stream- und Event-Interoperabilität

`ruda_driver_cuda::interop::{command, StreamCommand, record_allocation}` verwendet denselben Gerätedienst und CUDA-Kontext wie RUDA-Kernel. Stream-/Event-IDs sind von RUDA verwaltete Kennungen, keine rohen CUDA-Handles; Stream 0 ist der Standardstream.

- `command(device, StreamCommand::Create)` erzeugt eine Stream-ID. `Validate`, `Query` und `Synchronize` validieren, prüfen oder warten auf diesen Stream.
- `Record { stream, event: 0, timing }` erzeugt und zeichnet ein Event auf; zur erneuten Aufzeichnung übergeben Sie die zurückgegebene ID. `Wait { stream, event }` fügt eine GPU-seitige Abhängigkeit ein, ohne auf dem Host auf GPU-Abschluss zu warten.
- `EventQuery` prüft den Abschluss, `EventSynchronize` wartet und `EventDestroy` gibt das Event frei. `Elapsed { start, end }` verlangt zwei Events mit aktivierter Zeitmessung und liefert FP32-Millisekunden als Bits in einem `u64`; dekodieren Sie mit `f32::from_bits(value as u32)`.
- `DeviceSynchronize` wartet auf den Kontext und meldet verzögerte RUDA-Launch-Fehler. Befehle liefern `Result<u64, ServerError>`; der Abschlussstatus wird als 0 oder 1 kodiert.

`record_allocation(device, stream, handle)` hält eine Allokation bis zum Abschluss der bereits auf diesem Stream eingereichten Arbeit am Leben; es ersetzt keine Ausführungsabhängigkeit. Die Stream-Erstellung ist durch `streaming.max_streams` begrenzt und schlägt bei erschöpftem Pool fehl. Diese APIs importieren keine beliebigen externen CUDA-Streams oder -Kontexte.
