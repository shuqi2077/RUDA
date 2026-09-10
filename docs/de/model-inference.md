# Modellladen und Inferenz

[English](../en/model-inference.md) | [简体中文](../zh/model-inference.md) | [日本語](../ja/model-inference.md) | **Deutsch** | [Русский](../ru/model-inference.md)

[Dokumentation](README.md) · [Schulung](training.md) · [中文](../zh/model-inference.md)

ruLLM bietet Modellladen, Tokenisierung, zwischengespeicherte autoregressive Generierung und Anforderungsplanung. Sein Cargo-Paket ist `ruLLM`; Sein Rust-Importname ist `rullm`.

## Bereiten Sie ein lokales Modell vor

Bewahren Sie die Modelldateien in einem Verzeichnis auf:

|Datei|Zweck|
| --- | --- |
|`config.json`|Modellarchitektur und Parameter|
|`model.safetensors`|Einzeldateigewichtungen|
|`model.safetensors.index.json` und jeder aufgelistete Shard|Geteilte Gewichtungen anstelle einer einzelnen Gewichtsdatei|
|`tokenizer.json`|Text-Tokenisierung und -Dekodierung|
|`tokenizer_config.json`|ChatML-Vorlage für Qwen2/Qwen2.5-Pipelines erforderlich|
|`generation_config.json`|Optionale Konfiguration der Qwen2/Qwen2.5-Generation, die EOS-Einstellungen bereitstellt|
|`preprocessor_config.json`|Qwen3.5-Bildvorverarbeitungskonfiguration, erforderlich für die Bildeingabe|

Loader akzeptieren lokale Verzeichnisse und laden keine Modelle herunter. Wählen Sie den Loader für Ihre Architektur aus:

|Modell/Eingabe|Lader|
| --- | --- |
|Unquantisierter Lama-Text|`load_huggingface_llama_pipeline::<B>`|
|Unquantisierter Qwen2/Qwen2.5-Text|`load_huggingface_qwen2_pipeline::<B>`|
|AWQ GEMM INT4 Qwen2/Qwen2.5|`load_huggingface_awq_qwen2_pipeline::<R>`|
|Unquantisierter Qwen3.5-Text|`load_huggingface_qwen35_text::<B>`|
|Qwen3.5 Vision-Encoder|`load_huggingface_qwen35_vision::<B>`|
|Qwen3.5 Bilder und Text|`load_huggingface_qwen35_multimodal::<B>`|

`B` ist ein Tensor-Backend; `R` ist eine Gerätelaufzeit. Der Qwen3.5-Textlader verwendet eine `qwen3_5_text`-Konfiguration mit `layer_types` pro Schicht und lehnt Quantisierungskonfigurationen ab. Übergeben Sie AWQ-Dateien nicht an einen Gleitkomma-Loader.

## Führen Sie die Beispiele zur Textgenerierung aus

Führen Sie nach dem [Einrichten der Umgebung](getting-started.md) diese Befehle im Quellstammverzeichnis aus. Die relativen Pfade verweisen auf zwei lokale Modellverzeichnisse:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruLLM --features nvidia --example qwen2_generate -- ./models/qwen2 "Hello" 32 1
```

Das Qwen3.5-Beispiel verwendet standardmäßig Ruda IR → PTX, wenn `RUDA_CUDA_COMPILER` nicht festgelegt ist. Verwenden Sie `--release` für eine optimierte Ausführung. Nicht unterstützte direkte PTX-Vorgänge geben Fehler zurück, anstatt auf NVRTC zurückzugreifen. Die PTX-Version muss mit der Zielversion GPU und dem Treiber übereinstimmen. siehe die [PTX Backend-Referenz](ptx.md).

```powershell
Remove-Item Env:RUDA_CUDA_COMPILER -ErrorAction SilentlyContinue
$env:RUDA_PTX_VERSION = '8.0'
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

Der Pfad CUDA C++ / NVRTC bleibt mit einer expliziten Auswahl verfügbar:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruLLM --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

Argumente sind das Modellverzeichnis, die Rohtext-Eingabeaufforderung, die maximale Anzahl neuer Token und die Anzahl der Ausführungen. Die letzten beiden sind standardmäßig `32` und `1` für Qwen2 oder `8` und `1` für Qwen3.5. Die Laufanzahl muss positiv sein. Beispiele drucken JSON-Zeilen zum Laden und Generieren. Zu den Generationslinien gehören `text`, `generated_token_ids` und `stopped_on_eos`.

Diese Beispiele kodieren die Eingabeaufforderung direkt, ohne eine Chat-Vorlage hinzuzufügen. Bereiten Sie für Konversationsmodelle die Eingabeaufforderung mit `render_chat` wie folgt vor.

## Erzeugen Sie eine Chat-Antwort in einer Anwendung

Für ein Anwendungsverzeichnis neben dem Quellverzeichnis `RUDA` verwenden Sie diese Abhängigkeiten.

```toml
[dependencies]
rullm = { package = "ruLLM", path = "../RUDA/ruLLM", features = ["nvidia"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
half = "=2.7.1"

```

Dieses vollständige `src/main.rs` akzeptiert ein Modellverzeichnis und eine Benutzernachricht als Befehlszeilenargumente. Spätere Funktionsbeispiele verwenden ihren `B`-Alias, ihre Importe und Abhängigkeiten wieder:

```rust
use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::{
    GreedyGenerationConfig, Qwen2ChatMessage, Qwen2ChatRole,
    load_huggingface_qwen2_pipeline,
};
use std::{error::Error, io};

type B = Cuda<bf16, i32>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| io::Error::other("model directory required"))?;
    let prompt = args.next().ok_or_else(|| io::Error::other("prompt required"))?;
    let device = CudaDevice::default();
    let pipeline = load_huggingface_qwen2_pipeline::<B>(&directory, &device)?;
    let messages = [Qwen2ChatMessage::new(Qwen2ChatRole::User, prompt)];
    let prompt = pipeline.render_chat(&messages, true)?;
    let output = pipeline.generate_text(
        &prompt,
        GreedyGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
        },
        false,
        true,
        &device,
    )?;
    println!("{}", output.generated_text);
    Ok(())
}
```

`render_chat` verarbeitet Klartext-Qwen2/Qwen2.5 ChatML-Nachrichten. Durch die Übergabe von `true` wird das Präfix für die Assistentengenerierung hinzugefügt. Zu den Rollen gehören auch `System` und `Assistant`; Geben Sie Nachrichten in der Konversationsreihenfolge an. Dieser API akzeptiert keine Tool-Call-Nachrichten.

Die beiden booleschen Argumente für `generate_text` sind `add_special_tokens` und `skip_special_tokens`. Fügen Sie nach dem Rendern von ChatML keine weiteren Sondertoken hinzu. Beim Dekodieren können sie übersprungen werden. `generated_text` enthält nur die neue Antwort, während `text` sowohl Aufforderung als auch Antwort enthält.

## Temperatur, Top-K- und Top-P-Probenahme

Übergeben Sie die gerenderte Eingabeaufforderung an `generate_text_sampled`:

```rust
use rullm::{
    GeneratedText, HuggingFaceLoadError, HuggingFaceQwen2Pipeline,
    SamplingConfig, SamplingGenerationConfig,
};

fn sample_text(
    pipeline: &HuggingFaceQwen2Pipeline<B>,
    prompt: &str,
    device: &CudaDevice,
) -> Result<GeneratedText, HuggingFaceLoadError> {
    pipeline.generate_text_sampled(
        prompt,
        SamplingGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: pipeline.loaded.default_eos_token_ids.clone(),
            sampling: SamplingConfig {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                seed: Some(42),
            },
        },
        false,
        true,
        device,
    )
}
```

|Parameter|Werte und Wirkung|
| --- | --- |
|`temperature`|Endlich und positiv; scales meldet sich zuerst an|
|`top_k`|Hält die Token mit der höchsten Punktzahl als nächstes; 0 deaktiviert die Filterung, Bindungen an der Grenze bleiben erhalten|
|`top_p`|Filtert schließlich nach kumulativer Wahrscheinlichkeit; Bereich (0, 1], wobei 1 die Filterung deaktiviert|
|`seed`|Anforderungs-Lokaler Probenahme-Seed; `None` verwendet Systementropie|
|`max_new_tokens`|Begrenzt neue Token, mit Ausnahme der Eingabeaufforderung|
|`eos_token_ids`|Stoppt nach der Generierung eines aufgelisteten Tokens|

Verwenden Sie `generate_text` für die Decodierung mit maximalem Logit, anstatt die Temperatur auf Null zu setzen. Ein fester Startwert legt die zufällige Stichprobensequenz fest, nicht die numerische Konsistenz zwischen Modellen oder Backends.

## AWQ-Gewichte laden

Der AWQ-Loader verwendet `CudaRuntime` und nicht `Cuda<bf16, i32>` als generischen Parameter:

```rust
use ruda_driver_cuda::CudaRuntime;
use rullm::{HuggingFaceAwqQwen2Pipeline, load_huggingface_awq_qwen2_pipeline};

fn load_awq(
    directory: &str,
    device: &CudaDevice,
) -> Result<HuggingFaceAwqQwen2Pipeline<CudaRuntime>, HuggingFaceLoadError> {
    load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(directory, device)
}
```

Die zurückgegebene Pipeline stellt außerdem `render_chat`, `encode`, `decode`, `generate_text` und `generate_text_sampled` bereit. Für die AWQ-Konfiguration sind `quant_method = "awq"`, `bits = 4`, `version = "gemm"` und `zero_point = true` erforderlich. `group_size` muss positiv sein oder `-1`. Lineare Projektionen verwenden F16-Aktivierungen. Siehe [ruBLAS INT4](libraries/rublas.md) für gepackte Gewichtslayouts.

## Qwen3.5-Bildeingabe

`generate_image_files_greedy` benötigt Token-IDs für eine Eingabeaufforderung, JPEG/PNG-Pfade in Eingabeaufforderungsreihenfolge, einen Bildprozessor und Generierungsoptionen:

```rust
use rullm::{
    Qwen35ImageProcessor, TokenGenerationOutput,
    load_huggingface_qwen35_multimodal,
};

fn generate_from_images(
    directory: &str,
    prompt_ids: &[i32],
    paths: &[&str],
    device: &CudaDevice,
) -> Result<TokenGenerationOutput, Box<dyn Error>> {
    let model = load_huggingface_qwen35_multimodal::<B>(directory, device)?;
    let processor = Qwen35ImageProcessor::from_huggingface(directory)?;
    let output = model.generate_image_files_greedy(
        prompt_ids,
        paths,
        &processor,
        &GreedyGenerationConfig {
            max_new_tokens: 32,
            eos_token_ids: vec![model.text.config.eos_token_id],
        },
    )?;
    Ok(output)
}
```

Kodieren Sie `prompt_ids` mit dem `tokenizer.json` des Modells. Befolgen Sie die Eingabeaufforderungsvorlage des Modells und behalten Sie ein Platzhaltertoken pro Bild bei, indem Sie `image_token_id` aus `config.json` verwenden. Bildpfade müssen mit der Anzahl und Reihenfolge der Platzhalter übereinstimmen. Dieser Einstiegspunkt erweitert Platzhalter intern; Erweitern Sie sie nicht vorher. Es verarbeitet Standbilder und lehnt Video-Tokens ab.

Für vorhandene RGB8-Daten verwenden Sie `Qwen35RgbImage { pixels, height, width }` mit `generate_rgb_greedy`. Pixel sind verschachtelte RGB-Bytes ohne Zeilenauffüllung. Für die separate Vorverarbeitung geben `preprocess_files` und `preprocess_rgb` FP32 `patches`, ein zweidimensionales `shape` und ein bildbezogenes `grids` zurück.

Wenn Sie mehrere Anfragen bedienen, laden Sie das Modell und den Prozessor außerhalb der Anfrageschleife. Dekodieren Sie den zurückgegebenen `generated_token_ids` mit demselben Tokenizer.

## Caches und kontinuierliche Stapelverarbeitung

Generierungsfunktionen für einzelne Anforderungen erstellen und verwalten einen Cache für diese Generierung. Um Qwen3.5 manuell zu steuern, rufen Sie `model.new_cache()` auf, übergeben Sie den Prompt-Tensor an `forward_cached_last(tokens, &mut cache)` und übergeben Sie dann nur neue Token. Übermitteln Sie nicht das gesamte zwischengespeicherte Präfix erneut.

Um Ihren eigenen Batch-Executor zu integrieren, verwenden Sie `ContinuousBatchScheduler`:

1. Erstellen Sie es mit `ContinuousBatchScheduler::new(config, kv_config)`. `ContinuousBatchConfig` setzt `max_active_sequences` und `max_batch_tokens`. `PagedKvCacheConfig` legt `block_size`, `num_pages` und `max_sequence_length` fest. Alles muss positiv sein.
2. Senden Sie Anforderungen mit `submit(prompt_token_ids, generation)` und behalten Sie alle zurückgegebenen `RequestId` bei.
3. Rufen Sie `schedule()` für einen `ScheduledBatch` an. Sein `kind` zeichnet sich durch Prefill/Decode aus; `sequences` stellt die Token, die Startposition, die Kontextlänge und die Blocktabelle jeder Zeile bereit.
4. Führen Sie das Modell in Zeilenreihenfolge aus, schreiben Sie die entsprechenden KV-Seiten und wählen Sie das nächste Token jeder Zeile aus. Rufen Sie bei Erfolg `complete_batch(batch.id, &generated_token_ids)` mit einem Ergebnistoken pro Zeile auf. Rufen Sie bei einem Fehler `fail_batch(batch.id)` auf, um die Seitenreservierungen dieses Stapels zu stornieren.
5. Abgeschlossene Anfragen und generierte Token mit `pop_finished()` abrufen und dann mit der Planung fortfahren.

Der Scheduler verwaltet Anfragen und Seitenzuordnungsmetadaten. Der Ausführende des Aufrufers besitzt den Speicher und die Modellausführung des Geräts KV. Verwenden Sie `token_matrix()`, `context_lengths()` und `flattened_block_table()`, um Batch-Eingaben zu erstellen. Schließen Sie einen ausstehenden Stapel ab oder stornieren Sie ihn, bevor Sie `schedule()` erneut aufrufen.

API-Referenz: [ruLLM-Exporte](../../ruLLM/src/lib.rs), [Generierungsoptionen](../../ruLLM/src/generation.rs), [Planer](../../ruLLM/src/continuous_batch.rs).
