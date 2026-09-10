# ruLLM

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | **Русский**

**Английский** | [简体中文](../zh/README.md)

LLM Вывод, загрузка модели и генерация текста в Ruda.

ruLLM сочетает в себе токенизацию, загрузку модели, кэшированную авторегрессионную генерацию и планирование запросов с помощью тензоров Ruda и вычислительных библиотек.

- Cargo пакет: `ruLLM`
- Крейт Rust: `rullm`

## Краткое руководство

Настройте [среду NVIDIA](../../../docs/ru/getting-started.md) и подготовьте локальный [каталог модели Qwen3.5-0.8B](../../../docs/ru/model-inference.md#подготовьте-локальную-модель).

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "The capital of France is" 8 1
```

Замените `./models/qwen35` каталогом вашей модели. Последние аргументы выбирают максимальное количество новых токенов и количество запусков. В примере по умолчанию используется прямой PTX, когда `RUDA_CUDA_COMPILER` не установлен. Установите для него значение `nvrtc`, чтобы вместо этого использовать путь компиляции C++ CUDA. См. [Конфигурация PTX](../../../docs/ru/ptx.md) для выбора целевой версии.

## Документация

- [Загрузка и вывод модели](../../../docs/ru/model-inference.md)
- [Функции Cargo](../../Cargo.toml) · [Экспорт модулей](../../src/lib.rs)

## Загрузка моделей и инференс

[Документация](../../../docs/ru/README.md) · [Обучение](../../../docs/ru/training.md) · [中文](../zh/README.md)

ruLLM обеспечивает загрузку модели, токенизацию, кэшированную авторегрессионную генерацию и планирование запросов. Его пакет Cargo — `ruLLM`; его имя для импорта в Rust — `rullm`.

### Подготовьте локальную модель

Храните файлы модели в одном каталоге:

|Файл|Цель|
| --- | --- |
|`config.json`|Архитектура и параметры модели|
|`model.safetensors`|Веса отдельных файлов|
|`model.safetensors.index.json` и каждый указанный фрагмент|Сегментированные веса вместо одного файла весов|
|`tokenizer.json`|Токенизация и декодирование текста|
|`tokenizer_config.json`|ChatML Шаблон, необходимый для конвейеров Qwen2/Qwen2.5|
|`generation_config.json`|Дополнительная конфигурация поколения Qwen2/Qwen2.5, обеспечивающая настройки EOS|
|`preprocessor_config.json`|Конфигурация предварительной обработки изображения Qwen3.5, необходимая для ввода изображения|

Загрузчики принимают локальные каталоги и не загружают модели. Выберите загрузчик для вашей архитектуры:

|Модель/вход|Загрузчик|
| --- | --- |
|Неквантованный текст ламы|`load_huggingface_llama_pipeline::<B>`|
|Неквантованный текст Qwen2/Qwen2.5|`load_huggingface_qwen2_pipeline::<B>`|
|AWQ GEMM INT4 Qwen2/Qwen2.5|`load_huggingface_awq_qwen2_pipeline::<R>`|
|Неквантованный текст Qwen3.5|`load_huggingface_qwen35_text::<B>`|
|Видеокодер Qwen3.5|`load_huggingface_qwen35_vision::<B>`|
|Qwen3.5 изображения и текст|`load_huggingface_qwen35_multimodal::<B>`|

`B` — тензорный бэкенд; `R` — это среда выполнения устройства. Текстовый загрузчик Qwen3.5 использует конфигурацию `qwen3_5_text` с послойным `layer_types` и отклоняет конфигурации квантования. Не передавайте файлы AWQ в загрузчик с плавающей запятой.

### Запустите примеры генерации текста

После [настройки среды](../../../docs/ru/getting-started.md) запустите эти команды из корня исходного кода. Относительные пути указывают на два локальных каталога модели:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --locked -p ruda-llm --features nvidia --example qwen2_generate -- ./models/qwen2 "Hello" 32 1
```

В примере Qwen3.5 по умолчанию используется Ruda IR → PTX, если `RUDA_CUDA_COMPILER` не установлен. Используйте `--release` для оптимизации выполнения. Неподдерживаемые прямые операции PTX возвращают ошибки, а не возвращаются к NVRTC. Версия PTX должна соответствовать целевому GPU и драйверу; см. [Справочник по бэкенда PTX](../../../docs/ru/ptx.md).

```powershell
Remove-Item Env:RUDA_CUDA_COMPILER -ErrorAction SilentlyContinue
$env:RUDA_PTX_VERSION = '8.0'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

Путь CUDA C++/NVRTC остается доступным при явном выборе:

```powershell
$env:RUDA_CUDA_COMPILER = 'nvrtc'
cargo run --release --locked -p ruda-llm --features nvidia-ptx --example qwen35_generate -- ./models/qwen35 "Hello" 8 1
```

Аргументами являются каталог модели, текстовое приглашение, максимальное количество новых токенов и количество запусков. Последние два по умолчанию равны `32` и `1` для Qwen2 или `8` и `1` для Qwen3.5. Число пробегов должно быть положительным. В примерах печатаются строки JSON для загрузки и генерации. Линии генерации включают `text`, `generated_token_ids` и `stopped_on_eos`.

В этих примерах приглашение кодируется напрямую, без добавления шаблона чата. Для диалоговых моделей подготовьте подсказку с `render_chat`, как показано ниже.

### Создание ответа в чате в приложении

Для каталога приложения рядом с исходным каталогом `RUDA` используйте эти зависимости.

```toml
[dependencies]
rullm = { package = "ruLLM", path = "../RUDA/ruLLM", features = ["nvidia"] }
ruda-tensor-device = { path = "../RUDA/ruda-tensor-device", default-features = false, features = ["std", "cuda"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
half = "=2.7.1"

```

Этот полный `src/main.rs` принимает каталог модели и сообщение пользователя в качестве аргументов командной строки. Более поздние примеры функций повторно используют псевдоним `B`, импорт и зависимости:

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

`render_chat` обрабатывает текстовые сообщения Qwen2/Qwen2.5 ChatML. При передаче `true` добавляется префикс поколения помощника. Роли также включают `System` и `Assistant`; подавайте сообщения в порядке разговора. Этот API не принимает сообщения о вызове инструмента.

Два логических аргумента для `generate_text`: `add_special_tokens` и `skip_special_tokens`. Не добавляйте специальные токены повторно после рендеринга ChatML; декодирование может их пропустить. `generated_text` содержит только новый ответ, а `text` включает как приглашение, так и ответ.

### Выборка температуры, top-k и top-p

Передайте визуализированное приглашение `generate_text_sampled`:

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

|Параметр|Значения и эффект|
| --- | --- |
|`temperature`|Конечный и положительный; scales входит в систему первым|
|`top_k`|Оставляет токены с высокими показателями следующими; 0 отключает фильтрацию, связи на границе сохраняются|
|`top_p`|Наконец фильтруется по кумулятивной вероятности; диапазон (0, 1], где 1 отключает фильтрацию|
|`seed`|Запросить начальное значение локальной выборки; `None` использует системную энтропию|
|`max_new_tokens`|Ограничивает количество новых токенов, исключая запрос|
|`eos_token_ids`|Останавливается после создания любого токена из списка.|

Используйте `generate_text` для декодирования с максимальным логитом вместо установки температуры на ноль. Фиксированное начальное число фиксирует случайную последовательность выборки, а не числовую согласованность между моделями или серверными модулями.

### Загрузка весов AWQ

Загрузчик AWQ принимает в качестве общего параметра `CudaRuntime`, а не `Cuda<bf16, i32>`:

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

Возвращенный конвейер также предоставляет `render_chat`, `encode`, `decode`, `generate_text` и `generate_text_sampled`. Для конфигурации AWQ требуются `quant_method = "awq"`, `bits = 4`, `version = "gemm"` и `zero_point = true`. `group_size` должен быть положительным или `-1`. Линейные проекции используют активации F16. См. [ruBLAS INT4](../../../docs/ru/libraries/rublas.md) для получения информации о весе в упаковке.

### Ввод изображения Qwen3.5

`generate_image_files_greedy` принимает идентификаторы токенов для одного запроса, пути JPEG/PNG в порядке запроса, процессор изображений и параметры генерации:

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

Закодируйте `prompt_ids` с помощью `tokenizer.json` модели. Следуйте шаблону подсказки модели, сохраняя один токен-заполнитель для каждого изображения, используя `image_token_id` из `config.json`. Пути к изображениям должны соответствовать количеству и порядку заполнителей. Эта точка входа расширяет заполнители внутри; не расширяйте их заранее. Он обрабатывает неподвижные изображения и отклоняет видеотокены.

Для существующих данных RGB8 используйте `Qwen35RgbImage { pixels, height, width }` с `generate_rgb_greedy`. Пиксели представляют собой чередующиеся байты RGB без заполнения строк. Для раздельной предварительной обработки `preprocess_files` и `preprocess_rgb` возвращают FP32 `patches`, двумерный `shape` и `grids` для каждого изображения.

При обслуживании нескольких запросов загружайте модель и процессор вне цикла запросов. Декодируйте возвращенный `generated_token_ids` с помощью того же токенизатора.

### Кэши и непрерывная пакетная обработка

Функции генерации одного запроса создают и поддерживают кэш для этого поколения. Чтобы запустить Qwen3.5 вручную, вызовите `model.new_cache()`, передайте тензор приглашения `forward_cached_last(tokens, &mut cache)`, затем передавайте только новые токены. Не отправляйте повторно весь кэшированный префикс.

Чтобы интегрировать собственный пакетный исполнитель, используйте `ContinuousBatchScheduler`:

1. Создайте его с помощью `ContinuousBatchScheduler::new(config, kv_config)`. `ContinuousBatchConfig` устанавливает `max_active_sequences` и `max_batch_tokens`. `PagedKvCacheConfig` устанавливает `block_size`, `num_pages` и `max_sequence_length`. Все должно быть позитивно.
2. Отправляйте запросы с помощью `submit(prompt_token_ids, generation)` и сохраняйте каждый возвращенный `RequestId`.
3. Вызовите `schedule()` для получения `ScheduledBatch`. Его `kind` отличает предварительное заполнение/декодирование; `sequences` предоставляет токены каждой строки, начальную позицию, длину контекста и таблицу блоков.
4. Запустите модель в порядке строк, запишите соответствующие страницы KV и выберите следующий токен каждой строки. В случае успеха вызовите `complete_batch(batch.id, &generated_token_ids)` с одним токеном результата в каждой строке. В случае неудачи позвоните по `fail_batch(batch.id)`, чтобы отменить резервирование страниц этого пакета.
5. Получите выполненные запросы и сгенерированные токены с помощью `pop_finished()`, а затем продолжите планирование.

Планировщик управляет запросами и метаданными распределения страниц. Исполнитель вызывающего объекта владеет устройством хранения и исполнения модели KV. Используйте `token_matrix()`, `context_lengths()` и `flattened_block_table()` для создания пакетных входных данных. Завершите или отмените невыполненный пакет, прежде чем снова звонить по `schedule()`.

API Ссылка: [Экспорт ruLLM](../../src/lib.rs), [Параметры генерации](../../src/generation.rs), [Планировщик](../../src/continuous_batch.rs).
