# ruDNN

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | **Русский**

**Английский** | [简体中文](../zh/README.md)

Операторы нейронной сети для Ruda.

- Cargo пакет: `ruDNN`
- Крейт Rust: `rudnn`

## Feature

| Feature |Операции|
| --- | --- |
|`tensor-attention`| Механизм внимания |
|`tensor-convolution`|Свертка|
|`tensor-normalization`|LayerNorm, RMSNorm и softmax|
|`tensor-moe`|MoE маршрутизация, диспетчеризация и экспертные вычисления|
|`tensor-gated-delta`|Стробированное дельта-вычисление|
|`pooling`|Объединение в пул|
|`interpolation`|Интерполяция|
|`grid-sample`|Выборка сетки|
|`ctc`|CTC потеря|

## Краткое руководство

Сборка из рабочей области RUDA:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruDNN --features tensor-normalization
```

## Документация

- [Руководство пользователя](../../../docs/ru/libraries/rudnn.md)
- [Настройка среды](../../../docs/ru/getting-started.md)
- [Функции Cargo](../../Cargo.toml) · [Экспорт модулей](../../src/lib.rs)

## ruDNN Руководство пользователя

[Вычислительные библиотеки](../../../docs/ru/libraries/README.md) · [ruBLAS](../../../docs/ru/libraries/rublas.md) · [Тензоры и платформы](../../../docs/ru/tensor-framework.md) · [中文](../zh/README.md)

### 1. Обзор и возможности

ruDNN обеспечивает операции нейронной сети. Пакет Cargo — `ruDNN`, а крейт Rust — `rudnn`.

| Feature |Операции|
| --- | --- |
|`tensor-attention`|Тензорное внимание|
|`tensor-convolution`|Тензорная свертка|
|`pooling`, `interpolation`|Объединение и интерполяция|
|`grid-sample`, `ctc`|Выборка сетки и CTC|
|`tensor-moe`|Устройство MoE Маршрутизация и экспертные вычисления|
|`tensor-normalization`|Нормализация устройств общего назначения для слоев модели|
|`tensor-gated-delta`|Вычисления Gated-delta для гибридных архитектур, таких как Qwen3.5|

Внимание и свертка имеют соответствующие функции автонастройки. См. [Cargo.toml](../../Cargo.toml) и [экспорт модуля](../../src/lib.rs).

### 2. Механизм внимания и свёртка

#### Механизм внимания

`rudnn::attention::tensor::attention` принимает запрос, ключ, значение, необязательную маску, необязательные attn_bias, AttentionModuleOptions и AttentionStrategy. Он возвращает тензор устройства или AttentionSetupError.

Стратегии включают FlashBlackboxAccelerated, FlashUnit, Fallback и Autotune при включённом соответствующем feature. Без автонастройки по умолчанию используется Fallback, с ней — Autotune. Fallback использует несколько ядер на том же устройстве, а не бэкенд CPU.

Сопоставьте макет, маску и точность с выбранной стратегией. См. [интерфейс внимания](../../src/attention/tensor/base.rs).

#### Свертка

`rudnn::convolution::tensor::conv_forward` принимает входные данные, вес, необязательные bias, ConvOptions<N> и ConvStrategy. Он возвращает тензор устройства или ConvSetupError. Он преобразует входные данные в макет последнего канала для выполнения и преобразует выходные данные обратно. `conv_forward_nhwc` напрямую использует макет последнего канала.

Стратегии включают Direct, ImplicitGemm и дополнительный Autotune. Точка входа использует Direct для трехмерной свертки F32. Групповая свертка также использует Direct, если выбран ImplicitGemm. Таким образом, параметр стратегии не всегда является строгим требованием сохранить один алгоритм.

Один и тот же модуль предоставляет conv_data_backward и conv_weight_backward. Проверьте форму, варианты и требования к исполнению для каждого направления. См. [интерфейс свертки](../../src/convolution/tensor/base.rs).

### 3. Рабочий процесс MoE

`rudnn::moe` предоставляет следующую последовательность локальных вычислений:

Входные logits → softmax/top-k → компактное распределение → эксперты SwiGLU → взвешенное объединение.

|API|Цель|
| --- | --- |
|`route(logits, RoutingOptions)`|Создает RoutingPlan.|
|`RoutingPlan::expert_indices()`, `weights()`|Доступ к выбранным экспертам и весам.|
|`RoutingPlan::dispatch(input)`|Отправляет токены экспертом|
|`SwiGluExperts::new(gate, up, down)`|Создает экспертные веса без bias.|
|`SwiGluExperts::forward_dispatched`|Вычисляет отправленные токены|
|`DispatchedTokens::combine`|Восстанавливает порядок токенов и объединяет взвешенные результаты.|
|`SwiGluExperts::forward(input, logits, options)`|Выполняет полную локальную последовательность пересылки.|

Эти интерфейсы используют `RudaTensor<R>`. Точки входа вычислений возвращают `Result` с `MoeError` для недопустимых фигур, dtypes или других аргументов.

### 4. Контракт маршрутизации

Logits имеют форму [T, E] и используют неквантованные F32, F16 или BF16. `RoutingOptions` содержит `top_k: usize` и `renormalize: bool`, причём 1 ≤ top_k ≤ E.

Softmax вычисляет всех экспертов в FP32 перед выбором топ-k. Связи благоприятствуют более низким идентификаторам экспертов. При включенной перенормировке выбранные веса перенормируются, а затем приводятся к логитам dtype. Строки, содержащие NaN, положительную бесконечность или только отрицательную бесконечность, сохраняют веса NaN, а не используют равномерное распределение.

Выбранные экспертные индексы и веса имеют форму [T, top_k]. Индексы используют U32.

### 5. Веса экспертов и распределение

Входные токены имеют форму [T, H]; gate и up — [E, I, H]; down — [E, H, I]. Веса должны иметь общий dtype с плавающей точкой и находиться на одном устройстве. Число экспертов и размеры должны соответствовать маршрутизации и входу.

Распределение не отбрасывает токены ради соблюдения ограничения вместимости. Порядок атомарного назначения внутри эксперта не фиксирован; сохранённое отображение восстанавливает порядок токенов. Точка входа прямого прохода принимает заранее вычисленные logits, а не выполняет проекцию gate модели или загрузку файлов весов.

Источник: [маршрутизация](../../src/moe/routing.rs), [отправка и объединение](../../src/moe/dispatch.rs), [эксперты](../../src/moe/experts.rs) и [тесты](../../src/moe/tests.rs).

### 6. Вызов MoE

Включите `tensor-moe` в вашей зависимости `ruDNN`. Подготовьте тензоры устройств с помощью приведенных выше макетов, затем создайте экспертные веса и запустите прямую операцию:

```rust
use rudnn::moe::{RoutingOptions, SwiGluExperts};

let experts = SwiGluExperts::new(gate, up, down)?;
let options = RoutingOptions {
    top_k: 2,
    renormalize: true,
};
let output = experts.forward(input, logits, options)?;
```

При этом для каждого токена выбирается два эксперта, поэтому `E` должно быть не менее 2. `input` имеет форму `[T, H]`, `logits` имеет форму `[T, E]`, а `output` имеет форму `[T, H]`. Повторно используйте `experts` в последующих входных пакетах без повторного создания объекта веса.

Чтобы проверить маршрутизацию или вставить собственную обработку между этапами, вызовите их отдельно:

```rust
use rudnn::moe::route;

let routing = route(logits, options)?;
let selected_experts = routing.expert_indices();
let selected_weights = routing.weights();
let dispatched = routing.dispatch(input)?;
let expert_output = experts.forward_dispatched(&dispatched)?;
let output = dispatched.combine(expert_output)?;
```

Это альтернативные формы; второй начинается со свежей партии `input` и `logits`. `combine` использует сопоставление диспетчеризации для восстановления порядка токенов и объединяет экспертные выходные данные с использованием весов маршрутизации.

Форма, dtype или несоответствие устройств возвращают `MoeError`. Чтобы прочитать результаты на хосте, используйте `ruda_kernel::tensor::readback::into_data_sync(output)`, который ожидает результатов устройства и возвращает `TensorData`.

### 7. LayerNorm, RMSNorm и Softmax

Включите `tensor-normalization` и импортируйте эти функции из `rudnn::normalization`. Все они работают с конечной осью ввода и сохраняют форму:

|Функция|Ввод|Параметры|
| --- | --- | --- |
|`layer_norm(input, gamma, beta, epsilon)`|F32/F16/BF16|F32 вектор `gamma`, дополнительно F32 вектор `beta`|
|`rms_norm(input, gamma, epsilon)`|F32/F16/BF16|F32 вектор `gamma`|
|`softmax_last_axis(input)`|F32|Никаких дополнительных параметров.|

Входные данные должны быть неквантованными с непустой конечной осью. Для длины конечной оси H `gamma` и `beta` должны иметь форму `[H]`, не быть квантованными и иметь общее устройство ввода. `epsilon` должен быть конечным и положительным. Аффинные параметры остаются F32 даже для входов BF16/F16. В статистике и аффинной арифметике используется FP32, при этом выходные данные преобразуются во входные dtype.

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rudnn::normalization::{NormalizationError, layer_norm, rms_norm, softmax_last_axis};

fn normalize<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    beta: Option<RudaTensor<R>>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    layer_norm(input, gamma, beta, epsilon)
}

fn normalize_rms<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    rms_norm(input, gamma, epsilon)
}

fn probabilities<R: Runtime>(
    logits: RudaTensor<R>,
) -> Result<RudaTensor<R>, NormalizationError> {
    softmax_last_axis(logits)
}
```

### 8. Gated-delta: предварительное заполнение и рекуррентное вычисление

Включите `tensor-gated-delta`. Используйте `chunk_gated_delta_rule(input, chunk_size)` для предварительного заполнения фрагментированной последовательности и `gated_delta_rule(input)` для повторения каждого токена. Оба принимают `GatedDeltaInput<R>`:

|Поле| Форма/тип |
| --- | --- |
|`query`, `key`|`[B, H, T, K]`, соответствующий F32/F16/BF16|
|`value`|`[B, H, T, V]`, тот же dtype, что и в запросе|
|`beta`|`[B, H, T]`, тот же dtype, что и в запросе|
|`log_decay`|`[B, H, T]`, F32|
|`initial_state`|`[B, H, K, V]`, F32|
|`query_scale`| Конечный f32 согласно конфигурации модели |

Все тензоры должны быть неквантованными и находиться на одном устройстве. Поставка Q/K после нормализации для конкретной модели; точка входа не выполняет это за вас. Эта функция повторно использует предыдущий импорт `Runtime` и `RudaTensor`:

```rust
use rudnn::gated_delta::{
    GatedDeltaError, GatedDeltaInput, GatedDeltaOutput, chunk_gated_delta_rule,
};

fn delta_prefill<R: Runtime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    beta: RudaTensor<R>,
    log_decay: RudaTensor<R>,
    initial_state: RudaTensor<R>,
    query_scale: f32,
    chunk_size: usize,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    chunk_gated_delta_rule(
        GatedDeltaInput {
            query, key, value, beta, log_decay, initial_state, query_scale,
        },
        chunk_size,
    )
}
```

Возвращаемый `output` имеет форму `[B, H, T, V]` и тот же dtype, что у query. `final_state` имеет тип F32 и форму `[B, H, K, V]`. Для следующего сегмента той же последовательности передайте этот `final_state` как `initial_state`. Для разных последовательностей храните отдельные состояния. Начальное состояние не перезаписывается на месте.

`chunk_size` должен быть положительным, а треугольная рабочая область размером `4 × (chunk_size² + chunk_size)` байт не должна превышать лимит общей памяти устройства на рабочую группу. Точка входа дополняет последний блок; заполнение исключается из выхода. Недопустимые аргументы возвращают `GatedDeltaError`.

Для вызовов текста и изображений на уровне модели см. [Руководство по выводу ruLLM](../../../docs/ru/model-inference.md).
