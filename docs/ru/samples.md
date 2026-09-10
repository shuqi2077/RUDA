# Примеры и руководства

[English](../en/samples.md) | [简体中文](../zh/samples.md) | [日本語](../ja/samples.md) | [Deutsch](../de/samples.md) | **Русский**

[Документация](README.md) · [Краткое руководство](getting-started.md) · [中文](../zh/samples.md)

## 1. Пример выполнения NVIDIA

Точка входа: [ptx-runtime](../../ruda-driver-cuda/examples/ptx_runtime.rs).

В базовом случае сложение FP32 выполняется дважды для каждой длины: 1, 63, 64, 65 и 257. Он проверяет каждый результат и 16 хвостовых контрольных значений. Он демонстрирует выбор устройства, загрузку, привязку аргументов, хвостовые границы, обратное чтение и повторное выполнение.

См. [быстрый старт](getting-started.md) для создания и выбора бэкенда.

## 2. Целевые случаи

Поместите эти аргументы после `--` команды. Например, запустите тензорный случай с помощью:

```powershell
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --tensor
```

|Аргумент|Поведение|Пример источника|
| --- | --- | --- |
|`--tensor`|Проверка метаданных и макета тензора|[tensor.rs](../../ruda-driver-cuda/examples/ptx_runtime/tensor.rs)|
|`--shared`|Проверка общей памяти|[shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs)|
|`--half`|FP16/BF16 проверки|[half_precision.rs](../../ruda-driver-cuda/examples/ptx_runtime/half_precision.rs)|
|`--bitwise`|Проверка побитовых операций|[побитовый.rs](../../ruda-driver-cuda/examples/ptx_runtime/bitwise.rs)|
|`--shared-over-limit`|Диагностика ограничения общей памяти|[shared.rs](../../ruda-driver-cuda/examples/ptx_runtime/shared.rs)|
|`--expect-cold`|Компиляция утверждений произошла без обращения к дисковому кэшу|[Основная точка входа](../../ruda-driver-cuda/examples/ptx_runtime.rs)|
|`--expect-warm`|Подтверждает попадание в дисковый кэш без перекомпиляции.|[Основная точка входа](../../ruda-driver-cuda/examples/ptx_runtime.rs)|

`--shared-over-limit` занимает отдельную ветвь раннего возврата. Не комбинируйте его с проверками холодного/теплого кэша.

## 3. Холодные и теплые кэши

Используйте отдельный новый каталог кэша для каждого пути компиляции. Повторно используйте этот каталог для холодных и теплых запусков одного и того же пути.

После выбора пути компиляции, как описано в [Начало работы](getting-started.md), запустите эти команды в том же сеансе PowerShell:

```powershell
$env:RUDA_PTX_TEST_CACHE = 'target/ptx-example-cache-' + [guid]::NewGuid().ToString('N')
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-cold
cargo run --locked -p ruda-driver-cuda --features direct-ptx --example ptx-runtime -- --expect-warm
```

Сохраняйте неизменными компилятор, входные аргументы и путь к кэшу между запусками.

## 4. Примеры вычислительных библиотек

Подготовьте [среду сборки](getting-started.md) перед запуском примеров.

|Задача|Команда|Проверка вывода|
| --- | --- | --- |
|FP32 CSR Матрично-векторное умножение|`cargo run --locked -p ruSPARSE --features cuda --example csrmv`|Считывает и проверяет `[7.0, 2.0, 18.5]`|
|CUDA Кольцо AllReduce|`cargo run --locked -p ruCCL --features cuda --example all_reduce`|Четыре логических ранга в GPU 0, 257 элементов, сумма/среднее и сохранение входных данных.|

## 5. Обучение и вывод модели

- [Состояние обучения и сохранения](training.md): вперед, назад, накопление градиента и записи обучения.
- [Загрузка и вывод модели](model-inference.md): примеры команд Qwen2/Qwen3.5, чат, выборка, AWQ и ввод изображений.
