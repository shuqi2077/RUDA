# ruPRIM

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | **Русский**

**Английский** | [简体中文](../zh/README.md)

Параллельные примитивы, сокращения, сканирования и индексирование для Ruda.

- Cargo пакет: `ruPRIM`
- Крейт Rust: `ruprim`

## Feature

| Feature |Операции|
| --- | --- |
|`tensor-reduce`|Приведение всего тензора и оси|
|`tensor-reduce-autotune`|Автонастройка снижения|
|`tensor-scan`|Совокупная сумма, произведение, минимум и максимум|
|`elementwise`|Поэлементные операции|
|`indexing`|Выбор, нарезка, сбор и рассеяние|

## Краткое руководство

Сборка из рабочей области RUDA:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruPRIM --features tensor-reduce,tensor-scan,indexing
```

## Документация

- [Руководство пользователя](../../../docs/ru/libraries/ruprim.md)
- [Настройка среды](../../../docs/ru/getting-started.md)
- [Функции Cargo](../../Cargo.toml) · [Экспорт модулей](../../src/lib.rs)

## ruPRIM Руководство пользователя

[Вычислительные библиотеки](../../../docs/ru/libraries/README.md) · [Тензорная платформа](../../../docs/ru/tensor-framework.md) · [中文](../zh/README.md)

ruPRIM обеспечивает сокращение тензора устройства, совокупное сканирование, поэлементные операции и индексирование. На этой странице используется `RudaTensor<R>`, где R — среда выполнения устройства.

### 1. Настройте зависимости

Пакет Cargo — `ruPRIM`; его имя для импорта в Rust — `ruprim`. Включите функции для необходимых вам операций:

| Feature |Интерфейс|
| --- | --- |
|`tensor-reduce`|`ruprim::reduce::tensor`: сокращение всего тензора и оси|
|`tensor-reduce-autotune`| Автонастройка редукции; также включает tensor-reduce |
|`tensor-scan`|`ruprim::scan`: совокупная сумма, произведение, минимум, максимум.|
|`elementwise`|`ruprim::elementwise`: поэлементное вычисление|
|`indexing`| `ruprim::indexing`: выбор, срезы, gather, scatter; также включает elementwise |

В этой конфигурации каталог приложения размещается рядом с исходным каталогом `RUDA`. См. раздел [Начало работы](../../../docs/ru/getting-started.md) для настройки NVIDIA.

```toml
[dependencies]
ruprim = { package = "ruPRIM", path = "../RUDA/ruPRIM", default-features = false, features = ["std", "tensor-reduce", "tensor-scan", "indexing"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Сумма, сокращение строк и сканирование

Этот полный `src/main.rs` использует матрицу F32 `[[1, 2, 3], [4, 5, 6]]` для вычисления общего значения, средних значений строк, индексов argmax строк и сумм префиксов строк. Запустите `cargo run` из каталога приложения:

```rust
use ruda_core::tensor::{DType, TensorMetadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use ruprim::reduce::{
    components::instructions::ReduceOperationConfig,
    tensor::{KernelReduceStrategy, SumStrategy, reduce_dim, sum},
};
use ruprim::scan::cumsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        &device,
    );
    let total = sum(
        input.clone(),
        SumStrategy::Chained(KernelReduceStrategy::Unspecified),
    )?;
    assert_eq!(total.shape().as_slice(), &[1]);
    assert_eq!(into_data_sync(total).to_vec::<f32>()?, [21.0]);

    let means = reduce_dim(
        input.clone(), None, 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::Mean,
    )?;
    assert_eq!(means.shape().as_slice(), &[2, 1]);
    assert_eq!(into_data_sync(means).to_vec::<f32>()?, [2.0, 5.0]);

    let indices = reduce_dim(
        input.clone(), Some(DType::I32), 1,
        KernelReduceStrategy::Unspecified, ReduceOperationConfig::ArgMax,
    )?;
    assert_eq!(into_data_sync(indices).to_vec::<i32>()?, [2, 2]);

    let prefix = cumsum(input, 1);
    assert_eq!(into_data_sync(prefix).to_vec::<f32>()?, [1., 3., 6., 4., 9., 15.]);
    Ok(())
}
```

`sum` уменьшает все элементы и возвращает форму `[1]`. `reduce_dim` обрабатывает только выбранную ось, обычно сохраняя ранг и устанавливая длину этой оси равной 1. Таким образом, строка означает, что она имеет форму `[2, 1]`, а не `[2]`.

### 3. Операции и стратегии сокращения

Полный порядок параметров для тензорных интерфейсов:

|Функция|Цель|
| --- | --- |
|`sum(tensor, strategy)`|Целотензорная сумма|
|`sum_fallback(tensor, strategy)`|Целотензорная сумма; переключает OneShot на Chained, когда требуемое атомарное добавление недоступно|
|`reduce(tensor, output_dtype, strategy, config)`|По очереди уменьшает каждую ось и возвращает форму [1]|
|`reduce_dim(tensor, output_dtype, dim, strategy, config)`|Уменьшает выбранную ось.|

Выберите `config` с помощью `ReduceOperationConfig`:

|Операция|`reduce_dim` вывод|
| --- | --- |
|`Sum`, `Prod`, `Mean`|Сумма, произведение, среднее значение; выбранная длина оси 1|
|`Min`, `Max`, `MaxAbs`|Минимальное, максимальное, максимальное абсолютное значение; выбранная длина оси 1|
|`ArgMin`, `ArgMax`|Индексы с отсчетом от нуля в пределах выбранной оси; длина оси 1|
|`TopK(k)`|Верхние значения k вдоль выбранной оси; длина оси k|
|`ArgTopK(k)`|Соответствующие индексы внутри оси; длина оси k|

Сокращение индекса требует явного целочисленного вывода dtype, например `Some(DType::I32)`. Для уменьшения стоимости введите `None`; вывод сохраняет ввод dtype. Не используйте `output_dtype` в качестве общего параметра приведения. F16/BF16 Sum, Prod и Mean используют накопление FP32 в этом пути перед записью входных данных dtype. Ось должна быть допустимой, а TopK k должен находиться в `1..=axis length`. Используйте `reduce_dim` для TopK, а не целотензорный `reduce`, который сбрасывает окончательную форму на `[1]`.

Выберите `SumStrategy` следующим образом:

- `OneShot(ruda_count)`: явно устанавливает положительное количество рабочих групп и требует атомарного добавления для входа dtype.
- `Chained(KernelReduceStrategy::Unspecified)`: использует поэтапные сокращения, не требуя глобального атомарного сложения всей тензорной суммы; используется в примере.
- `Autotune`: доступно с `tensor-reduce-autotune`. Без этой функции по умолчанию используется `OneShot(4)`; при этом значением по умолчанию является `Autotune`.

`KernelReduceStrategy` предлагает `Unspecified`, `Specific(ReduceStrategy)` и `Autotune` с ограниченными возможностями. Используйте «Специальный», чтобы исправить низкоуровневую стратегию. Без автонастройки значением по умолчанию является «Не указано».

Интерфейсы редукции возвращают `Result<RudaTensor<R>, ReduceError>`. Ось вне допустимого диапазона даёт `InvalidAxis`; отсутствие атомарного сложения для OneShot — `MissingAtomicAdd`. `sum_fallback` заменяет лишь неподдерживаемый случай атомарного сложения OneShot и не переключает выполнение на CPU.

### 4. Совокупное сканирование

Все четыре функции принимают `(tensor, dim)`:

|Функция|Результат для одной входной строки [3, 1, 2]|
| --- | --- |
|`cumsum`|[3, 4, 6]|
|`cumprod`|[3, 3, 6]|
|`cummin`|[3, 1, 1]|
|`cummax`|[3, 3, 3]|

Это включающие префиксные операции. Выходная форма и dtype соответствуют входным; пакетные строки обрабатываются независимо. dim начинается с нуля и должен находиться в пределах входного ранга. Сканирует возвращаемые тензоры напрямую, а не `Result`.

Текущий алгоритм считывает соответствующий префикс для каждой выходной позиции, давая общее количество чтений O(n²) для оси длины n. Учитывайте эти затраты на длинных осях сканирования, а не оценивайте работу только по количеству входных элементов.

### 5. Отбор, сбор, нарезка и целочисленные степени.

Добавьте эту функцию в тот же файл и вызовите `indexing_example(&device)?;` перед возвратом функции main. Он повторно использует более ранний импорт:

```rust
fn indexing_example(device: &CudaDevice) -> Result<(), Box<dyn std::error::Error>> {
    use ruda_core::tensor::element::Scalar;
    use ruprim::{
        elementwise::binary::integer_power,
        indexing::{gather, select, slice},
    };

    let input = from_data::<CudaRuntime>(
        TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
        device,
    );
    let columns = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0], [2]), device,
    );
    let selected = select(input.clone(), 1, columns);
    assert_eq!(into_data_sync(selected).to_vec::<f32>()?, [3., 1., 6., 4.]);

    let positions = from_data::<CudaRuntime>(
        TensorData::new(vec![2i32, 0, 1, 1], [2, 2]), device,
    );
    let gathered = gather(1, input.clone(), positions);
    assert_eq!(into_data_sync(gathered).to_vec::<f32>()?, [3., 1., 5., 5.]);

    let sliced = slice(input.clone(), &[0..2, 1..3]);
    assert_eq!(into_data_sync(sliced).to_vec::<f32>()?, [2., 3., 5., 6.]);

    let squared = integer_power::scalar(input, Scalar::Int(2));
    assert_eq!(into_data_sync(squared).to_vec::<f32>()?, [1., 4., 9., 16., 25., 36.]);
    Ok(())
}
```

|Функция|Индексы и вывод|
| --- | --- |
|`select(tensor, dim, indices)`|Одномерные целочисленные индексы; применяет одни и те же позиции к другим измерениям, заменяя длину выбранной оси количеством индексов|
|`gather(dim, tensor, indices)`|dim — первый аргумент; выбирает входное значение для каждой выходной позиции, при этом выходная форма равна форме индексов|
|`slice(tensor, ranges)`|Полуоткрытые диапазоны `start..end` на ось; в примере выбираются столбцы 1 и 2 из всех строк|
|`slice_assign(tensor, slices, value)`|Использует `ruda_core::tensor::Slice`; записывает значение в выбранную область и возвращает обновленный тензор|

Индексы и данные должны совместно использовать одно устройство; индексы должны быть допустимыми целыми числами, отсчитываемыми от нуля. Для сбора используйте тензоры индексов с одинаковым рангом и соответствующими невыбранным измерениям. В каждой строке примера можно выбирать разные столбцы. Диапазоны срезов должны лежать в пределах их осей, а форма значения назначения должна соответствовать выбранной области.

Целочисленные степени находятся в `ruprim::elementwise::binary::integer_power`:

- `scalar(input, Scalar::Int(exponent))`: для всех элементов используется один целочисленный показатель степени.
- `tensor(base, exponents)`: использует базы с плавающей запятой и тензоры целочисленной экспоненты с широковещательно-совместимыми формами; вывод сохраняет базу dtype.

Используйте `Scalar::Int(-2)` для отрицательного показателя степени, а не для преобразования показателя степени в формат с плавающей запятой. Индексирующие и поэлементные интерфейсы напрямую возвращают тензоры устройств. Чтобы сохранить входные данные для других операций, передайте клонированные дескрипторы, как в примере, и сохраните тензор, возвращаемый каждой операцией.

API Ссылка: [Сокращение](../../src/reduce/tensor/base.rs), [Сканирование](../../src/scan/tensor.rs), [Индексирование](../../src/indexing/mod.rs), [Целые степени](../../src/elementwise/binary/integer_power.rs).
