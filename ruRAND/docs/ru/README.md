# ruRAND

[English](../../README.md) | [简体中文](../zh/README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | **Русский**

**Английский** | [简体中文](../zh/README.md)

Генерация случайных чисел для Ruda.

- Cargo пакет: `ruRAND`
- Крейт Rust: `rurand`

## Feature

| Feature |Операции|
| --- | --- |
|`tensor`|Равномерное, нормальное распределение и распределение Бернулли на тензорах устройств|

Используйте `rurand::tensor::{random_uniform, random_normal, random_bernoulli}` для выделения тензоров или функций в корне крейта для заполнения существующей памяти устройства. `rurand::seed` устанавливает общее случайное начальное число.

## Краткое руководство

Сборка из рабочей области RUDA:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --release --locked -p ruRAND --no-default-features --features std,tensor
```

## Документация

- [Руководство пользователя](../../../docs/ru/libraries/rurand.md)
- [Настройка среды](../../../docs/ru/getting-started.md)
- [Функции Cargo](../../Cargo.toml) · [Экспорт модулей](../../src/lib.rs)

## ruRAND Руководство пользователя

[Вычислительные библиотеки](../../../docs/ru/libraries/README.md) · [Среда выполнения API](../../../docs/ru/runtime-api.md) · [中文](../zh/README.md)

ruRAND генерирует тензоры устройств с равномерным, нормальным распределением и распределением Бернулли. Используйте `rurand::tensor` для выделения новых тензоров или одноименных функций в корне крейта для заполнения существующей памяти устройства.

### 1. Настройте зависимости

Пакет Cargo — `ruRAND`; его имя для импорта в Rust — `rurand`. Функция `tensor` включает тензорные интерфейсы устройств. В этой конфигурации каталог приложения размещается рядом с исходным каталогом `RUDA`. См. раздел [Начало работы](../../../docs/ru/getting-started.md) для настройки NVIDIA.

```toml
[dependencies]
rurand = { package = "ruRAND", path = "../RUDA/ruRAND", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

### 2. Генерация случайных тензоров

Этот полный `src/main.rs` генерирует тензоры F32 формы `[2, 3]` и воспроизводит вызов, сбрасывая начальное число. Запустите `cargo run` из каталога приложения:

```rust
use ruda_core::tensor::DType;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::readback::into_data_sync;
use rurand::tensor::{
    random_bernoulli, random_like_uniform, random_normal, random_uniform,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    rurand::seed(42);
    let uniform = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let first = into_data_sync(uniform.clone()).to_vec::<f32>()?;

    rurand::seed(42);
    let repeated = random_uniform::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let repeated = into_data_sync(repeated).to_vec::<f32>()?;
    assert_eq!(first, repeated);
    assert!(first.iter().all(|&x| (0.0..1.0).contains(&x)));

    let normal = random_normal::<CudaRuntime>([2, 3].into(), &device, 0.0, 1.0, DType::F32);
    let mask = random_bernoulli::<CudaRuntime>([2, 3].into(), &device, 0.25, DType::F32);
    let noise = random_like_uniform(&uniform, -0.1, 0.1, DType::F32);

    let mask = into_data_sync(mask).to_vec::<f32>()?;
    assert!(mask.iter().all(|&x| x == 0.0 || x == 1.0));
    println!("uniform={first:?}");
    println!("normal={:?}", into_data_sync(normal).to_vec::<f32>()?);
    println!("mask={mask:?}");
    println!("noise={:?}", into_data_sync(noise).to_vec::<f32>()?);
    Ok(())
}
```

`random_like_uniform` берет из эталона только форму и устройство. Он выделяет новый тензор без изменения и копирования опорных значений. Его последний аргумент явно выбирает выход dtype.

`random_bernoulli` возвращает нули и единицы в выбранном dtype, а не автоматически созданную логическую маску. В этом примере возвращаются значения F32 0.0 и 1.0.

### 3. Параметры и распределения

Первые три тензорные функции имеют общие параметры `shape: Shape`, `device: &R::Device` и последний параметр `dtype: DType`. Возвращенный `RudaTensor<R>` имеет указанную форму dtype и устройство.

|Порядок функций и параметров|Параметры распределения|
| --- | --- |
|`random_uniform(shape, device, lower_bound, upper_bound, dtype)`|f32 нижняя и верхняя границы|
|`random_normal(shape, device, mean, std, dtype)`|f32 среднее и стандартное отклонение; стандартное значение не является отклонением|
|`random_bernoulli(shape, device, probability, dtype)`|f32 вероятность возврата 1|
|`random_like_uniform(&reference, lower_bound, upper_bound, dtype)`|Базовый тензор, границы f32, выход dtype|

Предоставьте конечные параметры, соответствующие распределению: нижняя граница ниже верхней границы, неотрицательное нормальное стандартное отклонение и вероятность Бернулли в `[0, 1]`. Эти интерфейсы не проверяют эти диапазоны через `Result`; не используйте недопустимые параметры для запроса возврата ошибки.

При генерации однородного результата сначала создается F32 `u ∈ [0, 1)`, вычисляется `lower_bound + (upper_bound - lower_bound) × u`, а затем выполняется приведение к выходным данным dtype. Случай F32 `[0, 1)` исключает 1. Масштабирование до других интервалов или преобразование в более низкую точность может привести к округлению значения вблизи верхней границы этой границы.

Обычное генерирование использует преобразование Бокса-Мюллера FP32 перед приведением к выходу dtype. Выбор хранилища F64 не увеличивает внутреннюю точность случайных чисел с плавающей запятой. Бернулли сравнивает `u < probability`, поэтому вероятность 0 дает все нули, а вероятность 1 дает все единицы.

### 4. Начальные значения и порядок вызова

`rurand::seed(seed: u64)` сбрасывает случайное состояние общего хоста. Каждый вызов поколения извлекает новые семена из этого состояния и продвигает его:

- Задайте начальное число один раз, чтобы инициализировать последовательность, а затем сгенерируйте последующие пакеты.
- Чтобы воспроизвести последовательность, сбросьте то же начальное число и сохраните порядок вызова, форму, dtype и конфигурацию устройства.
- Не сбрасывайте одно и то же начальное значение перед каждым обучающим пакетом, если не предполагается повторение случайных входных данных.
- Параллельные потоки разделяют это состояние; чередование влияет на то, какие начальные числа получает каждый вызов. Это не отдельный объект-генератор для каждого потока или устройства.

Приведенное выше сравнение повторов использует один процесс, одно устройство и соответствующие аргументы. Сопоставление начальных чисел само по себе не подразумевает поэлементное равенство с другой случайной библиотекой или серверной частью.

### 5. Заполните существующий буфер.

Используйте функции запуска корневого каталога для повторного использования выходного распределения. Эта функция берет уже выделенный тензор устройства F32, записывает случайные значения в его хранилище и возвращает его:

```rust
use ruda_kernel::{
    dsl::{Runtime, prelude::LaunchError},
    tensor::RudaTensor,
};

fn fill_uniform<R: Runtime>(
    output: RudaTensor<R>,
    lower: f32,
    upper: f32,
) -> Result<RudaTensor<R>, LaunchError> {
    rurand::random_uniform(
        &output.client,
        lower,
        upper,
        output.clone().binding(),
        output.dtype.into(),
    )?;
    Ok(output)
}
```

Эта функция не вызывает тензорный распределитель. Другие дескрипторы, совместно использующие это хранилище, также наблюдают за записью. Соответствующими функциями для нормальных значений и значений Бернулли являются `rurand::random_normal(&client, mean, std, binding, dtype)` и `rurand::random_bernoulli(&client, probability, binding, dtype)`.

Функции тензорного уровня вызывают панику, если при запуске возвращается ошибка; базовая точка входа выше возвращает `Result<(), LaunchError>`. Успешная отправка не означает, что выполнение GPU завершено. Считайте тензор обратно или дождитесь синхронизации клиента. `into_data_sync` паникует при сбое чтения; используйте асинхронный `into_data(tensor).await` для распространения ошибок обратного чтения.

API Ссылка: [Тензорные интерфейсы](../../src/tensor/mod.rs), [Исходное состояние](../../src/state.rs), [Распределения](../../src/distributions/mod.rs).
