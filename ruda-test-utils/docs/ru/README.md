# руда-тест-утилиты

[English](../../README.md) | [日本語](../ja/README.md) | [Deutsch](../de/README.md) | **Русский**

Общие строительные блоки для тестов ядра в Ruda: построители тестовых
тензоров, сравнение эталонов на стороне хоста и унифицированный модуль рендеринга,
который красиво печатает тензоры (или сравнивает их) в рамках одной конфигурации.

---

## Конфигурация: `ruda-test.toml`

Настройте политику тестирования и тензорную печать с помощью файла `ruda-test.toml` в
корне рабочей области. При первом доступе загрузчик переходит из текущего рабочего
каталога до тех пор, пока не найдет файл, а затем кэширует конфигурацию.

Конфигурация состоит из двух разделов:

```toml
[test]
policy = "correct"   # "correct" | "strict" | "fail-if-run"

[print]
enabled = false       # toggle all printing
view = "table"        # "table" | "lines"
force-fail = true     # reject passing outcomes in correct/strict print mode
fail-only = false     # diff: only render cells where Δ > ε
show-expected = false # diff: render `got/expected` per cell (else just `got`)
filter = ""           # per-axis filter, same DSL as the slice helper
```

**Весь конвейер подчиняется одному правилу:** если `enabled = false` , ничего не печатается. Установите для него
значение `true` , запустите тест и посмотрите, как рендерятся ваши тензоры. Вот и все.

### `[test] policy`

|Политика|Нет ошибок|Числовая ошибка|Ошибка компиляции|
| ------------- | -------- | --------------- | ----------------- |
|`correct`|принять|не удалось|принять|
|`strict`|принять|не удалось|не удалось|
|`fail-if-run`|не удалось|принять|принять|

При включенной печати и `force-fail = true` политики `correct` и `strict`
отклоняют результаты передачи и ошибки компиляции, даже если рендеринг
был пропущен. Политика `fail-if-run` не изменяется при этом параметре.

---

## Рендеринг: один путь для всего

И `assert_equals_approx(actual, expected, ε)` , и бесплатный `print_tensors(label, &[&a, &b], Some(ε))` проходят через
один и тот же рендерер. Не существует «пути
различий» и «путь с красивой печатью»; сравнение фактического и
ожидаемого и красивая печать двух несвязанных тензоров одинаковой
формы — это буквально один и тот же вызов.

Правила:

- Один тензор → только значения, без цвета.
- Два тензора **одного ранга и формы** → конечные значения окрашены в зеленый цвет.
  , если `Δ ≤ max(ε, ε × |expected|)`, и красный в противном случае. С `show-expected = true` ячейка показывает
  `got/expected`; иначе просто `got`.
- Два тензора **разного ранга или формы** → молча пропускаются.
  Средство визуализации никогда не паникует из-за неправильного ввода.
- Ранг фильтра печати ≠ ранг тензора → рендеринг пропускается. Несоответствующий
  возвращает `ValidationResult::Error`.

```rust
use ruda_test_utils::print_tensors;

// Single tensor — table or lines per [print] view, no color.
print_tensors("input", &[&host], None);

// Two tensors — colored diff. Same path used by assert_equals_approx.
print_tensors("a vs b", &[&a, &b], Some(1e-3));
```

В представлении таблицы никогда не отображаются числа Δ/ε (информацию
несет цвет ячейки). В представлении линий они всегда отображаются.

### Пример представления таблицы (с `show-expected = true`)

```
=== diff  shape=[2, 3] ===
    |                 0                 1                 2
----+------------------------------------------------------
  0 | 0.000000/0.000000 1.000000/1.000000 2.000000/2.000000   ← green
  1 | 4.000000/3.000000 5.000000/4.000000 6.000000/5.000000   ← red
```

### Вид таблицы + `fail-only = true`

```
=== diff  shape=[2, 3] ===
    |        0        1        2
----+---------------------------
  0 |                            ← matching cells blanked out
  1 | 4.000000 5.000000 6.000000 ← red
```

### Просмотр линий + `fail-only = true`

```
=== diff  shape=[2, 3] ===
 index |      got | expected |        Δ |        ε | status
-----------------------------------------------------------
[1, 0] | 4.000000 | 3.000000 | 1.000000 | 0.003000 | FAIL    ← red
[1, 1] | 5.000000 | 4.000000 | 1.000000 | 0.004000 | FAIL    ← red
[1, 2] | 6.000000 | 5.000000 | 1.000000 | 0.005000 | FAIL    ← red
```

---

## Синтаксис фильтра

Используется как `[print] filter` , так и
`assert_equals_approx_in_slice` . Список тусклых записей, разделенных запятыми:

- `.` — подстановочный знак (любой индекс в этом диапазоне)
- `N` — единый индекс
- `M-K` — диапазон включительно

Пример для 4-D тензора: `.,.,10-20,30` выбирает все элементы, где dim
2 находится в `10..=20` , а dim 3 — это
точно `30` . Ранг фильтра должен быть равен рангу тензора.

Из Rust:

```rust
use ruda_test_utils::{DimFilter, assert_equals_approx_in_slice};

// Vec<Range<usize>> works (half-open, like Rust slices).
assert_equals_approx_in_slice(&actual, &expected, 0.001, vec![0..1, 0..3]);

// Or build the canonical TensorFilter explicitly.
let filter = vec![
    DimFilter::Exact(0),
    DimFilter::Range { start: 0, end: 2 }, // inclusive: 0..=2
];
assert_equals_approx_in_slice(&actual, &expected, 0.001, filter);
```

`parse_tensor_filter("0,0-2")` анализирует строку DSL в строку `TensorFilter`.

---

## Сообщения об ошибках

`assert_equals_approx` возвращает `ValidationResult` , собирая до **8**
несоответствий плюс совокупную статистику. Вызов `.as_test_outcome().enforce()` применяет
политику тестирования и вызывает панику при отклонении результата:

```
Test failed: Got incorrect results: 17/4096 elements mismatched
  (max |Δ|=0.014648, mean |Δ|=0.004112, worst at [3, 12]) — shape=[16, 256]
First mismatches:
  [0, 5]: got 1.234, expected 1.220, |Δ|=0.014 > ε=0.001
  ...
  ... and 9 more
```

Когда печать включена, вывод каждого элемента поступает на стандартный вывод;
тревожное сообщение сохраняет только совокупный заголовок, поэтому оно не дублирует дамп.

---

## Выполнение тестов

Запустите из корня рабочей области RUDA, выбрав одну среду выполнения теста:

```sh
cargo test --locked -p ruda-test-utils --test lib --features ruda-test-runtime/cuda
```

Функция времени выполнения может быть `cpu`, `cuda`, `hip` или `wgpu`.

---

## Создание тестовых входов

Два эквивалентных способа построения тестового тензора:

```rust
use ruda_kernel::dsl::prelude::RudaPrimitive;
use ruda_test_utils::{TestInput, StrideSpec, DataKind, Distribution};

// Long-form constructor.
let (handle, host) = TestInput::new(
    client.clone(),
    [4, 4],
    f32::as_type_native_unchecked().storage_type(),
    StrideSpec::RowMajor,
    DataKind::Random {
        seed: 0,
        distribution: Distribution::Uniform(-1.0, 1.0),
    },
)
.generate_with_f32_host_data();

// Fluent builder — `dtype` defaults to f32, `stride` defaults to RowMajor.
let (handle, host) = TestInput::builder(client.clone(), [4, 4])
    .uniform( 0, -1.0, 1.0)
    .generate_with_f32_host_data();
```

Настройки Builder (все необязательно):

|Установщик|По умолчанию|Эффект|
| --------------- | ---------------------- | --------------------------- |
|`.dtype(d)`|`f32`|Переопределить ввод dtype.|
|`.stride(spec)`|`StrideSpec::RowMajor`|Переопределить схему шага.|

Финализаторы Builder (каждый возвращает `TestInput`, готовый к генерации):

|Финализатор|Эквивалент `DataKind`|
| -------------------------- | ---------------------------------------------------------------- |
|`.arange()`|`Arange { scale: None }`|
|`.arange_scaled(s)`|`Arange { scale: Some(s) }`|
|`.eye()`|`Eye`|
|`.zeros()`|`Zeros`|
|`.uniform(seed, lo, hi)`|`Random { Uniform(lo, hi) }`|
|`.bernoulli(seed, p)`|`Random { Bernoulli(p) }`|
|`.normal(seed, mean, std)`|`Random { Normal { mean, std } }`|
|`.random(seed, dist)`|`Random { dist }`|
|`.linspace(start, end)`|`Custom { data }` с N равномерно распределенными значениями из `start..=end`|
|`.custom(data)`|`Custom { data }`|

После финализатора вызовите любой из:
`.generate()` , `.generate_with_f32_host_data()` , `.generate_with_bool_host_data()` ,
`.generate_test_tensor()` , `.f32_host_data()` , `.bool_host_data()` .
