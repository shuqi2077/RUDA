# ruda-test-utils

[English](../../README.md) | **日本語** | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

Ruda のカーネル テスト用の共有ビルディング
ブロック: テスト テンソル
ビルダー、ホスト側の参照比較、単一の構成でテンソルを整形表示 (または差分) する統合レンダラー。

---

## 構成: `ruda-test.toml`

ワークスペース ルートにある
`ruda-test.toml` ファイルを使用して、テスト
ポリシーとテンソル プリントを構成します。最初のアクセス時に、ローダーは現在の作業ディレクトリからファイルを見つけて設定をキャッシュするまで進みます。

設定には 2 つのセクションがあります。

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

**パイプライン全体は 1 つのルールに従います。**
`enabled = false` の場合、何も出力されません。これを `true` に設定し、テストを実行して、テンソルのレンダリングを確認します。それでおしまい。

### `[test] policy`

|ポリシー|エラーなし|数値エラー|コンパイルエラー|
| ------------- | -------- | --------------- | ----------------- |
|`correct`|承諾|失敗|承諾|
|`strict`|承諾|失敗|失敗|
|`fail-if-run`|失敗|承諾|承諾|

印刷が有効で `force-fail = true` の場合、
`correct` および `strict`
ポリシーは、レンダリングがスキップされた場合でも、合格した結果とコンパイル エラーを拒否します。 `fail-if-run` ポリシーは、この設定によって変更されません。

---

## レンダリング: すべてに 1 つのパス

`assert_equals_approx(actual, expected, ε)`
と無料の
`print_tensors(label, &[&a, &b], Some(ε))` は両方とも同じレンダラーを通過します。
「差分パス」と「きれいに印刷されたパス」はありません。実際の値と期待値の比較と、無関係な
2 つの同じ形状のテンソルをきれいに出力することは、文字通り同じ呼び出しです。

ルール:

- 1 つのテンソル → 値のみ、色なし。
- **同じランクと形状**の 2 つのテンソル → 有限値は緑色で表示されます
  `Δ ≤ max(ε, ε × |expected|)`の場合は赤、それ以外の場合は赤です。 `show-expected = true` の場合、セルは次のようになります。
  `got/expected`;それ以外の場合は、単に `got` です。
- **異なるランクまたは形状**の 2 つのテンソル → 黙ってスキップされました。の
  レンダラは、不正な入力に対してパニックを起こすことはありません。
- 印刷フィルターランク≠テンソルランク → レンダリングはスキップされます。不一致
  比較フィルターは `ValidationResult::Error` を返します。

```rust
use ruda_test_utils::print_tensors;

// Single tensor — table or lines per [print] view, no color.
print_tensors("input", &[&host], None);

// Two tensors — colored diff. Same path used by assert_equals_approx.
print_tensors("a vs b", &[&a, &b], Some(1e-3));
```

テーブル ビューには
Δ/ε の数値は表示されません (セルの色が情報を伝えます)。ラインビューには常にそれらが表示されます。

### テーブルビューの例 (`show-expected = true` を使用)

```
=== diff  shape=[2, 3] ===
    |                 0                 1                 2
----+------------------------------------------------------
  0 | 0.000000/0.000000 1.000000/1.000000 2.000000/2.000000   ← green
  1 | 4.000000/3.000000 5.000000/4.000000 6.000000/5.000000   ← red
```

### テーブルビュー + `fail-only = true`

```
=== diff  shape=[2, 3] ===
    |        0        1        2
----+---------------------------
  0 |                            ← matching cells blanked out
  1 | 4.000000 5.000000 6.000000 ← red
```

### ラインビュー + `fail-only = true`

```
=== diff  shape=[2, 3] ===
 index |      got | expected |        Δ |        ε | status
-----------------------------------------------------------
[1, 0] | 4.000000 | 3.000000 | 1.000000 | 0.003000 | FAIL    ← red
[1, 1] | 5.000000 | 4.000000 | 1.000000 | 0.004000 | FAIL    ← red
[1, 2] | 6.000000 | 5.000000 | 1.000000 | 0.005000 | FAIL    ← red
```

---

## フィルター構文

`[print] filter` と `assert_equals_approx_in_slice`
の両方で使用されます。 dim エントリのカンマ区切りリスト:

- `.` — ワイルドカード (そのディムに沿った任意のインデックス)
- `N` — 単一のインデックス
- `M-K` — 包含範囲

4 次元テンソルの例: `.,.,10-20,30` は、dim
2 が `10..=20` にあり、dim 3
が正確に `30` であるすべての要素を選択します。フィルター ランクはテンソル ランクと等しくなければなりません。

Rust より:

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

`parse_tensor_filter("0,0-2")` は、文字列 DSL を `TensorFilter` に解析します。

---

## 失敗メッセージ

`assert_equals_approx` は `ValidationResult`
を返し、最大 **8** の不一致と集計統計を収集します。
`.as_test_outcome().enforce()` を呼び出すと、テスト ポリシーが適用され、拒否された結果でパニックが発生します。

```
Test failed: Got incorrect results: 17/4096 elements mismatched
  (max |Δ|=0.014648, mean |Δ|=0.004112, worst at [3, 12]) — shape=[16, 256]
First mismatches:
  [0, 5]: got 1.234, expected 1.220, |Δ|=0.014 > ε=0.001
  ...
  ... and 9 more
```

印刷が有効な場合、要素ごとの出力は標準出力に出力されます。パニック
メッセージには集約ヘッダーのみが保持されるため、ダンプが複製されません。

---

## テストの実行

RUDA ワークスペース ルートから実行し、テスト ランタイムを 1 つ選択します。

```sh
cargo test --locked -p ruda-test-utils --test lib --features ruda-test-runtime/cuda
```

ランタイム機能は、`cpu`、`cuda`、`hip`、または `wgpu` である可能性があります。

---

## テスト入力の構築

テスト テンソルを構築する 2 つの同等の方法:

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

ビルダー セッター (すべてオプション):

|セッター|デフォルト|エフェクト|
| --------------- | ---------------------- | --------------------------- |
|`.dtype(d)`|`f32`|入力 dtype をオーバーライドします。|
|`.stride(spec)`|`StrideSpec::RowMajor`|ストライド レイアウトをオーバーライドします。|

ビルダー ファイナライザー (それぞれ、生成可能な `TestInput` を返します):

|ファイナライザー|同等 `DataKind`|
| -------------------------- | ---------------------------------------------------------------- |
|`.arange()`|`Arange { scale: None }`|
|`.arange_scaled(s)`|`Arange { scale: Some(s) }`|
|`.eye()`|`Eye`|
|`.zeros()`|`Zeros`|
|`.uniform(seed, lo, hi)`|`Random { Uniform(lo, hi) }`|
|`.bernoulli(seed, p)`|`Random { Bernoulli(p) }`|
|`.normal(seed, mean, std)`|`Random { Normal { mean, std } }`|
|`.random(seed, dist)`|`Random { dist }`|
|`.linspace(start, end)`|`Custom { data }` (`start..=end` からの N 等間隔の値)|
|`.custom(data)`|`Custom { data }`|

ファイナライザーの後に、 `.generate()` 、 `.generate_with_f32_host_data()`
、 `.generate_with_bool_host_data()` 、 `.generate_test_tensor()`
、 `.f32_host_data()` 、 `.bool_host_data()` のいずれかを呼び出します。
