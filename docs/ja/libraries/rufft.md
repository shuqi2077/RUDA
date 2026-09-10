# ruFFT ユーザーガイド

[English](../../en/libraries/rufft.md) | [简体中文](../../zh/libraries/rufft.md) | **日本語** | [Deutsch](../../de/libraries/rufft.md) | [Русский](../../ru/libraries/rufft.md)

[計算ライブラリ](README.md) · [Tensor フレームワーク](../tensor-framework.md) · [中文](../../zh/libraries/rufft.md)

ruFFT は、デバイス上で実信号 FFT と逆変換を計算します。 `rufft::tensor` インターフェイスは出力を割り当てます。デバイス バインディングを直接管理する場合は、`rfft_launch` および `irfft_launch` を使用します。

## 1. 依存関係を構成する

Cargo パッケージは `ruFFT` です。 Rust インポート名は `rufft` です。機能 `tensor` により、デバイス テンソル インターフェイスが有効になります。この構成では、アプリケーション ディレクトリが `RUDA` ソース ディレクトリの横に配置されます。 NVIDIA のセットアップについては、[はじめに](../getting-started.md) を参照してください。

```toml
[dependencies]
rufft = { package = "ruFFT", path = "../RUDA/ruFFT", default-features = false, features = ["std", "tensor"] }
ruda-core = { path = "../RUDA/ruda-core", default-features = false, features = ["std", "tensor-host-data"] }
ruda-kernel = { path = "../RUDA/ruda-kernel", default-features = false, features = ["frontend-std", "device-tensor"] }
ruda-driver-cuda = { path = "../RUDA/ruda-driver-cuda", default-features = false, features = ["std"] }
```

## 2. 順変換と逆変換

この完全な `src/main.rs` は、4 ポイント F32 信号をアップロードし、そのスペクトルを計算し、入力を再構築します。アプリケーション ディレクトリから `cargo run` を実行します。

```rust
use ruda_core::tensor::data::TensorData;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use rufft::tensor::{irfft, rfft};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::default();
    let input = vec![1.0f32, 0.0, -1.0, 0.0];
    let signal = from_data::<CudaRuntime>(TensorData::new(input.clone(), [4]), &device);
    let (real, imag) = rfft(signal, 0, None);

    let re = into_data_sync(real.clone()).to_vec::<f32>()?;
    let im = into_data_sync(imag.clone()).to_vec::<f32>()?;
    for (actual, expected) in re.iter().zip([0.0f32, 2.0, 0.0]) {
        assert!((actual - expected).abs() < 1e-5);
    }
    assert!(im.iter().all(|value| value.abs() < 1e-5));

    let restored = irfft(real, imag, 0, Some(input.len()));
    let restored = into_data_sync(restored).to_vec::<f32>()?;
    for (actual, expected) in restored.iter().zip(input) {
        assert!((actual - expected).abs() < 1e-5);
    }
    println!("real={re:?}, imag={im:?}, restored={restored:?}");
    Ok(())
}
```

4 点信号は、DC、1 つの正周波数ビン、およびナイキストの 3 つのビンを生成します。実数成分と虚数成分は別々のテンソルであり、インターリーブされた複素数値ではありません。順変換は正規化されていません。逆数には実際の FFT の長さの正規化が含まれるため、再度長さで除算しないでください。

この例では、`clone()` を使用して、逆変換のテンソル ハンドルを保持します。 `into_data_sync` はリードバックを待機し、ホスト データを返します。リードバック失敗時にパニックが発生します。

## 3. パラメータと出力形状

|関数|パラメータ|戻り値|
| --- | --- | --- |
|`rfft(signal, dim, n)`|信号、ゼロベースの軸、オプションの要求された長さ|2 つのデバイス テンソル `(real, imag)`|
|`irfft(real, imag, dim, n)`|実数部、虚数部、軸、オプションの出力長|実数値のデバイス テンソル|

デバイスの計算には F32 が使用されます。 F32 信号とスペクトルを提供します。 `dim` は入力ランクより小さくなければなりません。逆変換コンポーネントは、形状、dtype、およびデバイスが一致している必要があります。これらのインターフェイスは、`Result` ではなくテンソルを返します。引数のアサーションが失敗するか、パニックが発生します。

要求された長さ n の場合、実際の FFT 長さ N は、n 以上の 2 の最小累乗です。

- `n = None` では、`rfft` は dim に沿った入力長を使用します。それ以外の場合は、最初に切り捨てられるか n にゼロが埋め込まれ、次に N にゼロが埋め込まれます。
- ディムに沿った順方向出力の長さは `N / 2 + 1` です。他の寸法は変更されません。
- `n = None` では、`irfft` は `2 × (bin count - 1)` を使用します。明示的な n は、返される長さを指定します。
- 逆関数は、最初に両方のスペクトル コンポーネントを `N / 2 + 1` ビンに切り捨てるかゼロ埋めし、N ポイントの逆関数を計算してから、n にクロップします。
- 要求された長さは少なくとも 2 である必要があります。 1 点変換はデバイス カーネルの長さチェックに失敗します。

|入力/呼び出し|実際の変換長|寸法に沿った出力長さ|
| --- | --- | --- |
|長さ 8、`rfft(..., None)`|8|5|
|長さ8、`rfft(..., Some(6))`|8;最初の 6 つの入力ポイントとゼロパッドのみを保持します|5|
|長さ 6、`rfft(..., None)`|8、ゼロ埋め込み|5|
| 5 ビン、`irfft(..., None)` |8|8|
| 5 ビン、`irfft(..., Some(6))` |8|6|

したがって、`Some(6)` は 6 ポイントの DFT を要求しません。元の 6 点信号の往復の場合、`Some(6)` を逆信号に渡して、パディングされた 2 つの末尾点を削除します。

## 4. バッチと多次元入力

dim 以外の軸はバッチ次元です。たとえば、`[batch, channels, 1024]` を dim=2 に沿って変換すると、形状 `[batch, channels, 513]` の実数テンソルと虚数テンソルが生成されます。

この関数は、前述の `rfft` インポートを再利用し、デバイス テンソルを読み戻さずに受け入れます。

```rust
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

fn batched_rfft<R: Runtime>(
    signals: RudaTensor<R>,
    dim: usize,
    length: usize,
) -> (RudaTensor<R>, RudaTensor<R>) {
    rfft(signals, dim, Some(length))
}
```

単一の `rfft` 呼び出しは、多次元 FFT 全体ではなく、選択した軸のみを変換します。分離された実数成分と虚数成分をそれぞれ独立して `rfft` に渡して完全な複素信号として扱わないでください。

## 5. バッファと実行

Tensor インターフェイスは出力を割り当て、必要に応じてパディングとクロッピングを実行します。実際の長さが 4096 を超える場合は、コールを変更せずにステージングされたパスが自動的に使用されます。出力バッファを自分で管理するには、起動インターフェイスはクライアントを受け取り、`TensorBinding` 値、dim、および `StorageType` を入力/出力し、`Result<(), LaunchError>` を返します。実際の N に一致する出力レイアウトを割り当てます。

バッファーを管理するときにゼロ埋め入力が実体化されるのを避けるには、次のエントリー ポイントを使用します。

|関数|追加の長さパラメータ|
| --- | --- |
|`rfft_launch_padded(client, signal, real, imag, dim, signal_len, dtype)`|最初の signal_len 信号要素のみを読み取り、残りをゼロとして扱います。 N は出力スペクトル長から推測されます|
|`irfft_launch_padded(client, real, imag, signal, dim, spec_bins, dtype)`|最初の spec_bins 周波数ビンのみを読み取り、残りをゼロとして扱います。 N は出力信号の長さに由来します|

N は少なくとも 2 の 2 のべき乗でなければならず、実数と虚数の形状は一致する必要があります。 `signal_len` は入力軸の長さまたは N を超えることはできません。`spec_bins` は少なくとも 1 である必要があり、入力スペクトル軸の長さまたは `N / 2 + 1` を超えることはできません。出力バッファには依然として完全な割り当てが必要です。これらのインターフェイスは、ゼロ埋めされた入力末尾の実現を回避します。

API 参照: [Tensor インターフェイス](../../../ruFFT/src/tensor.rs)、[前方起動](../../../ruFFT/src/fft/rfft.rs)、[逆起動](../../../ruFFT/src/fft/irfft.rs)。
