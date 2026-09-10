# 実験的なデバイス上の融合 AdamW / AMSGrad

[English](../en/fused-adamw.md) | [简体中文](../zh/fused-adamw.md) | **日本語** | [Deutsch](../de/fused-adamw.md) | [Русский](../ru/fused-adamw.md)

これは **RUDA Rust/GPU の実行検証を待つオプトイン実装**です。
`ruda-optim` を拡張します。別のオプティマイザー ライブラリを作成したり、既存の `AdamW`
、モデル オプティマイザー アダプター、autograd グラフ、チェックポイント形式を変更したりすることはありません。

## この演算子を使用する理由

既存の汎用オプティマイザーは、テンソル演算から更新を構築します。この追加により、勾配アンスケール、モーメント、オプションの AMSGrad
最大値、bias 補正、分離された重み減衰、パラメーター更新が
1 つのデバイス
カーネルに明示的に結合されます。既存のフュージョン バックエンドが必ずしも多くのカーネルを使用することや、この実装が
PyTorch のフューズド オプティマイザーよりも高速であるとは主張しません。

各有限入力要素と 1 ベースの更新 `t`:

```text
g = stored_gradient / gradient_scale      # negate if maximize
m = beta1 * m_old + (1 - beta1) * g
v = beta2 * v_old + (1 - beta2) * g * g
v_used = max(v_max_old, v)                 # AMSGrad only; save uncorrected max
p_new = p * (1 - lr * weight_decay)
        - lr * (m / (1 - beta1^t)) / (sqrt(v_used / (1 - beta2^t)) + epsilon)
```

AMSGrad を使わない場合は
`v_used = v` です。epsilon
は平方根の外にあります。重み減衰を勾配やモーメントに加算することはありません。ホストは呼び出しごとに一度、FP64 の整数べき乗と
FP32 への係数キャストを用いてバイアス補正係数を計算します。デバイス演算は
FP32 で、コンパイラが演算を融合・再結合する場合があります。数値比較には許容誤差を使い、他のオプティマイザとのビット一致は保証しません。

計算式の参照: [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
。デフォルトは、ベータ、イプシロン、減衰に関して RUDA
の既存の `AdamWConfig`
と意図的に一致しています: beta=(0.9,0.999)、epsilon=1e-5、weight_decay=1e-4。別のフレームワークと比較する場合は、すべてのオプションを明示的に設定します。

## サポートされている契約

|アイテム|この実装|
|---|---|
|パラメータとモーメント|FP32 マスター/ステート|
|保存された勾配|FP32、FP16、または BF16;アップデートでFP32に昇格|
|形状|完全一致、高密度連続、ブロードキャストなし、保守的 <= u32 バイト範囲|
|モード|AdamW、AMSGrad、最大化、スカラー学習率、正の損失スケール|
|空のテンソル|起動、割り当て、またはステップ前進はありません|
|呼び出し元で検出されたオーバーフロー|`skip_update=true`: 起動、割り当て、またはステップ アドバンスはありません|
|入力/エイリアス|読み取り専用。出力は新しいバッファです|
|初期状態|最初の更新で計算されます。ゼロフィル起動は必要ありません|
|キュー|すべての入力に対して同じデバイスと送信キュー。不一致はエラーです|
|完了|非同期、既存のランタイムによって管理される|
|半精度モデルの重み|自動的に更新/キャストされません。呼び出し元はマスター出力を明示的にキャストします|

これは、FP8/FP4 オプティマイザー状態、GradScaler、自動有限勾配チェック、慎重な重み減衰、微分可能オプティマイザー、一般的なストライド
カーネル、可変ハイパーパラメーター マルチテンソル
オプティマイザー、自動 FSDP
統合ではありません**。 CUDA デバイス側のステップ
カウンターを備えたグラフ再生オプティマイザー。キャプチャされたホスト bias
係数を更新せずに再生することはサポートされていません。事前にフラット化されたバケットは、すべての要素がオプションとステップ数を共有する場合にのみ機能します。スパース グラデーション、任意のホスト
ポインター、サイレント CPU フォールバックは追加されません。

FP32 マスター状態は、エンドツーエンドの混合精度トレーニングが検証されたことを意味するものではありません。呼び出し元は、モデルのコピー、スケーリングの決定、および蓄積または分散された勾配を管理します。



## feature

- `fused-adamw`: オプションと明示的に呼び出される CPU 参照。
- `fused-adamw-device`: 汎用 RudaTensor デバイス ランチャーおよびカーネル、特定なし
  ハードウェア ドライバーはこの機能によって有効になります。
- `fused-adamw-cuda`: CUDA ランタイム、直接 PTX 機能および明示的な CUDA テスト/例。
  依然として `RUDA_CUDA_COMPILER` を使用して NVRTC または PTX を選択します。

デフォルトでは有効になっている機能はありません。新しい依存関係バージョンは導入されません。 `Cargo.lock` は、オプションの
ruda-optim 依存関係として既存の CUDA ドライバーのみを取得します。

## 低レベルの使用

```rust
use ruda_optim::fused_adamw::{AdamWOptions, StepControl, adamw_step};

// master: dense FP32 RudaTensor<R>, gradient: same-shape F32/F16/BF16 tensor.
// state: Option<AdamWState<R>>, initially None.
let options = AdamWOptions {
    learning_rate: 1e-3,
    weight_decay: 0.01,
    amsgrad: true,
    ..Default::default()
};
let result = adamw_step(
    &master, &gradient, state.as_ref(), &options,
    StepControl { gradient_scale: 128.0, skip_update: found_inf },
)?;
master = result.parameters;
state = result.state;
```

`found_inf` は呼び出し側が渡す値であり、ここでは計算しません。既存の
`AdamW::step` は従来の実装を使い続けます。この低レベルプリミティブは、モジュールの
`Parameter` を自動で置き換えたり、微分可能な更新グラフを構築したりしません。

入力検証はすべての入力を変更せずに行います。正常に返ったことはカーネルを投入したことを示すだけで、実行完了の証明ではありません。チェックポイントを保存する前にランタイムの同期結果を確認してください。デバイスに障害が起きた場合は未確定の出力を破棄し、外部で確定済みのチェックポイントを復元します。
`updated`
が
true
という理由だけで学習データのカーソルを進めないでください。

`AdamWState::into_parts/from_parts`
は、明示的なチェックポイント統合のためにステップ バッファーとモーメント
バッファーを公開します。それら自体はシリアル化または転送しません。マスターパラメータ、すべてのモーメント、ハイパーパラメータを保持し、一緒にステップを実行します。このモジュールは、高レベルの
`TrainingRecord` 形式を改良しません。

## 割り当てとパフォーマンスのモデル

この最初の実装では、エイリアスを保持し、新しい安全でない所有権ルールの追加を避けるために、out-of-place 更新を選択します。各アクティブ
ステップは 3
つの FP32 出力
(AMSGrad では
4 つ)
を割り当てます。中間デルタ テンソルはありませんが、**ゼロ割り当てステップの主張はありません**。古いバッファと新しいバッファは、存続期間中に重複する可能性があります。したがって、大規模なモデルにはメモリ バジェットが必要です。インプレースまたはアリーナ/バケットの再利用は将来の作業であり、ここでサイレントに有効にするわけではありません。

ベンチマークのみの段階的ベースラインは、4 つのカーネル、または AMSGrad の場合は 5
つのカーネルを明示的に送信します。融合されたパスは 1 を送信します。*定常状態* FP32 勾配ステップの場合:

|モデル|要素ごとの論理バイト|明示的なアップデートの開始|
|---|---:|---:|
|ステージ済み AdamW|48|4|
|融合 AdamW|28|1|
|ステージングされた AMSGrad|60|5|
|融合 AMSGrad|36|1|

28 バイトのカウントは、4 つの
FP32 読み取り (パラメーター、勾配、2 つのモーメント)
と 3 つの
FP32 書き込みです。 48 バイトのベースラインには、反復された勾配読み取りとデルタ
バッファーが含まれます。これはソースレベルのアカウンティングであり、**DRAM トラフィックや速度向上は測定されません**。キャッシュ、割り当て、算術および起動のオーバーヘッドにより、観察された結果が変化する可能性があります。すでに汎用の AdamW を融合しているバックエンドにはメリットがない可能性があります。

## 検証およびベンチマーク コマンド

```sh
# Python/NumPy/PyTorch formula oracle ONLY, does not execute RUDA.
python tools/run_adamw_regressions.py --suite oracle

# Standalone Rust config/reference tests, without Cargo registry resolution.
python tools/run_adamw_regressions.py --suite reference

# Cargo unit tests, including a comparison with RUDA's existing Host AdamW.
python tools/run_adamw_regressions.py --suite host

# Type-check the opt-in generic device implementation.
python tools/run_adamw_regressions.py --suite build

# Explicit hardware regression, separately under both compilers.
python tools/run_adamw_regressions.py --suite cuda --compiler both

# Small controlled A/B run, then increase elements after checking resources.
python tools/run_adamw_regressions.py --suite bench --compiler both \
    --elements 65536 --dtype bf16 --iterations 20 --samples 7 --amsgrad
```

実際のドライバー/GPU でサポートされている `RUDA_PTX_VERSION` を使用してください。
`--offline` には、キャッシュされた Cargo 依存関係が必要です。
`--timeout` は各コマンドを制限します。予測されたランタイムではありません。ランナーはコマンド ライン、ソース ハッシュ、ログ、ステータスを保存します。欠落している
Rust/Cargo は `blocked` です。利用できない
GPU は、明示的な CUDA スイートに失敗します。サイレントにバックエンドを切り替えたり、シミュレーターでの測定結果を要求したりするテストはありません。

この例では、時間指定されたセクションからの入力の作成/リードバックを除外し、両方のバリアントをウォームし、両方を同一の実体化状態から開始し、実行順序を交互に変更し、各測定バッチの前後で同期します。割り当てとホストの送信を含む、中央値/最小/最大
**ステップごとのウォールミリ秒** を報告します。これは、CUDA
イベント
GPU 専用のタイマーではありません。測定されたすべてのバッチでパラメータとモーメントが比較されます。
GPU、ドライバー、クロック/電源設定、ハードウェア負荷を
JSON と一緒に保存します。

コミットされた `pytorch_fixtures.json` は、実際の **CPU** PyTorch AdamW
で生成されます。 12 のケースは、3 つの勾配 dtype x
2 AMSGrad モード x 2 つの最大化モード、それぞれ 5
ステップをカバーします。 `oracle.py --write-fixtures` は明示的に再生成します。 CUDA テストでは、実際の RUDA
カーネル出力とこのデータ、および別の FP64 リファレンスを比較します。 Python フィクスチャ ジェネレーターに合格しても、CUDA テストには合格しません。

## 残りゲート

Rust
タイプ/マクロの拡張、CUDA
の実行、および測定されたパフォーマンスは引き続き必須です。新しいテストは、リリース前に以前の安全性回帰スイートと並行して合格する必要があります。さらなる最適化は、このトラフィック
モデルではなく、測定されたプロファイラーの結果から開始する必要があります。比較が完了するまで、新しいデフォルトのディスパッチャーを有効にしないでください。


オプトイン拡張については、[実験的なグラデーション チェックとグループ クリッピング](gradient-guard.md) を参照してください。
