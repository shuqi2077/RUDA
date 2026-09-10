# ミュオンおよび明示的ミュオン + AdamW グループ

[English](../en/muon.md) | [简体中文](../zh/muon.md) | **日本語** | [Deutsch](../de/muon.md) | [Русский](../ru/muon.md)

既存の `MuonConfig` 、 `Muon`
、 `MuonState` の実装は、複製ではなく拡張されています。詳細な範囲については、[中文完整契約约](../zh/muon.md) を参照してください。

## の使用法

```rust,ignore
use ruda_optim::{AdamWConfig, MuonAdamWConfig, MuonConfig, MuonMatrixLayout, MuonMomentumMode};
let mut optimizer = MuonAdamWConfig::new()
    .with_muon(MuonConfig::new()
        .with_momentum_mode(MuonMomentumMode::Ema)
        .with_stable_normalization(true)
        .with_matrix_layout(MuonMatrixLayout::InputOutput))
    .with_adamw(AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01))
    .init(&model, &[model.hidden.weight.id])?;
model = optimizer.try_step_with_lrs(0.02, 0.0003, model, gradients)?;
```

非表示の空でない **完全な
2D 行列**を明示的に選択します。埋め込み、分類子ヘッド、バイアス、および正規化パラメーターは、2D であっても通常、AdamW
を使用する必要があります。選択されていないすべてのパラメーターは、オプションの融合 AdamW
インターフェイスではなく、既存の高レベル AdamW 実装を使用します。学習率の例は、チューニングに関する推奨事項ではありません。

`Optimizer::step(lr, ...)` は Muon
に `lr` 、AdamW に
`lr * adamw_lr_ratio` を使います（比率の既定値は 0.015）。
`try_step_with_lrs` は独立したスケジュールを許可します。勾配がない場合、重み減衰とモメンタムの両方をスキップします。 `try_step_or_skip(..., true)` は状態を更新せず両グループをスキップします。先に両グループのメタデータを検査しますが、デバイス障害は非同期のままであり、デバイストランザクションではありません。共有パラメータは
ParamId ごとに一度だけ振り分けます。重なる重みを無関係な ID で表してはいけません。

## 数値選択と移行

既存のコンストラクターのデフォルトは、SGD モメンタム、テンソル dtype 正規化、および
AsStored スケーリングを保持します。 EMA モードはオプトインです: `m = beta*m + (1-beta)*g`
、ゼロ初期化。ネステロフは `(1-beta)*g + beta*m` を使用します。意図的な変換をせずに、EMA と従来の SGD
チェックポイント バッファを交換しないでください。有限 5 次ニュートン シュルツ多項式は正確な極分解ではありません。正確な恒等行列アサーションは無効です。

安定した正規化は、二乗和を作る前に最大絶対値を基準にスケーリングします。丸めが変わるため明示的な有効化が必要で、入力と状態には FP32
を要求します。他のすべての段階のオーバーフローを防ぐものではなく、非有限値も検査しません。暗黙の BF16 キャストは行いません。PyTorch
標準の Muon は
NS に BF16
を使うため、FP32 版はビット単位で等価ではありません。FP32 マスター重みや低精度モデルコピーの自動更新も提供しません。

RUDA Linear の格納順は
`[input, output]` なので、Original LR
スケーリングには必要に応じて InputOutput を使います。AsStored
は行を出力として解釈します。MatchRmsAdamW は転置に対して対称です。重み減衰には、形状による調整後ではなく元の学習率を使います。選択した Muon グループ全体に一つの構成を適用するため、論理レイアウトが混在する場合は個別に設定したオプティマイザが必要です。

プロジェクトの Config マクロは、新しいフィールドの
serde デフォルトを提供しません。古い JSON 構成には、古い選択を保持するために
`"momentum_mode":"Sgd"` 、 `"stable_normalization":false`
、 `"matrix_layout":"AsStored"` を追加する必要があります。プログラムによるデフォルトは引き続き利用可能です。単純なミュオン テンソル
レコードのレイアウトは変更されていません。混合レコードには、バージョン、構成、パラメーター ID/形状/dtype マニフェスト、および両方のオプティマイザー状態が含まれます。まず元の
ID でモデルを復元します。 FullPrecisionSettings は、正確な継続比較に必要です。変更されたグループ化/構成は拒否されます。

更新する前に外部でスケールを解除して勾配を確認し、レプリカ間でスキップの決定を同期します。このパッチは、以前の低レベルの勾配ガード API
を自動的に統合しません。暗黙的な `step_multi`
、Ruda 分散マーク付きテンソル、FSDP/TP
シャード、スパース勾配、および 4D 畳み込み再形成は実装されていません。任意のシャードを完全なマトリックスであるかのように直交化しないでください。

## 検証

```bash
cargo run --release --locked -p ruda-optim --example muon-training -- 20
python tools/run_muon_regressions.py --suite oracle
python tools/run_muon_regressions.py --suite reference
python tools/run_muon_regressions.py --suite host
python tools/run_muon_regressions.py --suite build
python tools/run_muon_regressions.py --suite cuda --compiler both
```

この例では、test-cuda で構築されていない限り、Host tensor
バックエンドを使用します。これはパフォーマンスのベンチマークではありません。リファレンス スイートは、rustc のみを使用して独立したスカラー オラクルをコンパイルします。
RUDA の実行は検証されません。ホスト スイートと CUDA
スイートは、実際の RUDA テンソル/グループ テストをコンパイルして実行します。ビルドにはデフォルト機能なしのチェックが含まれています。不足しているツールはブロックされます。インストール/フォールバックは試行されません。各コマンドには明示的なタイムアウトと個別のログがあります。

参照: [Muon 作者](https://github.com/KellerJordan/Muon)
、[PyTorch 公式インターフェース](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)、[修正された
v2.9 ソース](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py)。
