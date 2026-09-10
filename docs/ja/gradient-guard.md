# 勾配検査と融合 AdamW のクリッピング（実験的）

[English](../en/gradient-guard.md) | [简体中文](../zh/gradient-guard.md) | **日本語** | [Deutsch](../de/gradient-guard.md) | [Русский](../ru/gradient-guard.md)

明示的に有効化してください。既存の
`adamw_step` の公開シグネチャと既定のモデルオプティマイザは変更されません。

## 機能

`gradient_stats_sync` は累積勾配を走査し、元の値と FP32
でスケール解除した値の NaN/Inf を検査して、渡されたローカルグループ全体の
L2 ノルムを求めます。 `guarded_adamw_step`
はその共通ノルムを用いて既存の融合 AdamW/AMSGrad カーネル内でクリッピングし、元と同じサイズのクリッピング済み勾配を確保・書き込みしません。

処理順序は、格納型の変換 → FP32 で loss
scale の逆数を乗算 → 有限値の検査／ノルム → 共通の
FP32 クリッピング係数 → maximize の符号 →
AdamW です。意図的に FP16/BF16 のインプレース・クリッピングにはしていません。更新前に半精度の格納型へ丸め直すのではなく、実効勾配を FP32 のまま維持します。

有限勾配の場合:

```
clip = min(max_norm / (L2_norm + epsilon), 1)
g_effective = (float32(g_stored) * reciprocal(loss_scale)) * clip
```

`max_norm=None` はクリッピングを無効にしますが、有限値の検査は無効にしません。
`max_norm=0` は実効勾配をゼロにしますが、モーメント、重み減衰、ステップ数の更新は行います。
`Skip` 方針で
NaN/Inf がある場合、重み減衰とステップ数も含め、選択したグループ全体をスキップします。
`Error` 方針では、オプティマイザ更新カーネルを起動する前にエラーになります。すべてのメタデータ検査とステップのオーバーフロー検査は、統計処理の投入前に行います。

## 特徴と使い方

- `gradient-guard`: ホスト構成、統計/決定タイプ、および CPU オラクル。
- `gradient-guard-device`: 汎用デバイス削減と保護されたオプティマイザー。
- `gradient-guard-cuda`: CUDA 統合テストとベンチマーク。

```rust,ignore
use ruda_optim::fused_adamw::{
    AdamWEntry, AdamWOptions, StepControl, guarded_adamw_step,
    gradient_norm::GradientGuardOptions,
};

// master1/2: contiguous FP32; grad1/2: F32/F16/BF16, same device and queue.
// Accumulation must already be finished, using ONE loss scale for this step.
let entries = [
    AdamWEntry { parameters: &master1, gradients: &grad1, state: state1.as_ref() },
    AdamWEntry { parameters: &master2, gradients: &grad2, state: state2.as_ref() },
];
let pending = guarded_adamw_step(
    &entries, &AdamWOptions::default(),
    StepControl { gradient_scale: 128.0, skip_update: false },
    GradientGuardOptions { max_norm: Some(1.0), ..Default::default() },
)?;
// Compact stats readback has completed, but the UPDATE is still asynchronous.
// Await/synchronize the runtime and check completion before replacing committed
// model/state or saving a checkpoint. Keep the existing committed state on failure.
```

自動更新される
GradScaler、モデルアダプタ、FSDP 帰約、テンソルの平坦化、共有重みの重複除去、低精度モデルへの再キャストは追加しません。明示的に渡したパラメータだけを数えます。ローカルシャードのノルムをグローバルな
FSDP/TP
ノルムとして使ってはいけません。rank 間でスキップ判断を合意させる処理も行いません。

## 削減の実装とリソースコスト

グリッドストライド カーネルは、レーンごとに `(scale, sumsq, bad)`
トリプルを保持し、固定共有メモリ ツリーによって削減されます。大きな生の FP32
値を直接二乗することはありません。 `1e30` に近い有限値は、ノルム計算で偽の FP32
オーバーフローを生成しません。最大 1024 個の部分トリプルが
1 つの追加ブロックによって削減されます。 float アトミックはありません。これは自動デバイスのオートチューナーではありません。

この構成には、ブロックごとに 256 X
スレッドと 3072 バイトの共有メモリが必要です。サポートされていない構成は拒否され、サイレントに CPU
に送信されません。空のテンソルは仕事を送信しません。空ではない各テンソルは 1 つまたは 2
つのリダクション カーネルを送信し、1 つの 12
バイトのサマリーを返します。すべてのテンソル削減は、1 つのバッチ化されたホスト リードバック
API 呼び出しの前にキューに入れられます。ランタイムは複数の DMA コピーを実装する場合があります。ホストは、FP64
内のコンパクト サマリーを明示的に合計し、係数を決定します。テンソルには、アロケーター アライメント/メタデータを除き、最大 12
の KiB 部分ストレージと 12
バイトの最終サマリーが必要です。 `scratch_bytes` は、これらのテンソルごとの量を合計するものであり、ピーク アロケーターの測定値ではありません。

再関連付け、FMA、および非正規処理はバックエンドに依存します。規格は許容差を持ってテストされており、PyTorch
またはデバイス全体でビット同一であることは約束されていません。これは LAPACK
LASSQ の実装全体ではなく、数値的な互換性は宣伝されていません。クリッピングが無効になっている場合でも、大きな有限グラデーションは
AdamW の
2 番目の瞬間にオーバーフローする可能性があります。古いパラメータ/モーメントは有限であるかどうかスキャンされません。

## パフォーマンスに関する主張と明示的な制限

含まれているベースライン (同じノルム計算、その後 FP32
勾配を生成する個別のアンスケール + クリップ
カーネル、その後 AdamW の融合)
と比較すると、空ではないテンソルごとに 1 つのクリップ起動と
1 つの `4*N`
一時バイトが削除され、 `8*N` バイトの論理一時書き込み/読み取りが回避されます。これらはソースレベルのカウントであり、デバイスのメモリ トラフィックや加速度の測定値ではありません。保護されていない古いオプティマイザは、新しい統計コストを支払いません。診断を追加すると、ステップが遅くなる可能性があります。

この最初のバージョンでは、オプティマイザー
ステップごとにコンパクト リードバックでホストがブロックされます。グラフ
キャプチャは安全ではなく、計算/通信のオーバーラップが保証されず、多くの小さなテンソルではパフォーマンスが低下する可能性があります。安定したリダクションには追加の演算が含まれます。有効にする前にベンチマークを実行します。 PyTorch
融合/foreach AdamW
に対するクレームは行われません。元の
AdamW 出力割り当ては変更されません
(out-of-place FP32
マスターおよびモーメント)。統計または依存する更新の実行中に、別のエイリアス/キューを介してグラデーションを変更してはなりません。ランタイムエラーは、トランザクショングループのコミットではありません。更新の開始前に、非有限/検証スキップの決定のみが行われます。ランタイム割り当て/起動 API
はエラー コントラクトを保持します。

## 検証とベンチマーク

```bash
python tools/run_gradient_guard_regressions.py --suite oracle
python tools/run_gradient_guard_regressions.py --suite reference
python tools/run_gradient_guard_regressions.py --suite host
python tools/run_gradient_guard_regressions.py --suite build
python tools/run_gradient_guard_regressions.py --suite cuda --compiler both
python tools/run_gradient_guard_regressions.py --suite bench --compiler both --elements 65536 --tensors 4 --dtype bf16 --amsgrad
```

`oracle` には NumPy および
PyTorch CPU が必要です。 Python 数値モデルのみを実行し、RUDA
コードは実行しません。 `reference` は、Cargo レジストリにアクセスせずにスタンドアロンの Rust
テストをコンパイルします。 `build` は、古いデバイスと新しいデバイスの両方の機能をチェックします。 `cuda`
は、古い AdamW 回帰と新しいハードウェア テストの両方を実行します。不足しているツールはブロックされます。タイムアウトと失敗が記録されます。 `--dry-run`
は計画されたコマンドのみを出力します。 `--offline` には、キャッシュされた Cargo 依存関係が必要です。ツールのインストールは実行されません。

ベンチマークでは、ウォームアップ後にパスを切り替え、割り当て、基準、ホストのリードバック/決定、送信および最終的なデバイスの同期を含み、各測定バッチ後のパラメータとすべての瞬間をチェックします。比較のみに使用されるリードバックはタイミング外です。生の
JSON
サンプルを含む環境/ソース
バージョンを保持します。

## 数値参照

概念的参照、コピーされた実装ではありません:
- PyTorch AMP の例 (グラデーション クリッピングの前にスケールを解除):
  https://docs.pytorch.org/docs/main/notes/amp_examples.html
- PyTorch ローカル連結勾配ノルム コントラクト:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html
- スケーリングされた二乗和表現:
  https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq.html
