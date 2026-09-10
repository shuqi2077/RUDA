# ruCCL ユーザーガイド

[English](../../en/libraries/ruccl.md) | [简体中文](../../zh/libraries/ruccl.md) | **日本語** | [Deutsch](../../de/libraries/ruccl.md) | [Русский](../../ru/libraries/ruccl.md)

[計算ライブラリ](README.md) · [テンソルとフレームワーク](../tensor-framework.md) · [中文](../../zh/libraries/ruccl.md)

## 1. レイヤーとエントリーポイント

Cargo パッケージは `ruCCL`、Rust クレートは `ruccl` です。

ruCCL には、テンソル バックエンド集合体、ランク コア、およびインプロセス実装が含まれます。 `ruda-communication` は通信インフラを提供します。 `orchestrator` 機能により、オーケストレーション エントリ ポイントが有効になります。

## 2. テンソル集合体 API

|関数|の動作|
| --- | --- |
|`register<B>`|ピア、デバイス、および CollectiveConfig を登録します|
|`all_reduce<B>`|縮小結果を参加者に返します|
|`broadcast<B>`|送信者は Some(tensor) を渡します。レシーバーパス なし|
|`reduce<B>`|指定されたルートに縮小します。非 root 参加者は何も受け取りません|
|`finish_collective<B>`|ピアの集合セッションを終了します|
|`reset_collective<B>`|ローカル集合サービスをリセットし、登録および進行中の操作状態を破棄します|

インターフェイスは `B: ruda_tensor::Backend` および `B::FloatTensorPrimitive` を使用します。自動微分と統合する場合は、内部バックエンドを登録します。集合呼び出し自体は自動逆方向ルールを定義しません。

## 3. 登録と通話契約

`CollectiveConfig::default()` で構成を作成します。ローカル参加デバイスの数には、`with_num_devices` を使用します。構成方法を通じて戦略とマルチノード アドレスを構成します。

参加者はデバイス数に同意し、一意のピア ID を使用し、一致するコレクティブを同じ順序で呼び出す必要があります。形状、リダクション演算、ルート、その他のパラメータが一致する必要があります。各ブロードキャストには送信者が 1 人だけ必要です。

マルチノード実行の場合は、ノード数、グローバル アドレスとローカル アドレス、およびデータ サービス ポートを一緒に構成します。

## 4. エラーとライフサイクル

`CollectiveError` は、登録の重複または欠落、形状の不一致、一貫性のないリダクション操作またはルート、および無効なブロードキャスト送信者の数をカバーします。

通常の終了には `finish_collective` を使います。`reset_collective` は進行中の状態を破棄するもので、演算の完了、デバイスタスクのチェックポイント作成、損失のない復旧は行いません。

## 5. CUDA の例

`cuda` 機能は、CUDA テンソル バックエンドを有効にします。 `cargo run --locked -p ruCCL --features cuda --example all_reduce` を実行して、GPU 0 で 4 つの論理ランクを持つリング AllReduce を実行します。これは、257 の FP32 要素の合計/平均、入力の保存、およびセッションの終了をチェックします。

デバイス アダプターは [tensor_device](../../../ruCCL/src/tensor_device) にあります。オプティマイザー インターフェイスについては、[明示的なランク勾配削減](../../../ruda-optim/src/optim/grads/collective.rs) を参照してください。転送には、ゼロコピー P2P ではなく、ホストステージングされたパスが含まれます。

ソース: [集合 API](../../../ruCCL/src/api.rs)、[構成](../../../ruCCL/src/config.rs)、[ランク](../../../ruCCL/src/rank/mod.rs)、および [インプロセス実装](../../../ruCCL/src/in_process/mod.rs)。

## 6. 集合研修

`ruda-optim` で `collective` を有効にします。明示的に所有されているランク コミュニケーターを使用して、後方勾配を `GradientsParams` に変換し、`grads.all_reduce_with::<InnerBackend>(&communicator, ReduceOperation::Mean)?` を呼び出して、返された勾配を `optimizer.step` に渡します。パラメータ ID、グラデーション形状、dtype、および呼び出し順序はランク間で一致する必要があります。 autodiff トレーニングの場合、`InnerBackend` は `Autodiff` ラッパーなしのバックエンドです。

ソース ツリーから 2 ランクのトレーニング例を実行します。

```powershell
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- run ./collective-training-state
cargo run --locked -p ruda-optim --features collective,cuda --example collective_training -- resume ./collective-training-state
```

`run` にはまだ存在しないディレクトリが必要です。最初の更新後に各ランクのモデルとオプティマイザーを保存し、2 番目の更新を実行します。 `resume` はそのディレクトリを復元し、2 番目の更新を実行します。 CUDA が有効になっている場合、この例の両方の論理ランクは同じデフォルト デバイスを使用します。

完全な呼び出しシーケンスについては、[集合トレーニングの例](../../../ruda-optim/examples/collective_training.rs) を参照してください。スケジューラーの状態と保留中の累積勾配も保存するには、[トレーニングと状態の保存](../training.md) の `TrainingRecord` を使用します。
