# 明示的 rank・デバイス・レプリカ学習

[目次](README.md) · [学習](training.md) · [ruCCL](libraries/ruccl.md) · [接続コード](../en/distributed-training.md)

rank 数は GPU 数ではありません。register ベースの collective_training は二つの論理 rank を同じ既定 GPU に配置します。in_process Communicator はローカル device context、RankCommunicator は明示的 rank/world と TCP session、DataParallel は caller 所有 communicator 上のレプリカ学習です。PyTorch ruda:0 は単一ネイティブデバイスで、torch.distributed launcher の置換ではありません。ここでは Rust Tensor/Autodiff を扱います。

## 配置と起動

`CudaDevice { index }` の index はプロセス可視 CUDA ordinal。global rank ではありません。model と TensorDevice を同じ device で作成します。同一プロセスの GPU 0/1 を使うには明示的に異なる index を選択し、Default を二回使いません。launcher が可視性を変えた場合、そのプロセスの実際の列挙順を使います。

全 rank へ共通 rendezvous address、一つの UniqueId、異なる `0..world_size-1` rank、同一 world_size、local device を渡します。UniqueId は一度 new し as_bytes を配布、from_bytes で復元。rank ごとに new すると別 session です。

- coordinator は `TcpRendezvousServer::bind(address,id,world_size)?.run()`。全 worker と並行して動作させます。
- worker は `RankCommunicator::connect(initialize,address,id,rank,world_size,timeout,queue_name)`。initializer がその rank の TensorDevice を返します。
- TCP/heartbeat timeout、rail、transport は実際に対応する設定を統一します。標準の torchrun/NCCL 環境変数を自動認識する launcher はありません。
- 既定 TCP Tensor adapter は host-staged。TCP peer は自動的に GPU P2P/NVLink/RDMA になりません。

必要な Cargo 側は ruCCL、ruda-autodiff、ruda-nn、collective 有効の ruda-optim、cuda-default 有効の ruda-tensor-device。[完全な初期化関数](../en/distributed-training.md#rendezvous-and-rank-connection)と[接続 API](../../ruCCL/src/rank/communicator/connect.rs)を参照してください。別のネイティブ communicator は DataParallelCommunicator の metadata/broadcast/reduce 契約を実装します。

## 学習の順序

`DataParallel::initialize(communicator,model,root)` はパス、shape、dtype、凍結状態、共有 alias を照合して浮動パラメーターを broadcast。local parameter ID は保持され、rank 間で同値である必要はありません。optimizer の作成前、または一致する rank checkpoint の復元後に呼びます。initialize_with_buffers は I32/I64/Bool buffer も一度 broadcast し、各 forward の同期ではありません。

全 rank の collective 順、dtype/shape、勾配追跡、root と gather/scatter 軸を揃えます。peer が通信している間に一つの rank だけ通信を skip できません。可微分 collective の backward 順序も一致し、checkpoint 再計算で通信は再実行しません。

local loss **sum** を backward して累積します。境界で `reduce(&model,gradients,local_weight,policy)` を呼び、返った gradients で更新。全勾配和を有効 token/sample 総数 global_weight で割ります。rank 数や local mean の平均で割りません。reduce_fp32 は半精度パラメーターでも FP32 勾配を保持し、通常 reduce は最後に保存 dtype へ戻します。全 rank で同じ変種を使います。

MissingGradientPolicy::Error は非ゼロ local weight の勾配を要求、Zero は欠落をゼロ寄与にします。全体で未使用のパラメーターは更新に入れません。scheduler、clipping、optimizer と累積器 reset は暗黙に実行されません。構造/ID、凍結パラメーターや設定不一致は契約エラーです。

## checkpoint と再開

同じ完了境界で各 rank の model/optimizer/scheduler、必要な累積器、data/sampler cursor と RNG を保存。TrainingRecord は一貫した全 rank snapshot を自動選択しません。全 rank を同じ境界から復元してから新しい通信を始め、local ID と optimizer record を組にします。古い optimizer に新しい base を broadcast しません。world/device/source/transport 設定を外部 run 設定へ保存します。

```bash
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- run ../ruda-collective-state
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- resume ../ruda-collective-state
```

run は新規ディレクトリーを要求し第一 step を保存、resume はそこから第二 step を実行。未変更例の両 rank は同じ既定 device で、多 GPU 性能 baseline ではありません。通信/デバイス障害を一つの rank だけの無言再実行へ変換しません。長い実行前に実際の時間/メモリ、作業量、checkpoint と可視の永続進捗を準備し、記録は Git 外へ置きます。
