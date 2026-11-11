# Ruda ドキュメント

[English](../en/README.md) | [简体中文](../zh/README.md) | **日本語** | [Deutsch](../de/README.md) | [Русский](../ru/README.md)

[ドキュメントインデックス](../README.md) · [中文](../zh/README.md)

最初の GPU カーネルからライブラリ、テンソル トレーニング、ローカル モデル推論を計算します。まずは自分のタスクから始めて、プログラミング ガイドと API リファレンスを調べてください。

本書は `v0.3.0` リリース系列を説明します。Rust crate は独立したバージョンを維持します。正確なソースバージョンは各 Cargo manifest、公開済み rustdoc のスナップショットは [Rust API リファレンス](rust-api.md)を参照してください。対応する `ruda-torch==0.3.0` wheel または対応するソースを使用してください。以後の `main` のみにある API には対応するソースビルドが必要です。

## ここから始めてください

|あなたのタスク|読み取りパス|
| --- | --- |
|最初の GPU カーネルを実行します|[インストールとクイックスタート](getting-started.md) → [プログラミング ガイド](programming-guide.md)|
|行列、スパース、またはニューラル ネットワーク演算を使用する|[計算ライブラリ](libraries/README.md) → [テンソルとフレームワーク](tensor-framework.md)|
|モデルをトレーニングし、勾配を蓄積し、状態を保存する|[トレーニングと状態の保存](training.md)|
|ローカル モデルの読み込み、テキストの生成、または画像の処理|[モデルの読み込みと推論](model-inference.md)|

## はじめに

- [ネイティブ PyTorch クイックスタート](native-pytorch-quickstart.md)：インストール、データ転送、FP32 学習とエラー診断。
- [インストールとクイックスタート](getting-started.md): ソースのセットアップ、バックエンドの選択、および最初の例。
- [例とチュートリアル](samples.md): ベクトル加算、テンソル、共有メモリ、および半精度。

## プログラミング ガイド

- [Python 並列学習、データと復旧](parallel-and-data.md)：TP/FSDP/ZeRO、pipeline、sampler、checkpoint と調整。
- [Ruda プログラミング ガイド](programming-guide.md): ホストとデバイスのコード、実行階層、メモリ、同期、安全性。
- [テンソルとフレームワーク](tensor-framework.md): デバイス テンソル、ライブラリ ディスパッチ、融合、自動微分。
- [トレーニングと状態の保存](training.md): トレーニング ステップ、勾配の累積、学習率のスケジューリング、保存、および復元。
- [分散学習と rank・device](distributed-training.md): rendezvous、デバイス配置、collective 順序、重み付き勾配と rank ごとの再開。
- [アーキテクチャと Python Muon](architecture-training.md): mHC、DSA/CSA/HCA、圧縮 KV cache と hybrid model。
- [LoRA/NF4 微調整](finetuning.md): target 選択、重み準備、causal loss、checkpoint と再開。
- [PyTorch モデルコンパイラ](model-compiler.md): AOT forward/backward、native 分割、設定と cache。
- [固定アドレス PyTorch graph](static-pytorch-graphs.md): GraphOp、output の寿命と一階学習。
- [モデルの読み込みと推論](model-inference.md): ruLLM、テキストおよび画像入力、サンプリング、AWQ、および連続バッチ処理。

## コンパイルと実行

- [コンパイラー ガイド](compiler-guide.md): Rust カーネル フロントエンド、IR、CUDA C++/NVRTC、およびダイレクト PTX。
- [PTX バックエンドリファレンス](ptx.md): ターゲット構成、コンパイル出力、制約、およびエラー。
- [全スタック自動調整](stack-autotuning.md): 候補、検証、計時、policy と cache。

## API リファレンス

- [Rust API](rust-api.md)：crate とバージョン別の rustdoc。
- [Python 公開 API シグネチャ](python-api-signatures.md)：関数、コンストラクタ、メソッドとソース位置。
- [ランタイム API](runtime-api.md): デバイス クライアント、メモリ、送信、リードバック、および同期。
- [ドライバー API とバックエンド](driver-api.md): バックエンドの種類、デバイスの選択、およびランタイム統合。
- [ネイティブ PyTorch API](native-pytorch-api.md): コンポーネント版、tensor 契約、optimizer、stream と attention。
- [計算ライブラリ リファレンス](libraries/README.md): ライブラリの選択、Cargo の機能、およびエントリ ポイント。

## 計算ライブラリ

|ライブラリ|ガイド|
| --- | --- |
|ruBLAS|[線形代数とグループ化行列の乗算](libraries/rublas.md)|
|ruDNN|[ニューラル ネットワーク操作と MoE](libraries/rudnn.md)|
|ruTENSOR|[テンソル縮約、einsum と置換](libraries/rutensor.md)|
|ruPRIM|[削減、スキャン、およびインデックス作成](libraries/ruprim.md)|
|ruFFT|[高速フーリエ変換](libraries/rufft.md)|
|ruRAND|[乱数生成](libraries/rurand.md)|
|ruSPARSE|[スパース計算](libraries/rusparse.md)|
|ruCCL|[集団通信](libraries/ruccl.md)|

## デバッグと互換性

- [デバッグと診断](debugging.md): コンパイル エラー、非同期エラー、キャッシュ、および数値チェック。
- [互換性ガイド](compatibility.md): CUDA の概念、バックエンドの違い、API とコンパイルの境界。
- [寄稿](CONTRIBUTING.md): 問題と開発規約の報告。

最初のプロジェクトの場合は、クイックスタート、プログラミング ガイド、および関連するライブラリ ガイドに従ってください。バックエンドまたはカーネルを開発する場合は、コンパイラと API リファレンスを参照してください。
