# テンソルとフレームワーク

[English](../en/tensor-framework.md) | [简体中文](../zh/tensor-framework.md) | **日本語** | [Deutsch](../de/tensor-framework.md) | [Русский](../ru/tensor-framework.md)

[ドキュメント](README.md) · [計算ライブラリ](libraries/README.md) · [プログラミング ガイド](programming-guide.md) · [中文](../zh/tensor-framework.md)

## 1. レイヤー

|レイヤー|コンポーネント|責任|
| --- | --- | --- |
|共有データと契約|ルダコア|Dtype、shape、devices、およびコンパイルコントラクト|
|デバイス テンソル|ruda-kernel::tensor|ストレージ、メタデータ、割り当て、およびレイアウト|
|デバイス バックエンド|ruda-tensor-device|テンソル演算をライブラリを計算するためにディスパッチします|
|テンソル API|ruda テンソル|バックエンドの汎用テンソル インターフェイス|
|フュージョン|ルダ・フュージョン|オペレーションフュージョン|
|自動微分|ruda-autodiff|自動微分|
|モデルとトレーニングのコンポーネント|ruda-model、ruda-nn、ruda-optim、ruda-store、ruda-dataset|モデル、ネットワーク モジュール、オプティマイザー、ストレージ、およびデータ|

## 2. デバイス テンソル

`RudaTensor<R>` には、クライアント、ハンドル、メタ、デバイス、dtype、および qparams が含まれます。ストレージ ハンドルはシェイプ/ストライドとは別のものであり、量子化パラメータは別個に保存されます。

低レベルの操作では、入力がデバイスを共有していること、dtype が計算と一致していること、および量子化データが正しいパラメーターを保持していることを確認する必要があります。連続ストレージ、転置ビュー、実体化コピーは異なります。

割り当て、連続変換、再形成、順列、転送、リードバックについては、[デバイス テンソル モジュール](../../ruda-kernel/src/tensor/mod.rs) を参照してください。

## 3. NVIDIA バックエンド

`ruda-tensor-device/cuda` は `ruda_tensor_device::cuda` を有効にします。

`cuda-fusion` がない場合、`Cuda<F, I>` は `DeviceBackend<CudaRuntime, F, I, u8>` の別名になります。この機能を有効にすると、Fusion ラッパーが使用されます。 F のデフォルトは f32、I は i32 です。 [cuda.rs](../../ruda-tensor-device/src/cuda.rs) を参照してください。

これはテンソル バックエンドであり、CUDA ドライバー API ハンドルではありません。バックエンドを選択するときは、各オペレーションの dtype と機能要件を確認してください。

## 4. 計算ライブラリのディスパッチ

行列演算は ruBLAS、ニューラルネットワーク演算は ruDNN、帰約とインデックス操作は ruPRIM、FFT は ruFFT、乱数生成は ruRAND に委譲されます。デバイス Backend の[ディスパッチモジュール](../../ruda-tensor-device/src/dispatch)を参照してください。

## 5. スパーステンソル、量子化、およびバッチリードバック

`ruda_tensor::api::CsrTensor<B>` は、`SparseOps` を通じてスパース構造と浮動小数点値テンソルを組み合わせます。スパース/デンス乗算、転置、加算、収集、散乱加算、およびサンプリングされた演算を提供します。これは `rusparse::tensor::CsrTensor<R>` とは異なります。前者はバックエンドに対して汎用的であり、後者はランタイムに対してです。 [スパースガイド](libraries/rusparse.md)を参照してください。

量子化には、多次元ブロック scales、非最終軸に沿ったパッキングおよび部分パック、レイアウト変換、選択されたインデックス付け操作、および融合量子化リードバックが含まれます。論理形状はパックされたストレージ形状とは異なります。 FP8/FP4 エンコードを整数値として変換しないでください。 [カーネル量子化](../../ruda-kernel/src/quantization)、[量子化テンソル レイアウト](../../ruda-kernel/src/tensor/contiguous.rs)、および [融合トランザクション](../../ruda-fusion/src/ops/transaction.rs) を参照してください。運用対象範囲は制度によって異なります。

バッチ リードバックは、実際のデバイスとストリームごとに記述子を編成します。 [テンソル トランザクション](../../ruda-kernel/src/tensor/transaction.rs) を参照してください。

## 6. トレーニングとモデル推論

- [トレーニングと状態の保存](training.md): autodiff バックエンドを構成し、パラメーターを更新し、勾配を蓄積し、トレーニング状態を保存または復元します。
- [モデルの読み込みと推論](model-inference.md): ローカルの重みを読み込み、チャット プロンプトを構築し、サンプリングを使用して生成し、画像入力を処理します。
