# 固定アドレス PyTorch graph

[目次](README.md) · [モデル compiler](model-compiler.md) · [完全な node 表と例](../en/static-pytorch-graphs.md)

StaticGraph(inputs,nodes,outputs=None,infer_dependencies=False,track_completion=False,optimize=False,reuse_workspace=False,training=False) は明示 GraphOp を既存 RUDA CudaGraph で実行。任意モデル capture ではありません。

inputs は名前→固定アドレス連続 dense ruda:0 FP32/FP16/BF16、非空 rank1–8、uint32 要素数。最大256 node/512 tensor、options は bool。outputs は重複のない生成名、既定最後の node。既存名の上書き/将来 node の参照は禁止。

GraphOp(kind,output,left,right=None,scalar=0.) を使用。copy/unary は右入力なしと canonical zero、add の scalar は alpha、その他の二項は zero。add_scalar/mul_scalar/div_scalar は有限 FP32 scalar。mm/bmm は同 dtype rank2/3 と一致する inner/batch。RMSNorm weight は同 dtype 末幅、scalar は正 epsilon。softmax 軸は canonical 非負、keepdim sum/mean は非空 axis bitmask。

便利 constructor は copy/add/mul/silu/silu_mul/rms_norm。broadcast と implicit dtype promotion はありません。scalar/geometry は作成時固定、入力は内容のみ更新し pointer/shape/stride/device を交換しません。

optimize は全 plan 検証後に不要分岐を削除、左 SiLU の single-use mul を融合。不正な未使用 node は隠しません。silu_mul は低精度 SiLU の保存丸めを維持。reuse_workspace は最終利用後の同仕様 scratch のみで、input と返す output は再利用しません。

```python
import torch
import ruda_torch as r
x = torch.ones(2, 8).to('ruda:0').requires_grad_()
with r.StaticGraph({'x': x}, [r.GraphOp.silu('y', 'x')], training=True) as graph:
    graph.replay()['y'].float().mean().backward()
```

training=False は requires_grad 入力を拒否。True は native forward、一階の同 device 再計算 backward、入力 snapshot と独立結果を使い、backward/optimizer capture ではありません。元 input を backward 前に変更しません。

推論 output は次 replay で上書きする buffer。保存したい結果を clone し正しく順序づけます。作成 stream で replay。run_eager も同じ最適化 plan です。synchronize は wait、query は readiness、track_completion は query_completion/wait_completion を有効にします。info の workspace は node allocation だけでピーク VRAM ではありません。close は待機して解放し、保持 output は有効です。

base ABI10、graph API3 を両 native 部品で揃えます。attention/MoE node、第二 native device、AMD/Intel graph adapter、全 optimizer capture はありません。
