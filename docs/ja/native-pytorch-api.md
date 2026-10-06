# ネイティブ PyTorch API 契約

[目次](README.md) · [詳細なメソッド参照](../en/native-pytorch-api.md)

import ruda_torch は PrivateUse1 を登録し、別 backend と同じプロセスでは使えません。ruda:0 と torch.cuda の storage は別です。device_count=1、current_device=0。is_available は統合初期化を示し、すべての GPU 命令への対応を保証しません。execution_stats は累積 counter で timer ではありません。

Rust/C++ は base ABI 10、graph 3、training 4。router/sequence/NF4 decode/NF4 matmul は各 API 1、paged backward は互換 bridge の API 1/2、ordered は2。base ABI 一致だけでは任意拡張を提供しません。未対応演算はエラーで、汎用 CPU fallback はありません。

## 学習と dtype

融合 normalization は連続 dense ruda:0 FP32/FP16/BF16、未解決 conjugate/negative view 不可。統計は FP32、出力は input dtype、一階微分のみ。

| API | 条件 |
| --- | --- |
| rms_norm(x,weight=None,eps=None) | `[...,D]`、非空末尾。weight `[D]` 同デバイスで input dtype/FP32。eps None は finfo.eps。 |
| RMSNorm(width,eps=1e-5,elementwise_affine=True) | width 正、affine 既定 FP32。CPU constructor 後 RUDA へ移す。 |
| layer_norm / LayerNorm | 末軸のみ。任意 weight/bias `[D]` は input dtype/FP32、eps 有限正値。 |
| silu_mul(gate,up) | 同 shape/dtype/device、broadcast なし、保存丸め境界を保持。 |

AdamW 既定 lr .001、betas(.9,.999)、eps1e-8、decay .01、fused_step=False、hierarchical_stats=False、max_grad_norm=None。非重複連続 leaf parameter、dense matching gradient を使い、capturable/amsgrad/maximize/differentiable は対象外。FP32 master/moment。既定は gradient を unscale して4 byte readback、fused は gradient を保持し12 byte readback。一 active parameter 一 kernel。clipping/hierarchical は fused を要求し、非有限値は step 全体を skip。

GradScaler は RUDA AdamW/Muon/MuonAdamW 一 optimizer の FP32 scalar loss に使用。scale/backward → step → update の順。unscale_、closure、多 optimizer cycle はありません。init_scale65536、growth2、backoff.5、interval2000、範囲 `2**-24..2**24`。state は完了 cycle 間で保存。

## 実行と attention

Stream priority は0。current_stream/default_stream、stream context を使います。wait_event/wait_stream は device 順序、query は非待機、synchronize は host wait。timing-enabled Event の elapsed_time は完了後の ms。record_stream は寿命保持のみで依存を作りません。RUDA_TORCH_ASYNC は初回提出前に設定し process に cache されます。

PagedAttentionPlan は immutable schedule。Q `[queries,q_heads,key_dim]`、K/V `[pages,page_size,kv_heads,dim]`、同連続 floating storage、q_heads は kv_heads の倍数、dim<=1024、scale は有限正。MLA query `[queries,heads,rank]` と位置幅、cache `[pages,page_size,1,rank]`、position_dim<=256。caller が cache 更新/position encoding を所有し、元の QK scale を使います。splits=1 または2–32、workspace_bytes は forward scratch のみ。一階微分、atomic 既定、ordered は明示選択で autotune ではありません。

selected_router_weights は logits `[tokens,experts]` floating と indices `[tokens,top_k]` int32/int64、top_k<=min(experts,64)。FP32 weight と dlogits を返し、選択はしません。softmax は全 expert、sigmoid は pointwise、renormalize は選択 slot。重複 index は勾配を加算し、無効 index の行は forward/backward とも NaN。

## シーケンスと量子化

solve_triangular は同 floating dtype/device の A `[...,N,N]`、左 B `[...,N,K]` / 右 B `[...,K,N]`、batch broadcast と一階微分。flag は bool、RUDA sequence API 1。

gated_delta_rule は Q/K `[B,H,T,Dk]`、V `[B,H,T,Dv]`、beta/decay `[B,H,T]`、任意状態 `[B,H,Dk,Dv]`。Q/K/V dtype 一致、chunk_size64 が既定、output と final_state の両勾配。低精度 state は FP32。native chunk forward と同デバイス再計算 backward で、独立融合 backward ではありません。

learned_fake_quantize は bits2/4/8、floating input と同デバイス FP32 scale。block_shape None は scale 一つ、指定時は各軸の正 block と `product(ceil(shape/block))` scales。floor1e-8 以下は scale 勾配ゼロ。一階 floating output で packed NF4 ではありません。

関連：[モデル compiler](model-compiler.md)、[固定 graph](static-pytorch-graphs.md)、[LoRA/NF4](finetuning.md)、[mHC/CSA/HCA/Muon](architecture-training.md)、[分散学習](distributed-training.md)。非同期エラーは sync/readback で現れる場合があります。
