# mHC・圧縮 Attention・Python Muon

[目次](README.md) · [API](native-pytorch-api.md) · [詳細な例と定義](../en/architecture-training.md)

これらは同じデバイス上の微分可能なテンソル演算を組み合わせる部品です。公式 DeepSeek の重みローダー、モデル完全再現、FP4/FP8 Attention カーネルではありません。乱数初期化は CPU で行い、モジュールと入力を明示的に実行デバイスへ移します。FP16/BF16 の写像・スコア計算には FP32 を使います。

## mHC と残差を持たない分岐

`MHC(width, streams=4, sinkhorn_iterations=20, eps=1e-6, gate_init=0.01)` の状態は `[...,streams,width]`。幅・stream 数・反復数は正整数、eps は正、gate_init は有限値です。状態とパラメーターは同じデバイスに置きます。

| メソッド | 入出力 |
| --- | --- |
| `expand(x)` | `[...,width]` → 複製した stream 状態。 |
| `reduce(state)` | stream 軸を平均して `[...,width]` を返す。 |
| `coefficients(state)` | pre/post `[...,streams]`、residual `[...,streams,streams]` の `MHCCoefficients`。 |
| `pre(state)` | 分岐入力 `[...,width]` と係数。 |
| `post(state, branch_output, coefficients)` | 残差写像と分岐出力を合成。分岐出力から stream 軸だけを除く。 |
| `forward(state, branch, ...)` | pre → 分岐 → post。 |

`sinkhorn(logits, iterations=20)` は末尾が非空の正方行列である必要があります。log 空間で列、行の順に正規化します。有限反復は厳密な多様体射影ではありません。

`MHCResidual(branch,width,streams=4,checkpoint_branch=False,**options)` の branch は **自分で残差を加えてはいけません**。checkpoint_branch は学習時の非 reentrant 再計算です。`MHCSequential(width,branches,streams=4,**options)` は一度展開し、少なくとも一つの分岐を実行し、最後に平均します。

## DSA indexer

`LightningIndexer` と `DSAIndexer` は同じクラスです。既定値：`num_heads=4, head_dim=16, query_dim=None, topk=32, query_chunk_size=32, key_chunk_size=128, external_keys=False, detach_inputs=True, rope_dim=0, eps=1e-6`。query_dim を省略すると width。RoPE 次元は非負偶数で head_dim 以下です。

入力 `[B,T,width]`、query latent `[B,T,query_dim]`、prepared keys `[B,S,head_dim]`。`project_keys` は通常の token key を作成し、external_keys モードでは呼び出し側が渡します。`scores` は診断用の密な `[B,T,S]`、`select` は chunk を走査する top-k。chunk 化は中間メモリを減らしますが、スコア計算量の二次依存は残ります。

`allowed [B,T,S]`、key_valid `[B,S]`、query_valid `[B,T]` は同デバイス Bool。query_positions `[T]`、key_end_positions `[S]` は int64 の絶対位置です。圧縮 key は実際の終了位置を渡します。`forward(...,causal=True)` は `IndexerOutput(indices,scores,valid)` を返し、無効 index は -1。causal=False は明示的な非因果選択です。

離散 top-k には勾配がありません。既定では入力特徴を detach します。`indexer_kl_loss` または `distillation_loss` を損失へ明示的に加えます。teacher は非負 `[B,T,S]` または `[B,T,H,S]`、detach され、空行の寄与はゼロです。LM 損失だけでは selector を学習できません。

## CSA/HCA とキャッシュ

CSA は重複する学習済み圧縮、DSA 選択、局所窓を合成します。HCA は非重複圧縮と全可視圧縮要素を使い、indexer を持ちません。一つの softmax に局所/圧縮要素を入れ、完成したブロックだけを因果的に参照します。

| オプション | 既定値 / 条件 |
| --- | --- |
| head_dim | width/num_heads。省略時は割り切れること。 |
| compress_ratio | CSA 4、HCA 128。正整数。 |
| topk、window_size | 32、128。 |
| query_rank、index_heads、index_dim | width、4、16。 |
| rope_dim、rope_base | 0、10000。次元は偶数で Attention/indexer に収まる。YaRN ではない。 |
| output_groups、output_rank | 1、width/output_groups。groups は head 数を割り切る。 |
| query_chunk_size、key_chunk_size | 32、128。 |
| attention_sink、eps | True、1e-6。sink は分母への寄与で KV 値ではない。 |

`forward(x,valid_mask=None,return_aux=False,indexer_warmup=False)` は非空 `[B,T,width]`、mask は同デバイス Bool `[B,T]`、True が有効。return_aux は `AttentionOutput(output,indexer_loss)` を返します。warm-up は全可視圧縮要素を使うため密な計算コストを持ちます。indexer のみの学習では補助損失だけを backward。HCA の補助項は主モデルから切り離されたゼロです。

`LearnedKVCompressor(width,head_dim,ratio,overlap=False,eps=1e-6)` は floor(T/ratio) 個の `(compressed,valid)`。未完成 tail は出力せず、全無効ブロックはゼロ。`RotaryEmbedding` は `[B,T,D]` / `[B,T,H,D]` と `[T]` 位置を使い、frequency は FP32、inverse=True で逆回転します。

`forward_cached(x,cache=None,valid_mask=None)` と compressor.append は **eval と no_grad/inference_mode の両方**が必要です。戻り値 `(output,new_cache)` を次の chunk へ渡します。別レイヤー、変更されたパラメーター、異なる batch/device/dtype の cache は拒否し、None から再開します。cache 更新は関数的です。`reorder` は同デバイスの一次元 int64 beam index、tensor_bytes は論理保持量でピーク VRAM ではありません。圧縮履歴は系列長とともに増えます。

## 混合モデルと Muon

`MHCTransformerBlock` の状態は `[B,T,streams,width]`。attention_kind は csa/hca、FFN 幅は既定 4*width。`HybridAttentionLanguageModel(vocab_size,width,num_heads,num_layers,streams=4,csa_ratio=4,hca_ratio=128,tie_embeddings=False)` は CSA/HCA を交互に使い、int32/int64 `[B,T]` から `[B,T,V]` logits を返します。return_aux は indexer 損失の和も返します。モデル全体に forward_cached はなく、個々の Attention が提供します。

`next_token_loss` は有効な source/target ペアの次 token 損失。mask された token も正しい語彙 ID が必要で、-100 は使いません。無効 label を使う SFT は[微調整](finetuning.md)の別契約です。

Python `Muon` は単一デバイスで完全な行列を更新します。既定 lr .02、momentum .95、nesterov True、momentum_mode sgd、ns_steps 5、係数 `(3.4445,-4.775,2.0315)`、eps 1e-7、adjust_lr original、matrix_layout as_stored、stable_normalization True、flatten False。反復数は1–99、係数は有限。flatten=True のみ高次元を `[先頭次元,-1]` にします。SGD/EMA の Nesterov は正 momentum とゼロ dampening を要求します。

original の LR 係数は `sqrt(max(1,rows/cols))`、match_rms_adamw は `.2*sqrt(max(rows,cols))`。input_output は係数の軸意味を交換し、保存パラメーターを転置しません。max_grad_norm は任意の正値で、勾配保存値は変更しません。FP16/BF16 master は FP32、有限性 readback のため step は capture できません。非有限更新は全体を skip、commit 中のデバイス障害はトランザクションではありません。

`MuonAdamW.from_model(model,muon_modules=...,adamw_modules=...,lr=.02,adamw_lr=.001)` は実際の module オブジェクトを指定します。Embedding と明示除外が AdamW 優先、共有パラメーターは一度だけ。他の trainable パラメーターも AdamW。対象行列がない場合はエラー。手動 group は use_muon を明示します。分散勾配を同期してから完全行列を正規化し、TP/FSDP shard を独立行列として更新しません。Python state は Rust Muon Record と互換ではありません。
