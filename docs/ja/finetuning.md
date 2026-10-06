# モデル非依存 LoRA・NF4 微調整

[目次](README.md) · [API](native-pytorch-api.md) · [完全な API 契約](../en/finetuning.md)

Rust モデルは ruda_nn::LoRALinearConfig、PyTorch モデルは inject_lora、ローカル HF safetensors は load_hf_nf4_model / load_nf4_safetensors を使います。モデル family、target、backbone/head を推測せず、重みダウンロードや不足デバイス演算の追加はしません。[ネイティブ部品](../../ruda-torch/README.md)を準備し、streaming は safetensors、HF helper は transformers/accelerate も必要です。

## 準備の順序と対象

1. named_modules から正確な完全 module path を選びます。API target_modules は非空の名前列または文字列 all-linear。suffix/regex ではありません。
2. CPU の連続 FP32/FP16/BF16 weight を quantize_nf4 で量子化するか、meta 上のモデルへ safetensors を逐テンソル load します。
3. inject_lora を実行し、**その後** requires_grad=True のパラメーターだけで optimizer を作成します。元の全パラメーターは凍結され、二重 injection はエラー。

LoRALinear は `base(x)+(alpha/rank)*B(A(x))`。rank=16、alpha=16、adapter_dtype=FP32 が既定。rank は正整数、alpha は有限正値、dtype は FP32/FP16/BF16。A `[rank,in]`、B `[out,rank]` は B=0 から開始します。Python dropout 引数はありません。共有 module alias は保持、root Linear は container に包みます。

Rust LoRALinearConfig::new(rank,alpha).with_dropout(p).init(base) は既存 base ID を保持します。alpha は有限であればよく、dropout は `[0,1)`。merge は adapter を消費して凍結 dense 層を作り、optimizer の再開状態変換ではありません。

## NF4 と streaming

quantize_nf4 は injection/デバイス移動の前に実行。block_size=64 は正偶数、tile_rows=128 は正。embedding/head と tied weight は対象から除外します。変換失敗時、完了した層は変換済みのままです。

NF4Linear は dense shadow を持たず、row-major 二コード/byte、最初の値が高 nibble、flat block の FP32 absmax を保存。scales/codebook は dtype 移動後も FP32。凍結 base の勾配はなく、一階 input 勾配のみ。bitsandbytes/PEFT、AWQ INT4、learned fake quantization とは別形式です。

pack_nf4 は非空連続 CPU `[out,in]` を受け、packed uint8 長 `ceil(out*in/2)` と scales 長 `ceil(out*in/block_size)` を返します。NF4Linear の入力 `[...,in]` と weight device は一致します。FP16/BF16 は matmul API 1 があれば融合解量子化 GEMM、FP32/古い任意 capability は bounded tile decode を使います。decode API 1 は必要で、失敗した融合処理を別経路で再実行しません。

load_nf4_safetensors は全 parameter が meta のモデルと正確な重み名/shape を要求。model.safetensors または index JSON と shards を読み、非 persistent buffer は constructor が実体化しておきます。parameter_dtypes/buffer_dtypes は完全浮動 tensor 名の override。保護した weight は NF4 target にできず、共有 dtype の矛盾を拒否。失敗後は fresh meta model からやり直します。

load_hf_nf4_model は local architecture を meta で作り、tie_weights を再適用し、stream と injection を実行。auto_class は AutoModelForCausalLM、remote code は許可しません。config_kwargs/model_kwargs、parameter_dtypes を明示できます。両 loader の dtype 既定は BF16、**T4 は FP16 を明示**します。

## 因果監督と累積

SFTCollator は CPU 上で右 padding。input_ids/labels `[B,T]` int64、attention_mask Bool を返します。入力は等長 input_ids/labels または messages。labels は語彙 ID/-100、事前 shift しません。assistant-only の chat は tokenizer が assistant mask を宣言する必要があります。train_on_prompt=True だけが全 template token を監督。pad_token_id と max_length は実際の値を与え、truncate=False が既定です。

CausalLMFinetuner は明示的 backbone/head を受け、backbone が `[B,T,D]` または last_hidden_state を返します。head は dense/NF4/LoRA Linear。token_chunk_size=32、activation_checkpointing=True、preserve_rng_state=True が既定。checkpoint_modules は backbone 相対の非重複 path。再計算 RNG の保持と再起動 checkpoint は別です。

chunked_lm_cross_entropy は完全な語彙へ token chunk ごとに投影、`[B,T,V]` 全体は保持しません。shift=True、ignore_index=-100、reduction=mean/sum、recompute=True。mean は有効 target 数で割り、無効全体は微分可能なゼロ。labels は同デバイス int32/int64。

SFTTrainer.train_step は collated CPU microbatch 列を受け、local loss sum を **ウィンドウ全体の token 数**で割って backward、optimizer を一度更新し gradient を消去。mean の平均ではありません。scheduler は update が skip されないときだけ進みます。step/cursor は skip 時も進みます。

## 保存と実行

adapter_state_dict / load_adapter_state_dict は CPU A/B と厳密な名前・rank・alpha・寸法/base kind。finetune_state_dict / load_finetune_state_dict は optimizer 型/順序、任意 scaler、CPU と使用した CUDA RNG、step/data_state を追加し、base は複製しません。grad を None にした optimizer 境界だけで保存し、部分累積は保存しません。復元は同じ base_id と全 adapter/optimizer 配置、data cursor を使います。

SFTTrainer.save は next.pt を書いて確認、latest を previous へ回転。resume は同一 run_config/scheduler を要求。write_progress は local progress.json で、RUDA GPU peak は未知。Python/NumPy、外部 sampler、独立 RUDA RNG は low-level API が自動保存しません。merge_lora は eval dense 層だけで、NF4/tied weight は融合しません。

[CLI](../../ruda-torch/python/examples/finetune_causal_lm.py)では model/data/output/base-id/backbone/head、正確な targets、max-length/steps を明示。output は repo 外。targets は path 列で all-linear 特殊文字列ではありません。まず実際の shape で有界 step を実行し、同一設定へ --resume を追加して再開。--steps は新しい総目標。ファイルは一度だけ走査し、暗黙 repeat/shuffle はありません。--chat、--truncate、--train-on-prompt は意図した監督方針でのみ選択し、--compile は全演算の native 化を意味しません。
