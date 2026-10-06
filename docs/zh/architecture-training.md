# mHC、压缩注意力与 Python Muon

[文档目录](README.md) · [原生 API](native-pytorch-api.md) · [English](../en/architecture-training.md)

这些是可训练、同设备张量操作组合，不是 DeepSeek 官方预训练模型加载器或 FP4／FP8 注意力内核。随机初始化模块先在 CPU 上构造，再明确搬移模块与输入。FP16／BF16 映射和评分运算使用 FP32；显式 FP64 输入在受支持路径保留 FP64。组合模型能否执行仍取决于设备算子支持。

## Hyper-connection 与无残差分支

`MHC(width, streams=4, sinkhorn_iterations=20, eps=1e-6, gate_init=0.01, device=None, dtype=None)` 保存可训练的映射、门控及 bias。宽度／流数／迭代数为正整数，epsilon 为正，gate 初始化有限；状态为 `[...,streams,width]`，与参数同设备。

| 方法／类型 | 契约 |
| --- | --- |
| `expand(x)` | `[...,width]` → 克隆的 `[...,streams,width]` 状态。 |
| `reduce(state)` | 平均 stream 轴 → `[...,width]`。 |
| `coefficients(state)` | `MHCCoefficients(pre, post, residual)`，形状为 `[...,streams]`、`[...,streams]`、`[...,streams,streams]`。 |
| `pre(state)` | 返回合并后的 `[...,width]` 分支输入及系数。 |
| `post(state, branch_output, coefficients)` | 组合残差映射与分支更新；分支输出除 stream 外保留全部维度。 |
| `forward(state, branch, *args, **kwargs)` | pre → 无残差分支 → post。 |
| `sinkhorn(logits, iterations=20)` | 非空末尾方阵，在 log 空间先列后行归一化；有限迭代不是精确流形投影。 |

`MHCResidual(branch, width, streams=4, checkpoint_branch=False, **mhc_options)` 包装一个**不自行加残差输入**的 `nn.Module`。`checkpoint_branch=True` 在有梯度的训练中使用非重入式重算。`MHCSequential(width, branches, streams=4, **options)` 要求至少一个分支，只展开一次，最后平均 streams；参数正常参与 state_dict、自动微分与优化器。

```python
import torch
from torch import nn
import ruda_torch as r

block = r.MHCSequential(64, [nn.Sequential(nn.Linear(64, 64), nn.SiLU())], streams=4)
block = block.to('ruda:0')
x = torch.ones(2, 8, 64).to('ruda:0').requires_grad_()
y = block(x)  # [2,8,64]，分支不能再次加 x
```

## DSA lightning indexer

`LightningIndexer` 与 `DSAIndexer` 为别名。width 为正，默认 `num_heads=4, head_dim=16, query_dim=None, topk=32, query_chunk_size=32, key_chunk_size=128, external_keys=False, detach_inputs=True, rope_dim=0, eps=1e-6`；省略 query_dim 时使用 width。RoPE 维度为非负偶数，不超过 head_dim。

输入 `[B,T,width]`、可选 query latent `[B,T,query_dim]`、准备好的 keys `[B,S,head_dim]`。`project_keys(x, pos=None)` 生成 token keys，`external_keys=True` 时由调用者提供。`scores` 为诊断／warm-up 生成稠密 `[B,T,S]`；选择路径则分块扫描 query／key，限制中间存储但不消除二次方评分计算量。

`select` 返回离散 top-k int64 索引，无效槽位为 `-1`。同设备 Bool mask：`allowed [B,T,S]`、`key_valid [B,S]`、`query_valid [B,T]`；绝对 `query_positions [T]`、`key_end_positions [S]` 为 int64。只有结束位置不晚于 query 的条目可见。`forward(..., causal=True)` 返回 `IndexerOutput(indices,scores,valid)`，valid 为 `indices >= 0`；外部压缩 keys 须提供真实绝对结束位置。`causal=False` 明确使用非因果／交叉注意力选择。

Top-k 决策没有梯度；默认 detach 输入特征，但 indexer 参数可通过重算的已选评分和 `indexer_kl_loss(scores, teacher, valid=None)` 训练。teacher 为非负 `[B,T,S]` 或 `[B,T,H,S]`，detach 后汇总 head 质量，空行贡献零。`distillation_loss(..., indices=None)` 使用稠密评分，传入 indices 时重算已选评分。辅助损失须明确加入目标，语言模型损失本身不训练离散选择器。

## CSA、HCA 与压缩参数

`CompressedSparseAttention`／`CSA(width,num_heads,**options)` 组合重叠的可学习分块压缩、DSA 选择与局部窗口。`HeavilyCompressedAttention`／`HCA` 使用不重叠压缩，覆盖全部因果可见压缩条目，不含 indexer。两者对局部与压缩条目使用同一个 softmax，只暴露已经完成的压缩块。

| 选项 | 默认／限制 |
| --- | --- |
| `head_dim` | `width/num_heads`，省略时 heads 须整除 width。 |
| `compress_ratio` | CSA 4、HCA 128，正整数。 |
| `topk`、`window_size` | CSA 选 32 个压缩条目，局部窗口 128。 |
| `query_rank` | width，正的 query bottleneck 宽度。 |
| `index_heads`、`index_dim` | CSA indexer 默认 4、16。 |
| `rope_dim`、`rope_base` | 0、10000；维度为偶数并容纳于 attention／indexer 通道，是基础 RoPE 而非 YaRN。 |
| `output_groups`、`output_rank` | 1、`width/output_groups`；groups 整除 heads，rank 为正。 |
| `query_chunk_size`、`key_chunk_size` | 32、128，正整数；key 分块控制 CSA indexing。 |
| `attention_sink`、`eps` | True、`1e-6`；sink 是可学习的分母贡献，不是另一个 KV 值。 |

`forward(x, valid_mask=None, return_aux=False, indexer_warmup=False)` 输入非空 `[B,T,width]`，可选同设备 Bool `[B,T]` mask，True 表示有效位置；输出特征形状相同。`return_aux=True` 返回 `AttentionOutput(output,indexer_loss)`。`indexer_warmup=True` 关注全部可见压缩条目，通过 detach 的注意力质量训练 indexer，仍有稠密压缩注意力计算量；仅 warm-up indexer 时只对辅助损失反传。HCA 辅助项为不连接主模型梯度的零。

`LearnedKVCompressor(width,head_dim,ratio,overlap=False,eps=1e-6,...)` 对 `[B,T,width]` 返回 `(compressed,valid)`，条目数 `floor(T/ratio)`。不输出未完成 tail，全无效块为零；overlap 使用不同的前块／当前块学习路径。`RotaryEmbedding(rope_dim,base=10000.,device=None)` 对 `[B,T,D]` 或 `[B,T,H,D]` 的尾部通道及 `[T]` positions 旋转；非持久 frequency buffer 在类型搬移时保持 FP32，`inverse=True` 逆旋转。

## Prefill、decode 与缓存所有权

缓存操作同时要求 **eval() 和 no_grad()／inference_mode()**；训练使用可微分的全序列 forward。`attention.forward_cached(x,cache=None,valid_mask=None)` 接受任意非空 chunk，返回 `(output,new_cache)`，下个 chunk 使用返回的 cache，不与其他层混用。

```python
attention = r.CSA(64, 4, compress_ratio=4, window_size=16).to('ruda:0').eval()
with torch.no_grad():
    prefill, cache = attention.forward_cached(x[:, :6].detach())
    decode, cache = attention.forward_cached(x[:, 6:].detach(), cache)
    retained_bytes = cache.tensor_bytes
```

`CompressedAttentionCache` 保存 owner、参数版本、已见位置、压缩历史／有效性、可选 index keys、最后 `window_size-1` 个局部条目和 compressor tails。参数、层、batch size、dtype 或设备改变会拒绝缓存复用，重新从 `cache=None` 开始；更新为函数式，原 cache 不原地改变。

`cache.reorder(batch_indices)` 用同设备一维 int64 索引返回重排／分叉 beam 缓存。`tensor_bytes` 是逻辑保留张量字节，不是分配器峰值或固定容量，压缩历史仍随序列增长。`LearnedKVCompressor.append(x,valid_mask=None,state=None)` 同样返回压缩新增条目、有效性与新 `CompressionState`；克隆 tail，避免小 tail 保留整个 prompt 的分配。

## 混合模型与损失

`MHCTransformerBlock(width,num_heads,streams=4,attention_kind='csa',feedforward_width=None,sinkhorn_iterations=20,eps=1e-6,**attention_options)` 对 `[B,T,streams,width]` 用各自独立 mHC 连接的 attention／门控 FFN，保持形状。FFN 默认宽度为 `4*width`，attention_kind 为 csa 或 hca。

`HybridAttentionLanguageModel(vocab_size,width,num_heads,num_layers,streams=4,csa_ratio=4,hca_ratio=128,tie_embeddings=False,...)` 交替使用 CSA／HCA，非空 int32／int64 `[B,T]` tokens 输出 `[B,T,V]` logits，可选 Bool `[B,T]` valid mask。`return_aux=True` 额外返回 indexer loss 之和。尺寸由调用者选择，不是已发布大模型超参；整个模型不提供合并的 `forward_cached`，该方法属于单个注意力层。

`next_token_loss(logits,tokens,valid_mask=None)` 对源／目标位置**都有效**的配对平均错位交叉熵，长度一或全无效时为连接计算图的零。被 mask 的 token 位置也须含合法词表 ID，不能用 `-100`；显式忽略标签使用[微调分块损失](finetuning.md)，两者监督契约不同。

## Python Muon 与明确的 AdamW 排除项

`Muon(params,lr=0.02,momentum=0.95,weight_decay=0.,nesterov=True,momentum_mode='sgd',dampening=0.,ns_steps=5,ns_coefficients=(3.4445,-4.775,2.0315),eps=1e-7,adjust_lr='original',matrix_layout='as_stored',stable_normalization=True,flatten=False,max_grad_norm=None)` 为单设备 eager 优化器，梯度须为稠密、同 shape／dtype／device。Muon 矩阵须二维，显式 `flatten=True` 才将更高维参数整理为 `[首维,-1]`。

- ns_steps 为 1–99，三个系数有限；`muon_orthogonalize` 是有限次 quintic Newton–Schulz，不是精确极分解。半精度／BF16 主参数与动量为 FP32。
- momentum_mode 为 sgd／ema；Nesterov 要求正 momentum、零 dampening，EMA 也要求零 dampening。
- original 调整系数为 `sqrt(max(1,rows/cols))`；match_rms_adamw 为 `0.2*sqrt(max(rows,cols))`。input_output 只交换缩放时的行／列语义，不转置参数存储。
- 可选正 max_grad_norm 内部裁剪反缩放后的梯度，存储梯度保持不变。有限性回读使 step 不可图捕获；非有限活跃梯度／拟更新值跳过整个更新，不推进 decay／计数，设备提交失败不是事务。
- 状态保存主参数、动量、算法及参数版本，建状态后外部改参须重置／加载匹配状态。分布式同步之后才正交化**完整矩阵**，不能对局部 TP／FSDP shard 分别当成完整矩阵。

`MuonAdamW.from_model` 接收模块对象，不猜测名称：

```python
model = r.HybridAttentionLanguageModel(256, 64, 4, 2, hca_ratio=8).to('ruda:0')
optimizer = r.MuonAdamW.from_model(
    model, muon_modules=list(model.layers),
    adamw_modules=[model.embedding, model.head], lr=0.02, adamw_lr=0.001,
)
```

只有指定的可训练矩阵进入 Muon，embedding、显式排除项及其余可训练参数使用 AdamW。共享参数只出现一次，AdamW 排除优先；没有合格指定矩阵会报错。手动构造时各组须显式 `use_muon=True/False`。保存模型、优化器与可选 scaler，不混用 Rust Muon Record 和 Python state 格式。

源码：[mHC](../../ruda-torch/python/ruda_torch/mhc.py)、[索引／压缩／缓存](../../ruda-torch/python/ruda_torch/sparse_attention.py)、[混合模型](../../ruda-torch/python/ruda_torch/hybrid_model.py)、[Python Muon](../../ruda-torch/python/ruda_torch/optim.py)。已有[组合示例](../../ruda-torch/python/examples/train_hybrid_attention.py)与真实预训练权重微调、性能 baseline 是不同任务。
