# Muon 与显式 Muon + AdamW 参数分组

[English](../en/muon.md) | **简体中文** | [日本語](../ja/muon.md) | [Deutsch](../de/muon.md) | [Русский](../ru/muon.md)

## 已有内容与本次补充

基线已经包含 `ruda-optim/src/optim/muon/mod.rs`，公开 `MuonConfig`、`Muon`、`MuonState` 和两种学习率缩放。
这次不是重复实现一个新库，也不是声称从零创造 Muon。保留已有 Tensor/矩阵乘法路径，补上：

- `MuonError`、`validate`、`try_build`、`try_init`、`validate_step`、`try_step`；阻止形状广播掩盖梯度错误。
- 显式 `MuonMomentumMode::Ema`，以及 FP32 的稳定范数计算选项。
- `MuonMatrixLayout::InputOutput`，适配 RUDA Linear 的逻辑矩阵方向。
- `MuonAdamWConfig` / `MuonAdamW`：隐藏层矩阵显式选择，其余参数使用现有高层 AdamW。
- 分组记录、配置/参数身份/形状/类型检查、整组跳步、缺失梯度跳过、训练示例和专项测试。

## 什么参数应该用 Muon

Muon 对隐藏层权重的矩阵更新做有限次 Newton–Schulz 多项式迭代。这里不是求精确的正交矩阵，
不能把 `U U^T` 必须接近单位阵当作所有输入的正确性标准。本次替换了旧测试中这种过强假设。

需要主动选出隐藏层的 **完整、非空二维矩阵**。Embedding、输出 head、bias 和归一化参数通常交给 AdamW。
不能只按 `ndim == 2` 自动选择，因为 embedding 和输出 head 也是矩阵。

## 推荐的单设备 FP32 使用方式

```rust,ignore
use ruda_optim::{
    AdamWConfig, GradientsParams, MuonAdamWConfig, MuonConfig,
    MuonMatrixLayout, MuonMomentumMode, Optimizer,
};

let mut optimizer = MuonAdamWConfig::new()
    .with_muon(MuonConfig::new()
        .with_momentum_mode(MuonMomentumMode::Ema)
        .with_stable_normalization(true)
        .with_matrix_layout(MuonMatrixLayout::InputOutput))
    .with_adamw(AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01))
    .init(&model, &[model.hidden.weight.id])?;

let gradients = GradientsParams::from_grads(loss.backward(), &model);
model = optimizer.try_step_with_lrs(0.02, 0.0003, model, gradients)?;
```

`try_step_with_lrs` 的两个学习率相互独立。它们是示例值，不是模型通用的最优超参数。
现有 `Optimizer::step(lr, ...)` 也能使用：Muon 采用 `lr`，AdamW 采用 `lr * adamw_lr_ratio`；默认 ratio 为 0.015。
原有学习率调度器因此可以调节两组的共同倍率。更新完成后的异步错误仍由设备同步接口报告。

可运行示例包含完整的自动求导过程、隐藏层和带 bias 的输出 head：

```bash
cargo run --release --locked -p ruda-optim --example muon-training -- 20
```

默认例子使用 `ruda-tensor-host`，不需要 CUDA。它只是用法与数值演示，不是性能 benchmark。

## 数值契约与兼容性

**旧模式继续保留。** `MuonConfig::new()` 保持 SGD 动量、原先的范数路径以及 `AsStored` 方向。
原来的 `build/init` 方法仍然存在；无效配置现在会提前报错/在旧接口中 panic。需要 `Result` 的调用者使用 `try_*`。

SGD 模式的首步动量为 `g`，后续为 `beta * m + (1 - dampening) * g`。
EMA 模式从零开始，动量为 `beta * m + (1 - beta) * g`；开启 Nesterov 时更新方向为
`(1 - beta) * g + beta * m`。EMA 不允许 dampening。两种动量状态不能不经转换混用。

NS 默认 5 步，系数 `(3.4445, -4.775, 2.0315)`。参数方向是高矩阵时先转置，使用较小的 Gram 矩阵，随后转回。
这一点基线已经存在，不属于本轮新增的性能成果。每一步 NS 仍有三次矩阵乘法，不是像融合 AdamW 那样单个逐元素 kernel。

稳定范数选项先用最大绝对值缩放，再平方求和。实数运算下它与 `g / max(norm(g), epsilon)` 等价，
但浮点舍入次序不同，因此默认不开启。该选项明确只接收 FP32 参数/梯度/动量。它防止计算平方和时的溢出，
不保证任意超参数或动量累加都不会溢出。

代码不会自动把 NS 转成 BF16。PyTorch 2.10 的 `torch.optim.Muon` 内部 NS 使用 BF16，
所以本实现的 FP32 路径不承诺逐位相同。也不自动创建 FP32 master 参数或把结果写回低精度模型副本。

`InputOutput` 只改变 `Original` 学习率缩放的长宽比，不改变张量形状。
例如 RUDA Linear 的 `[2,8]` 权重表示 2 输入、8 输出，Original 倍率按 `sqrt(8/2)` 取 2；
`AsStored` 的倍率则为 1。`MatchRmsAdamW` 对长宽交换对称。衰减始终使用未做形状调整的学习率。
混合多个不同逻辑布局的隐藏参数时，不要全部套同一方向配置；本轮只有一个 Muon 组和一个 AdamW 组。

**配置文件迁移：** 本仓库的 `Config` 派生宏不会为新增字段自动生成 serde 缺省值。
因此旧 Muon JSON 配置需要显式补上以下三项才能读取；原有 Tensor 动量记录结构本身没有改变：

```json
{
  "momentum_mode": "Sgd",
  "stable_normalization": false,
  "matrix_layout": "AsStored"
}
```

以上只是需要合并到原完整配置的字段片段。不要用它替换其他超参数。
新的混合优化器记录包含 schema 版本、配置标识、参数分组、形状/类型和两组状态。
必须先恢复带原 ParamId 的模型，再创建同配置的优化器并 `try_load_record`。
不同分组、配置、几何和类型会拒绝载入。需要比较精确续训时使用 FullPrecisionSettings；记录不是进程恢复器。

## AMP、分布式与跳步边界

`try_step_or_skip(muon_lr, adamw_lr, model, gradients, true)` 不更新任何参数或优化器状态。
缺失的梯度也不触发该参数的动量或权重衰减。所有输入的元数据先检查，再开始更新两组。
这不是 GPU 上的原子事务，设备执行错误仍需要上层恢复。

本轮不自动扫描 NaN/Inf，不自动反缩放梯度，不进行全组裁剪，也不管理 loss scale。
调用方必须先反缩放和检查，并在所有副本间统一跳步决定。上一轮 `fused_adamw::gradient-guard`
接收的是另一套底层设备 Tensor 接口，本补丁没有悄悄将两套接口连通。

不实现分片 Muon。矩阵正交化是非线性的，对每个 shard 单独更新再拼回通常不是完整矩阵更新。
`step_multi` 明确拒绝；带 Ruda distributed 标记的模块也拒绝。手动同步后的完整本地矩阵可以作为输入，
但本轮没有测试 DDP/FSDP/TP、跨节点一致性或状态迁移。也不自动展开卷积四维权重、处理稀疏梯度或任意多维批次。

## 验证入口

```bash
python tools/run_muon_regressions.py --suite oracle
python tools/run_muon_regressions.py --suite reference
python tools/run_muon_regressions.py --suite host
python tools/run_muon_regressions.py --suite build
python tools/run_muon_regressions.py --suite cuda --compiler both
```

- `oracle`：NumPy FP32 公式与独立 PyTorch FP64 表达式，外加未经修改的 PyTorch Muon 对照。
- `reference`：只用 rustc 编译的独立 FP64 测试，不执行 RUDA。
- `host`：编译并执行真实 RUDA Tensor/分组优化器测试。
- `build`：默认配置、no-default-features 和完整示例的构建检查。
- `cuda`：真实 CUDA Tensor 后端上分别运行 NVRTC 和直接 PTX 路径。不自动回退 CPU。

日志与源码哈希保存在独立结果目录；缺少工具记为 blocked。加 `--dry-run` 查看命令，
`--offline` 禁止 Cargo 联网，`--timeout` 是每条命令的时间上限，不是预计时长。

正式采用前先在独立分支运行 `host/build`，然后运行目标 GPU 测试及你的真实模型验证。

## 来源

- Muon 原始算法与参数选择：https://github.com/KellerJordan/Muon
- PyTorch 官方接口：https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- 固定版本参考代码：https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py
- Moonshot 的学习率缩放工作：https://arxiv.org/abs/2502.16982

本轮分组/校验/测试为新写实现；没有把第三方 Python 优化器当作设备端实现包装，也没有复制原作者的全文件。
