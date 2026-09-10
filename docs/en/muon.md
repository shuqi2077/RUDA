# Muon and explicit Muon + AdamW groups

**English** | [简体中文](../zh/muon.md) | [日本語](../ja/muon.md) | [Deutsch](../de/muon.md) | [Русский](../ru/muon.md)

The existing `MuonConfig`, `Muon`, `MuonState` implementation is extended, not duplicated.
See [中文完整契约](../zh/muon.md) for the detailed scope.

## Usage

```rust,ignore
use ruda_optim::{AdamWConfig, MuonAdamWConfig, MuonConfig, MuonMatrixLayout, MuonMomentumMode};
let mut optimizer = MuonAdamWConfig::new()
    .with_muon(MuonConfig::new()
        .with_momentum_mode(MuonMomentumMode::Ema)
        .with_stable_normalization(true)
        .with_matrix_layout(MuonMatrixLayout::InputOutput))
    .with_adamw(AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01))
    .init(&model, &[model.hidden.weight.id])?;
model = optimizer.try_step_with_lrs(0.02, 0.0003, model, gradients)?;
```

Select hidden nonempty **complete 2D matrices** explicitly. Embeddings, classifier heads,
biases and normalization parameters should normally use AdamW, even when they are 2D.
All unselected parameters use the existing high-level AdamW implementation, NOT the optional fused-AdamW interface.
The example learning rates are not tuning recommendations.

`Optimizer::step(lr, ...)` uses `lr` for Muon and `lr * adamw_lr_ratio` for AdamW (ratio default 0.015).
`try_step_with_lrs` allows independent schedules. Missing gradients skip both decay and momentum.
`try_step_or_skip(..., true)` skips both groups without updating state.
Metadata is checked for both groups first, but device failures remain asynchronous: this is not a device transaction.
Tied parameters are routed once by ParamId; callers must not represent overlapping weights with unrelated IDs.

## Numerical choices and migration

Existing constructor defaults preserve SGD momentum, tensor-dtype normalization and AsStored scaling.
EMA mode is opt-in: `m = beta*m + (1-beta)*g`, zero initialized; Nesterov uses `(1-beta)*g + beta*m`.
Do not interchange EMA and legacy SGD checkpoint buffers without a deliberate conversion.
The finite quintic Newton-Schulz polynomial is not an exact polar decomposition; an exact identity-matrix assertion is invalid.

Stable normalization first scales by the maximum magnitude before forming the sum of squares.
It is opt-in because rounding changes, and explicitly requires FP32 inputs/state.
It does not protect against overflow in every other stage and does not inspect nonfinite values.
No implicit BF16 cast is performed. Native PyTorch Muon uses BF16 for NS; the FP32 variant is not bit-equivalent.
No automatic FP32 master weights or low-precision model-copy update is provided.

RUDA Linear stores `[input, output]`, so use InputOutput for Original LR scaling when appropriate.
AsStored interprets rows as outputs. MatchRmsAdamW is symmetric under transpose.
Weight decay uses the original, not shape-adjusted, learning rate.
One configuration covers the selected Muon group; mixed logical layouts need separately configured optimizers.

The project's Config macro does not supply serde defaults for new fields.
Old JSON configs must add `"momentum_mode":"Sgd"`, `"stable_normalization":false`,
`"matrix_layout":"AsStored"` to preserve the old choices. Programmatic defaults remain available.
Simple Muon tensor-record layout is unchanged. Mixed records include a version, configuration,
parameter identity/shape/dtype manifest and both optimizer states. Restore the model with original IDs first.
FullPrecisionSettings is needed for precise continuation comparisons. Changed grouping/configuration is rejected.

Unscale and check gradients externally before updating, and synchronize skip decisions across replicas.
This patch does not automatically integrate the prior low-level gradient-guard API.
Implicit `step_multi`, Ruda distributed-marked tensors, FSDP/TP shards, sparse gradients and 4D convolution reshaping
are not implemented. Never orthogonalize arbitrary shards as though they were the full matrix.

## Validation

```bash
cargo run --release --locked -p ruda-optim --example muon-training -- 20
python tools/run_muon_regressions.py --suite oracle
python tools/run_muon_regressions.py --suite reference
python tools/run_muon_regressions.py --suite host
python tools/run_muon_regressions.py --suite build
python tools/run_muon_regressions.py --suite cuda --compiler both
```

The example uses the Host tensor backend unless built with test-cuda. It is not a performance benchmark.
The reference suite compiles an independent scalar oracle with rustc only; it does not validate RUDA execution.
The host and CUDA suites compile and run real RUDA tensor/group tests. Build includes a no-default-features check.
Missing tools are BLOCKED; no installation/fallback is attempted. Each command has an explicit timeout and separate log.

References: [Muon authors](https://github.com/KellerJordan/Muon),
[PyTorch official interface](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html),
[fixed v2.9 source](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py).
