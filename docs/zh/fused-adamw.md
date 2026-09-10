# 实验性融合 AdamW / AMSGrad

本次在现有 `ruda-optim` 中增加可选路径，不新建重复的优化器库，也不改原来的
`AdamW`、模型优化器适配器和 checkpoint 格式。**新 Rust/GPU 路径尚未编译运行验收。**

## 增加的实际能力

一个设备 kernel 完成梯度反缩放、第一/第二动量更新、可选 AMSGrad 历史最大值、
偏差校正、解耦权重衰减和参数更新。参数与动量固定 FP32，梯度可以是 FP32、
FP16 或 BF16。支持 maximize；非连续布局、广播和不匹配的设备/执行队列明确报错。

首次更新在该 kernel 内初始化动量，不再额外清零。空张量或外部传入的
`skip_update=true` 不分配、不启动 kernel、不增加步数。该标志需要调用者提供，
这里没有自动实现 GradScaler 或梯度有限值扫描，也没有设备侧步数计数器；
不能直接把固定偏差校正系数的图捕获下来重复回放，当成多个更新步骤。

采用非原地输出保留输入及别名；每步仍需分配三份 FP32 输出，有 AMSGrad 时是四份。
这不是“零分配优化器”，大模型要计算新旧状态同时存活的显存。低精度模型权重的
回写/转换仍由调用者完成，不能称为完整混合精度训练已经验收。

## 优化点与边界

基准中的**显式分步对照实现**普通 AdamW 使用 4 个 kernel，AMSGrad 使用 5 个；
新的融合更新各使用 1 个。FP32 梯度、已有动量的情况下，源码层面数据流量模型为
48 → 28 字节/参数；AMSGrad 是 60 → 36 字节/参数。

这些是按读写次数推算的量，不是实测显存流量，也不是加速倍数。原有图融合后端
可能已经合并了 AdamW，新的代码不一定更快。因此没有自动替换旧优化器。

## 用法

启用 `ruda-optim` 的 `fused-adamw-device` feature，然后调用：

```rust
use ruda_optim::fused_adamw::{AdamWOptions, StepControl, adamw_step};

let options = AdamWOptions {
    learning_rate: 1e-3,
    amsgrad: true,
    ..Default::default()
};
let update = adamw_step(
    &master, &gradient, state.as_ref(), &options,
    StepControl { gradient_scale: 128.0, skip_update: found_inf },
)?;
master = update.parameters;
state = update.state;
```

`master` 和 `gradient` 是 `RudaTensor<R>`；`state` 初始为 `None`。
`found_inf` 是上层检查结果，不由本函数扫描生成。所有输入必须在相同执行队列。
返回只是提交成功；保存 checkpoint、判断训练步成功和统计耗时前，必须检查设备同步结果。
设备失败时丢弃尚未确认完成的输出，使用上层已经提交的 checkpoint 恢复。

## 验证顺序

```sh
python tools/run_adamw_regressions.py --suite oracle
python tools/run_adamw_regressions.py --suite reference
python tools/run_adamw_regressions.py --suite host
python tools/run_adamw_regressions.py --suite build
python tools/run_adamw_regressions.py --suite cuda --compiler both
python tools/run_adamw_regressions.py --suite bench --compiler both --elements 65536 --dtype bf16 --amsgrad
```

第一项只是 Python/NumPy 与真实 CPU PyTorch 公式对照，**不能证明 RUDA 的 Rust 或 GPU 实现正确**。
`reference` 使用独立 rustc 测试，不需要整个 Cargo 依赖树；`host` 还对照原有 Host AdamW。
CUDA 测试分别运行 NVRTC 和直接 PTX，并检验参数、动量、精度、尾部长度、跳步、状态恢复和拒绝路径。

基准先预热，使用同一份初始状态，交替执行顺序，前后同步，记录样本中位数及最小/最大值。
记录的是包含分配、主机提交和同步的每步耗时，不是仅设备执行耗时；每批测量后比较参数和动量。
缺少工具会记录 blocked，不填写虚构结果。

完整契约和限制见 [英文说明](../en/fused-adamw.md)。
