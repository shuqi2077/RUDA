# 梯度检测、全组 L2 裁剪与融合 AdamW（实验性）

新功能默认关闭，原来的 `adamw_step` 签名和模型优化器默认分派不变。

## 本次补上的能力

扫描一组已累积的梯度，在 FP32 中反缩放，检查原始值和反缩放结果是否包含 NaN/Inf，
计算这组梯度「当作拼成一个向量」的 L2 范数，再把同一个裁剪系数直接并入 AdamW。
不生成一份完整的「反缩放／裁剪后梯度」，也不单独启动裁剪 kernel。

例如两个参数张量的梯度分别是 `[3]` 和 `[4]`，全组范数为 5，而不是分别裁剪成 1。
限值为 1 时，两者使用约 0.2 的共同系数。

顺序固定为：存储精度转 FP32 → 乘以 loss scale 的倒数 → 非有限值检查与范数 → 裁剪 → AdamW。
FP16/BF16 梯度不会在裁剪后再舍入回半精度；这是 FP32 master 更新契约，不承诺与
原位裁剪半精度梯度逐位一致。

- `max_norm=None`：不裁剪，仍检查非有限值。
- `max_norm=0`：有效梯度归零，但动量、权重衰减和步数仍更新。
- `NonFinitePolicy::Skip`：一个张量有坏梯度，整个所选组都跳过，不更新权重衰减和步数。
- `NonFinitePolicy::Error`：在任何参数更新 kernel 提交前返回错误。
- `StepControl::skip_update=true`：保留原来的外部跳步，不扫描、不分配设备输出。

整个组的形状、类型、布局、设备／队列、状态和步数溢出在扫描前统一检查。
参数和梯度均按只读输入处理；每个真实参数只应传入一次，绑定权重不自动去重。

## 接口与开关

`gradient-guard` 提供主机策略和独立参考公式；`gradient-guard-device` 增加设备实现；
`gradient-guard-cuda` 增加 CUDA 测试与基准。它们均是 `ruda-optim` 的可选 feature。

```rust,ignore
use ruda_optim::fused_adamw::{
    AdamWEntry, AdamWOptions, StepControl, guarded_adamw_step,
    gradient_norm::GradientGuardOptions,
};
let entries = [
    AdamWEntry { parameters: &master1, gradients: &grad1, state: state1.as_ref() },
    AdamWEntry { parameters: &master2, gradients: &grad2, state: state2.as_ref() },
];
let pending = guarded_adamw_step(
    &entries, &AdamWOptions::default(),
    StepControl { gradient_scale: 128.0, skip_update: false },
    GradientGuardOptions { max_norm: Some(1.0), ..Default::default() },
)?;
// 本调用已等到统计结果回到主机，但参数更新仍是异步的。
// 确认设备完成成功后再提交 pending 中的参数、状态或保存 checkpoint。
```

本次不自动更新 loss scale，不安装高层优化器适配器，不自动把 master 转回模型低精度。
统计是**单设备／单队列所选参数组**的范数，不是分布式全局范数；不能把 FSDP 的一个局部
分片范数当成全模型范数，也没有跨 rank 的一致跳步协议。

## 性能设计和代价

设备采用稳定的 `(最大尺度, 归一化平方和, 非有限标记)` 归约。
避免直接在 FP32 中计算 `1e30 * 1e30`，不会因此把有限梯度误报成范数溢出。
固定 256 线程、3072 字节共享内存；每个非空张量 1～2 次归约 kernel，最多 1024 个中间三元组。
中间区最多 12 KiB，另有 12 字节最终统计；不包含分配器对齐和元数据。

所有张量的归约先提交，再通过一次批量读回 API 回传每张量 12 字节。主机用 FP64 合并并
作出裁剪／跳步决定。批量 API 不等于硬件只有一次 DMA。

与附带的「相同归约 → 独立反缩放／裁剪 → FP32 临时梯度 → 融合 AdamW」基线相比，
每个非空张量减少一次裁剪 kernel、一个 `4×元素数` 字节临时缓冲区，以及 `8×元素数`
字节的逻辑临时读写。**这是源码计数，不是 GPU 性能测试，也不是实测显存流量。**
原有无 guard 的 AdamW 本来不做这些检查；增加 guard 可能更慢。

目前每步显式等待统计读回主机，不支持 graph capture，不承诺无主机同步或计算通信重叠；
许多小张量可能明显受启动和同步开销影响。稳定归约也比直接平方求和多算术。
没有宣称比 PyTorch fused/foreach AdamW 更快。原来的非原地参数／动量输出分配保持不变。

浮点累加顺序和非正规数处理受后端影响，测试使用容差而非逐位等价。
禁用裁剪时，巨大但有限的梯度仍可能让 AdamW 二阶动量溢出；本功能不扫描旧参数或动量。
输入别名不得在扫描／更新期间从其他队列改写。运行时故障不等于可自动回滚的组事务；
只有参数校验和非有限值跳步保证在更新前决策。

## 验收命令

```bash
python tools/run_gradient_guard_regressions.py --suite oracle
python tools/run_gradient_guard_regressions.py --suite reference
python tools/run_gradient_guard_regressions.py --suite host
python tools/run_gradient_guard_regressions.py --suite build
python tools/run_gradient_guard_regressions.py --suite cuda --compiler both
python tools/run_gradient_guard_regressions.py --suite bench --compiler both --elements 65536 --tensors 4 --dtype bf16 --amsgrad
```

`oracle` 是 NumPy 归约模拟、FP64 独立范数和真实 PyTorch CPU 公式对照，**不是运行 RUDA**。
`reference` 使用 rustc 编译不依赖 Cargo 仓库的独立测试；`build` 检查新旧两种 feature；
`cuda` 在 NVRTC/直接 PTX 两条路径跑旧 AdamW 回归和新测试。缺工具记 blocked，超时或失败
单独记录。`--dry-run` 只列命令；`--offline` 需要本地已缓存依赖，不自动安装环境。

基准先预热，再交替跑两条路径；计时包含分配、归约、统计读回、主机决策、提交和设备同步。
每批都比较参数及所有动量，比较读回不计入时间。首选小规模配置，确认正确和资源用量再扩大。

详细契约及外部算法参考见 [英文说明](../en/gradient-guard.md)。
