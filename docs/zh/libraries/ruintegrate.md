# ruINTEGRATE：数值积分与常微分方程

[计算库](README.md) · [English](../../en/libraries/ruintegrate.md)

**状态：实验性源码。默认只有 CPU FP64，不提供 GPU 后端。**
`ruintegrate` 是与训练框架、GPU 驱动无依赖的数值库，不隐式下载设备张量。

## 能力

**一维有限区间积分**：`integrate` 使用 Gauss–Kronrod 15/7 点规则，估计误差最大的子区间先二分。它可以计算曲线下的面积，例如高斯函数积分。支持反向区间、零长度区间、绝对/相对容差、子区间和函数调用预算。

**非刚性常微分方程初值问题**：`solve_ivp` 使用 Dormand–Prince 5(4)，根据四阶/五阶结果差异调整步长。输入是“变化率如何由当前状态决定”，输出状态随时间如何变化；例如弹簧振子的位置和速度。

后续[数值扩展](science-extended.md)已写入一阶隐式 BDF1、有限差分 Jacobian、事件定位和无限区间变换。仍不是完整 SciPy/GSL：不包含偏微分方程网格、奇异/振荡积分专用方法、公开稠密轨迹插值、复数 ODE 或 ODE 自动微分。

## 例子

```rust
use ruintegrate::{integrate, solve_ivp};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = integrate(|x| Ok((-x*x).exp()), -2.0, 2.0, Default::default())?;
    if !q.converged() { return Err(format!("integration stopped: {:?}", q.status).into()); }
    println!("integral={} estimated_error={}", q.integral, q.estimated_absolute_error);

    // 位置'=速度，速度'=-位置。
    let ode = solve_ivp(|_, y, dy| {
        dy[0] = y[1];
        dy[1] = -y[0];
        Ok(())
    }, 0.0, 6.0, &[1.0, 0.0], Default::default())?;
    if !ode.reached_end() { return Err(format!("ODE stopped: {:?}", ode.status).into()); }
    println!("state={:?}", ode.state);
    Ok(())
}
```

## 数值与资源契约

误差都是**估计值而非数学保证**。有限采样可能遗漏窄峰、尖点或振荡；知道奇点或不连续点时应主动分段或改用专用方法。不接受非有限边界、非有限函数值及无法表示的中间结果；极小区间半宽下溢会报错，而不是误报积分为零。

积分收敛条件：总估计误差 <= `max(atol, rtol*abs(integral))`。每个区间最初 15 次函数调用，每次二分新增 30 次。本版为减少误差累计，重新汇总当前分区结果；极多子区间的汇总开销尚未优化。

RK45 使用每个分量的 `atol + rtol*max(abs(y),abs(y_next))` 作尺度，再取最大归一化误差。
接受步使用五阶结果；拒绝步不修改已提交状态。缓存接受终点的导数，首次调用后每次尝试增加六次函数计算。

回调可能在失败/被拒绝的试探点调用，因此必须是固定、无训练状态推进等副作用的数学函数。
ODE 回调必须覆盖全部导数项；缓冲区先填入 NaN，漏写会报错。运行超过预算不是成功：必须检查 report 状态。

默认只保存最终状态，复用导数与暂存缓冲区。启用 `save_trajectory` 时受 `max_output_points` 上限约束，达到上限返回 `OutputLimit`，不会无限增长。
支持向前/向后积分和最大步长。`min_step` 为用户的正步长下限，最后一步到终点可更短；因浮点精度无法推进时返回 `StepTooSmall`。

```bash
cargo run --locked -p ruintegrate --example integrate-demo
```

RK45 原理对照 [SciPy RK45](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html)。
Gauss–Kronrod 节点/权重及规则可参考 [Netlib QUADPACK](https://www.netlib.org/quadpack/)。
本次实现独立编写，不复制这些项目源码；不宣称与它们相同的选择策略、接口或逐位数值。
