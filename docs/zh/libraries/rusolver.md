# ruSOLVER：线性方程与矩阵分解

[计算库](README.md) · [English](../../en/libraries/rusolver.md)

**状态：实验性源码。**
默认执行位置为 CPU，使用 FP64。`tensor`/`cuda` 是显式开启的设备路径，不把 CPU 结果冒充 GPU 结果。

## 范围

| 能力 | API | 当前执行范围 |
|---|---|---|
| 带部分行主元的 LU 分解 | `Lu::factor` | CPU FP64，非空方阵 |
| 方程求解、转置求解、显式求逆、符号/对数行列式 | `Lu::solve/solve_transpose/inverse/slogdet` | 可复用分解、多个右端项 |
| 对称正定矩阵 Cholesky | `Cholesky::factor/solve/log_determinant` | CPU FP64，检查对称性，不悄悄加正则项 |
| 带列主元的 Householder QR 与最小二乘 | `Qr::factor/least_squares` | CPU FP64，行数不小于列数；求解要求满列秩 |
| 实对称矩阵特征分解 | `symmetric_eigen` | CPU FP64，循环 Jacobi；返回升序特征值 |
| 预条件共轭梯度 | `conjugate_gradient` | CPU FP64，调用者保证算子及预条件器对称正定 |
| 已有 ruSPARSE CSR 接入 | `sparse::CsrF32Operator` | `sparse` feature；FP32 系数显式提升到 FP64 计算 |
| 小矩阵批量 Cholesky + 方程求解 | `tensor::cholesky_solve_batched` | `tensor` feature，原生 FP32 设备 kernel |

后续[数值扩展](science-extended.md)已经写入 SVD/最小范数解、复数分解、稀疏 LU、分布式 CG、主机自动微分及额外 GPU 求解路径；不包含完整 LAPACK/cuSOLVER API 或分布式直接矩阵分解。没有替换原 `ruda-tensor::api::linalg::lu`；这里提供独立的、可复用因子的低层求解接口。

## 基本用法

```rust
use rusolver::{Matrix, Lu, relative_residual};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Matrix::new(2, 2, vec![4.0, 1.0, 1.0, 3.0])?;
    let b = Matrix::new(2, 1, vec![6.0, 7.0])?;
    let factor = Lu::factor(a.view(), Default::default())?;
    let x = factor.solve(b.view())?;
    println!("x = {:?}", x.values()); // 约 [1, 2]
    println!("relative residual = {}",
        relative_residual(a.view(), x.values(), b.values())?);
    Ok(())
}
```

`A x = b` 是已知 A、b，求 x。不要为了求 x 先构造 A 的逆矩阵：`solve` 直接使用因子。
多个右端项放在 B 的多列中；同一个 A 可先分解一次，再多次 `solve`。这是减少重复工作，不是已实测加速比。

```bash
cargo run --locked -p ruda-solver --example solver-demo
cargo run --locked -p ruda-solver --features sparse --example sparse-poisson
```

## 存储、精度与失败语义

`Matrix` 是非空、行优先、有限值的自有 FP64 矩阵。`MatrixView::strided` 接受不可变的正步长 view，可显式转置；不隐式下载设备张量。
FP32 输入通过 `Matrix::from_f32` 明确提升；这不会增加原始数据已有的有效精度。

LU 约定为 **P A = L U**，`pivots[k]` 为第 k 步交换的行。QR 约定为 **A[:, permutation] = Q R**，Q 为薄矩阵。不要把不同库的 P 定义直接混用。

`Tolerance` 使用 `max(absolute, relative * scale)`。LU 的 scale 是 A 的最大绝对元素；默认 relative=1e-12，因此病态但形式上可逆的矩阵可能被判为数值奇异。调整容差不会自动修复病态性。本版不估计条件数。QR 用初始最大列范数作为秩阈值尺度，秩亏输入返回 `RankDeficient`，不会伪装成最小范数解。

Cholesky 和特征分解检查整个输入的对称性，容差内的差异按下三角定义有效对称矩阵。Cholesky 默认拒绝非正主元，不隐藏添加 jitter。允许配置额外正主元阈值。

范数用缩放平方和；内积使用补偿求和。但并非任意精度算法：若乘积、中间和、解或实际范数超出 FP64 表示范围，会报错。CG 对极端尺度的内积也可能下溢并明确返回 breakdown；需要调用者缩放/平衡问题。

CG 默认每 32 步重算真实残差 `b-Ax`，在准备报告收敛前也重算；替换残差时重启方向。返回 `MaxIterations` 不表示成功。对正曲率的检查不能证明任意 `LinearOperator` 一定对称正定。矩阵自由算子和预条件器必须在整个求解中保持固定，并完整覆盖输出缓冲区。

## 设备路径

```bash
# CUDA toolkit、驱动与 NVIDIA GPU 环境。
RUDA_CUDA_COMPILER=nvrtc cargo run --locked -p ruda-solver --features cuda --example solver-cuda
RUDA_CUDA_COMPILER=ptx cargo run --locked -p ruda-solver --features cuda --example solver-cuda
```

入口接收 `RudaTensor<R>`：A `[batch,n,n]`，B `[batch,n,nrhs]`，限制 n=1..32、nrhs=1..8，未量化 FP32、行优先连续布局、同设备且同执行队列。空 batch 不提交 kernel。

一个 GPU 线程顺序处理一个系统，多个系统在 batch 中独立并行。这是小矩阵的功能基线，**不是**针对大矩阵的 blocked/warp-cooperative 高吞吐 Cholesky；没有宣称比 cuSOLVER 快。
非空 batch 提交一个 kernel，融合数值检查、Cholesky 和所有 RHS 求解。输入只读，输出新分配 factor、solution、info。算法中的临时矩阵写在设备缓冲区中，因此仍有设备内存访问，不应称作“全寄存器求解”。

`BatchedCholeskyOptions::diagonal_shift` 显式指定求解 `(A+shift I)X=B`；默认 0。
`info` 每个系统一个 I32：0 成功，正数为从 1 开始的失败主元，-1 输入非有限，-2 非对称，-3 算术非有限。失败系统的 factor/solution 清零，但这些零不是有效解。
**调用 `check_status_sync()` 后再认为结果有效**；只返回提交对象不表示设备已成功完成。该检查会同步并读回 status，不计算主机替代结果。

这一路没有自动微分、低精度混合求解、跨卡分解或默认框架接入。

## 算法参考

这是根据公开数学方法独立写出的实现；未复制 LAPACK/cuSOLVER/GSL 源码，不宣称与其实现或数值舍入逐位相同。

- [LAPACK GETRF](https://netlib.org/lapack/explore-html/db/d04/group__getrf_gaea332d65e208d833716b405ea2a1ab69.html)：LU 部分主元的职责；本库不是其 blocked BLAS-3 实现。
- [LAPACK QR 列主元](https://www.netlib.org/lapack/lug/node42.html)：QR 与最小二乘；本版明确拒绝秩亏解。
- [LAPACK POTRF](https://www.netlib.org/lapack/explore-html/d2/d09/group__potrf.html)：对称正定分解。
- [Templates for Linear Systems](https://www.netlib.org/templates/templates.html)：CG 的适用条件与停止判据。
