# RUDA 数值科学扩展：执行范围与使用指南

扩展原有 ruSOLVER、ruINTEGRATE，
并通过可选 `solver-host` 功能接入现有 ruda-autodiff；不另建 Tensor/自动微分引擎。

## 支持矩阵

| 能力 | 执行位置 | 当前实现和边界 |
|---|---|---|
| SVD | CPU，实 FP64 | 缩放单边 Jacobi，薄分解；高/宽/秩亏矩阵；伪逆、最小范数多 RHS 求解 |
| 复数分解 | CPU，每分量 FP64 | 部分行主元 LU/伴随求解，Hermitian Cholesky，无列主元 Householder QR/满列秩最小二乘 |
| 稀疏直接求解 | CPU，实 FP64 | 稀疏行 BTreeMap LU，行主元、填充预算、多 RHS/转置求解；不转为稠密因子 |
| 分布式求解 | CPU，实 FP64 | 均衡连续行分片 CSR 共轭梯度，Jacobi 预条件；向量 all-gather，标量全局归约；可适配已有 ruCCL TCP |
| 自动微分公式 | CPU，实 FP64 | solve、Cholesky、薄 SVD（含向量）、对称特征分解（含向量）、固定列主元的 QR 反向拉回 |
| 自动微分图 | Ruda `Autodiff<Host>`，FP64 | solve、Cholesky、奇异值、对称特征值接入现有图；一阶反向，不接受 GPU tensor |
| GPU LU + solve | 设备，FP32 | A=[batch,n,n]，B=[batch,n,nrhs]；n≤32，nrhs≤8；部分行主元 |
| GPU QR | 设备，FP32 | [batch,m,n]，1≤n≤32、n≤m≤64；薄 Householder，无列主元 |
| GPU 特征分解 | 设备，FP32 | 实对称，n≤32；循环 Jacobi，升序特征值；有收敛上限 |
| GPU 共轭梯度 | 设备，FP32 | 稠密 SPD，n≤128、单 RHS、零初始猜测；可选 Jacobi，真实残差复核 |
| 刚性 ODE | CPU，FP64 | 自适应隐式 Euler/BDF1，Newton + 回溯；解析或有限差分 Jacobian；默认最多256维 |
| 事件定位 | CPU，FP64 | RK45/BDF1 接受步端点 Hermite 插值 + 二分；方向、多事件、终止事件、预算 |
| 无限区间积分 | CPU，FP64 | 半无限/全实轴有理变换 + 原有 GK15；全轴两尾分别积分，不当作柯西主值 |

FP64 是64位浮点数。`Complex64` 的 **64 指每个分量**，总存储128位，不等同 CUDA 的
单精度复数类型。SPD 指实对称正定矩阵。RHS 是方程右端项；多 RHS 表示同一个 A 求多个 b。

## 1. SVD / 伪逆

```rust
use rusolver::{Matrix, Svd};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = Matrix::new(2, 3, vec![1.0,0.0,0.0, 0.0,2.0,0.0])?;
let factor = Svd::factor(a.view(), Default::default())?;
let b = Matrix::new(2, 1, vec![1.0,4.0])?;
let x = factor.solve(b.view())?; // [1,2,0]，欠定系统的最小范数解
let pinv = factor.pseudo_inverse()?;
# Ok(()) }
```

返回 U=[m,k]、s=[k]、VT=[k,n]，k=min(m,n)。不构造完整 AᵀA。
默认截断低于 `max(absolute, relative * ||A||F)` 的奇异值，因此可能是明确的截断近似。
绝对阈值按输入单位定义。被截断的零空间使用确定性正交补全。
达到扫描预算仍未正交时返回 NonConvergence，而不是冒充收敛。
不能保证任意病态矩阵的所有微小奇异值都获得高相对精度。

## 2. 复数 / 稀疏

```rust
use rusolver::complex::{Complex64 as C, ComplexMatrix, ComplexLu};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = ComplexMatrix::new(2,2,vec![C::new(2.0,1.0),C::ONE,C::ZERO,C::new(3.0,-1.0)])?;
let b = ComplexMatrix::new(2,1,vec![C::ONE,C::new(2.0,0.0)])?;
let lu = ComplexLu::factor(&a, Default::default())?;
let x = lu.solve(&b)?;
let y = lu.solve_adjoint(&b)?; // Aᴴ y=b，不是只做普通转置
# Ok(()) }
```

`SparseLu::factor_csr` 接收零基 CSR，允许无序列索引和重复项（相加），因子一直采用稀疏行。
`max_factor_nonzeros` 限制存储的因子项数，不是进程总字节预算；BTreeMap 和工作区另有开销。
此实现没有列重排序、超节点或符号分解复用；行主元搜索仍可能有平方级开销，填充也可能很大。
`--features sparse` 可用 `SparseLu::from_rusparse` 接收既有零基/一基 ruSPARSE 结构。

## 3. 分布式 CG

`distributed::distributed_cg` 不把全矩阵收集到根节点。每个 rank 保留自己的一组 CSR 行，
向量按 rank 顺序收集，内积/范数全局汇总。支持非均匀分块、空 rank、初始解、迭代上限与错误传播。
只支持约定好的均衡连续行分配；A 必须是 SPD，此处不做全局正定性的完整预检。
每个 rank 仍持有 O(n) 全向量，当前不是只交换邻域的稀疏 halo 实现，也不是延迟隐藏 CG。

`RucclCommunicator` 需要专用会话，不能与其他操作在同一个会话上交错。
失败会 abort 会话；重试需要新建。`SolverCommunicator` 供其他有序、有超时和失败唤醒能力的实现接入。
不承诺真实多 GPU/RDMA/分布式直接 LU。

## 4. 自动微分

底层 `rusolver::adjoint` 返回保存前向因子的 pullback，使用解析反向公式，不通过有限差分求生产梯度。
数值有限差分仅用于测试。solve 的反向为 dB=A⁻ᵀdX，dA=−dB Xᵀ。
Cholesky 的梯度按完整对称 A 的约定返回，忽略输出 L 上三角的梯度。
SVD/特征向量梯度需要分离的谱；此版连奇异值/特征值图入口也保守拒绝重复谱。
SVD 还要求满秩且远离零奇异值。QR 导数固定前向的列主元置换，不对离散主元变化求导。

开启 `ruda-autodiff/solver-host` 后：

```rust
use ruda_autodiff::solver_host::{solve_host, cholesky_host,
    singular_values_host, symmetric_eigenvalues_host};
// 输入是 Tensor<Autodiff<ruda_tensor_host::Host>, 2>，实际 dtype 必须是 F64。
// 返回的 Tensor 可与现有 Tensor 运算组合后调用 backward()。
```

FP64 Host 数据要显式建立，不能用默认FP32数据假装64位。示例测试在
`ruda-autodiff/src/solver_host.rs`，其中直接组合 `.sum().backward()`。
目前是**一阶**：反向产生 Host primitive，不构造二阶图。
现有 Backward trait 不返回 Result；前向尽量提前验证，反向遇到非法 cotangent/算术失败会带上下文 panic，
不会静默生成零梯度。复数、稀疏、分布式和 GPU 的自动图接入不在本轮范围。
QR/完整 SVD/特征向量已有显式 pullback，但未提供多输出图包装。

## 5. GPU API 和状态

```rust
use rusolver::tensor::{lu_solve_batched, qr_batched,
    symmetric_eigen_batched, conjugate_gradient_batched};
// 例：let output=lu_solve_batched(&a,&b,Default::default())?;
//     output.check_status_sync()?;
// 只有通过状态检查后才能把 solution 当作成功结果。
```

每个非空 batch 一次设备 kernel；一个设备线程处理一个系统。A/B 不原地修改。
scratch、输出和状态都在设备上；不读取矩阵到 CPU 重算。状态检查显式同步且只读状态。
0=成功；LU/QR 的正数是失败主元/秩位置；−1=非有限输入，−2=非对称，−3=非有限中间值，
−4=未收敛/预算用尽，−5=CG曲率或预条件失败。CG 的−4保留近似解但不代表成功；
LU/QR/eigen失败清零相关输出，LU pivots=-1。
CG 每32次以及疑似收敛/结束时复核真实残差；没有自动放宽误差或静默添加对角项。
GPU路径是小矩阵功能基线；无分块 GEMM、warp协作、分布式 GPU 求解或实测加速比。

```sh
cargo run --release --locked -p ruda-solver --features cuda --example solver-cuda-advanced
```

## 6. 刚性方程、事件、无限积分

```rust
use ruintegrate::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut jac = |_:f64, _:&[f64], j:&mut[f64]| { j[0]=-1000.0; Ok(()) };
let report = solve_bdf1(|_,y,d| { d[0]=-1000.0*y[0]; Ok(()) },
    Some(&mut jac), 0.0, 1.0, &[1.0], Default::default())?;
assert!(report.ode.reached_end());
let q = integrate_infinite(|x| Ok((-x*x).exp()),
    InfiniteInterval::WholeLine{split:0.0}, Default::default())?;
assert!(q.converged());
# Ok(()) }
```

BDF1 每步比较一个整步和两个半步，接受两个半步结果，不做改变阻尼特征的 Richardson 外推。
Newton 使用解析或数值 Jacobian，带回溯，线性系统使用 host LU；失败尝试缩短步长。
默认只保留最终状态，可选轨迹有容量上限。它是一阶稠密方法，不是完整变阶 BDF/Radau，
高维稀疏刚性问题还需要稀疏 Jacobian 与预条件策略。

事件回调填写每个事件函数的值，支持 `Any/Increasing/Decreasing` 与 terminal。
方向按实际时间正向定义，反向积分也一致。初始精确零会报告；终止事件会停在定位的状态。
采用接受步的端点状态/导数做三次 Hermite 插值后二分，根容差不是 ODE 真实解误差保证。
步内多根、相切而不变号的根可能漏过，应按问题限制 max_step。

无限区间采用 x=a±t/(1−t)；全实轴分两条尾部分别检查收敛，不把发散的正负尾相消成主值。
仍然是误差估计，不是积分存在性证明；不包含专门的振荡尾、奇点或主值算法。

兼容提醒：`OdeStatus` 新增 `Event`，穷尽 match 要补分支；ruINTEGRATE 现在依赖本地rusolver，
直接rustc调用必须加 `--extern rusolver=... -L dependency=...`，提供的runner已处理。
