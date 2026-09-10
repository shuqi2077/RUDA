# ruSOLVER：小矩阵 kernel 优化（实验性）

## 为什么先优化这两条

原有 `cholesky_solve_batched`、`lu_solve_batched` 都是一个线程处理一个系统，分解和三角求解在全局输出数组上反复读写。本批只针对这两个有明确瓶颈的路径，不新增重复数学库，不同时改 QR/特征分解/CG。

新增显式入口：

```rust
use rusolver::tensor::{cholesky_solve_batched_warp, lu_solve_batched_warp};

// a: [batch, n, n], b: [batch, n, nrhs]；同设备同执行队列的连续 FP32 张量。
let result = cholesky_solve_batched_warp(&a, &b, Default::default())?;
result.check_status_sync()?; // 显式同步并只读回状态；矩阵不转到 CPU 求解。
let solution = result.solution;

let lu = lu_solve_batched_warp(&a, &b, Default::default())?;
lu.check_status_sync()?;
```

新路径由 `warp-solvers` feature 显式开启（自动包含 `tensor`）；NVIDIA 测试/示例启用 `cuda,warp-solvers`。只启用原 `cuda` 不会编译或启用新 kernel。脚本会自动传入这两个 feature。原入口不变，仍调用原来的串行基线，没有默认自动选择或无声回退。

## 优化结构

| 项目 | 原基线 | 新路径 |
|---|---|---|
| 一个矩阵由谁算 | 一个线程 | 一个 32-thread block，也就是一个完整 warp |
| 中间因子和右端项 | 在全局设备数组上反复访问 | 装载一次到 shared memory，计算后统一写回 |
| 分解中的独立行 | 顺序计算 | 不同 lane 同时处理不同目标行 |
| 多右端项求解 | 列与列顺序执行 | 每列由一个 lane 顺序求解，列之间并行 |
| LU 主元选择 | 逐行寻找最大绝对值 | warp max 后再 min 选行号；相等时仍选最小行号 |
| 输入输出访问 | 不同线程相隔一个矩阵 | 相邻 lane 读取/写出相邻元素 |
| shared 行距 | 不适用 | n 为偶数用 n+1，否则用 n；行距为奇数 |

32 个 32-bit bank 的布局模型下，奇数行距让同列不同的行落在不同 bank。此结论只针对该访问模式，不代表整个 kernel 没有 bank conflict，也不是带宽实测。

矩阵维度和 shared pitch 是编译期参数，既控制 shared 大小，也参与 kernel 特化。内容缓存键额外包含新 kernel、包装层与规划器源码。

### 不改变的数值与失败契约

- 不启用 TF32、Tensor Core、低精度转换或倒数近似来冒充等价优化。
- Cholesky 内积仍按递增 k 累加，仅并行互不依赖的行；保持显式对角 shift，不偷偷加 jitter。
- LU 保留按全矩阵最大绝对值缩放、部分行主元、最低行号平局规则和最终 U 恢复尺度。
- 状态码保持原 API：0 成功，正数为首个失败主元，-1 非有限输入，-2 Cholesky 非对称，-3 非有限中间结果。
- 失败系统的因子/解全部归零，LU pivots 全为 -1；无效结果不能作为解使用。
- 输入只读；新结果仍然分配独立输出缓冲区，不是原地或零分配实现。
- 主机验证形状、精度、连续性、有效 buffer 范围、同设备/同队列、参数有限性和目标能力，之后才分配/提交。
- 空 batch 不启动 kernel。非空 batch 一次 kernel。

没有全局求和顺序重排并不意味着保证跨编译器/硬件逐位相同。设备验收用结果容差、相对残差和因子重建，同时比较原串行路径。

## 同步设计

一个 block 只处理一个系统。只有整个 block 越界时才在开头统一退出，不能让部分线程提前退出 barrier。
所有 plane max/min/any/broadcast 都在全 32 lane 参与的路径上调用；错误码经 plane 操作统一，再决定整个 block 是否进入下一阶段。

显式 `sync_ruda()` 分隔：装载→校验、校验→分解、主元写入/换行→行更新、列更新→下一列、求解→恢复尺度、计算→统一写回。
尤其 LU 在主元读取与换行之间、在 RHS 求解与 U 恢复尺度之间有明确 barrier。不能只依赖 warp 看似同步执行。

## 支持与取舍

`n=1..32`、`nrhs=1..8`，FP32，行优先连续布局。需要固定 32-lane plane、Plane::Ops、至少 32 个 X 线程、足够 shared memory 和一维 grid 容量。不支持的设备返回错误，请调用者显式选择原基线；不自动转 CPU。

最大 shared 声明量：

| n=32, nrhs=8 | 字节/block |
|---|---:|
| Cholesky 因子+RHS | 5,248 |
| LU 因子+RHS+主元 | 5,376 |

成功路径在源码层只对 A/B 各读取一次，因子/解各写一次，另写状态/主元，逻辑全局数据量分别为 10,244 和 10,372 字节/系统。**这不包含分配器、metadata、编译器附加访问或实际 cache-line 流量，不能当成测得 DRAM 字节。**

一个 warp 一个 block 也有代价：每 SM 的驻留 block 上限可能限制可用 warp 数；较小 n、较小 batch、单 RHS 的阶段仍可能利用率不足，同步和分配开销也可能抵消收益。没有测量前不设默认切换阈值，不声称一定加速，更不声称 32 倍加速。

## 编译和验证

```bash
# 仅需 rustc，编译并执行真实 launch-plan 的 host 测试。
python tools/run_kernel_optimization_regressions.py --suite plan

# Cargo 集成；build 会展开新的设备宏、编译测试和 benchmark。
python tools/run_kernel_optimization_regressions.py --suite host
python tools/run_kernel_optimization_regressions.py --suite build

# 真实 GPU：原始两组设备测试 + 新 warp 测试，两条编译路径分别跑。
python tools/run_kernel_optimization_regressions.py --suite cuda --compiler both

# 真实 GPU：先 memcheck，再 racecheck、synccheck，错误码强制非零。
python tools/run_kernel_optimization_regressions.py --suite sanitizer --compiler both
```

`--suite sanitizer` 先正常构建测试，再用 Compute Sanitizer 的 `--target-processes all` 跟踪 cargo 启动的测试子进程。需要该工具支持当前操作系统/驱动，不通过时不能发布为稳定优化。

Python 的独立算法/布局模型：

```bash
python -m pip install -r tools/kernel_optimization/requirements.txt
python tools/run_kernel_optimization_regressions.py --suite oracle
python -m unittest discover -s tools/kernel_optimization -p test_runner.py -v
```

模型对比 FP32 串行递推、带 shared padding/32-lane 所有权的递推与 SciPy FP64。还检查模型的 shared 阶段是否发生跨 lane 的未同步读写、多写者和越界。**它不执行 Rust 源码，不是 GPU racecheck，不是形式证明。**

## 性能比较

```bash
# 先做数值与 sanitizer，再用小批次跑短基准。
python tools/run_kernel_optimization_regressions.py --suite bench --compiler both --batch 256 --order 16 --rhs 4

# 可另跑 n=8/16/32，batch=1/32/256/4096，rhs=1/4/8；每次生成独立报告。
```

基准先预热两条路径，每个 sample 交换运行次序，记录每次耗时、样本平均值的中位数/最小值/最大值、配置、设备、源码和执行后数值检查。每个被计时的输出都在计时结束后与 FP64 解、归一化残差和因子重建核对。

计时是**同步的 API 墙钟时间**：包含输出分配、主机提交、设备执行与同步；不包含首次编译/预热、输入上传和结果读回检查。不是 CUDA event 纯 kernel 时间。speedup<1 会原样记录，不能只发布最好的一组。

没有复用求解因子、预分配输出池、tensor graph 自动路由、科学算子自动微分或大矩阵 tiled LU/Cholesky。这些是不同优化任务，不混入本次。

参考（不是从这些文档复制实现）：
- NVIDIA CUDA Programming Guide, shared memory banks and synchronization: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- NVIDIA Compute Sanitizer, memcheck/racecheck/synccheck: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
