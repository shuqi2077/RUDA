# 原生 PyTorch 静态子图：v24 融合与局部工作区复用

本版本以已交付 v23 累计源码为基线，原生 `ruda-torch → Rust GPU 内核 → CudaGraph → RUDA 设备服务/直接 PTX` 路线不变。新增计算仍使用 RUDA 内核语言；没有 CUDA C++ 内核、NVCC、独立 Python PTX 运行时或 CPU 生产回退。

## 支持范围和显式选项

`StaticGraph` 接受固定地址、连续、非空、rank 1..8 的原生 ruda:0 FP32/FP16/BF16 张量。支持 copy、同形状 add/mul、SiLU、最后一维同精度权重 RMSNorm，以及本轮新增的 silu_mul。图内最多 256 节点、512 张量。输入不允许 requires_grad，不支持广播、隐式类型提升、指针重绑定或隐藏内存副作用。

新增两个独立参数，默认都为 False，避免在新 GPU 路径未经验证时替换已测配置：

```python
import torch
import ruda_torch as r

x = torch.randn(1, 4096, dtype=torch.bfloat16).to('ruda')
up = torch.randn_like(x)
with r.StaticGraph(
    {'x': x, 'up': up},
    [r.GraphOp.silu('activation', 'x'),
     r.GraphOp.mul('y', 'activation', 'up')],
    optimize=True,
    reuse_workspace=True,
) as graph:
    y = graph.replay()['y']
    graph.synchronize()
    print(graph.info)
```

`optimize=True` 先完整校验用户提供的图，再按请求的输出移除无用分支，最后将单使用者的 `mul(silu(x), up)` 合成一节点。无用但非法的节点仍报错，不会通过剪枝隐藏错误。所有原输入仍保留绑定检查。

如果 SiLU 的输出被请求返回、被多个边消费，或位于乘法右侧，不进行本轮融合。右侧不交换乘法操作数，避免扩大位级语义假设。显式 `GraphOp.silu_mul('y','gate','up')` 也可用，其定义与原来左 SiLU、右 up 的组合一致。

## 融合内核与低精度

新增 `kernels::static_silu_mul`，一个线程计算一个连续元素，检查尾部后直接读 gate/up、写最终输出。仍使用原 pointwise SiLU 的 FP32 表达式。与仅把两条数学公式拼起来不同，**先将 SiLU 结果转换到原存储类型，再提升到 FP32 做乘法**，最后转回输出类型。FP16/BF16 的原中间舍入边界保留。

两个原内核成为一个内核；不分配 activation 张量。形状 [1,4096]、BF16 时，两份输出从 16 KiB 变成 8 KiB，另省去中间结果 8 KiB 写回和 8 KiB 读取。这是计划与逻辑流量计算，不是 GPU 测量或模型显存比例。没有本轮生成 PTX 的正确性/性能证据之前，不能假设编译器及硬件已经满足所有舍入要求，真实 GPU 测试必须运行。

## 只在结束使用后复用缓冲

`reuse_workspace=True` 单独启用生命周期规划，与融合可分开比较：

* 只复用形状和精度完全相同的完整 scratch 缓冲；不做大缓冲切片、不复用输入。
* 原值的最后一个消费者必须严格早于新值的生产者。**同一个内核读完再原位覆盖，不在支持范围内。**
* 所有请求返回的结果拥有单独分配，不与任何中间值共用；未剪枝的无消费者输出也保守保留到末尾。
* C++ 桥接会从经验证的节点再次计算生命周期，拒绝提前覆盖、未来读取冲突、同存储不同视图，以及输入/输出别名。Python 规划不是唯一安全屏障。
* 顺序图天然保留访问顺序；`infer_dependencies=True` 的已有内存依赖推导会看到真实复用地址，保留相关读写约束。复用可能增加这些约束，降低潜在并发，不能保证更快。

九个连续 SiLU 节点只返回最后一个结果、形状 [1,4096]、BF16 时，九份输出从 72 KiB 降为两份交替 scratch 加一份独立结果，共 24 KiB。内核数仍为九，不应当成融合收益。

`info.workspace_bytes` 统计独立节点输出分配；`logical_workspace_bytes` 统计不复用时的逻辑输出；`unoptimized_workspace_bytes` 为原图统计。均不包含输入、权重、驱动图、参数元数据、编译缓存和分配器驻留，不是峰值显存。

## 重放检查与生命周期

C++ 每次重放不再构造 Descriptor 的 shape/stride 向量，也不再把两组元数据复制到新 vector 后比较。改为读取已有张量和借用数组视图，仍检查设备、存储所有权、地址、偏移、形状、步长、精度、视图标记、梯度和当前流。构图阶段才建立固定快照。

原来的版本计数更新、推理模式写入限制、错误传播、关闭等待和队列保障保留。返回结果会在后续重放中覆盖，保存历史仍须先复制并安排正确顺序。复用 scratch 不意味着输出可与下一轮写入并发读取。关闭之后，外部持有的结果张量仍有效。全局异步默认值没有更改。

`run_eager()` 执行的是这个计划的同一批内核。优化开启后，它也使用融合内核与复用缓冲；要比较未融合原图，另建 `optimize=False, reuse_workspace=False` 的计划，不能把 optimized eager 当原图对照。

## 构建与验证

当前基础张量 ABI 为 10；**附加图接口为 2**，Rust 和 C++ 两侧必须一起重建。接口不匹配会报错，防止把新 opcode 传给旧实现。

```bash
cargo build --locked --release -p ruda-torch-native
export RUDA_TORCH_LIBRARY="$PWD/target/release/libruda_torch_native.so"
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python

export RUDA_CUDA_COMPILER=ptx
# RUDA_PTX_VERSION 必须使用实际驱动支持且已验证的值。
python ruda-torch/tools/validate_static_graph.py --build --benchmark \
  --output ./v24-static-graph-validation
```

每种同步模式 77 项真实 GPU 测试；默认在两个独立进程分别使用同步和异步。缺失运行标记、跳过、零执行、测试失败均拒收。`--sanitizers all` 使用四种设备检查工具；不会修改全局异步默认。原底层 v22 验收入口保留。

新基准 `benchmark_static_graph_optimized.py` 分别比较原图、只融合、只复用、两者开启；所有路径预热，每组 7 轮轮换顺序，构图单独计时，正确性在计时外检查。报告是主机墙钟延迟与计划字节，不是整模型 tok/s 或峰值显存。

## 尚未完成

本版本没有把矩阵乘法、注意力或 MoE 添加到 StaticGraph 前端，也没有自动模型捕获、训练、第二设备或 AMD/Intel 原生图适配。原生 Rust/新 PTX/GPU 验收和真实模型端到端对照仍是硬门槛；此前上传的 T4 基线结果不能代替 v24 验收。

PyTorch 的变更与别名契约参考（本次没有复制其源码）：
https://docs.pytorch.org/tutorials/advanced/python_custom_ops
https://docs.pytorch.org/docs/main/cpp_extension.html
