# 原生 PyTorch 固定地址静态子图

[文档目录](README.md) · [模型编译](model-compiler.md) · [English](../en/static-pytorch-graphs.md)

`ruda-torch → Rust GPU 内核 → CudaGraph → RUDA 设备服务/直接 PTX` 使用已有原生运行时，不安装 CPU 生产回退。`StaticGraph` 为显式固定地址计划，不是任意模型捕获或 `torch.cuda.graph` 的替代品；普通模型使用[通用编译入口](model-compiler.md)。

## 支持范围和显式选项

`StaticGraph(inputs, nodes, outputs=None, infer_dependencies=False, track_completion=False, optimize=False, reuse_workspace=False, training=False)` 接受命名输入：固定地址、连续、非空、rank 1..8、元素数不超过 uint32 的原生 `ruda:0` FP32／FP16／BF16 张量。图内最多 256 节点、512 张量，不支持广播、隐式类型提升、指针重绑定或隐藏内存副作用。`training=False` 拒绝 requires_grad 输入，显式训练模式见下文。

| 节点 | 元数据／scalar 契约 |
| --- | --- |
| copy、一元逐点操作 | 单输入，无 right，scalar 为规范零；完整一元列表见 [`UNARY_CODES`](../../ruda-torch/python/ruda_torch/_graph_spec.py)。 |
| add／mul／div、激活 backward、silu_mul | 同 shape／dtype 的两个张量，不广播；add 的 scalar 为 alpha，其他 scalar 为零。 |
| add_scalar／mul_scalar／div_scalar | 单输入与有限 FP32 scalar，无 right；减法通过负的 add scalar／alpha 显式表示。 |
| mm／bmm | 同类型二维／三维输入，内维与批次匹配，不广播，scalar 为零。 |
| rms_norm | 可选末轴同类型 `[width]` weight，scalar 为有限正 FP32 epsilon。 |
| softmax／log_softmax 及其 backward | scalar 为规范的非负轴，backward 的梯度与输出张量规格匹配。 |
| sum_keepdim／mean_keepdim | scalar 为非空、范围内的归约轴位掩码，不是结果 shape；归约轴保留为一。 |

通用构造器为 `GraphOp(kind, output, left, right=None, scalar=0.)`；便捷方法包括 copy、add(alpha=...)、mul、silu、silu_mul、rms_norm(eps=...)。`outputs` 为不同的已生产节点名序列，默认最后一个节点；不允许覆盖名称或读取未来节点。scalar 构图时固定，改变尺寸或 scalar 须重建。准确编码与检查见 [`_graph_spec.py`](../../ruda-torch/python/ruda_torch/_graph_spec.py)。

`optimize`、`reuse_workspace` 为两个独立选项，默认均为 False；其他图选项也须为显式 bool：

```python
import torch
import ruda_torch as r

x = torch.ones(1, 4096, dtype=torch.float16).to('ruda:0')
up = torch.ones_like(x)
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

如果 SiLU 的输出被请求返回、被多个边消费，或位于乘法右侧，不进行融合。右侧不交换乘法操作数，避免扩大位级语义假设。显式 `GraphOp.silu_mul('y','gate','up')` 也可用，其定义与左 SiLU、右 up 的组合一致。

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

`info.workspace_bytes` 统计独立节点输出分配；`logical_workspace_bytes` 统计不复用时的逻辑输出；`unoptimized_workspace_bytes` 为原图统计。均不包含输入、权重、驱动图、参数元数据、编译缓存、分配器驻留或训练快照，不是峰值显存。

## 重放检查与生命周期

C++ 每次重放不再构造 Descriptor 的 shape/stride 向量，也不再把两组元数据复制到新 vector 后比较。改为读取已有张量和借用数组视图，仍检查设备、存储所有权、地址、偏移、形状、步长、精度、视图标记、梯度和当前流。构图阶段才建立固定快照。

版本计数、推理模式写入限制、错误传播、关闭等待和队列契约仍适用。推理模式返回结果会在后续重放中覆盖，保存历史须先复制并安排正确顺序；训练模式返回独立结果。复用 scratch 不意味着输出可与下一轮写入并发读取。关闭之后，外部持有的结果张量仍有效。全局异步默认值没有更改。

`synchronize()` 在主机等待所属队列，`query()` 检查是否就绪。`track_completion=True` 开启逐次完成跟踪，使用 `query_completion()`／`wait_completion()`；外部跨流输入写入和输出读取仍须显式排序。`close()` 或上下文退出等待并释放图资源，关闭后不能再运行。

`run_eager()` 执行的是这个计划的同一批内核。优化开启后，它也使用融合内核与复用缓冲；要比较未融合原图，另建 `optimize=False, reuse_workspace=False` 的计划，不能把 optimized eager 当原图对照。

## 构建与验证

当前基础张量 ABI 为 10，**附加图接口为 3**；Rust／C++ 组件须匹配，可从同一源码构建或使用 [Linux／Colab 预编译包](../../ruda-torch/README.md#linuxcolab-precompiled-bundle)。接口不匹配会报错。

```bash
cargo build --locked --release -p ruda-torch-native
export RUDA_TORCH_LIBRARY="$PWD/target/release/libruda_torch_native.so"
python -m pip install --no-build-isolation --no-deps -e ./ruda-torch/python

export RUDA_CUDA_COMPILER=ptx
# RUDA_PTX_VERSION 必须使用实际驱动支持且已验证的值。
python ruda-torch/tools/validate_static_graph.py --build --benchmark \
  --output ./v24-static-graph-validation
```

验证器默认在两个独立进程分别使用同步和异步，不将缺少执行、跳过或失败当作通过。`--sanitizers all` 使用已配置的设备检查工具，不改变全局异步默认；输出留在本地，不作为使用文档或提交内容。

新基准 `benchmark_static_graph_optimized.py` 分别比较原图、只融合、只复用、两者开启；所有路径预热，每组 7 轮轮换顺序，构图单独计时，正确性在计时外检查。报告是主机墙钟延迟与计划字节，不是整模型 tok/s 或峰值显存。

## 显式一阶训练

`training=True` 为原生前向增加 autograd 边，保存每次前向的输入快照，并返回独立输出。反向在同设备重算支持的操作；反向及优化器更新不捕获为原生图。仅支持一阶导数，需要额外存储。

```python
x = torch.ones(2, 8).to('ruda:0').requires_grad_()
with r.StaticGraph(
    {'x': x}, [r.GraphOp.silu('y', 'x')], training=True,
) as graph:
    graph.replay()['y'].float().mean().backward()
    print(x.grad)
```

对应反向完成前不能修改原输入或参数；快照不绕过 PyTorch 的版本检查。关闭训练图前完成全部反向。`StaticGraph.from_model` 返回通用 AOT 可调用 wrapper，不是固定输入计划。没有显式 attention／MoE 节点、第二原生设备或 AMD／Intel 图适配，不捕获整个优化器；矩阵乘及 graph API 3 操作按上表支持。

PyTorch 的变更与别名契约参考（本次没有复制其源码）：
https://docs.pytorch.org/tutorials/advanced/python_custom_ops
https://docs.pytorch.org/docs/main/cpp_extension.html
