# v20：原生执行图的分支依赖与共享显存检查

默认异步未改变，原生 ABI 仍为 9，Cargo.lock 未改变。

## 1. 找到的限制与实现

v19 `execution/graph.rs` 为每个节点只连接 `last_node`，强制构成一条串行链。即使两个节点只读取同一输入、写入不同结果，也不能表达它们相互独立。

v20 增加：

```rust
// nodes 必须由真实内核的 prepare 构造，同设备、同固定执行队列。
// 示例：0、1 是独立分支，2 在两个分支都结束后运行。
let mut graph = unsafe {
    CudaGraph::build_dag(&client, nodes, vec![vec![], vec![], vec![0, 1]])?
};
graph.replay()?;
std::fs::write("ruda-graph.dot", graph.to_dot())?;
```

`build_dag_tracked` 同时启用 v19 的完成事件。`dependencies()` 返回不可修改的拓扑描述，`edge_count()` 返回依赖边数，`to_dot()` 导出纯结构文本，不包含设备地址、句柄和运行时标量。DOT 不是实际执行时间线或性能剖析结果。

原来的 `build`、`build_tracked` 仍然构建串行链，原调用的行为不变。批量标量更新、完成查询和关闭语义不变。新接口不会自动捕获 PyTorch 模型，也不会自动改写任何模型的投影代码。

## 2. 不是仅添加一份依赖描述

`CudaGraph` 验证依赖列表后，将它交给现有设备服务。设备服务完成读写冲突预检查，再使用原编译、分配、参数、加载路径准备内核。原生 `KernelGraph` 从已建立的节点中解析每个父节点的真实 `CUgraphNode`，将完整父列表传给现有 `cuGraphAddKernelNode` / `_v2`。

独立根节点没有虚构的先后边；汇合节点包含全部指定的前驱。驱动实例化、上传、重放仍走现有路径，一次重放仍是一次图启动。没有增加 CUDA C++、NVCC、NVRTC 编译步骤，没有使用主机循环模拟图执行。

这只是解除不必要的先后约束。实际能否重叠执行由设备资源和驱动调度决定，不能据此报告 2 倍、3 倍提速。若一个节点已经占满计算或显存带宽，分支化也可能没有收益。

## 3. 先验证依赖，再接受并行机会

依赖列表按构建顺序编号：节点 i 只可依赖编号小于 i 的节点。这样不需要隐式重排，能在创建原生图之前拒绝环、自依赖、未来节点、重复父节点、数量不匹配及越界下标。调用者需要先把节点按依赖顺序排列。

对于新 DAG 接口，读写属性来自真实 `kernel_definition()` 的 `KernelArg.visibility`，不是调用者随意填写的“只读”标签。没有可检查中间表示的源码型任务，以及不匹配的缓冲描述、TMA、cluster 配置，会明确拒绝。

共享显存按 RUDA 管理分配身份和字节视图区间检查：

| 情形 | 处理 |
|---|---|
| 两节点只读同一范围 | 允许没有依赖 |
| 同一分配的两个互不重叠视图 | 允许没有依赖 |
| 重叠范围中至少一方可写，且无依赖路径 | 拒绝，不擅自插入边 |
| 有直接或传递依赖的重叠访问 | 允许 |
| 无效偏移、同一分配出现矛盾的总大小 | 拒绝 |

检查保守地把 ReadWrite 视为可能写入，包括原子操作和条件写入。它不验证内核内的线程竞争，不验证隐藏指针，也不证明输入数值或索引合法；这些仍受 `unsafe build` 的原有契约约束。它不能代替 GPU 内存检查工具。

构建限制：1～4096 个节点，最多 65536 条依赖边、65536 个显式缓冲访问。祖先位图最大约 2 MiB，只在构图阶段使用。别名候选比较有 1,048,576 次的上限，超限明确拒绝；只读组和完整串行关系有跳过扫描的路径。图构建可能比旧链式构图更贵，没有新增每次重放的别名分析。

## 4. 生命周期与作用范围

复用 v19 的显存保留、模块所有权、错误中止及完成事件；没有删除同步保障或提前释放图中缓冲。拓扑在实例化后固定，更新标量不能改边、换指针、改形状或更换内核。

仍限定单个 NVIDIA 设备、固定执行队列、固定地址和启动尺寸的显式 RUDA 内核。没有实现任意流捕获、图内 memcpy/memset/分配、动态形状、跨设备图、Managed Memory、IPC、AMD/Intel 图后端或完整 CUDA ABI 替代。

## 5. 测试与验收

新增 29 项生产 Rust 规划器测试，包含全部 1024 个五节点有序 DAG 的传递关系对照，以及视图、读写冲突、预算上限等边界。

新增 14 项真实 GPU 用例，涵盖共享只读输入、缺失汇合边、读写/写写冲突、相邻和重叠视图、尾部保护、批量更新、旧串行接口、完成跟踪和错误队列。另有一个可选基准，测三个形状/分支数量组合，以相同仿射运算比较串行图与依赖图；交错四轮，显式同步并核对输出。不是大模型或注意力性能基准。

```bash
# 不依赖 GPU 的生产 Rust 模块编译和测试，必须有 rustc。
python tools/gpu_validation/rust_host_v20.py --output ./v20-rust-host

# 完整原生验收：也会先运行 v19 的严格验收链。
export RUDA_CUDA_COMPILER=ptx
# 按实际驱动配置 RUDA_PTX_VERSION=major.minor
python tools/gpu_validation/validate_v20.py --output ./v20-hardware-validation
python tools/gpu_validation/validate_v20.py --benchmark --output ./v20-benchmark
python tools/gpu_validation/validate_v20.py --sanitizer memcheck --output ./v20-memcheck
```

脚本遇到工具/设备缺失、零项测试、忽略项目或不完整用例清单都会失败。

## 6. 原生接口依据

- NVIDIA CUDA Programming Guide，Graph Structure：依赖约束执行顺序，满足依赖后的实际调度由驱动决定。
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- NVIDIA Driver Graph Management：节点依赖及 executable 更新语义。
  https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html

未升级锁定的 cudarc 依赖。
