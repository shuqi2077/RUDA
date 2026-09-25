# v19：执行图批量标量更新与可选完成事件

## 1. 批量更新：一个设备服务操作处理多个节点

新增原生 Rust 公共接口：

```rust
// replacements: Vec<(usize, PreparedKernel<CudaRuntime>)>
// 每个准备对象都来自原内核的 prepare；只改变运行时标量。
unsafe { graph.update_nodes(replacements)?; }
// 或选择另一入口，更新后在同一个设备服务操作中重放一次：
unsafe { graph.update_nodes_and_replay(replacements_for_replay)?; }
```

两个参数是不同的拥有型对象，不能消费同一个 Vec 两次。完整可运行示例在 `ruda-driver-cuda/examples/native_graph_batch.rs`。

例如 32 个节点都需要更新，原来需要 32 次 `update_node` 与 1 次 `replay` 的主机设备服务提交；新入口把它们合成一次服务提交。结构验证仍逐节点执行，驱动仍可能收到 32 次节点参数更新；没有把这说成一次驱动调用或一次融合内核。图内部的 GPU 内核数量不变。

节点下标可以不排序，但必须唯一且有效。空批次拒绝，不把“传错空列表”悄悄解释为重放。图支持范围继续是 1～4096 个固定节点。更新不能改变内核及编译期特化、网格、线程块、缓冲区分配身份、偏移、队列、形状、步长或非标量元数据。

### 先检查整批，再提交修改

生产路径分为两阶段：

1. 检查全部下标、所有节点结构、每个标量存储边界；收集真正改变的节点。最后一个节点错误，也不会修改第一个节点。
2. 为整批解析一次执行队列与保留缓冲区，核对一次地址，然后执行改变节点的参数更新。成功后按请求重放一次，最后提交主机签名副本。

这保证结构拒绝时原图不变，并防止设备服务在同一批修改中间插入重放。**不是驱动级回滚事务**：驱动更新或异步复制可能在中途失败，先前更新无法保证撤销。此时图被标为不可执行，继续保留资源，必须关闭并重新构建，不能继续重放一个部分更新的图。

同值比较保持 v18 的位模式语义，包括有符号零与 NaN 的有效载荷。完全同值且不要求重放时，在完整验证后直接返回，不切换驱动上下文、不扫描地址、不更新参数。要求重放时即使全部同值，也恰好提交一次图执行。

设备元数据模式仍通过现有固定页主机暂存和同队列异步复制，只写标量前缀。工作区、张量缓冲和图无需因更新而重新分配；但主机 Vec、参数准备和固定页暂存仍可能分配。不能宣传整个 API 零分配。

## 2. 可选完成事件，不改变原有查询语义

```rust
let mut graph = unsafe { CudaGraph::build_tracked(&client, prepared_nodes)? };
// ... 更新并提交若干次 ...
let done = graph.query_completion()?;
graph.wait_completion()?;
```

普通 `CudaGraph::build` 不创建该事件。需要跟踪时才使用 `build_tracked`，因此原有用户不会默认多出一次事件记录。跟踪模式每次重放后记录一个禁用计时的驱动事件；对只修改设备元数据而不重放的批次，复制完成位置也会记录。

`query_completion` / `wait_completion` 跟踪最近一次图操作及其之前的队列依赖，不受事件记录后排入同队列的其他任务影响。它是复用的“最新位置”标记，不是每一次重放独立的句柄，也不提供跨设备同步、计时或外部事件导入。调用仍需要主机设备服务调度，不承诺无锁或恒定时延。

兼容性保持：`query()`、`synchronize()`、`try_close()`、`close()` 仍检查或等待**整条固定队列**。本轮没有以新事件为由提前释放共享缓冲区，也没有把默认异步开关打开。非跟踪图调用新查询接口会明确报错，不偷偷分配事件或退化为全队列查询。

### 防止旧事件错误认证新工作

构建时先完成图上传提交，再记录初始事件；避免未记录事件看起来已完成。启动或元数据上传之前先使旧标记失效。成功后重新记录；记录失败会使图不可重放。这样以前已经完成的标记不会被拿来证明新工作完成。关闭时仍先等待整队列，再销毁图与事件，最后释放保留的显存引用。

跟踪有代价：新增事件对象和每轮事件记录。批量更新减少主机往返，但性能是否更好、跟踪模式的额外开销多大，必须实测。默认不跟踪。

## 3. 原生接入与范围

修改落在 `ruda-driver-cuda/src/graph.rs`、`src/execution/server/graph.rs`、`src/execution/graph.rs`，复用现有 PTX 编译器、模块、分配器和物理队列。新增 `src/execution/graph_batch.rs` 是无 GPU 依赖的生产验证模块，可直接用 `rustc --test` 验证，不是另写的 Python 模拟。

本轮没有新增 CUDA C++ 算子、NVCC、NVRTC 编译流程或独立 Python 执行器。NVIDIA 原生图与事件仍依赖厂商驱动；PTX 不是免驱动执行方式。没有新增 AMD/Intel 图后端、CUDA ABI 兼容、Managed Memory、IPC、跨设备图、流捕获、动态形状、指针重绑定或自动 PyTorch 模型捕获。原生 ABI 保持 9，Rust 运行时需要重建。

## 4. 验收与对照

```bash
# 只编译并运行本轮无依赖的生产 Rust 验证模块：
python tools/gpu_validation/rust_host_v19.py --output ./v19-rust-host

# 完整原生 GPU 验收；缺工具、缺设备、零项执行或跳过都不能算通过：
export RUDA_CUDA_COMPILER=ptx
# RUDA_PTX_VERSION 使用实际 GPU/驱动支持的 major.minor，不应盲目复制示例。
python tools/gpu_validation/validate_v19.py --output ./v19-hardware-validation

# 可选同设备计时与内存检查：
python tools/gpu_validation/validate_v19.py --benchmark --output ./v19-benchmark
python tools/gpu_validation/validate_v19.py --sanitizer memcheck --output ./v19-memcheck
```

新增 14 项生产 Rust 规划器测试、14 项 GPU 正确性测试，以及含 2、8、32 个节点的三组对照基准。GPU 用例覆盖最后节点无效时不改前面节点、重复下标、空批次、同值仍重放、在途参数版本、尾部保护、错误队列、跟踪生命周期。基准比较同精度、同工作量、相同 warmup 的逐个更新与批量更新；结尾同步并验证输出，没有预设提速门槛。运行顺序固定，正式性能评估应多轮交错比较，避免温度或时钟偏差。

## 5. 外部接口依据

以下只用于核对 API 行为，不是 RUDA 的测试结果：

- NVIDIA Driver Graph Management（节点更新只作用于未来提交；图对象需串行化访问）：https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html
- NVIDIA Event Management（记录捕获此前工作；重录取代原状态；未记录事件为空集合）：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html

没有升级 cudarc 或改变 Cargo.lock，真实接口版本继续由工程锁文件决定。
