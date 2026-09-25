# v18：固定地址执行图的标量更新与生命周期优化

这里的接口不是 CUDA Runtime/Driver ABI 替换，也不是自动 PyTorch 模型捕获。v17 的基本构图契约保留；其“标量永久固定”限制由本页明确的受限更新接口取代。

## 调用链与实际修改

`ruda-kernel` 的 `PreparedKernel` 记录标量区占用的对齐字数；`ruda-driver-cuda::graph::CudaGraph` 接受同一内核的替换准备对象；设备服务验证身份后更新指定节点。现有编译器、分配器、物理执行队列和 PTX 加载路径继续复用，没有独立运行时。

新增 API：

```rust
// replacement 必须用原有 kernel::prepare 构造，原缓冲区、形状和网格均不变。
// unsafe 的原因：标量可能是数组索引或循环上界，调用者仍负责内核语义安全。
unsafe { graph.update_node(0, replacement)?; }
graph.replay()?;

// 或者将一次节点更新与一次图重放合并为一次设备服务提交：
unsafe { graph.update_and_replay(0, replacement)?; }

let entire_queue_ready = graph.query()?;
if graph.try_close()? {
    // 图已关闭。
} else {
    // 没有主动等待 GPU；图仍持有资源，可以稍后重试。
}
```

两段使用 replacement 的示例是替代方案，准备对象会被消费，不可重复使用同一对象。完整可运行示例位于 `ruda-driver-cuda/examples/native_graph_update.rs`。

## 允许变化与明确拒绝

只允许运行时标量数据变化。内核身份、编译期特化、线程块、启动网格、标量布局、数组长度、张量形状/步长、动态元数据、缓冲区分配身份、偏移、容量和所属设备/队列都必须匹配。即使两个分配碰巧有相同设备地址，也不能据此替换。编号按最初 `Vec<PreparedKernel>` 中的顺序。

CPU 端先完整检查结构，再进行任何参数修改。不匹配只返回错误，原图仍可使用；驱动更新或异步复制失败会禁止继续重放该图，但保留资源以便显式关闭。多次 `update_node` 不是原子批量事务：某一次失败不会撤销先前成功的节点更新。

更新只是修改参数，不自行执行计算；`update_and_replay` 才同时提交一次重放。同值检查采用位模式比较，保留有符号零和 NaN 的有效位模式，不用浮点相等判断。`update_and_replay` 的同值情况依旧重放一次。

## 两种参数传递模式

1. 按值传递：保存节点句柄、函数、固定启动配置、设备指针数值和常量元数据的主机副本，调用 `cuGraphExecKernelNodeSetParams[_v2]`。不重新编译、不重建图、不重新分配设备缓冲。指向参数的地址数组从稳定自有存储建立，不能引用已被复用的启动暂存。
2. 设备元数据：保持原来的元数据设备分配，只在同一物理队列中写入标量前缀，然后重放。采用固定页主机暂存，并交给现有队列延迟释放机制。形状/步长等后缀不改，不上传整个模型或激活。

当前直接 PTX 后端在已有配置中启用按值元数据，因此主线主要走第一种。第二种保留给现有的非按值编译配置；单块显卡上的默认验收不代表两种配置都已覆盖。主机参数准备、类型/形状检查、驱动调用仍有成本；设备元数据路径还可能申请主机固定页暂存，不能宣传整个更新路径零分配。

驱动文档说明 executable node 参数更新只影响后续 launch，不改已入队或正在执行的 launch；设备元数据写入则靠相同队列顺序保证先前重放先完成读取。这两者不可混为一谈。

## 重复缓冲区检查

v17 为每次使用保留并检查一次 binding。v18 按“分配标识 + 来源队列 + 总大小 + 起止偏移”去重，同一视图只进入一次重放依赖解析/地址检查。不同偏移或不同分配不能合并。仍保留每个独立视图，仍核对实际设备地址没有迁移。

这减少主机端重复引用与检查，不减少图的内核节点数，也不直接缩小模型权重/KV Cache。主机端节点签名和参数副本会占用额外内存。

## 完成查询与关闭

`query()` 调用真实 `cuStreamQuery`，`CUDA_ERROR_NOT_READY` 返回 `false`，其他错误不能被当成“尚未完成”。查询覆盖图的整个固定队列，图之后的普通工作也会影响结果；不是独立图实例的完成事件。调用仍需等待主机设备服务调度，不承诺无锁或恒定耗时。

`try_close()` 先查询；忙时返回 false，不主动执行 GPU 同步，图及缓冲区继续保留。队列空闲时销毁执行对象和模板，再释放引用。`close()` 与析构仍保留等待逻辑，防止提前释放。原有 unsafe build 契约继续禁止未同步的外部原始队列访问。

## 验证与运行

```bash
export RUDA_CUDA_COMPILER=ptx
# 根据实际驱动设置，不应照抄不兼容的版本。
export RUDA_PTX_VERSION=8.0

cargo run --release --locked -p ruda-driver-cuda \
  --no-default-features --features std,direct-ptx --example native-graph-update

python tools/gpu_validation/validate_v18.py --output ./v18-hardware-validation
python tools/gpu_validation/validate_v18.py --benchmark --output ./v18-benchmark
python tools/gpu_validation/validate_v18.py --sanitizer memcheck --output ./v18-memcheck
```

新增 10 个 Rust 主机布局测试、11 个必须执行的 GPU 测试函数，以及包含 3 个尺寸的可选同设备基准。GPU 测试覆盖并行在途重放不被未来标量污染、混合类型标量、尾部保护、错误拒绝后继续使用、查询/关闭和共享视图。CI 提供 Rust 编译入口。

基准比较“更新、重放分开提交”与“更新并重放一次提交”，使用相同尺寸、相同计算和校验结果，记录完整队列结束后的耗时；不设凭空提速门槛。

## 仍未实现

任意流捕获、节点拓扑变化、指针重绑定、动态网格、多节点原子更新、图内分配/拷贝节点、跨队列/跨设备图、自动 PyTorch 模型接入、AMD/Intel 原生图后端、Managed Memory 和 IPC。本轮不修改 ABI 9 或默认异步设置，不宣称完整模型/GPU 兼容验收。

## 外部接口参考（用于核对语义，不是本项目测试结果）

- NVIDIA Driver Graph API： https://docs.nvidia.com/cuda/archive/12.0.1/cuda-driver-api/group__CUDA__GRAPH.html
- NVIDIA 图对象线程安全约定： https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/graphs-thread-safety.html
- cudarc 驱动接口： https://docs.rs/cudarc/latest/cudarc/driver/sys/index.html

实际依赖版本仍以仓库 `Cargo.lock` 为准。本轮没有升级 cudarc。
