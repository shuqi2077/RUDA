# v21：执行图内存依赖推导与区间检查

## 范围

本轮是**公共 GPU 运行时的构图优化**，不是新增 Tensor Core 算子，不是 CUDA C++/ABI 兼容层。原生 ABI 仍为 9，默认异步行为不变，主线仍可选择直接 PTX。

## 1. 修正同一分配中的区间检查瓶颈

v20 对同一分配的访问逐对枚举，即使两个视图完全不相交，也先消耗检查预算。4096 个独立、不相交的写视图，潜在组合数为 `4096 × 4095 / 2 = 8,386,560`；旧实现会在超过 1,048,576 次检查后报错，并不会完成全部枚举。这是一个合法显存布局被预算拒绝的具体场景。

v21 在构图时先按分配和访问模式整理字节区间，再按起点扫描。结束位置不大于当前起点的区间立即退出活动集合；读集合与写集合分开，读/读组合不枚举。只对实际重叠且至少一方可写的不同节点计入比较预算。

相同节点、相同读写属性的相邻/重叠区间可以合并；不跨空隙合并，不将只读部分扩大成写入部分。不相交的 4096 个写视图产生零个重叠候选对。排序、集合维护、范围检查仍然存在，不能称为零开销或推理提速。

空区间忽略，半开区间使用 `[start, end)`，检查 `u64` 边界，不通过 `end + 1` 实现。现有显式 DAG 仍严格拒绝未排序的真实写冲突，不暗中添加依赖。

## 2. 从已有内核序列推导内存依赖

新增 `CudaGraph::build_inferred` 和 `build_inferred_tracked`。调用者仍必须提供真实 `PreparedKernel`，原序列必须具有正确的顺序语义。

服务器从真实内核中间表示读取缓冲区的可读/可写属性，以分配身份和精确视图范围建立冲突：

- 先写后读：读必须看到前面写入的结果。
- 先读后写：不能先覆盖尚未读取的数据。
- 先写后写：保留最终写入顺序。

只读共享和不相交区域不产生顺序约束。推导按原节点编号定向，不根据扫描时遇到地址的先后顺序定向。得到冲突关系后，利用祖先位集合去掉已经被间接路径覆盖的直接边，不改变可达关系。

新增入口：

```rust
// nodes 是原有 RUDA prepare 得到的内核序列，地址与形状固定。
let mut graph = unsafe { CudaGraph::build_inferred(&client, nodes)? };
graph.replay()?;
std::fs::write("graph.dot", graph.to_dot())?;
```

三个节点分别计算 `A = f(X)`、`B = g(X)`、`Y = h(A, B)` 时，会得到 A、B 两个独立分支，Y 等待两者。调用者不用手写 `vec![vec![], vec![], vec![0, 1]]`。需要完成事件时选择 `build_inferred_tracked`。

**这不是 PyTorch 自动捕获。** 不扫描现有 Python 模型，不改变 `torch.compile` 分发；不支持隐藏指针、I/O、外部信号、设备动态并行或没有真实 IR 的任务。存在非内存先后关系时继续使用显式 `build_dag` 或顺序 `build`。检查不了单个内核内部的竞争、越界或编译器错误。

## 3. 边界和资源

最多 4096 个节点、65,536 个原始缓冲区访问和 65,536 条最终依赖边。显式 DAG 的实际重叠检查预算保留 1,048,576；可选推导最多处理 16,777,216 个实际重叠候选。预算耗尽明确返回错误，不换成 CPU 算子，不悄悄强制串行。

推导的冲突与祖先位集合在最大节点数时合计约 4 MiB，另有区间、节点和向量开销；全部是一次性主机构图工作。连续写入同一范围的 4096 个节点可约简为 4095 条链式依赖。空位集合按机器字跳过，不逐个测试所有无关前驱。

重放不重新推导或重新扫描区间。现有固定地址检查、参数更新身份检查、缓冲区保活、事件和关闭协议保留。图接口本身仍限制同设备、同固定执行队列；解除了逻辑先后约束，不保证硬件同时执行。

## 4. 示例与验收

```bash
# 必须有 rustc；编译并运行真正的生产拓扑模块，不是 Python 替身。
python tools/gpu_validation/rust_host_v21.py --output ./v21-rust-host

# 有 Rust、驱动和实际 GPU 后，选择与设备匹配的 PTX 版本。
export RUDA_CUDA_COMPILER=ptx
# export RUDA_PTX_VERSION=<major.minor>
python tools/gpu_validation/validate_v21.py --output ./v21-hardware-validation

# 可选同设备对照；需要先保证数值通过。
python tools/gpu_validation/validate_v21.py --benchmark --output ./v21-benchmark
python tools/gpu_validation/validate_v21.py --sanitizer memcheck --output ./v21-memcheck
```

生产 Rust 模块有 52 项测试，其中新增 23 项；包括 10,000 组推导/独立闭包对比和 2,000 组显式 DAG 检查对比。另新增 12 项 GPU 数值/生命周期/错误处理测试及 1 个显式忽略的基准测试（三组尺寸）。

验收工具对缺编译器、缺驱动、缺命名测试、零执行、失败和跳过均拒绝通过。

## 官方语义参考

NVIDIA Driver Graph API 规定图对象不是线程安全对象；本实现仍由同一个设备服务串行操作图对象。驱动负责遵循图依赖，而不是保证任何两个可并行节点一定并发。

https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html

真实设备内存/同步检查仍需 Compute Sanitizer 等工具；主机内存依赖推导不能替代它们。

https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
