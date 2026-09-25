# v17：原生固定缓冲区执行图和启动参数复用

公共 C++/Rust ABI 仍为 9；未修改默认异步，未自动替换 PyTorch 模型入口。

## 这次到底是哪种 Graph

通过已有 `#[ruda(launch)]` / `#[ruda(launch_unchecked)]` 内核生成的 `prepare` 入口，取得一个拥有参数和固定队列的 `PreparedKernel`。`CudaGraph::build` 调用同一个 RUDA 编译器、模块缓存、元数据布局和显存管理，使用 Driver API 为每个内核添加节点，串联依赖，实例化并上传执行图。

重放主体调用一次 `cuGraphLaunch`，不是循环调用普通内核启动。原来的 N 个 GPU 内核仍然执行，Graph 不等于算子融合，也不等于只剩一次驱动调用：队列依赖和缓冲区检查还会进行。当前重放还逐个解析所保留的绑定，主机成本并非与节点数无关。

这是**显式构图与重放**，不是 `cuStreamBeginCapture` 的任意流捕获；不接受任意 Python 闭包、`torch.cuda.graph` 或完整 CUDA C++ 程序。没有把 Managed Memory、IPC、跨卡通信算进本轮成果。

## 支持范围

一个客户端、一个固定逻辑/物理队列、1～4096 个普通 RUDA 内核节点、静态启动网格。图内按提交顺序执行。图外其他线程/设备对缓冲区的访问须有正确的同步。

所有节点使用 Checked 内核执行模式；不支持动态网格回读、张量映射/TMA、图内分配/释放、主机回调、拷贝节点、条件/子图、节点参数更新、跨设备和跨队列节点。TMA 是张量内存映射与异步传输相关接口，本版本遇到相关参数直接拒绝。空图和零/超界网格也直接拒绝。

构图会编译内核、分配/上传内部元数据并执行图上传，但不会执行图中的计算内核。设备标量、形状元数据和宿主标量参数在构图时固定。重复推理中变化的 token、位置或有效历史长度必须由适配器写入同地址设备缓冲并由内核读取；改变宿主标量或换一块新张量不会更新图。当前没有替模型自动完成这种适配。

NVIDIA 入口沿用直接 PTX 编译通路，验收强制 `RUDA_CUDA_COMPILER=ptx`；上游已有的 NVRTC 构建依赖及其他编译入口没有删除。没有新增 CUDA C++ 内核、NVCC 或 cuBLAS/cuDNN 调用。AMD/Intel Graph 不在本轮实现内。

## 显存与生命周期

图保留每个输入、输出、中间结果、视图以及内部设备元数据的原生引用。普通调用不因此全局进入捕获。重放重新走既有队列依赖解析，并检查实际设备地址仍与构图时相同；发生变化时拒绝，不能悄悄继续使用旧指针。

图析构顺序先于 `CudaContext` 的模块卸载。显式 `close` 等待固定队列、销毁执行图与模板，再释放引用。驱动等待失败时保留引用并报告错误；析构失败时宁可留下明确记录的资源，也不提前释放仍可能被 GPU 使用的存储。`Drop` 会等待，不应每个 token 新建/销毁图；优先使用返回错误的显式 `close`。

图会延长缓冲区寿命，可能增加稳定显存占用；不能把 Graph 描述成普遍减少显存。复制构图后需要原位写入的普通张量也可能触发上游写时复制，调用者必须明确处理固定地址约定。

## 启动参数复用

普通启动和图节点共用 `LaunchArguments`。设备服务持有两组可增长向量：设备指针值与参数地址；每次清空长度、保留容量，在容量足够时不再为这两组数据重新申请主机堆内存。

必须先填完所有指针值，再生成其地址，避免向量扩容使早先参数地址失效。指针值取自 `GpuResource.ptr`，不依赖 `GpuStorage` 的环形参数槽。可选的空动态元数据槽仍为零，Tensor Map、指针和常量元数据的顺序保持原约定。驱动在启动/添加节点时复制参数值；设备缓冲引用另行持有。

这不是 GPU 显存池，不能据此声称整个提交路径零分配；其他层仍可能分配。

## 可运行入口

在应用补丁后的仓库根目录，准备项目原有 Rust 构建依赖和真实 NVIDIA 驱动：

```bash
export RUDA_CUDA_COMPILER=ptx
# 按实际驱动选择；8.0 只是示例，不是自动检测结论。
export RUDA_PTX_VERSION=8.0
cargo run --release --locked -p ruda-driver-cuda \
  --no-default-features --features std,direct-ptx --example native-graph
```

例子在 `ruda-driver-cuda/examples/native_graph.rs`，完整展示准备两个内核、构图、重复启动、读回和关闭；构图后释放外部输入引用，仍由图持有原始显存引用。

严格验收：

```bash
python tools/gpu_validation/validate_v17.py --output ./v17-hardware-validation
python tools/gpu_validation/validate_v17.py --benchmark --output ./v17-benchmark
python tools/gpu_validation/validate_v17.py --sanitizer memcheck --output ./v17-memcheck
```

脚本要求实际 Rust 工具链、NVIDIA 驱动 API 12.0+、图函数符号和指定 PTX 版本；执行 4 个 Rust 主机单元测试及 8 个真实 GPU 测试函数。GPU 用例内部覆盖多种尺寸、构图不执行、重复启动、外部引用释放、后续普通启动改写主机参数缓冲、原位更新、关闭后启动拒绝、非法网格、动态网格拒绝与错误队列拒绝。没有设备会失败，不会跳过。

额外基准是 3 组同设备同内核序列的普通提交与 Graph 重放，单列构图成本。计时包含主机提交和等待完成，不是纯 GPU 时间，固定测试顺序也不是全面统计性能研究。结果须从硬件实测取得，没有预置提速值。

## 参考依据

以下只用于接口约定，不是对本代码的验收证明：

- NVIDIA Driver Graph Management：`cuGraphAddKernelNode` 的参数复制、图实例化/上传、启动和销毁。https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html
- CUDA 12.0+ 的 Kernel Node 参数/API 版本：https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/structCUDA__KERNEL__NODE__PARAMS__v2.html
- Stream Capture 是另外的 API 和执行约束，本版本没有实现任意流捕获。https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html

代码为 CUDA 12.0+ 绑定选择 `cuGraphAddKernelNode_v2`，旧绑定选择 v1；这只表示实现了条件编译分支，不构成全部版本兼容认证。
