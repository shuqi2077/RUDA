# Runtime API 参考

[文档首页](README.md) · [编程指南](programming-guide.md) · [Driver API](driver-api.md) · [English](../en/runtime-api.md) | [日本語](../ja/runtime-api.md) | [Deutsch](../de/runtime-api.md) | [Русский](../ru/runtime-api.md)

本页按设备、内存和执行职责介绍当前通用运行时入口。模块位于 `ruda::runtime`，需要 `ruda/runtime` feature；具体后端还需对应驱动 crate。

## 1. 核心类型

| 类型 | 职责 | 定义 |
| --- | --- | --- |
| `Runtime` | 关联 Compiler、Server、Device，获取设备客户端 | [backend.rs](../../ruda-runtime/src/runtime/backend.rs) |
| `ComputeClient<R>` | 内存分配、Kernel 提交、回读、同步及能力查询 | [client.rs](../../ruda-runtime/src/runtime/client.rs) |
| `ComputeServer` | 设备后端执行契约 | [server 模块](../../ruda-runtime/src/runtime/server/mod.rs) |
| `RudaTensor<R>` | 设备存储与张量元数据 | [张量定义](../../ruda-kernel/src/tensor/base.rs) |

`R::client(&device)` 获取对应运行时的客户端。`R::Device` 决定设备类型；同一个泛型参数并不意味着不同物理设备的存储可以直接互用。

`ComputeClient::init(device, server)` 注册新 server，若同一设备已注册该 server 类型则 panic。`load(device)` 要求兼容 server 已初始化，两者都不替代常规的 `R::client(&device)` 设备初始化。

## 2. 内存与传输

以下方法属于 `ComputeClient<R>`：

| 方法 | 行为 |
| --- | --- |
| `create_from_slice(&[u8])` | 从主机字节切片创建设备数据，返回 Handle |
| `empty(usize)` | 按字节数分配存储，不承诺清零 |
| `create_tensor_from_slice`、`empty_tensor` | 使用 shape 和元素大小创建张量存储布局 |
| `read_one(Handle)` | 同步回读单个句柄，返回 `Result<Bytes, ServerError>` |
| `read_async(Vec<Handle>)` | 返回异步回读结果 |
| `read(Vec<Handle>)` | 同步回读多个句柄，错误时 panic |
| `memory_usage()` | 查询运行时记录的内存使用情况 |

`read_one_unchecked` 在回读失败时 panic；不要仅凭方法名将它和取消 Kernel 边界检查混为一谈。张量存在非连续布局时应使用张量回读接口，而不是把原始字节直接当作连续元素。

`read_tensor(Vec<CopyDescriptor>)` 返回 `Vec<Bytes>`，失败时 panic；`read_tensor_async` 返回结果为 `Result<Vec<Bytes>, ServerError>` 的 future。描述符必须使用运行时兼容布局：通过 `Runtime::can_read_tensor` 检查，不支持的张量布局先转连续再回读，客户端不自动重排任意视图。`memory_usage()` 返回 `Result<MemoryUsage, ServerError>`，是 server 分配器的记账，不是主机 RSS、全部物理显存或峰值测量。

## 3. 执行控制

| 方法 | 行为 |
| --- | --- |
| `launch` | 以 Checked 模式提交 Kernel |
| `launch_unchecked` | unsafe 接口；实际检查模式受 BoundsCheckMode 配置控制 |
| `flush` | 提交积压命令，返回 `Result` |
| `sync` | 返回等待执行完成的 future |
| `set_stream` | unsafe 地设置客户端使用的 StreamId |

`launch` 本身不返回设备计算结果。异步编译或执行错误可能在后续回读、同步中被观察到。高层宏生成的启动接口还涉及参数构造，不能用本页的客户端方法签名替代其完整调用契约。

`flush()` 不保证设备完成。须等待或解析 `sync()` 返回的 future，才能等待客户端所解析执行流的完成；仅创建 future 不会等待。unsafe 切换流不建立生产／消费依赖。

| 队列方法 | 契约 |
| --- | --- |
| `execution_stream()` | 解析显式设置的流，否则使用调用线程当前流。 |
| `same_execution_queue(&other)` | 比较设备／server 身份和当前解析的流，不检查完成状态。 |
| `fixed_execution_queue()` | 克隆客户端并固定当前解析的流，不新建流、不等待；适合跨调用持有工作区的计划。 |

## 4. 能力与分析

`properties()` 提供设备属性，`features()` 提供特性集合；在选用 dtype、原子操作或矩阵指令前查询相应能力。`enumerate_devices`、`enumerate_all_devices` 和计数方法用于枚举。`profile` 是运行时分析入口，计时范围应区分提交、执行和传输。

`device_id()` 返回运行时设备身份，`properties_fingerprint()` 返回缓存的硬件／能力身份，不逐次探测驱动。共享选择通过 `runtime_environment(&client)` 使用它们，见[全栈调优 API](stack-autotuning.md#10-控制器与缓存-api)。部署标签应在首次使用前固定；改动标签不会重新配置现有控制器或已记忆环境。

## 5. 错误与安全

底层 unchecked 启动要求调用者排除越界访问及不终止的循环。布局、绑定长度和跨流生命周期也必须与实际 Kernel 一致。检查配置见[调试指南](debugging.md)，设备级接入见 [Driver API](driver-api.md)。
