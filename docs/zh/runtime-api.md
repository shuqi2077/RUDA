# Runtime API 参考

[文档首页](README.md) · [编程指南](programming-guide.md) · [Driver API](driver-api.md) · [English](../en/runtime-api.md)

本页按设备、内存和执行职责介绍当前通用运行时入口。模块位于 `ruda::runtime`，需要 `ruda/runtime` feature；具体后端还需对应驱动 crate。

## 1. 核心类型

| 类型 | 职责 | 定义 |
| --- | --- | --- |
| `Runtime` | 关联 Compiler、Server、Device，获取设备客户端 | [backend.rs](../../ruda/src/runtime/backend.rs) |
| `ComputeClient<R>` | 内存分配、Kernel 提交、回读、同步及能力查询 | [client.rs](../../ruda/src/runtime/client.rs) |
| `ComputeServer` | 设备后端执行契约 | [server 模块](../../ruda/src/runtime/server/mod.rs) |
| `RudaTensor<R>` | 设备存储与张量元数据 | [张量定义](../../ruda-kernel/src/tensor/base.rs) |

`R::client(&device)` 获取对应运行时的客户端。`R::Device` 决定设备类型；同一个泛型参数并不意味着不同物理设备的存储可以直接互用。

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

## 3. 执行控制

| 方法 | 行为 |
| --- | --- |
| `launch` | 以 Checked 模式提交 Kernel |
| `launch_unchecked` | unsafe 接口；实际检查模式受 BoundsCheckMode 配置控制 |
| `flush` | 提交积压命令，返回 `Result` |
| `sync` | 返回等待执行完成的 future |
| `set_stream` | unsafe 地设置客户端使用的 StreamId |

`launch` 本身不返回设备计算结果。异步编译或执行错误可能在后续回读、同步中被观察到。高层宏生成的启动接口还涉及参数构造，不能用本页的客户端方法签名替代其完整调用契约。

## 4. 能力与分析

`properties()` 提供设备属性，`features()` 提供特性集合；在选用 dtype、原子操作或矩阵指令前查询相应能力。`enumerate_devices`、`enumerate_all_devices` 和计数方法用于枚举。`profile` 是运行时分析入口，计时范围应区分提交、执行和传输。

## 5. 错误与安全

底层 unchecked 启动要求调用者排除越界访问及不终止的循环。布局、绑定长度和跨流生命周期也必须与实际 Kernel 一致。检查配置见[调试指南](debugging.md)，设备级接入见 [Driver API](driver-api.md)。
