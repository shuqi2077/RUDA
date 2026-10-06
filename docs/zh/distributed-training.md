# 明确的 rank、设备与副本训练

[文档目录](README.md) · [训练](training.md) · [ruCCL](libraries/ruccl.md) · [English](../en/distributed-training.md)

## 区分执行接口

| 接口 | 所有权／启动方式 |
| --- | --- |
| `ruccl::register` 与已有 collective_training 示例 | worker 线程中的旧注册式逻辑 rank，示例两个 rank 使用同一默认设备。 |
| `in_process::Communicator<TensorDevice<B>>` | 明确的本地设备上下文，不需要 TCP rendezvous 进程。 |
| `RankCommunicator<TensorDevice<B>>` | 明确 rank／world size、rendezvous 会话和 rank 本地张量设备，可连接不同应用进程。 |
| `ruda_optim::data_parallel::DataParallel` | 调用者拥有 communicator 的副本模型／梯度语义，默认张量传输经过主机暂存。 |
| `DataParallel<B,C>` | 通过明确实现的设备原生 communicator 复用训练语义，例如 rust-ascend HCCL 适配器。 |

rank 数不是 GPU 数。PyTorch `ruda:0` 只暴露一个原生设备，不是可直接替换 torch.distributed 的多设备 launcher；这里描述 Rust 张量／自动微分训练。接口不自动分片模型、不实现 TP／PP／FSDP／ZeRO、不安装 scheduler，也不自动决定 rank／设备映射。

## 将 rank 映射到实际设备

`ruda_tensor_device::cuda::CudaDevice { index }` 选择进程可见的 CUDA ordinal，与 collective rank 独立。模型及其 TensorDevice 使用同一 ordinal。同一进程两张可见 GPU 的 worker 可明确选 0、1；两次 Default::default() 则都选零。launcher 改变每个进程的设备可见性时，编号以该进程实际可见设备列表为准，不能直接把全局 rank 当设备序号。

启动前准备 CUDA／HIP／原生环境；各 rank 使用兼容源码、dtype、模型结构及集合通信选项。类型见[后端指南](backend-composition.md)，Autodiff<B> 与 communicator 的底层后端须匹配，InnerBackend 是去掉自动微分 wrapper 的后端。

## Rendezvous 与 rank 连接

应用 launcher 提供统一可达地址、一个共享 UniqueId、不同的 `0..world_size-1` rank、相同 world size 及各自本地设备。**只生成一次** `UniqueId::new()`，分发其 as_bytes，再通过 from_bytes 重建；各 rank 独立生成新 ID 会得到不同会话。

协调端使用已有 API：

```rust
use ruccl::rank::{NetworkError, TcpRendezvousServer, UniqueId};

fn serve(address: &str, id: UniqueId, world_size: usize) -> Result<(), NetworkError> {
    TcpRendezvousServer::bind(address, id, world_size)?.run()
}
```

协调端与全部 worker 并发启动。run 接受 rank 通道并服务到断开，不能在唯一 worker 线程先阻塞 serve 再启动 peer。支持明确的 collective timeout、可选 heartbeat timeout、rail 和 transport；有效值见[网络定义](../../ruCCL/src/rank/network.rs)。地址、ID、rank／device 配置由应用传入，不自动读取 torchrun／NCCL 启动约定。

CUDA rank 可这样连接并在创建优化器之前广播参数：

```rust
use std::time::Duration;
use ruccl::{
    rank::{UniqueId, communicator::RankCommunicator},
    tensor_device::{TensorDevice, TensorDeviceError},
};
use ruda_autodiff::Autodiff;
use ruda_nn::{Linear, LinearConfig};
use ruda_optim::data_parallel::{DataParallel, DataParallelError};
use ruda_tensor_device::cuda::{Cuda, CudaDevice};

type Inner = Cuda<f32>;
type Training = Autodiff<Inner>;

fn initialize_rank(
    address: &str, id: UniqueId, rank: u32, world_size: u32, local_gpu: usize,
) -> Result<(DataParallel<Training>, Linear<Training>), DataParallelError> {
    let device = CudaDevice { index: local_gpu };
    let execution_device = device.clone();
    let communicator = RankCommunicator::connect(
        move || Ok::<_, TensorDeviceError>(TensorDevice::<Inner>::new(execution_device)),
        address, id, rank, world_size, Duration::from_secs(30), "ruda-training",
    )?;
    let model = LinearConfig::new(128, 64).init(&device);
    DataParallel::<Training>::initialize(communicator, model, 0)
}
```

层尺寸仅示范初始化，实际使用时换为真实副本模型。依赖为 ruCCL（Rust 导入 ruccl）、ruda-autodiff、ruda-nn、启用 collective 的 ruda-optim，以及启用 cuda-default 的 ruda-tensor-device。连接错误经 TensorDeviceError 转换，rank 元数据与设备配置由应用提供；见[连接方法](../../ruCCL/src/rank/communicator/connect.rs)。

默认 TCP 张量适配器下载／主机暂存／上传载荷，TCP peer 交换不自动等于 GPU P2P／NVLink／RDMA。自定义设备原生 DataParallelCommunicator 须提供有序元数据、匹配的浮点／整数广播及归约；换 transport 不改变 token 加权、共享别名或本地参数身份。

## 副本初始化与集合调用顺序

`DataParallel::initialize(communicator,model,root)` 检查参数路径、shape、dtype、冻结标记及共享别名，再广播浮点参数。本地 ID 可不同且会保留，路径／别名建立公共顺序。全部模型张量位于 communicator 设备，创建优化器之前或恢复匹配的 rank 本地存档之后调用。

`initialize_with_buffers` 还会一次性广播 I32／I64 和 Bool 参数 buffer，保留宽度／ID／别名，不表示每次 forward 自动同步。各 rank 的 root 和初始化变体须一致；底层直接集合接口则由调用者自行匹配明确操作顺序与身份。

各累积窗口的 rank 须保持相同 collective 顺序、shape／dtype、梯度跟踪、gather／scatter 轴及 broadcast root。不能一边跳过集合调用而让 peer 等待。可微分集合操作的[反向接口](training.md#副本训练与可微分张量集合通信)也要求通信顺序一致，不在 checkpoint 重算时重复通信。

## 按有效 token 加权累积

对**本地损失和**反传，在本地累积并统计整个窗口的有效样本／token 数。累积边界调用 `session.reduce(&model,gradients,local_weight,policy)`，之后用返回的 gradients 更新；结果还提供准确 global_weight。它将各 rank 梯度和除以有效权重总数，不是除以 rank 数，也不是先平均局部均值。

`reduce_fp32` 为显式 FP32 主参数优化器保留 FP32 梯度，也接受半精度参数的 FP32 累积；普通 reduce 在 FP32 归一化后转回参数类型。各 rank 须选择相同变体和 missing-gradient 策略：

- Error：local weight 非零时要求梯度存在。
- Zero：明确为缺失局部梯度贡献零，全球未使用参数仍不进入更新。

未知／冻结参数梯度、改变的本地 ID／结构、不同策略或无效全局权重均为契约错误。不会隐式执行 scheduler、裁剪、优化器或累积器清空；见 [DataParallel 源码](../../ruda-optim/src/data_parallel.rs)和[训练状态](training.md)。

## 保存与恢复每个 rank

在共同完成的训练边界，各 rank 保存自己的模型、优化器、scheduler、使用中的累积状态、数据／sampler 位置及 RNG。rank／world size、设备映射、源码／配置与 transport 身份保存在外部运行配置中。TrainingRecord 和混合类型 capture_with_dtypes／restore_with_dtypes 保留相应 Rust 状态，不自动选择一致的多 rank 快照。

重启时先从同一预期边界恢复**全部** rank 本地存档，再开始新的集合调用。本地模型 ID 与自己的优化器记录保持配对，不能让不同 rank 恢复到不同 step，也不能对旧优化器广播新的 base；一个 rank 的权重不等于完整训练续跑状态。恢复相同 communicator／model 配置和数据游标，拓扑／transport 变化须明确准备兼容环境，不从旧 rank 编号推断。

已有有界双 rank 示例：

```bash
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- run ../ruda-collective-state
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- resume ../ruda-collective-state
```

run 要求新目录，第一步后保存各 rank 模型／优化器再执行第二步；resume 恢复第一步边界再执行第二步。未改动示例的两个 rank 使用同一默认设备；[进程内张量示例](../../ruCCL/examples/tensor_collectives.rs)也创建三个默认设备上下文，都不是多 GPU 性能基线。

## 错误与长任务准备

契约／网络／设备错误明确传播，应用不能悄悄只对某个 rank 重跑有状态 step。保持端点／timeout 配置一致，终止或重启前核实最近有效 checkpoint。长任务前评估实际模型／batch／sequence 内存及可比较 step 耗时，准备边界恢复和持久可见进度；日志／进程存活不能代替工作量信息和恢复方案。checkpoint／进度／结果留在 Git 外。
