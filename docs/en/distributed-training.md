# Explicit ranks, devices and replicated training

[Documentation](README.md) · [Training](training.md) · [ruCCL](libraries/ruccl.md) · [中文](../zh/distributed-training.md)

## Distinguish the execution interfaces

| Interface | Ownership / launch model |
| --- | --- |
| `ruccl::register` and existing `collective_training` example | Legacy registered logical ranks in worker threads; the example uses the same default device for both ranks. |
| `in_process::Communicator<TensorDevice<B>>` | Explicit local device contexts, without a TCP rendezvous process. |
| `RankCommunicator<TensorDevice<B>>` | Explicit rank/world size, rendezvous session and rank-local tensor device; can connect separate application processes. |
| `ruda_optim::data_parallel::DataParallel` | Replicated model/gradient semantics on a caller-owned communicator; default tensor transport is host-staged. |
| `DataParallel<B,C>` | Same training semantics with an explicitly implemented device-native communicator, such as rust-ascend's HCCL adapter. |

Rank count is not GPU count. The PyTorch `ruda:0` integration exposes one native device and is not a drop-in multi-device `torch.distributed` launcher. This guide concerns Rust tensor/autodiff training. None of these entry points automatically shards a model, implements TP/PP/FSDP/ZeRO, installs a scheduler or discovers a desired rank/device assignment.

## Map ranks to actual devices

`ruda_tensor_device::cuda::CudaDevice { index }` selects a process-visible CUDA ordinal, independently of its collective rank. Construct each model and its `TensorDevice` on that same ordinal. In one process with two visible GPUs, two rank workers can explicitly select indices 0 and 1; using `Default::default()` twice selects ordinal zero twice. When the launcher changes device visibility per process, ordinal numbering must follow that process's actual visible device list, not a global rank assumption.

Prepare the CUDA/HIP/native environment before starting workers. All ranks must use compatible source, dtype, model structure and collective settings. Use the [backend guide](backend-composition.md) for backend types; do not mix `Autodiff<B>` with an unrelated communicator backend. `InnerBackend` is the backend without the Autodiff wrapper.

## Rendezvous and rank connection

The application launcher supplies one reachable address, one shared `UniqueId`, distinct ranks `0..world_size-1`, matching world size and per-rank local device selection. Generate the ID **once** with `UniqueId::new()` and distribute its `as_bytes()`; reconstruct with `UniqueId::from_bytes`. Generating a new ID independently on every rank creates different sessions.

The coordinator uses the existing API:

```rust
use ruccl::rank::{NetworkError, TcpRendezvousServer, UniqueId};

fn serve(address: &str, id: UniqueId, world_size: usize) -> Result<(), NetworkError> {
    TcpRendezvousServer::bind(address, id, world_size)?.run()
}
```

Run the coordinator concurrently with all workers; `run()` accepts their channels and serves until disconnect. Do not block the only worker thread inside serve before launching peers. It accepts explicit collective timeout, optional heartbeat timeout, rail and transport configuration; [network definitions](../../ruCCL/src/rank/network.rs) specify valid values. The rendezvous address, ID and rank/device assignment are application inputs, not automatically consumed `torchrun`/NCCL environment conventions.

For a CUDA tensor rank, this initialization function connects its session and broadcasts model parameters before optimizer creation:

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

The layer dimensions illustrate initialization only; replace the model construction with your actual replica. Workspace dependencies are `ruCCL` (Rust import `ruccl`), `ruda-autodiff`, `ruda-nn`, `ruda-optim` with `collective`, and `ruda-tensor-device` with `cuda-default`. Connection error conversion uses `TensorDeviceError`; rank metadata/device configuration must be provided by the application. Source: [connection methods](../../ruCCL/src/rank/communicator/connect.rs).

The default TCP tensor adapter downloads/stages/uploads tensor payloads. TCP peer exchange is not automatically GPU P2P/NVLink/RDMA. An alternative device-native `DataParallelCommunicator` must provide ordered metadata plus matching floating/integer broadcast and reduction contracts; changing the transport does not change token weighting, tied aliases or local parameter identities.

## Replica initialization and collective order

`DataParallel::initialize(communicator, model, root)` validates parameter paths, shapes, dtypes, frozen flags and tied-alias topology, then broadcasts floating parameters. Local IDs may differ across ranks and are preserved; paths/topology establish the shared order. Every model tensor must reside on its communicator's device. Call before constructing optimizers, or after restoring matching rank-local checkpoints.

`initialize_with_buffers` also broadcasts I32/I64 and Bool parameter buffers once, preserving their width/IDs/aliases; it is not per-forward buffer synchronization. All ranks must use the same root and initialization variant. This is different from lower-level APIs whose caller must align explicit operation order and identities directly.

For every forward/backward accumulation window, all ranks agree on collective order, shape/dtype, gradient tracking, gather/scatter axis and broadcast root. One rank cannot skip a collective while peers enter it. Broadcast/all-gather/reduce-scatter's [differentiable interfaces](training.md#replicated-training-and-differentiable-tensor-collectives) also require matching backward communication order; they do not replay communication during checkpoint recomputation.

## Token-weighted accumulation

Backpropagate **local loss sums**, accumulate locally and track their effective sample/token count. At the accumulation boundary call `session.reduce(&model, gradients, local_weight, policy)`, then update using its returned `gradients`. The result also exposes exact `global_weight`. It divides the sum of all ranks' gradients by total effective weight, not by number of ranks or by a mean of local means.

`reduce_fp32` retains FP32 gradients for explicit FP32-master updates and accepts FP32 accumulation for half-storage parameters. Ordinary reduce casts back to the parameter storage type after FP32 normalization. All ranks must choose the same variant and missing-gradient policy:

- `MissingGradientPolicy::Error`: require local gradients where the rank has nonzero weight.
- `Zero`: explicitly contribute zero for missing local gradients. Globally unused parameters remain absent.

Unknown/frozen parameters in the gradient container, changed local IDs/structure, inconsistent policy, or invalid global weighting are contract errors. Scheduler, clipping, optimizer update and accumulator reset are not implicit. See [DataParallel source](../../ruda-optim/src/data_parallel.rs) and [training state](training.md).

## Save and resume every rank

At a common completed training boundary, save each rank's own model, optimizer, scheduler, accumulation state if used, data/sampler position and RNG state. Keep rank/world size, device mapping, source/configuration and transport identity with the external run configuration. `TrainingRecord` and mixed-storage `capture_with_dtypes` / `restore_with_dtypes` retain the relevant Rust model/optimizer state; they do not select a consistent multi-rank snapshot automatically.

On restart, restore **all** rank-local checkpoints from the same intended boundary before starting new collective work. Preserve each local model's IDs with its optimizer record. Do not restore one rank to a different step, broadcast a new base over a resumed optimizer, or treat a single rank's weights as the whole training continuation. Bring each communicator/model pair back with identical collective configuration, then resume the saved data positions. Topology/transport changes require an explicit compatible setup, not reuse inferred from an old rank number.

The existing bounded two-rank example can be used from the source tree:

```bash
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- run ../ruda-collective-state
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- resume ../ruda-collective-state
```

`run` requires a new directory. It saves rank-local model/optimizer after the first step and performs the second; `resume` restores that saved boundary and performs the second. Both ranks in this unmodified example use the same default device. The [in-process tensor example](../../ruCCL/examples/tensor_collectives.rs) likewise creates three default-device contexts; neither command is a multi-GPU performance baseline.

## Failures and long-run preparation

Contract/network/device failures are explicit; an application must not silently rerun a stateful step on only one rank. Configure reachable endpoints and timeouts consistently and verify the nearest valid checkpoint before terminating or restarting a run. Before long work, assess actual model/batch/sequence memory and comparable step time, prepare step-boundary recovery and visible persistent progress. Logs or live processes alone are not a workload estimate or recovery plan. Keep checkpoint/progress/results outside Git.
