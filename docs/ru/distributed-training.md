# Явные ranks, устройства и обучение реплик

[Содержание](README.md) · [Обучение](training.md) · [ruCCL](libraries/ruccl.md) · [Код подключения](../en/distributed-training.md)

Число ranks не равно числу GPU. Зарегистрированный collective_training использует два логических rank на одном default-устройстве. in_process::Communicator владеет локальными device contexts; RankCommunicator задаёт ranks/world и TCP-session; DataParallel реализует реплики через communicator вызывающей стороны. PyTorch ruda:0 — одно native-устройство, не замена torch.distributed launcher. Здесь описано Rust Tensor/Autodiff-обучение, не автоматические TP/PP/FSDP/ZeRO.

## Размещение и запуск

CudaDevice { index } выбирает CUDA ordinal, видимый процессу, независимо от global rank. Модель и TensorDevice создавайте на одном выбранном устройстве. Для двух GPU одного процесса явно задайте0/1; два Default выбирают0 дважды. Если launcher меняет видимость, используйте фактический список устройств данного процесса.

Передайте общую достижимую rendezvous-адресацию, одну UniqueId, разные ranks `0..world_size-1`, одинаковый world_size и local device. Создайте UniqueId::new один раз, раздайте as_bytes и восстановите from_bytes. Независимый new на каждом rank создаёт разные sessions.

- Координатор: TcpRendezvousServer::bind(address,id,world_size)?.run() одновременно со всеми workers.
- Worker: RankCommunicator::connect(initialize,address,id,rank,world_size,timeout,queue_name), initializer возвращает его TensorDevice.
- Согласуйте TCP/heartbeat timeout, rails и transport. Автоматический launcher по соглашениям torchrun/NCCL не предоставляется.
- TCP Tensor adapter по умолчанию host-staged. TCP peer не означает автоматически GPU P2P/NVLink/RDMA.

Cargo: ruCCL, ruda-autodiff, ruda-nn, ruda-optim с collective, ruda-tensor-device с cuda-default. [Полная функция инициализации](../en/distributed-training.md#rendezvous-and-rank-connection), [API подключения](../../ruCCL/src/rank/communicator/connect.rs). Другой native transport реализует DataParallelCommunicator с упорядоченными metadata и контрактами broadcast/reduce.

## Последовательность обучения

DataParallel::initialize(communicator,model,root) проверяет пути, shapes/dtypes, frozen-флаги, shared aliases и broadcast floating-параметры. Локальные ID сохраняются и могут различаться между ranks. Вызывайте перед созданием optimizer либо после восстановления совместимых локальных checkpoint. initialize_with_buffers один раз передаёт I32/I64/Bool, не перед каждым forward.

Ranks согласуют порядок collectives, shape/dtype, tracking градиента, root и gather/scatter ось. Нельзя пропустить коллективную операцию на одном rank, пока peers входят в неё. Дифференцируемые collectives требуют одинаковый backward-порядок; checkpoint-рекомпутация не повторяет коммуникацию.

Накапливайте градиенты локальных **сумм** loss. На границе reduce(&model,gradients,local_weight,policy), затем обновление по возвращённым gradients. Сумма всех градиентов делится на общее число эффективных tokens/samples global_weight, не на число ranks или среднее локальных средних. reduce_fp32 сохраняет FP32 даже для Half, обычный reduce в конце возвращает dtype параметра. Выбирайте одинаковую версию на всех ranks.

MissingGradientPolicy::Error требует локальные градиенты при ненулевом весе, Zero явно подставляет нулевой вклад. Глобально неиспользованные параметры не обновляются. Scheduler/clipping/optimizer/reset не выполняются скрыто. Смена структуры/ID, неизвестные/frozen градиенты и разная конфигурация — ошибки контракта.

## Сохранение и продолжение

На общей завершённой границе сохраняйте для каждого rank модель, optimizer, scheduler, используемую аккумуляцию, data/sampler cursor и RNG. TrainingRecord не выбирает автоматически согласованный snapshot всех ranks. Восстановите всех с одной границы до новой коммуникации, сочетайте локальные ID с их optimizer record. Не broadcast новый base поверх старого optimizer. World/device/source/transport сохраняются во внешней конфигурации.

```bash
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- run ../ruda-collective-state
cargo run --locked -p ruda-optim --features collective,cuda \
  --example collective_training -- resume ../ruda-collective-state
```

run требует новую директорию и сохраняет первый step; resume выполняет второй с этого состояния. Оба rank неизменённого примера используют одно default-устройство, это не multi-GPU baseline. Ошибку не превращайте в скрытый повтор только одного rank. До долгого запуска оцените реальные ресурсы/работу, checkpoint и видимый постоянный прогресс; результаты и checkpoint держите вне Git.
