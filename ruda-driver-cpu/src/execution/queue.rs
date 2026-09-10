use crate::{
    execution::jit::MlirEngine,
    execution::{
        runner::KernelRunner,
        schedule::{BindingsResource, ScheduleTask},
    },
};
use ruda_core::bytes::Bytes;
use ruda_kernel::dsl::{RudaDim, server::ExecutionMode};
use ruda::runtime::{logging::ServerLogger, storage::BytesResource};
use std::sync::{Arc, OnceLock, mpsc::SyncSender};

static INSTANCE: OnceLock<CpuExecutionQueue> = OnceLock::new();

#[derive(Clone)]
/// There is a single execution queue instance for the whole CPU runtime.
///
/// This type allows users to send tasks to the global execution queue.
pub struct CpuExecutionQueue {
    sender: SyncSender<QueueItem>,
}

enum QueueItem {
    Task(ScheduleTask),
    Flush(std::sync::mpsc::SyncSender<()>),
}

impl CpuExecutionQueue {
    /// Adds a new task to the queue.
    pub fn add(&self, task: ScheduleTask) {
        self.sender.send(QueueItem::Task(task)).unwrap();
    }

    /// Flushes the queue, making sure all enqueued tasks before this point are executed.
    pub fn flush(&self) {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.sender.send(QueueItem::Flush(sender)).unwrap();
        receiver.recv().unwrap()
    }

    /// Resolves the global execution queue instance.
    pub fn get(logger: Arc<ServerLogger>) -> Self {
        INSTANCE.get_or_init(|| Self::init(logger)).clone()
    }

    fn init(logger: Arc<ServerLogger>) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(32);

        std::thread::spawn(move || {
            let mut server = CpuExecutionQueueServer {
                runner: KernelRunner::new(logger),
            };

            loop {
                match receiver.recv() {
                    Ok(item) => match item {
                        QueueItem::Task(task) => server.execute_task(task),
                        QueueItem::Flush(sender) => sender.send(()).unwrap(),
                    },
                    Err(err) => panic!("{err:?}"),
                }
            }
        });

        Self { sender }
    }
}

struct CpuExecutionQueueServer {
    runner: KernelRunner,
}

impl CpuExecutionQueueServer {
    fn execute_task(&mut self, task: ScheduleTask) {
        match task {
            ScheduleTask::Write { data, buffer } => self.write(data, buffer),
            ScheduleTask::Execute {
                mlir_engine,
                bindings,
                kind,
                ruda_dim,
                ruda_count,
            } => self.kernel(mlir_engine, bindings, kind, ruda_dim, ruda_count),
        }
    }

    fn write(&mut self, data: Bytes, mut buffer: BytesResource) {
        buffer.write().copy_from_slice(&data);
    }

    fn kernel(
        &mut self,
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        kind: ExecutionMode,
        ruda_dim: RudaDim,
        ruda_count: [u32; 3],
    ) {
        self.runner
            .execute_data(mlir_engine, bindings, kind, ruda_dim, ruda_count)
    }
}
