use crate::{execution::jit::MlirEngine, execution::stream::CpuStream};
use ruda_core::bytes::Bytes;
use ruda_kernel::dsl::{
    RudaDim, ExecutionMode, MemoryConfiguration, ir::MemoryDeviceProperties,
    server::MetadataBindingInfo,
};
use ruda::runtime::{
    logging::ServerLogger,
    storage::BytesResource,
    stream::{StreamFactory, scheduler::SchedulerStreamBackend},
};
use std::sync::Arc;

/// Defines tasks that can be scheduled on a cpu stream.
pub enum ScheduleTask {
    /// Represents a task to write data to a buffer.
    Write { data: Bytes, buffer: BytesResource },
    /// Represents a task to execute a kernel.
    Execute {
        mlir_engine: MlirEngine,
        bindings: BindingsResource,
        kind: ExecutionMode,
        ruda_dim: RudaDim,
        ruda_count: [u32; 3],
    },
}

impl core::fmt::Debug for ScheduleTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write { data, buffer } => f
                .debug_struct("Write")
                .field("data", data)
                .field("buffer", buffer)
                .finish(),
            Self::Execute {
                mlir_engine: _,
                bindings: _,
                kind,
                ruda_dim,
                ruda_count,
            } => f
                .debug_struct("Execute")
                .field("kind", kind)
                .field("ruda_dim", ruda_dim)
                .field("ruda_count", ruda_count)
                .finish(),
        }
    }
}

/// Represents a collection of resources and bindings for a compute task.
#[derive(Debug)]
pub struct BindingsResource {
    /// List of cpu resources used in the task.
    pub resources: Vec<BytesResource>,
    /// Metadata for uniform bindings.
    pub info: MetadataBindingInfo,
}

/// Represents a cpu backend for scheduling tasks on streams.
#[derive(Debug)]
pub struct ScheduledCpuBackend {
    /// Factory for creating cpu streams.
    factory: CpuStreamFactory,
}

/// Factory for creating cpu streams with specific configurations.
#[derive(Debug)]
pub struct CpuStreamFactory {
    memory_properties: MemoryDeviceProperties,
    memory_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
}

impl StreamFactory for CpuStreamFactory {
    type Stream = CpuStream;

    fn create(&mut self) -> Self::Stream {
        CpuStream::new(
            self.memory_properties.clone(),
            self.memory_config.clone(),
            self.logger.clone(),
        )
    }
}

impl ScheduledCpuBackend {
    /// Creates a new [`ScheduledCpuBackend`] with the given configurations.
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        Self {
            factory: CpuStreamFactory {
                memory_properties,
                memory_config,
                logger,
            },
        }
    }
}

impl SchedulerStreamBackend for ScheduledCpuBackend {
    type Task = ScheduleTask;
    type Stream = CpuStream;
    type Factory = CpuStreamFactory;

    fn enqueue(task: Self::Task, stream: &mut Self::Stream) {
        stream.enqueue_task(task);
    }

    fn flush(stream: &mut Self::Stream) {
        let _ = stream
            .flush(ruda_kernel::dsl::server::StreamErrorMode {
                ignore: true,
                flush: false,
            })
            .ok();
    }

    fn factory(&mut self) -> &mut Self::Factory {
        &mut self.factory
    }
}
