mod base_float;
mod base_quantized;
mod base_int;
mod base_bool;
mod numeric_float;
mod numeric_int;
mod boolean;
mod integer;
mod float;
mod module;

#[cfg(test)]
mod tests;

use core::sync::atomic::{AtomicU64, Ordering};

use super::{RouterTensor, RunnerClient};
use crate::{
    binary_bool_ops, binary_float_cmp_ops, binary_float_ops, binary_int_cmp_ops, binary_int_ops,
    reduce_float_dim_ops, reduce_float2int_dim_ops, reduce_int_dim_ops, scalar_float_cmp_ops,
    scalar_float_ops, scalar_int_cmp_ops, scalar_int_ops, unary_float_ops, unary_int_ops,
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use ruda_tensor::{Backend, DType, ExecutionError, Shape, TensorData, tensor::IndexingUpdateOp};
use ruda_tensor::graph::{
    BackendIr, BaseOperationIr, BoolOperationIr, FloatOperationIr, HandleContainer, IntOperationIr,
    ModuleOperationIr, NumericOperationIr, OperationIr, TensorId, TensorIr, TensorStatus,
};
use ruda_core::{future::DynFut, stub::Mutex};

/// A runner's context contains a [handle container](HandleContainer) to manage
/// (i.e., fetch and update) existing tensors.
pub struct RunnerContext<B: BackendIr> {
    /// Handle container to retrieve tensors based on their intermediate representation.
    handles: HandleContainer<B::Handle>,
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

impl<B: BackendIr> RunnerContext<B> {
    /// Create a new (uninitialized) empty tensor and returns its corresponding [tensor id](TensorId).
    fn create_empty_handle(&mut self) -> TensorId {
        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        TensorId::new(value)
    }
}

/// A runner is responsible for executing tensor operations for a given [intermediate backend](BackendIr).
#[derive(Clone)]
pub struct Runner<B: BackendIr> {
    // Mutex for the mutable handles
    context: Arc<Mutex<RunnerContext<B>>>,
    device: B::Device,
}

impl<B: BackendIr> core::fmt::Debug for Runner<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Runner")
            .field("device", &self.device)
            .finish()
    }
}

impl<B: BackendIr> Runner<B> {
    /// Create a new runner.
    pub fn new(device: B::Device) -> Self {
        Self {
            context: Arc::new(Mutex::new(RunnerContext {
                handles: HandleContainer::new(),
            })),
            device,
        }
    }

    /// Get the tensor handle for the given [tensor representation](TensorIr).
    pub fn get_tensor_handle(&self, tensor: &TensorIr) -> B::Handle {
        let handles = &mut self.context.lock().unwrap().handles;
        handles.get_tensor_handle(tensor).handle
    }

    /// Create a tensor with the given handle and shape.
    pub fn register_tensor<C: RunnerClient>(
        &self,
        handle: B::Handle,
        shape: Shape,
        dtype: DType,
        client: C,
    ) -> RouterTensor<C> {
        let mut ctx = self.context.lock().unwrap();
        let id = ctx.create_empty_handle();

        ctx.handles.register_handle(id, handle);
        core::mem::drop(ctx);

        RouterTensor::new(id, shape, dtype, client)
    }

    /// Register a tensor from its data and id.
    pub fn register_tensor_data_id(&self, id: TensorId, data: TensorData) {
        let mut ctx = self.context.lock().unwrap();
        let dtype = data.dtype;

        if dtype.is_float() {
            let tensor = B::float_from_data(data, &self.device);
            ctx.handles.register_float_tensor::<B>(&id, tensor)
        } else if dtype.is_int() || dtype.is_uint() {
            let tensor = B::int_from_data(data, &self.device);
            ctx.handles.register_int_tensor::<B>(&id, tensor)
        } else if dtype.is_bool() {
            let tensor = B::bool_from_data(data, &self.device);
            ctx.handles.register_bool_tensor::<B>(&id, tensor)
        } else if let DType::QFloat(_) = dtype {
            let tensor = B::q_from_data(data, &self.device);
            ctx.handles.register_quantized_tensor::<B>(&id, tensor)
        }

        core::mem::drop(ctx);
    }

    /// Register a tensor and returns its intermediate representation.
    pub fn register_tensor_data_desc(&self, data: TensorData) -> TensorIr {
        let mut ctx = self.context.lock().unwrap();
        let id = ctx.create_empty_handle();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        if dtype.is_float() {
            let tensor = B::float_from_data(data, &self.device);
            ctx.handles.register_float_tensor::<B>(&id, tensor)
        } else if dtype.is_int() || dtype.is_uint() {
            let tensor = B::int_from_data(data, &self.device);
            ctx.handles.register_int_tensor::<B>(&id, tensor)
        } else if dtype.is_bool() {
            let tensor = B::bool_from_data(data, &self.device);
            ctx.handles.register_bool_tensor::<B>(&id, tensor)
        } else if let DType::QFloat(_) = dtype {
            let tensor = B::q_from_data(data, &self.device);
            ctx.handles.register_quantized_tensor::<B>(&id, tensor)
        }

        core::mem::drop(ctx);

        TensorIr {
            id,
            shape,
            status: TensorStatus::ReadWrite,
            dtype,
        }
    }
}

// This is a Remote Runner
impl<B: BackendIr> RunnerClient for Runner<B> {
    type Device = B::Device;

    /// Execute a tensor operation.
    fn register_op(&self, op: OperationIr) {
        // Remove unused tensor handles
        let mut ctx = self.context.lock().unwrap();

        let handles = &mut ctx.handles;
        match &op {
            // For every op: get the input(s), execute the operation and register the output(s)
            OperationIr::BaseFloat(op) => self.run_base_float(handles, op),
            OperationIr::BaseInt(op) => self.run_base_int(handles, op),
            OperationIr::BaseBool(op) => self.run_base_bool(handles, op),
            OperationIr::NumericFloat(_dtype, op) => self.run_numeric_float(handles, op),
            OperationIr::NumericInt(_dtype, op) => self.run_numeric_int(handles, op),
            OperationIr::Bool(op) => self.run_boolean(handles, op),
            OperationIr::Int(op) => self.run_integer(handles, op),
            OperationIr::Float(_dtype, op) => self.run_float(handles, op),
            OperationIr::Module(op) => self.run_module(handles, op),
            OperationIr::Custom(_) => {
                panic!("Can't execute custom operation here")
            }
            OperationIr::Init(_) => {
                // Nothing to do.
            }
            OperationIr::Drop(repr) => {
                handles.remove_handle(repr.id);
            }
            #[cfg(feature = "distributed")]
            OperationIr::Distributed(_op) => todo!(),
        }
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let mut ctx = self.context.lock().unwrap();

        enum Output<B: Backend> {
            Float(B::FloatTensorPrimitive),
            Int(B::IntTensorPrimitive),
            Bool(B::BoolTensorPrimitive),
            Quantized(B::QuantizedTensorPrimitive),
        }

        let tensor = if tensor.dtype.is_float() {
            let tensor = ctx.handles.get_float_tensor::<B>(&tensor);
            Output::<B>::Float(tensor)
        } else if tensor.dtype.is_int() || tensor.dtype.is_uint() {
            let tensor = ctx.handles.get_int_tensor::<B>(&tensor);
            Output::Int(tensor)
        } else if tensor.dtype.is_bool() {
            let tensor = ctx.handles.get_bool_tensor::<B>(&tensor);
            Output::Bool(tensor)
        } else if let DType::QFloat(_) = tensor.dtype {
            Output::Quantized(ctx.handles.get_quantized_tensor::<B>(&tensor))
        } else {
            unimplemented!()
        };

        match tensor {
            Output::Float(val) => Box::pin(B::float_into_data(val)),
            Output::Int(val) => Box::pin(B::int_into_data(val)),
            Output::Bool(val) => Box::pin(B::bool_into_data(val)),
            Output::Quantized(val) => Box::pin(B::q_into_data(val)),
        }
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let desc = self.register_tensor_data_desc(data);
        RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        B::sync(&self.device)
    }

    fn seed(&self, seed: u64) {
        B::seed(&self.device, seed)
    }

    fn create_empty_handle(&self) -> TensorId {
        let mut ctx = self.context.lock().unwrap();
        ctx.create_empty_handle()
    }

    fn dtype_usage(&self, dtype: DType) -> ruda_tensor::DTypeUsageSet {
        B::dtype_usage(&self.device, dtype)
    }
}
