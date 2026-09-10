use ruda_compiler::mlir::shared_memories::SharedMemories;
use ruda_core::arguments::cpu::{BuiltinArray, LineMemRef};
use ruda_compiler::mlir::shared_memories::SharedMemory;
use crate::execution::schedule::BindingsResource;
use ruda::runtime::{memory_management::{ManagedMemoryHandle, MemoryManagement}, storage::{BytesResource, BytesStorage}};
use std::sync::Arc;

pub struct SharedMlirData {
    pub args_zero_indirection: Vec<LineMemRef>,
    pub info: Vec<u64>,
    pub args_first_indirection: Vec<*mut ()>,
    // Own all allocations referenced by the JIT indirection tables.
    _resources: Vec<BytesResource>,
    _shared_handles: Vec<ManagedMemoryHandle>,
}

unsafe impl Send for SharedMlirData {}
unsafe impl Sync for SharedMlirData {}

pub struct MlirData {
    pub shared_mlir_data: Arc<SharedMlirData>,
    pub args_second_indirection: Vec<*mut ()>,
    pub builtin: BuiltinArray,
}

unsafe impl Send for MlirData {}

impl Clone for MlirData {
    fn clone(&self) -> Self {
        Self {
            shared_mlir_data: Arc::clone(&self.shared_mlir_data),
            args_second_indirection: self.args_second_indirection.clone(),
            builtin: self.builtin.clone(),
        }
    }
}

impl MlirData {
    pub fn new(
        bindings: BindingsResource,
        shared_memories: &SharedMemories,
        memory_management_shared_memory: &mut MemoryManagement<BytesStorage>,
    ) -> Self {
        let BindingsResource { resources, info } = bindings;

        let builtin = BuiltinArray::default();
        let max_buffer_size = resources.len() + shared_memories.0.len() + 1;

        let args_zero_indirection = Vec::with_capacity(max_buffer_size);
        let args_first_indirection = Vec::with_capacity(max_buffer_size);
        let mut args_second_indirection = Vec::with_capacity(max_buffer_size + BuiltinArray::len());
        let info = info.data;

        let mut shared_mlir_data = SharedMlirData {
            args_zero_indirection,
            args_first_indirection,
            info,
            _resources: Vec::new(),
            _shared_handles: Vec::new(),
        };

        let mut push_undirected = |line_memref: LineMemRef| {
            shared_mlir_data.args_zero_indirection.push(line_memref);
            let undirected =
                shared_mlir_data.args_zero_indirection.last_mut().unwrap() as *mut LineMemRef;
            shared_mlir_data
                .args_first_indirection
                .push(undirected as *mut ());
            let undirected =
                shared_mlir_data.args_first_indirection.last_mut().unwrap() as *mut *mut ();

            args_second_indirection.push(undirected as *mut ());
        };

        for resource in &resources {
            let (ptr, len) = resource.get_write_ptr_and_length();
            let line_memref = LineMemRef::new(ptr, len);
            push_undirected(line_memref);
        }

        let mut smem_handles = Vec::with_capacity(shared_memories.0.len());
        let mut all_resources = resources;
        for shared_memory in shared_memories.0.iter() {
            let handle = match shared_memory {
                SharedMemory::Array { ty, length, .. } => {
                    let length = (ty.size() * *length) as u64;
                    memory_management_shared_memory.reserve(length).unwrap()
                }
                SharedMemory::Value { ty, .. } => {
                    let length = ty.size() as u64;
                    memory_management_shared_memory.reserve(length).unwrap()
                }
            };

            smem_handles.push(handle.clone());

            let handle = memory_management_shared_memory
                .get_resource(handle.binding(), None, None)
                .expect("Failed to find resource");
            let (ptr, len) = handle.get_write_ptr_and_length();
            let line_memref = LineMemRef::new(ptr, len);
            push_undirected(line_memref);
            all_resources.push(handle);
        }

        let ptr = shared_mlir_data.info.as_mut_ptr() as *mut u8;
        let line_memref = LineMemRef::new(ptr, shared_mlir_data.info.len());
        push_undirected(line_memref);

        shared_mlir_data._resources = all_resources;
        shared_mlir_data._shared_handles = smem_handles;
        let shared_mlir_data = Arc::new(shared_mlir_data);

        Self {
            shared_mlir_data,
            args_second_indirection,
            builtin,
        }
    }

    pub fn push_builtin(&mut self) {
        for arg in self.builtin.dims.iter_mut() {
            self.args_second_indirection
                .push(arg as *mut u32 as *mut ());
        }
    }
}
