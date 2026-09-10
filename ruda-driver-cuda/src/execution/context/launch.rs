use super::*;

impl CudaContext {
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        const_info: Option<*mut c_void>,
    ) -> Result<(), LaunchError> {
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.binding));
        let kernel = self.module_names.get(&kernel_id).unwrap();
        let mut empty_dynamic_metadata: u64 = 0;
        if kernel.dynamic_metadata_index == Some(resources.len()) {
            bindings.push(&mut empty_dynamic_metadata as *mut u64 as *mut c_void);
        }
        bindings.extend(const_info);

        let ruda_dim = kernel.ruda_dim;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and ruda dimensions are validated by
        // the caller.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (ruda_dim.x, ruda_dim.y, ruda_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                &mut bindings,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?;
        };

        Ok(())
    }
}
