use super::*;

/// Device-service-owned scratch: reuse host allocations after warmup and own
/// pointer values rather than depending on GpuStorage's wrapping pointer ring.
#[derive(Debug, Default)]
pub(super) struct LaunchArguments {
    pointers: Vec<u64>,
    bindings: Vec<*mut c_void>,
}
impl LaunchArguments {
    fn prepare<'a>(
        &'a mut self, resources: &[GpuResource], maps: &[CUtensorMap],
        empty_dynamic_metadata: bool, const_info: Option<*mut c_void>,
    ) -> &'a mut [*mut c_void] {
        self.pointers.clear();
        self.bindings.clear();
        self.pointers.extend(resources.iter().map(|resource| resource.ptr));
        if empty_dynamic_metadata { self.pointers.push(0); }
        // Construct addresses only AFTER every pointer value is appended: Vec
        // growth would otherwise invalidate already constructed argument slots.
        self.bindings.extend(maps.iter().map(|map| map as *const _ as *mut c_void));
        self.bindings.extend(self.pointers.iter_mut().map(|p| p as *mut u64 as *mut c_void));
        self.bindings.extend(const_info);
        &mut self.bindings
    }
}

impl CudaContext {
    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        const_info: Option<*mut c_void>,
        graph: Option<&mut crate::execution::graph::KernelGraph>,
    ) -> Result<(), LaunchError> {
        let kernel = self.module_names.get(&kernel_id).unwrap();
        let bindings = self.launch_arguments.prepare(resources, tensor_maps,
            kernel.dynamic_metadata_index == Some(resources.len()), const_info);
        let ruda_dim = kernel.ruda_dim;
        let block = (ruda_dim.x, ruda_dim.y, ruda_dim.z);
        let shared = u32::try_from(kernel.shared_mem_bytes).map_err(|_| LaunchError::Unknown {
            reason: "dynamic shared memory byte count exceeds driver ABI".into(),
            backtrace: BackTrace::capture(),
        })?;
        if let Some(graph) = graph {
            // Native graph construction, not immediate dispatch. CUDA copies
            // parameter values here; device allocation pins are owned by graph.
            return graph.add_kernel(kernel.func, dispatch_count, block, shared, bindings);
        }
        // The driver copies the argument values before returning. Reusing this
        // scratch for the next queued kernel does not change earlier arguments.
        unsafe {
            cudarc::driver::result::launch_kernel(
                kernel.func, dispatch_count, block, shared, stream.sys, bindings,
            ).map_err(|err| LaunchError::Unknown {
                reason: format!("{err:?}"), backtrace: BackTrace::capture(),
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn launch_arguments_own_values_and_preserve_order() {
        let mut args = LaunchArguments::default();
        let resources = [GpuResource::new(17, std::ptr::null_mut(), 4),
                         GpuResource::new(33, std::ptr::null_mut(), 4)];
        let mut scalar = 71u64;
        let params = args.prepare(&resources, &[], true, Some(&mut scalar as *mut _ as *mut c_void));
        let actual: Vec<u64> = params.iter().map(|&p| unsafe { *(p as *const u64) }).collect();
        assert_eq!(actual, vec![17, 33, 0, 71]);
    }
    #[test]
    fn launch_arguments_reuse_capacity_and_clear_optional_slots() {
        let mut args = LaunchArguments::default();
        let resources = (0..64).map(|i| GpuResource::new(i, std::ptr::null_mut(), 4)).collect::<Vec<_>>();
        args.prepare(&resources, &[], true, None);
        let pointer_base = args.pointers.as_ptr();
        let binding_base = args.bindings.as_ptr();
        for _ in 0..32 {
            let params = args.prepare(&resources[..3], &[], false, None);
            assert_eq!(params.len(), 3);
            assert_eq!(args.pointers.as_ptr(), pointer_base);
            assert_eq!(args.bindings.as_ptr(), binding_base);
        }
        assert!(args.prepare(&[], &[], false, None).is_empty());
    }
}
