use super::*;

impl CpuServer {
    pub(super) fn prepare_bindings(&mut self, bindings: KernelArguments) -> BindingsResource {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let resources = bindings
            .buffers
            .into_iter()
            .map(|binding| {
                let stream = self.scheduler.stream(&binding.stream);
                stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        BindingsResource {
            resources,
            info: bindings.info,
        }
    }

    pub(super) fn prepare_task(
        &mut self,
        kernel: Box<dyn RudaTask<CpuCompiler>>,
        count: RudaCount,
        bindings: BindingsResource,
        kind: ExecutionMode,
    ) -> Result<ScheduleTask, CompilationError> {
        let ruda_count = match count {
            RudaCount::Static(x, y, z) => [x, y, z],
            RudaCount::Dynamic(binding) => {
                let stream = self.scheduler.stream(&binding.stream);
                let resource = stream
                    .memory_management
                    .get_resource(binding.memory, binding.offset_start, binding.offset_end)
                    .unwrap();

                let _ = stream
                    .flush(ruda_kernel::dsl::server::StreamErrorMode {
                        ignore: true,
                        flush: false,
                    })
                    .ok();

                let bytes = resource.read();
                let x = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
                let y = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                let z = u32::from_ne_bytes(bytes[8..12].try_into().unwrap());
                [x, y, z]
            }
        };

        self.prepare_task_inner(kernel, ruda_count, bindings, kind)
    }

    fn prepare_task_inner(
        &mut self,
        kernel: Box<dyn RudaTask<CpuCompiler>>,
        ruda_count: [u32; 3],
        bindings: BindingsResource,
        kind: ExecutionMode,
    ) -> Result<ScheduleTask, CompilationError> {
        let kernel_id = kernel.id();
        let kernel = if let Some(kernel) = self.compilation_cache.get(&kernel_id) {
            kernel
        } else {
            let kernel = kernel.compile(
                &mut Default::default(),
                &MlirCompilerOptions::default(),
                kind,
                kernel.address_type(),
            )?;
            self.compilation_cache
                .insert(kernel_id.clone(), CpuKernel::new(kernel));
            self.compilation_cache
                .get_mut(&kernel_id)
                .expect("Just inserted")
        };

        let ruda_dim = kernel.mlir.ruda_dim;

        let mlir_engine = kernel.mlir.repr.clone().unwrap();

        let task = ScheduleTask::Execute {
            mlir_engine,
            bindings,
            kind,
            ruda_dim,
            ruda_count,
        };

        Ok(task)
    }
}
