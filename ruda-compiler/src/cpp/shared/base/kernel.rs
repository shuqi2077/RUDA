use super::*;

impl<D: Dialect, P: super::super::DialectProcessors> CppCompiler<D, P> {
    pub(super) fn compile_ir(
        mut self,
        value: KernelDefinition,
        address_type: StorageType,
    ) -> ComputeKernel<D> {
        let metadata = self.build_metadata(&value);
        self.info = ruda_core::arguments::Info::new(&value.scalars, metadata, address_type);

        let instructions = self.compile_scope(&mut value.body.clone());
        let tensor_maps = value
            .tensor_maps
            .into_iter()
            .map(|b| self.compile_binding(b))
            .collect();
        let buffers = value
            .buffers
            .into_iter()
            .map(|b| self.compile_binding(b))
            .collect();
        let scalars = value
            .scalars
            .into_iter()
            .map(|binding| (self.compile_storage_type(binding.ty), binding.count))
            .collect::<Vec<_>>();

        // translation flags
        let flags = Flags {
            indexes: D::builtin_rules(&self.flags.indexes),
            inst_wmma: self.flags.inst_wmma,
            op_pipeline: self.flags.op_pipeline,
            op_barrier: self.flags.op_barrier,
            elem_fp4: self.flags.elem_fp4,
            elem_fp6: self.flags.elem_fp6,
            elem_fp8: self.flags.elem_fp8,
            elem_bf16: self.flags.elem_bf16,
            elem_f16: self.flags.elem_f16,
            elem_tf32: self.flags.elem_tf32,
            inst_tma: self.flags.inst_tma,
            inst_tma_im2col: self.flags.inst_tma_im2col,
            inst_async_copy: self.flags.inst_async_copy,
            inst_ptx_wrappers: self.flags.inst_ptx_wrappers,
            use_grid_constants: self.compilation_options.supports_features.grid_constants,
            has_info: self.info.has_info(),
            has_dynamic_meta: self.info.has_dynamic_meta,
            static_meta_length: self.info.metadata.static_len() as usize,
            ruda_dim: value.ruda_dim,
            cluster_dim: value.options.cluster_dim,
            address_type: self.compile_type(address_type.into()),
        };

        let mut opt = Optimizer::shared_only(value.body, value.ruda_dim);
        let shared_allocs = opt.analysis::<SharedLiveness>();
        let shared_memories = shared_allocs
            .allocations
            .values()
            .map(|alloc| match alloc.smem {
                crate::optimizer::SharedMemory::Array {
                    id,
                    length,
                    ty,
                    align,
                } => SharedMemory::Array {
                    index: id,
                    item: self.compile_type(ty),
                    length,
                    align,
                    offset: alloc.offset,
                },
                crate::optimizer::SharedMemory::Value { id, ty, align } => SharedMemory::Value {
                    index: id,
                    item: self.compile_type(ty),
                    align,
                    offset: alloc.offset,
                },
            })
            .collect();

        let body = Body {
            instructions,
            shared_memories,
            pipelines: self.pipelines,
            barriers: self.barriers,
            const_arrays: self.const_arrays,
            local_arrays: self.local_arrays,
            info_by_ptr: !self.compilation_options.supports_features.grid_constants,
            has_dynamic_meta: self.info.has_dynamic_meta,
            address_type: self.addr_type,
        };

        let mut cluster_dim = value.options.cluster_dim;
        if !self.compilation_options.supports_features.clusters {
            cluster_dim = None;
        }

        ComputeKernel {
            tensor_maps,
            buffers,
            scalars,
            meta_static_len: self.info.metadata.static_len() as usize,
            ruda_dim: value.ruda_dim,
            body,
            extensions: self.extensions,
            flags,
            items: self.items,
            kernel_name: value.options.kernel_name,
            cluster_dim,
            info: self.info.clone(),
        }
    }

    pub(super) fn build_metadata(&mut self, value: &KernelDefinition) -> ruda_core::arguments::Metadata {
        let mut num_ext = 0;

        let mut all_meta: Vec<_> = value
            .buffers
            .iter()
            .chain(value.tensor_maps.iter())
            .map(|buf| (buf.id, buf.has_extended_meta))
            .collect();

        all_meta.sort_by_key(|(id, _)| *id);

        for (_, has_extended_meta) in &all_meta {
            self.ext_meta_positions.push(num_ext);
            if *has_extended_meta {
                num_ext += 1;
            }
        }

        let num_meta = all_meta.len();

        ruda_core::arguments::Metadata::new(num_meta as u32, num_ext)
    }

    pub(crate) fn ext_meta_position(&self, var: gpu::Variable) -> u32 {
        let id = var.index().expect("Variable should have index");
        self.ext_meta_positions[id as usize]
    }
}
