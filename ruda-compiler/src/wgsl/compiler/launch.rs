use super::*;

impl<P: WgslLowering> WgslCompiler<P> {
    pub(super) fn compile_shader(
        &mut self,
        mut value: kernel::KernelDefinition,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<wgsl::ComputeShader, CompilationError> {
        let errors = value.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile wgsl kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        self.strategy = mode;
        self.kernel_name = value.options.kernel_name.clone();

        let num_meta = value.buffers.len();

        self.ext_meta_pos = Vec::new();
        let mut num_ext = 0;

        for binding in value.buffers.iter() {
            self.ext_meta_pos.push(num_ext);
            if binding.has_extended_meta {
                num_ext += 1;
            }
        }

        let metadata = Metadata::new(num_meta as u32, num_ext);
        self.info = Info::new(&value.scalars, metadata, address_type);

        let address_type = self.compile_storage_type(address_type);
        let instructions = self.compile_scope(&mut value.body);
        let extensions = register_extensions(&instructions);
        let body = wgsl::Body {
            instructions,
            id: self.id,
            address_type,
        };

        Ok(wgsl::ComputeShader {
            address_type,
            buffers: value
                .buffers
                .into_iter()
                .map(|mut it| {
                    // This is safe when combined with the unroll transform that adjusts all indices.
                    // Must not be used alone
                    if it.ty.vector_size() > MAX_VECTOR_SIZE {
                        it.ty = it.ty.with_vector_size(MAX_VECTOR_SIZE);
                    }
                    self.compile_binding(it)
                })
                .collect(),
            scalars: value
                .scalars
                .into_iter()
                .map(|binding| (self.compile_storage_type(binding.ty), binding.count))
                .collect(),
            shared_arrays: self.shared_arrays.clone(),
            shared_values: self.shared_values.clone(),
            constant_arrays: self.const_arrays.clone(),
            local_arrays: self.local_arrays.clone(),
            static_meta_len: self.info.metadata.static_len() as usize,
            info: self.info.clone(),
            workgroup_size: value.ruda_dim,
            global_invocation_id: self.global_invocation_id || self.id,
            local_invocation_index: self.local_invocation_index,
            local_invocation_id: self.local_invocation_id,
            num_workgroups: self.id
                || self.num_workgroups
                || self.num_workgroup_no_axis
                || self.workgroup_id_no_axis,
            workgroup_id: self.workgroup_id || self.workgroup_id_no_axis,
            subgroup_size: self.subgroup_size,
            subgroup_id: self.subgroup_id,
            num_subgroups: self.num_subgroups,
            subgroup_invocation_id: self.subgroup_invocation_id,
            body,
            extensions,
            num_workgroups_no_axis: self.num_workgroup_no_axis,
            workgroup_id_no_axis: self.workgroup_id_no_axis,
            workgroup_size_no_axis: self.workgroup_size_no_axis,
            subgroup_instructions_used: self.subgroup_instructions_used,
            f16_used: self.f16_used,
            kernel_name: value.options.kernel_name,
        })
    }
}
