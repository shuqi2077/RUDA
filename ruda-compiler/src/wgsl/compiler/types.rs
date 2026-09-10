use super::*;

impl<P: WgslLowering> WgslCompiler<P> {
    pub(super) fn compile_type(&mut self, item: ruda::Type) -> Item {
        match item {
            ruda::Type::Scalar(ty) => wgsl::Item::Scalar(self.compile_storage_type(ty)),
            ruda::Type::Vector(ty, size) => {
                let elem = self.compile_storage_type(ty);
                match size {
                    2 => wgsl::Item::Vec2(elem),
                    3 => wgsl::Item::Vec3(elem),
                    4 => wgsl::Item::Vec4(elem),
                    _ => panic!("Unsupported vectorizations scheme {:?}", item.vector_size()),
                }
            }
            ruda::Type::Semantic(_) => unimplemented!("Can't compile semantic type"),
        }
    }

    pub(super) fn compile_storage_type(&mut self, ty: ruda::StorageType) -> wgsl::Elem {
        match ty {
            ruda::StorageType::Scalar(ty) => self.compile_elem(ty),
            ruda::StorageType::Atomic(ty) => match ty {
                ruda::ElemType::Float(i) => match i {
                    ruda::FloatKind::F32 => wgsl::Elem::AtomicF32,
                    kind => panic!("atomic<{kind:?}> is not a valid WgpuElement"),
                },
                ruda::ElemType::Int(i) => match i {
                    ruda::IntKind::I32 => wgsl::Elem::AtomicI32,
                    kind => panic!("atomic<{kind:?}> is not a valid WgpuElement"),
                },
                ruda::ElemType::UInt(kind) => match kind {
                    ruda::UIntKind::U32 => wgsl::Elem::AtomicU32,
                    kind => panic!("{kind:?} is not a valid WgpuElement"),
                },
                other => panic!("{other:?} is not a valid WgpuElement"),
            },
            ruda::StorageType::Packed(_, _) => {
                unimplemented!("Packed types not yet supported in WGSL")
            }
            ruda::StorageType::Opaque(ty) => match ty {
                ruda::OpaqueType::Barrier(_) => {
                    unimplemented!("Barrier objects not supported in WGSL")
                }
            },
        }
    }

    pub(super) fn compile_elem(&mut self, value: ruda::ElemType) -> wgsl::Elem {
        match value {
            ruda::ElemType::Float(f) => match f {
                ruda::FloatKind::E2M1
                | ruda::FloatKind::E2M3
                | ruda::FloatKind::E3M2
                | ruda::FloatKind::E4M3
                | ruda::FloatKind::E5M2
                | ruda::FloatKind::UE8M0 => panic!("Minifloat is not a valid WgpuElement"),
                ruda::FloatKind::F16 => {
                    self.f16_used = true;
                    wgsl::Elem::F16
                }
                ruda::FloatKind::BF16 => panic!("bf16 is not a valid WgpuElement"),
                ruda::FloatKind::TF32 => panic!("tf32 is not a valid WgpuElement"),
                ruda::FloatKind::Flex32 => wgsl::Elem::F32,
                ruda::FloatKind::F32 => wgsl::Elem::F32,
                ruda::FloatKind::F64 => wgsl::Elem::F64,
            },
            ruda::ElemType::Int(i) => match i {
                ruda::IntKind::I32 => wgsl::Elem::I32,
                ruda::IntKind::I64 => wgsl::Elem::I64,
                kind => panic!("{kind:?} is not a valid WgpuElement"),
            },
            ruda::ElemType::UInt(kind) => match kind {
                ruda::UIntKind::U32 => wgsl::Elem::U32,
                ruda::UIntKind::U64 => wgsl::Elem::U64,
                kind => panic!("{kind:?} is not a valid WgpuElement"),
            },
            ruda::ElemType::Bool => wgsl::Elem::Bool,
        }
    }

    pub(super) fn ext_meta_pos(&self, var: &ruda::Variable) -> u32 {
        let pos = var.index().expect("Variable should have index");
        self.ext_meta_pos[pos as usize]
    }

    pub(crate) fn compile_variable(&mut self, value: ruda::Variable) -> wgsl::Variable {
        let item = value.ty;
        match value.kind {
            ruda::VariableKind::GlobalInputArray(id) => {
                wgsl::Variable::GlobalInputArray(id, self.compile_type(item))
            }
            ruda::VariableKind::GlobalScalar(id) => {
                wgsl::Variable::GlobalScalar(id, self.compile_storage_type(item.storage_type()))
            }
            ruda::VariableKind::LocalMut { id } | ruda::VariableKind::Versioned { id, .. } => {
                wgsl::Variable::LocalMut {
                    id,
                    item: self.compile_type(item),
                }
            }
            ruda::VariableKind::LocalConst { id } => wgsl::Variable::LocalConst {
                id,
                item: self.compile_type(item),
            },
            ruda::VariableKind::GlobalOutputArray(id) => {
                wgsl::Variable::GlobalOutputArray(id, self.compile_type(item))
            }
            ruda::VariableKind::Constant(value) => {
                wgsl::Variable::Constant(value, self.compile_type(item))
            }
            ruda::VariableKind::SharedArray {
                id,
                length,
                unroll_factor,
                alignment,
            } => {
                let item = self.compile_type(item);
                if !self.shared_arrays.iter().any(|s| s.index == id) {
                    self.shared_arrays.push(SharedArray::new(
                        id,
                        item,
                        (length * unroll_factor) as u32,
                        alignment.map(|it| it as u32),
                    ));
                }
                wgsl::Variable::SharedArray(id, item, length as u32)
            }
            ruda::VariableKind::Shared { id } => {
                let item = self.compile_type(item);
                if !self.shared_values.iter().any(|s| s.index == id) {
                    self.shared_values.push(SharedValue::new(id, item));
                }
                wgsl::Variable::SharedValue(id, item)
            }
            ruda::VariableKind::ConstantArray { id, length, .. } => {
                let item = self.compile_type(item);
                wgsl::Variable::ConstantArray(id, item, length as u32)
            }
            ruda::VariableKind::LocalArray {
                id,
                length,
                unroll_factor,
            } => {
                let item = self.compile_type(item);
                if !self.local_arrays.iter().any(|s| s.index == id) {
                    self.local_arrays.push(LocalArray::new(
                        id,
                        item,
                        (length * unroll_factor) as u32,
                    ));
                }
                wgsl::Variable::LocalArray(id, item, length as u32)
            }
            ruda::VariableKind::Builtin(builtin) => match builtin {
                ruda::Builtin::AbsolutePos => {
                    self.id = true;
                    wgsl::Variable::Id
                }
                ruda::Builtin::UnitPos => {
                    self.local_invocation_index = true;
                    wgsl::Variable::LocalInvocationIndex
                }
                ruda::Builtin::UnitPosX => {
                    self.local_invocation_id = true;
                    wgsl::Variable::LocalInvocationIdX
                }
                ruda::Builtin::UnitPosY => {
                    self.local_invocation_id = true;
                    wgsl::Variable::LocalInvocationIdY
                }
                ruda::Builtin::UnitPosZ => {
                    self.local_invocation_id = true;
                    wgsl::Variable::LocalInvocationIdZ
                }
                ruda::Builtin::RudaPosX => {
                    self.workgroup_id = true;
                    wgsl::Variable::WorkgroupIdX
                }
                ruda::Builtin::RudaPosY => {
                    self.workgroup_id = true;
                    wgsl::Variable::WorkgroupIdY
                }
                ruda::Builtin::RudaPosZ => {
                    self.workgroup_id = true;
                    wgsl::Variable::WorkgroupIdZ
                }
                ruda::Builtin::RudaPosCluster
                | ruda::Builtin::RudaPosClusterX
                | ruda::Builtin::RudaPosClusterY
                | ruda::Builtin::RudaPosClusterZ => self.constant_var(1),
                ruda::Builtin::AbsolutePosX => {
                    self.global_invocation_id = true;
                    wgsl::Variable::GlobalInvocationIdX
                }
                ruda::Builtin::AbsolutePosY => {
                    self.global_invocation_id = true;
                    wgsl::Variable::GlobalInvocationIdY
                }
                ruda::Builtin::AbsolutePosZ => {
                    self.global_invocation_id = true;
                    wgsl::Variable::GlobalInvocationIdZ
                }
                ruda::Builtin::RudaDimX => wgsl::Variable::WorkgroupSizeX,
                ruda::Builtin::RudaDimY => wgsl::Variable::WorkgroupSizeY,
                ruda::Builtin::RudaDimZ => wgsl::Variable::WorkgroupSizeZ,
                ruda::Builtin::ClusterDim
                | ruda::Builtin::ClusterDimX
                | ruda::Builtin::ClusterDimY
                | ruda::Builtin::ClusterDimZ => self.constant_var(1),
                ruda::Builtin::RudaCountX => {
                    self.num_workgroups = true;
                    wgsl::Variable::NumWorkgroupsX
                }
                ruda::Builtin::RudaCountY => {
                    self.num_workgroups = true;
                    wgsl::Variable::NumWorkgroupsY
                }
                ruda::Builtin::RudaCountZ => {
                    self.num_workgroups = true;
                    wgsl::Variable::NumWorkgroupsZ
                }
                ruda::Builtin::RudaPos => {
                    self.workgroup_id_no_axis = true;
                    wgsl::Variable::WorkgroupId
                }
                ruda::Builtin::RudaDim => {
                    self.workgroup_size_no_axis = true;
                    wgsl::Variable::WorkgroupSize
                }
                ruda::Builtin::RudaCount => {
                    self.num_workgroup_no_axis = true;
                    wgsl::Variable::NumWorkgroups
                }
                ruda::Builtin::PlaneDim => {
                    self.subgroup_size = true;
                    wgsl::Variable::SubgroupSize
                }
                ruda::Builtin::PlanePos => {
                    self.subgroup_id = true;
                    wgsl::Variable::SubgroupId
                }
                ruda::Builtin::PlaneCount => {
                    self.num_subgroups = true;
                    self.subgroup_instructions_used = true;
                    wgsl::Variable::NumSubgroups
                }
                ruda::Builtin::UnitPosPlane => {
                    self.subgroup_invocation_id = true;
                    wgsl::Variable::SubgroupInvocationId
                }
            },
            ruda::VariableKind::Matrix { .. } => {
                panic!("Cooperative matrix-multiply and accumulate not supported.")
            }
            ruda::VariableKind::Pipeline { .. } => {
                panic!("Pipeline not supported.")
            }
            ruda::VariableKind::BarrierToken { .. } => {
                panic!("Barrier not supported.")
            }
            ruda::VariableKind::TensorMapInput(_) => panic!("Tensor map not supported."),
            ruda::VariableKind::TensorMapOutput(_) => panic!("Tensor map not supported."),
        }
    }

    pub(super) fn constant_var(&mut self, value: u32) -> wgsl::Variable {
        let var = ruda::Variable::constant(value.into(), UIntKind::U32);
        self.compile_variable(var)
    }

    pub(super) fn compile_binding(&mut self, value: kernel::KernelArg) -> wgsl::KernelArg {
        wgsl::KernelArg {
            id: value.id,
            visibility: value.visibility,
            location: wgsl::Location::Storage,
            item: self.compile_type(value.ty),
            size: value.size,
        }
    }
}
