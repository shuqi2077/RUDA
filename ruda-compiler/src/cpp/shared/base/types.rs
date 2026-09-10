use super::*;

impl<D: Dialect, P: super::super::DialectProcessors> CppCompiler<D, P> {
    pub(super) fn compile_variable(&mut self, value: gpu::Variable) -> Variable<D> {
        let item = value.ty;
        match value.kind {
            gpu::VariableKind::GlobalInputArray(id) => {
                Variable::GlobalInputArray(id, self.compile_type(item))
            }
            gpu::VariableKind::GlobalScalar(id) => Variable::GlobalScalar {
                id,
                elem: self.compile_storage_type(item.storage_type()),
            },
            gpu::VariableKind::TensorMapInput(id) => {
                self.flags.inst_tma = true;
                Variable::TensorMap(id)
            }
            gpu::VariableKind::TensorMapOutput(id) => {
                self.flags.inst_tma = true;
                Variable::TensorMap(id)
            }
            gpu::VariableKind::LocalMut { id } => Variable::LocalMut {
                id,
                item: self.compile_type(item),
            },
            gpu::VariableKind::Versioned { id, .. } => Variable::LocalMut {
                id,
                item: self.compile_type(item),
            },
            gpu::VariableKind::LocalConst { id } => Variable::LocalConst {
                id,
                item: self.compile_type(item),
            },
            gpu::VariableKind::GlobalOutputArray(id) => {
                Variable::GlobalOutputArray(id, self.compile_type(item))
            }
            gpu::VariableKind::Constant(value) => {
                Variable::Constant(value, self.compile_type(item))
            }
            gpu::VariableKind::SharedArray { id, length, .. } => {
                let item = self.compile_type(item);
                Variable::SharedArray(id, item, length)
            }
            gpu::VariableKind::Shared { id } => {
                let item = self.compile_type(item);
                Variable::Shared(id, item)
            }
            gpu::VariableKind::ConstantArray {
                id,
                length,
                unroll_factor,
            } => {
                let item = self.compile_type(item);
                Variable::ConstantArray(id, item, length * unroll_factor)
            }
            gpu::VariableKind::Builtin(builtin) => match builtin {
                gpu::Builtin::AbsolutePos => {
                    self.flags.indexes.absolute_pos = true;
                    let item = self.compile_type(item);
                    Variable::AbsolutePos(item.elem)
                }
                gpu::Builtin::RudaPosCluster
                    if self.compilation_options.supports_features.clusters =>
                {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterRank
                }
                gpu::Builtin::RudaPosClusterX
                    if self.compilation_options.supports_features.clusters =>
                {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterIndexX
                }
                gpu::Builtin::RudaPosClusterY
                    if self.compilation_options.supports_features.clusters =>
                {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterIndexY
                }
                gpu::Builtin::RudaPosClusterZ
                    if self.compilation_options.supports_features.clusters =>
                {
                    self.flags.indexes.cluster_pos = true;
                    Variable::ClusterIndexZ
                }
                // Fallback if clusters aren't supported, ID is always 0 since clusters are always
                // (1, 1, 1) if unsupported
                gpu::Builtin::RudaPosCluster
                | gpu::Builtin::RudaPosClusterX
                | gpu::Builtin::RudaPosClusterY
                | gpu::Builtin::RudaPosClusterZ => const_u32(0),
                gpu::Builtin::AbsolutePosX => {
                    self.flags.indexes.absolute_pos_tuple = true;
                    Variable::AbsolutePosX
                }
                gpu::Builtin::AbsolutePosY => {
                    self.flags.indexes.absolute_pos_tuple = true;
                    Variable::AbsolutePosY
                }
                gpu::Builtin::AbsolutePosZ => {
                    self.flags.indexes.absolute_pos_tuple = true;
                    Variable::AbsolutePosZ
                }
                gpu::Builtin::RudaDim => {
                    self.flags.indexes.ruda_dim = true;
                    Variable::RudaDim
                }
                gpu::Builtin::RudaDimX => {
                    self.flags.indexes.ruda_dim_tuple = true;
                    Variable::RudaDimX
                }
                gpu::Builtin::RudaDimY => {
                    self.flags.indexes.ruda_dim_tuple = true;
                    Variable::RudaDimY
                }
                gpu::Builtin::RudaDimZ => {
                    self.flags.indexes.ruda_dim_tuple = true;
                    Variable::RudaDimZ
                }
                gpu::Builtin::ClusterDim => const_u32(self.cluster_dim.num_elems()),
                gpu::Builtin::ClusterDimX => const_u32(self.cluster_dim.x),
                gpu::Builtin::ClusterDimY => const_u32(self.cluster_dim.y),
                gpu::Builtin::ClusterDimZ => const_u32(self.cluster_dim.z),
                gpu::Builtin::RudaPos => {
                    self.flags.indexes.ruda_pos = true;
                    let item = self.compile_type(item);
                    Variable::RudaPos(item.elem)
                }
                gpu::Builtin::RudaPosX => {
                    self.flags.indexes.ruda_pos_tuple = true;
                    Variable::RudaPosX
                }
                gpu::Builtin::RudaPosY => {
                    self.flags.indexes.ruda_pos_tuple = true;
                    Variable::RudaPosY
                }
                gpu::Builtin::RudaPosZ => {
                    self.flags.indexes.ruda_pos_tuple = true;
                    Variable::RudaPosZ
                }
                gpu::Builtin::RudaCount => {
                    self.flags.indexes.ruda_count = true;
                    let item = self.compile_type(item);
                    Variable::RudaCount(item.elem)
                }
                gpu::Builtin::RudaCountX => {
                    self.flags.indexes.ruda_count_tuple = true;
                    Variable::RudaCountX
                }
                gpu::Builtin::RudaCountY => {
                    self.flags.indexes.ruda_count_tuple = true;
                    Variable::RudaCountY
                }
                gpu::Builtin::RudaCountZ => {
                    self.flags.indexes.ruda_count_tuple = true;
                    Variable::RudaCountZ
                }
                gpu::Builtin::UnitPos => {
                    self.flags.indexes.unit_pos = true;
                    Variable::UnitPos
                }
                gpu::Builtin::UnitPosX => {
                    self.flags.indexes.unit_pos_tuple = true;
                    Variable::UnitPosX
                }
                gpu::Builtin::UnitPosY => {
                    self.flags.indexes.unit_pos_tuple = true;
                    Variable::UnitPosY
                }
                gpu::Builtin::UnitPosZ => {
                    self.flags.indexes.unit_pos_tuple = true;
                    Variable::UnitPosZ
                }
                gpu::Builtin::PlaneDim => {
                    self.flags.indexes.plane_dim = true;
                    Variable::PlaneDim
                }
                gpu::Builtin::PlanePos => {
                    self.flags.indexes.plane_pos = true;
                    Variable::PlanePos
                }
                gpu::Builtin::PlaneCount => {
                    self.flags.indexes.plane_count = true;
                    Variable::PlaneCount
                }
                gpu::Builtin::UnitPosPlane => {
                    self.flags.indexes.unit_pos_plane = true;
                    Variable::UnitPosPlane
                }
            },
            gpu::VariableKind::LocalArray {
                id,
                length,
                unroll_factor,
            } => {
                let item = self.compile_type(item);
                if !self.local_arrays.iter().any(|s| s.index == id) {
                    self.local_arrays
                        .push(LocalArray::new(id, item, length * unroll_factor));
                }
                Variable::LocalArray(id, item, length)
            }
            gpu::VariableKind::Matrix { id, mat } => {
                self.flags.inst_wmma = true;
                Variable::WmmaFragment {
                    id,
                    frag: self.compile_matrix(mat),
                }
            }
            gpu::VariableKind::Pipeline { id, num_stages } => {
                self.flags.op_pipeline = true;
                let pipeline = Variable::Pipeline { id };
                if !self.pipelines.iter().any(|s| s.pipeline_id() == id) {
                    self.pipelines.push(PipelineOps::Init {
                        pipeline,
                        num_stages,
                    });
                }
                pipeline
            }
            gpu::VariableKind::BarrierToken { id, level } => {
                self.flags.op_barrier = true;
                Variable::BarrierToken { id, level }
            }
        }
    }

    pub(super) fn compile_matrix(&mut self, matrix: gpu::Matrix) -> Fragment<D> {
        Fragment {
            ident: self.compile_matrix_ident(matrix.ident),
            m: matrix.m as u32,
            n: matrix.n as u32,
            k: matrix.k as u32,
            elem: self.compile_storage_type(matrix.storage),
            layout: self.compile_matrix_layout(matrix.layout),
        }
    }

    pub(super) fn compile_matrix_ident(&mut self, ident: gpu::MatrixIdent) -> FragmentIdent<D> {
        match ident {
            gpu::MatrixIdent::A => FragmentIdent::A,
            gpu::MatrixIdent::B => FragmentIdent::B,
            gpu::MatrixIdent::Accumulator => FragmentIdent::Accumulator,
        }
    }

    pub(super) fn compile_matrix_layout(&mut self, layout: gpu::MatrixLayout) -> Option<FragmentLayout<D>> {
        match layout {
            gpu::MatrixLayout::ColMajor => Some(FragmentLayout::ColMajor),
            gpu::MatrixLayout::RowMajor => Some(FragmentLayout::RowMajor),
            gpu::MatrixLayout::Undefined => None,
        }
    }

    pub(super) fn compile_binding(&mut self, binding: ruda_core::kernel::KernelArg) -> KernelArg<D> {
        KernelArg {
            id: binding.id,
            item: self.compile_type(binding.ty),
            size: binding.size,
            vis: binding.visibility,
        }
    }

    pub(super) fn compile_type(&mut self, ty: gpu::Type) -> Item<D> {
        let item = match ty {
            gpu::Type::Scalar(ty) => Item::new(self.compile_storage_type(ty), 1, false),
            gpu::Type::Vector(ty, vector_size) => {
                Item::new(self.compile_storage_type(ty), vector_size, false)
            }
            gpu::Type::Semantic(_) => Item::new(Elem::Bool, 1, true),
        };
        if item.elem != super::Elem::TF32 {
            self.items.insert(item);
            self.items.insert(item.optimized());
        } else {
            // TF32 is represented as `float` in C++
            let mut item = item;
            item.elem = super::Elem::F32;
            self.items.insert(item);
        }

        item
    }

    pub(super) fn compile_storage_type(&mut self, value: gpu::StorageType) -> Elem<D> {
        match value {
            gpu::StorageType::Scalar(ty) => self.compile_elem(ty),
            gpu::StorageType::Atomic(ty) => Elem::Atomic(ty.into()),
            gpu::StorageType::Packed(gpu::ElemType::Float(kind), 2) => match kind {
                FloatKind::E2M1 => {
                    self.flags.elem_fp4 = true;
                    Elem::FP4x2(FP4Kind::E2M1)
                }
                FloatKind::E2M3 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6x2(FP6Kind::E2M3)
                }
                FloatKind::E3M2 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E3M2)
                }
                FloatKind::E4M3 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8x2(FP8Kind::E4M3)
                }
                FloatKind::E5M2 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8x2(FP8Kind::E5M2)
                }
                FloatKind::UE8M0 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8x2(FP8Kind::UE8M0)
                }
                FloatKind::F16 => {
                    self.flags.elem_f16 = true;
                    Elem::F16x2
                }
                FloatKind::BF16 => {
                    self.flags.elem_bf16 = true;
                    Elem::BF16x2
                }
                other => unimplemented!("Unsupported storage type: packed<{other:?}, 2>"),
            },
            gpu::StorageType::Packed(other, factor) => {
                unimplemented!("Unsupported storage type: packed<{other}, {factor}>")
            }
            gpu::StorageType::Opaque(ty) => match ty {
                gpu::OpaqueType::Barrier(level) => {
                    self.flags.op_barrier = true;
                    Elem::Barrier(level)
                }
            },
        }
    }

    pub(super) fn compile_elem(&mut self, value: gpu::ElemType) -> Elem<D> {
        match value {
            gpu::ElemType::Float(kind) => match kind {
                gpu::FloatKind::E2M1 => {
                    self.flags.elem_fp4 = true;
                    Elem::FP4(FP4Kind::E2M1)
                }
                gpu::FloatKind::E2M3 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E2M3)
                }
                gpu::FloatKind::E3M2 => {
                    self.flags.elem_fp6 = true;
                    Elem::FP6(FP6Kind::E3M2)
                }
                gpu::FloatKind::E4M3 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::E4M3)
                }
                gpu::FloatKind::E5M2 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::E5M2)
                }
                gpu::FloatKind::UE8M0 => {
                    self.flags.elem_fp8 = true;
                    Elem::FP8(FP8Kind::UE8M0)
                }
                gpu::FloatKind::F16 => {
                    self.flags.elem_f16 = true;
                    Elem::F16
                }
                gpu::FloatKind::BF16 => {
                    self.flags.elem_bf16 = true;
                    Elem::BF16
                }
                gpu::FloatKind::TF32 => Elem::TF32,
                gpu::FloatKind::Flex32 => Elem::F32,
                gpu::FloatKind::F32 => Elem::F32,
                gpu::FloatKind::F64 => Elem::F64,
            },
            gpu::ElemType::Int(kind) => match kind {
                gpu::IntKind::I8 => Elem::I8,
                gpu::IntKind::I16 => Elem::I16,
                gpu::IntKind::I32 => Elem::I32,
                gpu::IntKind::I64 => Elem::I64,
            },
            gpu::ElemType::UInt(kind) => match kind {
                gpu::UIntKind::U8 => Elem::U8,
                gpu::UIntKind::U16 => Elem::U16,
                gpu::UIntKind::U32 => Elem::U32,
                gpu::UIntKind::U64 => Elem::U64,
            },
            gpu::ElemType::Bool => Elem::Bool,
        }
    }
}
