use super::*;

impl<P: WgslLowering> WgslCompiler<P> {
    pub(super) fn compile_scope(&mut self, scope: &mut ruda::Scope) -> Vec<wgsl::Instruction> {
        let mut instructions = Vec::new();

        let const_arrays = scope
            .const_arrays
            .drain(..)
            .map(|(var, values)| ConstantArray {
                index: var.index().unwrap(),
                item: self.compile_type(var.ty),
                size: values.len() as u32,
                values: values
                    .into_iter()
                    .map(|val| self.compile_variable(val))
                    .collect(),
            })
            .collect::<Vec<_>>();
        self.const_arrays.extend(const_arrays);

        let processors = P::processors(self.strategy, self.kernel_name.clone(), MAX_VECTOR_SIZE);
        let processing = scope.process([&*processors[0], &*processors[1], &*processors[2]]);

        for mut var in processing.variables {
            if var.ty.vector_size() > MAX_VECTOR_SIZE {
                var.ty = var.ty.with_vector_size(MAX_VECTOR_SIZE);
            }
            instructions.push(wgsl::Instruction::DeclareVariable {
                var: self.compile_variable(var),
            });
        }

        processing
            .instructions
            .into_iter()
            .for_each(|op| self.compile_operation(&mut instructions, op.operation, op.out, scope));

        instructions
    }

    pub(super) fn compile_operation(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        operation: ruda::Operation,
        out: Option<ruda::Variable>,
        scope: &mut ruda::Scope,
    ) {
        match operation {
            ruda::Operation::Copy(variable) => instructions.push(wgsl::Instruction::Assign {
                input: self.compile_variable(variable),
                out: self.compile_variable(out.unwrap()),
            }),
            ruda::Operation::Arithmetic(op) => {
                self.compile_arithmetic(op, out, instructions, scope)
            }
            ruda::Operation::Comparison(op) => self.compile_cmp(op, out, instructions),
            ruda::Operation::Bitwise(op) => self.compile_bitwise(op, out, instructions),
            ruda::Operation::Operator(op) => self.compile_operator(op, out, instructions),
            ruda::Operation::Atomic(op) => instructions.push(self.compile_atomic(op, out)),
            ruda::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, out)),
            ruda::Operation::Branch(val) => self.compile_branch(instructions, val),
            ruda::Operation::Synchronization(val) => {
                self.compile_synchronization(instructions, val)
            }
            ruda::Operation::Plane(op) => self.compile_subgroup(instructions, op, out),
            ruda::Operation::CoopMma(_) => {
                panic!("Cooperative matrix-multiply and accumulate isn't supported on wgpu.")
            }
            ruda::Operation::NonSemantic(ruda::NonSemantic::Comment { content }) => {
                self.compile_comment(instructions, content)
            }
            ruda::Operation::NonSemantic(_) => {}
            ruda::Operation::Barrier(_) => {
                panic!("Barrier isn't supported on wgpu.")
            }
            ruda::Operation::Tma(_) => panic!("TMA isn't supported on wgpu."),
            ruda::Operation::Marker(_) => {}
        }
    }

    pub(super) fn compile_subgroup(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        subgroup: ruda::Plane,
        out: Option<ruda::Variable>,
    ) {
        self.subgroup_instructions_used = true;

        let out = out.unwrap();
        let op = match subgroup {
            ruda::Plane::Elect => Subgroup::Elect {
                out: self.compile_variable(out),
            },
            ruda::Plane::All(op) => Subgroup::All {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::Any(op) => Subgroup::Any {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::Ballot(op) => Subgroup::Ballot {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },

            ruda::Plane::Broadcast(op) => Subgroup::Broadcast {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },

            ruda::Plane::Sum(op) => Subgroup::Sum {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },

            ruda::Plane::ExclusiveSum(op) => Subgroup::ExclusiveSum {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::InclusiveSum(op) => Subgroup::InclusiveSum {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::Prod(op) => Subgroup::Prod {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::ExclusiveProd(op) => Subgroup::ExclusiveProd {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::InclusiveProd(op) => Subgroup::InclusiveProd {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::Min(op) => Subgroup::Min {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::Max(op) => Subgroup::Max {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::Plane::Shuffle(op) => Subgroup::Shuffle {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::Plane::ShuffleXor(op) => Subgroup::ShuffleXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::Plane::ShuffleUp(op) => Subgroup::ShuffleUp {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::Plane::ShuffleDown(op) => Subgroup::ShuffleDown {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
        };

        instructions.push(wgsl::Instruction::Subgroup(op));
    }

    pub(super) fn compile_branch(&mut self, instructions: &mut Vec<wgsl::Instruction>, branch: ruda::Branch) {
        match branch {
            ruda::Branch::If(mut op) => instructions.push(wgsl::Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            ruda::Branch::IfElse(mut op) => instructions.push(wgsl::Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            ruda::Branch::Switch(mut op) => instructions.push(wgsl::Instruction::Switch {
                value: self.compile_variable(op.value),
                instructions_default: self.compile_scope(&mut op.scope_default),
                cases: op
                    .cases
                    .into_iter()
                    .map(|(val, mut scope)| {
                        (self.compile_variable(val), self.compile_scope(&mut scope))
                    })
                    .collect(),
            }),
            ruda::Branch::Return => instructions.push(wgsl::Instruction::Return),
            // No unreachable hint in WGSL
            ruda::Branch::Unreachable => instructions.push(wgsl::Instruction::Return),
            ruda::Branch::Break => instructions.push(wgsl::Instruction::Break),
            ruda::Branch::RangeLoop(mut range_loop) => {
                instructions.push(wgsl::Instruction::RangeLoop {
                    i: self.compile_variable(range_loop.i),
                    start: self.compile_variable(range_loop.start),
                    end: self.compile_variable(range_loop.end),
                    step: range_loop.step.map(|it| self.compile_variable(it)),
                    inclusive: range_loop.inclusive,
                    instructions: self.compile_scope(&mut range_loop.scope),
                })
            }
            ruda::Branch::Loop(mut op) => instructions.push(wgsl::Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }

    pub(super) fn compile_synchronization(
        &mut self,
        instructions: &mut Vec<wgsl::Instruction>,
        synchronization: ruda::Synchronization,
    ) {
        match synchronization {
            ruda::Synchronization::SyncRuda => {
                instructions.push(wgsl::Instruction::WorkgroupBarrier)
            }
            ruda::Synchronization::SyncPlane => {
                panic!("Synchronization within a plane is not supported in WGSL")
            }
            ruda::Synchronization::SyncStorage => {
                instructions.push(wgsl::Instruction::StorageBarrier)
            }
            ruda::Synchronization::SyncAsyncProxyShared => panic!("TMA is not supported in WGSL"),
        };
    }

    pub(super) fn compile_comment(&mut self, instructions: &mut Vec<wgsl::Instruction>, content: String) {
        instructions.push(wgsl::Instruction::Comment { content })
    }

    pub(super) fn compile_metadata(
        &mut self,
        metadata: ruda::Metadata,
        out: Option<ruda::Variable>,
    ) -> wgsl::Instruction {
        let out = out.unwrap();
        match metadata {
            ruda::Metadata::Rank { var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.info.metadata.rank_index(position);
                wgsl::Instruction::Metadata {
                    out: self.compile_variable(out),
                    info_offset: self.compile_variable(offset.into()),
                }
            }
            ruda::Metadata::Stride { dim, var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.info.metadata.stride_offset_index(position);
                wgsl::Instruction::ExtendedMeta {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            ruda::Metadata::Shape { dim, var } => {
                let position = self.ext_meta_pos(&var);
                let offset = self.info.metadata.shape_offset_index(position);
                wgsl::Instruction::ExtendedMeta {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            ruda::Metadata::Length { var } => match var.kind {
                ruda::VariableKind::GlobalInputArray(id) => {
                    let offset = self.info.metadata.len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                ruda::VariableKind::GlobalOutputArray(id) => {
                    let offset = self.info.metadata.len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                _ => wgsl::Instruction::Length {
                    var: self.compile_variable(var),
                    out: self.compile_variable(out),
                },
            },
            ruda::Metadata::BufferLength { var } => match var.kind {
                ruda::VariableKind::GlobalInputArray(id) => {
                    let offset = self.info.metadata.buffer_len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                ruda::VariableKind::GlobalOutputArray(id) => {
                    let offset = self.info.metadata.buffer_len_index(id);
                    wgsl::Instruction::Metadata {
                        out: self.compile_variable(out),
                        info_offset: self.compile_variable(offset.into()),
                    }
                }
                _ => wgsl::Instruction::Length {
                    var: self.compile_variable(var),
                    out: self.compile_variable(out),
                },
            },
        }
    }
}
