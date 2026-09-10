use super::*;

impl<D: Dialect, P: super::super::DialectProcessors> CppCompiler<D, P> {
    pub(super) fn compile_scope(&mut self, scope: &mut gpu::Scope) -> Vec<Instruction<D>> {
        let mut instructions = Vec::new();

        let const_arrays = scope
            .const_arrays
            .drain(..)
            .map(|(var, values)| ConstArray {
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

        let checked_io: Box<dyn Processor> = P::checked_io(
            self.strategy,
            self.kernel_name.clone(),
        );
        let dialect_processors = P::processors();
        let mut processors: Vec<&dyn Processor> = vec![&*checked_io];
        processors.extend(dialect_processors.iter().map(|it| &**it));

        let processing = scope.process(processors);

        for var in processing.variables {
            instructions.push(Instruction::DeclareVariable {
                var: self.compile_variable(var),
            });
        }

        processing
            .instructions
            .into_iter()
            .for_each(|op| self.compile_instruction(&mut instructions, op));

        instructions
    }

    pub(super) fn update_debug_loc(
        &mut self,
        instructions: &mut Vec<Instruction<D>>,
        inst: &gpu::Instruction,
    ) {
        if !matches!(inst.operation, Operation::NonSemantic(_)) {
            match &inst.source_loc {
                Some(loc) if Some(loc) != self.source_loc.as_ref() => {
                    self.source_loc = Some(loc.clone());
                    instructions.push(Instruction::Line {
                        file: loc.source.file.clone(),
                        line: loc.line,
                    });
                }
                _ => {}
            }
        }
    }

    pub(super) fn compile_cmma(&mut self, cmma: gpu::CoopMma, out: Option<gpu::Variable>) -> Instruction<D> {
        self.flags.inst_wmma = true;

        let out = self.compile_variable(out.unwrap());

        let inst = match cmma {
            gpu::CoopMma::Fill { value } => WmmaInstruction::Fill {
                frag: out,
                value: self.compile_variable(value),
            },
            gpu::CoopMma::Load {
                value,
                stride,
                offset,
                layout,
            } => WmmaInstruction::Load {
                frag: out,
                offset: self.compile_variable(offset),
                value: self.compile_variable(value),
                stride: self.compile_variable(stride),
                layout: layout.and_then(|l| self.compile_matrix_layout(l)),
            },
            gpu::CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => WmmaInstruction::Execute {
                frag_a: self.compile_variable(mat_a),
                frag_b: self.compile_variable(mat_b),
                frag_c: self.compile_variable(mat_c),
                frag_d: out,
                warp_size: self.compilation_options.warp_size,
            },
            gpu::CoopMma::ExecuteManual {
                matrix,
                registers_a,
                registers_b,
                registers_c,
            } => WmmaInstruction::ExecuteManual {
                shape: MmaShape::new(matrix.m as u32, matrix.n as u32, matrix.k as u32),
                frag_a: self.compile_variable(registers_a),
                frag_b: self.compile_variable(registers_b),
                frag_c: self.compile_variable(registers_c),
                frag_d: out,
            },
            gpu::CoopMma::ExecuteScaled {
                matrix,
                registers_a,
                registers_b,
                registers_c,
                scales_a,
                scales_b,
                scales_factor,
            } => WmmaInstruction::ExecuteScaled {
                shape: MmaShape::new(matrix.m as u32, matrix.n as u32, matrix.k as u32),
                frag_a: self.compile_variable(registers_a),
                frag_b: self.compile_variable(registers_b),
                frag_c: self.compile_variable(registers_c),
                frag_d: out,

                scales_a: self.compile_variable(scales_a),
                scales_b: self.compile_variable(scales_b),
                scales_factor: scales_factor as u32,
            },
            gpu::CoopMma::Store {
                mat,
                stride,
                offset,
                layout,
            } => {
                self.flags.indexes.unit_pos = true;
                self.flags.indexes.plane_pos = true;
                WmmaInstruction::Store {
                    output: out,
                    offset: self.compile_variable(offset),
                    frag: self.compile_variable(mat),
                    stride: self.compile_variable(stride),
                    layout: self
                        .compile_matrix_layout(layout)
                        .expect("Layout required for store instruction"),
                }
            }
            gpu::CoopMma::LoadMatrix {
                buffer,
                offset,
                vector_size,
                factor,
                transpose,
            } => WmmaInstruction::LdMatrix {
                output: out,
                buffer: self.compile_variable(buffer),
                offset: self.compile_variable(offset),
                vector_size,
                factor: factor as u32,
                transpose,
            },
            gpu::CoopMma::StoreMatrix {
                offset,
                vector_size,
                registers,
                factor,
                transpose,
            } => WmmaInstruction::StMatrix {
                registers: self.compile_variable(registers),
                buffer: out,
                offset: self.compile_variable(offset),
                vector_size,
                factor: factor as u32,
                transpose,
            },
            gpu::CoopMma::Cast { input } => WmmaInstruction::Cast {
                input: self.compile_variable(input),
                output: out,
            },
            gpu::CoopMma::RowIndex { .. } | gpu::CoopMma::ColIndex { .. } => {
                panic!("Row/Col index should be handled by processors")
            }
        };

        D::register_wmma_instruction_extension(&mut self.extensions, &inst);

        Instruction::Wmma(inst)
    }

    pub(super) fn compile_metadata(
        &mut self,
        metadata: gpu::Metadata,
        out: Option<gpu::Variable>,
    ) -> Instruction<D> {
        let out = out.unwrap();
        match metadata {
            gpu::Metadata::Stride { dim, var } => {
                let position = self.ext_meta_position(var);
                let offset = self.info.metadata.stride_offset_index(position);
                Instruction::ExtendedMetadata {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Shape { dim, var } => {
                let position = self.ext_meta_position(var);
                let offset = self.info.metadata.shape_offset_index(position);
                Instruction::ExtendedMetadata {
                    info_offset: self.compile_variable(offset.into()),
                    dim: self.compile_variable(dim),
                    out: self.compile_variable(out),
                }
            }
            gpu::Metadata::Rank { var } => {
                let out = self.compile_variable(out);
                let pos = self.ext_meta_position(var);
                let offset = self.info.metadata.rank_index(pos);
                super::Instruction::Metadata {
                    info_offset: self.compile_variable(offset.into()),
                    out,
                }
            }
            gpu::Metadata::Length { var } => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(out);

                match input {
                    Variable::Slice { .. } => Instruction::SliceLength { input, out },
                    Variable::SharedArray(_id, _item, length) => {
                        Instruction::ConstLength { length, out }
                    }
                    _ => {
                        let id = input.id().expect("Variable should have id");
                        let offset = self.info.metadata.len_index(id);
                        Instruction::Metadata {
                            info_offset: self.compile_variable(offset.into()),
                            out,
                        }
                    }
                }
            }
            gpu::Metadata::BufferLength { var } => {
                let input = self.compile_variable(var);
                let out = self.compile_variable(out);

                match input {
                    Variable::Slice { .. } => Instruction::SliceLength { input, out },
                    _ => {
                        let id = input.id().expect("Variable should have id");
                        let offset = self.info.metadata.buffer_len_index(id);
                        Instruction::Metadata {
                            info_offset: self.compile_variable(offset.into()),
                            out,
                        }
                    }
                }
            }
        }
    }

    pub(super) fn compile_branch(&mut self, instructions: &mut Vec<Instruction<D>>, branch: gpu::Branch) {
        match branch {
            gpu::Branch::If(mut op) => instructions.push(Instruction::If {
                cond: self.compile_variable(op.cond),
                instructions: self.compile_scope(&mut op.scope),
            }),
            gpu::Branch::IfElse(mut op) => instructions.push(Instruction::IfElse {
                cond: self.compile_variable(op.cond),
                instructions_if: self.compile_scope(&mut op.scope_if),
                instructions_else: self.compile_scope(&mut op.scope_else),
            }),
            gpu::Branch::Switch(mut op) => instructions.push(Instruction::Switch {
                value: self.compile_variable(op.value),
                instructions_default: self.compile_scope(&mut op.scope_default),
                instructions_cases: op
                    .cases
                    .into_iter()
                    .map(|(val, mut block)| {
                        (self.compile_variable(val), self.compile_scope(&mut block))
                    })
                    .collect(),
            }),
            gpu::Branch::Return => instructions.push(Instruction::Return),
            gpu::Branch::Break => instructions.push(Instruction::Break),
            gpu::Branch::Unreachable => instructions.push(Instruction::Unreachable),
            gpu::Branch::RangeLoop(mut range_loop) => instructions.push(Instruction::RangeLoop {
                i: self.compile_variable(range_loop.i),
                start: self.compile_variable(range_loop.start),
                end: self.compile_variable(range_loop.end),
                step: range_loop.step.map(|it| self.compile_variable(it)),
                inclusive: range_loop.inclusive,
                instructions: self.compile_scope(&mut range_loop.scope),
            }),
            gpu::Branch::Loop(mut op) => instructions.push(Instruction::Loop {
                instructions: self.compile_scope(&mut op.scope),
            }),
        };
    }

    pub(super) fn compile_atomic(
        &mut self,
        value: gpu::AtomicOp,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::AtomicOp::Load(op) => {
                instructions.push(Instruction::AtomicLoad(self.compile_unary(op, out)))
            }
            gpu::AtomicOp::Store(op) => {
                instructions.push(Instruction::AtomicStore(self.compile_unary(op, out)))
            }
            gpu::AtomicOp::Swap(op) => {
                instructions.push(Instruction::AtomicSwap(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Add(op) => {
                instructions.push(Instruction::AtomicAdd(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Sub(op) => {
                instructions.push(Instruction::AtomicSub(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Max(op) => {
                instructions.push(Instruction::AtomicMax(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Min(op) => {
                instructions.push(Instruction::AtomicMin(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::And(op) => {
                instructions.push(Instruction::AtomicAnd(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Or(op) => {
                instructions.push(Instruction::AtomicOr(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::Xor(op) => {
                instructions.push(Instruction::AtomicXor(self.compile_binary(op, out)))
            }
            gpu::AtomicOp::CompareAndSwap(op) => instructions.push(Instruction::AtomicCAS {
                input: self.compile_variable(op.input),
                cmp: self.compile_variable(op.cmp),
                val: self.compile_variable(op.val),
                out: self.compile_variable(out),
            }),
        }
    }
}
