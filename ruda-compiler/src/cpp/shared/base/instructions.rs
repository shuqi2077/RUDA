use super::*;

impl<D: Dialect, P: super::super::DialectProcessors> CppCompiler<D, P> {
    pub(super) fn compile_instruction(
        &mut self,
        instructions: &mut Vec<Instruction<D>>,
        instruction: gpu::Instruction,
    ) {
        self.update_debug_loc(instructions, &instruction);
        let out = instruction.out;
        match instruction.operation {
            gpu::Operation::Copy(variable) => {
                instructions.push(Instruction::Assign(UnaryInstruction {
                    input: self.compile_variable(variable),
                    out: self.compile_variable(out.unwrap()),
                }));
            }
            gpu::Operation::Arithmetic(op) => {
                self.compile_arithmetic(op, out, instruction.modes, instructions)
            }
            gpu::Operation::Comparison(op) => self.compile_comparison(op, out, instructions),
            gpu::Operation::Bitwise(op) => self.compile_bitwise(op, out, instructions),
            gpu::Operation::Operator(op) => self.compile_operator(op, out, instructions),
            gpu::Operation::Atomic(op) => self.compile_atomic(op, out, instructions),
            gpu::Operation::Metadata(op) => instructions.push(self.compile_metadata(op, out)),
            gpu::Operation::Branch(val) => self.compile_branch(instructions, val),
            gpu::Operation::Synchronization(val) => match val {
                gpu::Synchronization::SyncRuda => instructions.push(Instruction::SyncThreads),
                gpu::Synchronization::SyncPlane => instructions.push(Instruction::SyncWarp),
                gpu::Synchronization::SyncStorage => instructions.push(Instruction::SyncThreads),
                gpu::Synchronization::SyncAsyncProxyShared => {
                    self.flags.inst_tma = true;
                    instructions.push(Instruction::ProxyAsyncToSharedFence)
                }
            },
            gpu::Operation::Plane(op) => {
                self.flags.indexes.plane_dim_checked = true;
                let out = self.compile_variable(out.unwrap());
                match op {
                    gpu::Plane::Sum(op) => {
                        let instruction = WarpInstruction::ReduceSum {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction));
                    }
                    gpu::Plane::InclusiveSum(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::InclusiveSum {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::InclusiveProd(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::InclusiveProd {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::ExclusiveSum(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::ExclusiveSum {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::ExclusiveProd(op) => {
                        self.flags.indexes.unit_pos_plane = true;
                        instructions.push(Instruction::Warp(WarpInstruction::ExclusiveProd {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Prod(op) => {
                        let instruction = WarpInstruction::ReduceProd {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    gpu::Plane::Max(op) => {
                        let instruction = WarpInstruction::ReduceMax {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    gpu::Plane::Min(op) => {
                        let instruction = WarpInstruction::ReduceMin {
                            input: self.compile_variable(op.input),
                            out,
                        };
                        D::register_warp_instruction_extension(&mut self.extensions, &instruction);
                        instructions.push(Instruction::Warp(instruction))
                    }
                    gpu::Plane::Elect => {
                        if self.compilation_options.supports_features.elect_sync {
                            self.flags.inst_ptx_wrappers = true;
                            instructions.push(Instruction::Warp(WarpInstruction::Elect { out }))
                        } else {
                            instructions
                                .push(Instruction::Warp(WarpInstruction::ElectFallback { out }))
                        }
                    }
                    gpu::Plane::All(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::All {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Any(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Any {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Ballot(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Ballot {
                            input: self.compile_variable(op.input),
                            out,
                        }))
                    }
                    gpu::Plane::Broadcast(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Broadcast {
                            input: self.compile_variable(op.lhs),
                            id: self.compile_variable(op.rhs),
                            out,
                        }))
                    }
                    gpu::Plane::Shuffle(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::Shuffle {
                            input: self.compile_variable(op.lhs),
                            src_lane: self.compile_variable(op.rhs),
                            out,
                        }))
                    }
                    gpu::Plane::ShuffleXor(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::ShuffleXor {
                            input: self.compile_variable(op.lhs),
                            mask: self.compile_variable(op.rhs),
                            out,
                        }))
                    }
                    gpu::Plane::ShuffleUp(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::ShuffleUp {
                            input: self.compile_variable(op.lhs),
                            delta: self.compile_variable(op.rhs),
                            out,
                        }))
                    }
                    gpu::Plane::ShuffleDown(op) => {
                        instructions.push(Instruction::Warp(WarpInstruction::ShuffleDown {
                            input: self.compile_variable(op.lhs),
                            delta: self.compile_variable(op.rhs),
                            out,
                        }))
                    }
                }
            }
            gpu::Operation::CoopMma(cmma) => instructions.push(self.compile_cmma(cmma, out)),
            gpu::Operation::NonSemantic(debug) => match debug {
                gpu::NonSemantic::Print {
                    format_string,
                    args,
                } => instructions.push(Instruction::Printf {
                    format_string,
                    args: args
                        .into_iter()
                        .map(|arg| self.compile_variable(arg))
                        .collect(),
                }),
                gpu::NonSemantic::Comment { content } => {
                    instructions.push(Instruction::Comment { content })
                }
                // Don't need to handle scopes
                _ => {}
            },
            gpu::Operation::Barrier(barrier_ops) => match barrier_ops {
                gpu::BarrierOps::Declare { barrier } => {
                    let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type()
                    else {
                        unreachable!()
                    };
                    let barrier = self.compile_variable(barrier);
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Declare {
                        barrier,
                        level,
                    }));
                }
                gpu::BarrierOps::Init {
                    barrier,
                    is_elected,
                    arrival_count,
                } => {
                    let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type()
                    else {
                        unreachable!()
                    };
                    let barrier = self.compile_variable(barrier);
                    let arrival_count = self.compile_variable(arrival_count);
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Init {
                        barrier,
                        is_elected: self.compile_variable(is_elected),
                        arrival_count,
                        level,
                    }));
                }
                gpu::BarrierOps::InitManual {
                    barrier,
                    arrival_count,
                } => {
                    let barrier = self.compile_variable(barrier);
                    let arrival_count = self.compile_variable(arrival_count);
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::InitManual {
                            barrier,
                            arrival_count,
                        },
                    ));
                }
                gpu::BarrierOps::MemCopyAsync {
                    barrier,
                    source,
                    source_length,
                    offset_source,
                    offset_out,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsync {
                            barrier: self.compile_variable(barrier),
                            source: self.compile_variable(source),
                            destination: self.compile_variable(out.unwrap()),
                            source_length: self.compile_variable(source_length),
                            offset_source: self.compile_variable(offset_source),
                            offset_out: self.compile_variable(offset_out),
                            cooperative: false,
                        },
                    ));
                }
                gpu::BarrierOps::MemCopyAsyncCooperative {
                    barrier,
                    source,
                    source_length,
                    offset_source,
                    offset_out,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsync {
                            barrier: self.compile_variable(barrier),
                            source: self.compile_variable(source),
                            destination: self.compile_variable(out.unwrap()),
                            source_length: self.compile_variable(source_length),
                            offset_source: self.compile_variable(offset_source),
                            offset_out: self.compile_variable(offset_out),
                            cooperative: true,
                        },
                    ));
                }
                gpu::BarrierOps::MemCopyAsyncTx {
                    barrier,
                    source,
                    source_length,
                    offset_source,
                    offset_out,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsyncTx {
                            barrier: self.compile_variable(barrier),
                            source: self.compile_variable(source),
                            destination: self.compile_variable(out.unwrap()),
                            source_length: self.compile_variable(source_length),
                            offset_source: self.compile_variable(offset_source),
                            offset_out: self.compile_variable(offset_out),
                        },
                    ));
                }
                gpu::BarrierOps::CopyAsync {
                    source,
                    source_length,
                    offset_source,
                    offset_out,
                    copy_length,
                    checked,
                } => {
                    self.flags.inst_async_copy = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::CopyAsync {
                            source: self.compile_variable(source),
                            destination: self.compile_variable(out.unwrap()),
                            source_length: self.compile_variable(source_length),
                            offset_source: self.compile_variable(offset_source),
                            offset_out: self.compile_variable(offset_out),
                            copy_size: copy_length,
                            checked,
                        },
                    ));
                }
                gpu::BarrierOps::TmaLoad {
                    barrier,
                    tensor_map,
                    offset_out,
                    indices,
                } => {
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::MemCopyAsyncTensorGlobalToShared {
                            barrier: self.compile_variable(barrier),
                            smem_buffer: self.compile_variable(out.unwrap()),
                            smem_offset: self.compile_variable(offset_out),
                            tensor_map: self.compile_variable(tensor_map),
                            indices: indices
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                        },
                    ));
                }
                gpu::BarrierOps::TmaLoadIm2col {
                    barrier,
                    tensor_map,
                    offset_out,
                    indices,
                    offsets,
                } => {
                    self.flags.inst_tma_im2col = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::TmaLoadIm2col {
                            barrier: self.compile_variable(barrier),
                            smem_buffer: self.compile_variable(out.unwrap()),
                            smem_offset: self.compile_variable(offset_out),
                            tensor_map: self.compile_variable(tensor_map),
                            indices: indices
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                            offsets: offsets
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                        },
                    ));
                }
                gpu::BarrierOps::Arrive { barrier } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Arrive {
                        barrier: self.compile_variable(barrier),
                        token: self.compile_variable(out.unwrap()),
                    }))
                }
                gpu::BarrierOps::ArriveTx {
                    barrier,
                    arrive_count_update,
                    transaction_count_update,
                } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::ArriveTx {
                        barrier: self.compile_variable(barrier),
                        token: self.compile_variable(out.unwrap()),
                        arrive_count_update: self.compile_variable(arrive_count_update),
                        transaction_count_update: self.compile_variable(transaction_count_update),
                    }))
                }
                gpu::BarrierOps::CommitCopyAsync { barrier } => {
                    self.flags.inst_async_copy = true;
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::ArriveCopyAsync {
                            barrier: self.compile_variable(barrier),
                        },
                    ))
                }
                gpu::BarrierOps::ExpectTx {
                    barrier,
                    transaction_count_update,
                } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::ExpectTx {
                        barrier: self.compile_variable(barrier),
                        transaction_count_update: self.compile_variable(transaction_count_update),
                    }))
                }
                gpu::BarrierOps::Wait { barrier, token } => {
                    instructions.push(Instruction::Barrier(super::barrier::BarrierOps::Wait {
                        barrier: self.compile_variable(barrier),
                        token: self.compile_variable(token),
                    }))
                }
                gpu::BarrierOps::WaitParity { barrier, phase } => instructions.push(
                    Instruction::Barrier(super::barrier::BarrierOps::WaitParity {
                        barrier: self.compile_variable(barrier),
                        phase: self.compile_variable(phase),
                    }),
                ),
                gpu::BarrierOps::ArriveAndWait { barrier } => {
                    let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type()
                    else {
                        unreachable!()
                    };
                    instructions.push(Instruction::Barrier(
                        super::barrier::BarrierOps::ArriveAndWait {
                            barrier: self.compile_variable(barrier),
                            level,
                        },
                    ))
                }
            },
            gpu::Operation::Tma(tma_ops) => {
                self.flags.inst_tma = true;
                match tma_ops {
                    gpu::TmaOps::TmaStore {
                        source,
                        coordinates,
                        offset_source,
                    } => {
                        instructions.push(Instruction::MemCopyAsyncTensorSharedToGlobal {
                            smem_buffer: self.compile_variable(source),
                            smem_offset: self.compile_variable(offset_source),
                            tensor_map: self.compile_variable(out.unwrap()),
                            indices: coordinates
                                .into_iter()
                                .map(|it| self.compile_variable(it))
                                .collect(),
                        });
                    }
                    gpu::TmaOps::CommitGroup => {
                        instructions.push(Instruction::BulkCommitGroup);
                    }
                    gpu::TmaOps::WaitGroup { max_pending } => {
                        instructions.push(Instruction::BulkWaitGroup { max_pending });
                    }
                    gpu::TmaOps::WaitGroupRead { max_pending } => {
                        instructions.push(Instruction::BulkWaitGroupRead { max_pending });
                    }
                }
            }
            gpu::Operation::Marker(_) => {}
        }
    }
}
