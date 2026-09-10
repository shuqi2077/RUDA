use super::*;

impl<P: WgslLowering> WgslCompiler<P> {
    pub(super) fn compile_cmp(
        &mut self,
        value: ruda::Comparison,
        out: Option<ruda::Variable>,
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            ruda::Comparison::Equal(op) => instructions.push(wgsl::Instruction::Equal {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Comparison::Lower(op) => instructions.push(wgsl::Instruction::Lower {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Comparison::Greater(op) => instructions.push(wgsl::Instruction::Greater {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Comparison::LowerEqual(op) => instructions.push(wgsl::Instruction::LowerEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Comparison::GreaterEqual(op) => {
                instructions.push(wgsl::Instruction::GreaterEqual {
                    lhs: self.compile_variable(op.lhs),
                    rhs: self.compile_variable(op.rhs),
                    out: self.compile_variable(out),
                })
            }
            ruda::Comparison::NotEqual(op) => instructions.push(wgsl::Instruction::NotEqual {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Comparison::IsNan(op) => instructions.push(wgsl::Instruction::IsNan {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Comparison::IsInf(op) => instructions.push(wgsl::Instruction::IsInf {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
        }
    }

    pub(super) fn compile_bitwise(
        &mut self,
        value: ruda::Bitwise,
        out: Option<ruda::Variable>,
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        let out = out.unwrap();
        match value {
            ruda::Bitwise::BitwiseOr(op) => instructions.push(wgsl::Instruction::BitwiseOr {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::BitwiseAnd(op) => instructions.push(wgsl::Instruction::BitwiseAnd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::BitwiseXor(op) => instructions.push(wgsl::Instruction::BitwiseXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::CountOnes(op) => instructions.push(wgsl::Instruction::CountBits {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::ReverseBits(op) => instructions.push(wgsl::Instruction::ReverseBits {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::ShiftLeft(op) => instructions.push(wgsl::Instruction::ShiftLeft {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::ShiftRight(op) => instructions.push(wgsl::Instruction::ShiftRight {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::BitwiseNot(op) => instructions.push(wgsl::Instruction::BitwiseNot {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::LeadingZeros(op) => instructions.push(wgsl::Instruction::LeadingZeros {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Bitwise::TrailingZeros(op) => {
                instructions.push(wgsl::Instruction::TrailingZeros {
                    input: self.compile_variable(op.input),
                    out: self.compile_variable(out),
                })
            }
            ruda::Bitwise::FindFirstSet(op) => instructions.push(wgsl::Instruction::FindFirstSet {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
        }
    }

    pub(super) fn compile_operator(
        &mut self,
        value: ruda::Operator,
        out: Option<ruda::Variable>,
        instructions: &mut Vec<wgsl::Instruction>,
    ) {
        if matches!(value, ruda::Operator::NativeAddress(_) | ruda::Operator::NativeLoad(_) | ruda::Operator::NativeStore(_)) {
            panic!("WGSL does not expose native device addresses");
        }
        let out = out.unwrap();
        match value {
            ruda::Operator::NativeAddress(_) | ruda::Operator::NativeLoad(_) | ruda::Operator::NativeStore(_) => unreachable!(),
            ruda::Operator::Cast(op) => instructions.push(wgsl::Instruction::Assign {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Operator::Index(op) | ruda::Operator::UncheckedIndex(op) => {
                instructions.push(wgsl::Instruction::Index {
                    lhs: self.compile_variable(op.list),
                    rhs: self.compile_variable(op.index),
                    out: self.compile_variable(out),
                });
            }
            ruda::Operator::IndexAssign(op) | ruda::Operator::UncheckedIndexAssign(op) => {
                instructions.push(wgsl::Instruction::IndexAssign {
                    index: self.compile_variable(op.index),
                    rhs: self.compile_variable(op.value),
                    out: self.compile_variable(out),
                })
            }
            ruda::Operator::And(op) => instructions.push(wgsl::Instruction::And {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Operator::Or(op) => instructions.push(wgsl::Instruction::Or {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Operator::Not(op) => instructions.push(wgsl::Instruction::Not {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Operator::Reinterpret(op) => instructions.push(wgsl::Instruction::Bitcast {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Operator::InitVector(op) => instructions.push(wgsl::Instruction::VecInit {
                inputs: op
                    .inputs
                    .into_iter()
                    .map(|var| self.compile_variable(var))
                    .collect(),
                out: self.compile_variable(out),
            }),
            ruda::Operator::CopyMemory(op) => instructions.push(wgsl::Instruction::Copy {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
            }),
            ruda::Operator::CopyMemoryBulk(op) => instructions.push(wgsl::Instruction::CopyBulk {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
                len: op.len as u32,
            }),
            ruda::Operator::Select(op) => instructions.push(wgsl::Instruction::Select {
                cond: self.compile_variable(op.cond),
                then: self.compile_variable(op.then),
                or_else: self.compile_variable(op.or_else),
                out: self.compile_variable(out),
            }),
        }
    }

    pub(super) fn compile_atomic(
        &mut self,
        atomic: ruda::AtomicOp,
        out: Option<ruda::Variable>,
    ) -> wgsl::Instruction {
        let out = out.unwrap();
        match atomic {
            ruda::AtomicOp::Add(op) => wgsl::Instruction::AtomicAdd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Sub(op) => wgsl::Instruction::AtomicSub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Max(op) => wgsl::Instruction::AtomicMax {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Min(op) => wgsl::Instruction::AtomicMin {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::And(op) => wgsl::Instruction::AtomicAnd {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Or(op) => wgsl::Instruction::AtomicOr {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Xor(op) => wgsl::Instruction::AtomicXor {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Load(op) => wgsl::Instruction::AtomicLoad {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Store(op) => wgsl::Instruction::AtomicStore {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::Swap(op) => wgsl::Instruction::AtomicSwap {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            },
            ruda::AtomicOp::CompareAndSwap(op) => wgsl::Instruction::AtomicCompareExchangeWeak {
                lhs: self.compile_variable(op.input),
                cmp: self.compile_variable(op.cmp),
                value: self.compile_variable(op.val),
                out: self.compile_variable(out),
            },
        }
    }
}
