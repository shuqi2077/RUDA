use super::*;

impl<D: Dialect, P: super::super::DialectProcessors> CppCompiler<D, P> {
    pub(super) fn compile_arithmetic(
        &mut self,
        value: gpu::Arithmetic,
        out: Option<gpu::Variable>,
        modes: InstructionModes,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Arithmetic::Add(op) => {
                instructions.push(Instruction::Add(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::SaturatingAdd(op) => {
                instructions.push(Instruction::SaturatingAdd(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Mul(op) => {
                instructions.push(Instruction::Mul(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Div(op) => {
                let op = self.compile_binary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::AllowReciprocal
                        | FastMath::ReducedPrecision
                        | FastMath::UnsignedZero
                        | FastMath::NotInf,
                    Instruction::Div(op),
                    Instruction::FastDiv(op),
                ))
            }
            gpu::Arithmetic::Sub(op) => {
                instructions.push(Instruction::Sub(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::SaturatingSub(op) => {
                instructions.push(Instruction::SaturatingSub(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::MulHi(op) => {
                let instruction = Instruction::HiMul(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Modulo(op) => {
                instructions.push(Instruction::Modulo(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Abs(op) => {
                instructions.push(Instruction::Abs(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Exp(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Exp(op),
                    Instruction::FastExp(op),
                ));
            }
            gpu::Arithmetic::Log(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Log(op),
                    Instruction::FastLog(op),
                ));
            }
            gpu::Arithmetic::Log1p(op) => {
                instructions.push(Instruction::Log1p(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Cos(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Cos(op),
                    Instruction::FastCos(op),
                ));
            }
            gpu::Arithmetic::Sin(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Sin(op),
                    Instruction::FastSin(op),
                ));
            }
            gpu::Arithmetic::Tan(op) => {
                instructions.push(Instruction::Tan(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Tanh(op) => {
                let op = self.compile_unary(op, out);
                let instruction = Instruction::Tanh(op);
                D::register_instruction_extension(&mut self.extensions, &instruction);
                if self.compilation_options.supports_features.fast_tanh {
                    instructions.push(self.select_fast_float(
                        out.ty,
                        modes,
                        FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                        instruction,
                        Instruction::FastTanh(op),
                    ))
                } else {
                    instructions.push(instruction);
                }
            }
            gpu::Arithmetic::Sinh(op) => {
                let instruction = Instruction::Sinh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Cosh(op) => {
                let instruction = Instruction::Cosh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::ArcCos(op) => {
                let instruction = Instruction::ArcCos(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::ArcSin(op) => {
                let instruction = Instruction::ArcSin(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::ArcTan(op) => {
                let instruction = Instruction::ArcTan(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::ArcSinh(op) => {
                let instruction = Instruction::ArcSinh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::ArcCosh(op) => {
                let instruction = Instruction::ArcCosh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::ArcTanh(op) => {
                let instruction = Instruction::ArcTanh(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Degrees(op) => {
                let instruction = Instruction::Degrees(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Radians(op) => {
                let instruction = Instruction::Radians(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::ArcTan2(op) => {
                let instruction = Instruction::ArcTan2(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Powf(op) => {
                let op = self.compile_binary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Powf(op),
                    Instruction::FastPowf(op),
                ))
            }
            gpu::Arithmetic::Powi(op) => {
                instructions.push(Instruction::Powi(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Hypot(op) => {
                instructions.push(Instruction::Hypot(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Rhypot(op) => {
                instructions.push(Instruction::Rhypot(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Sqrt(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Sqrt(op),
                    Instruction::FastSqrt(op),
                ))
            }
            gpu::Arithmetic::InverseSqrt(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::InverseSqrt(op),
                    Instruction::FastInverseSqrt(op),
                ))
            }
            gpu::Arithmetic::Erf(op) => {
                let instruction = Instruction::Erf(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Max(op) => {
                let instruction = Instruction::Max(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Min(op) => {
                let instruction = Instruction::Min(self.compile_binary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
            gpu::Arithmetic::Clamp(op) => instructions.push(Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(out),
            }),
            gpu::Arithmetic::Recip(op) => {
                let elem = op.input.ty.elem_type();
                let input = self.compile_variable(op.input);
                let out = self.compile_variable(out);
                let lhs = match elem {
                    gpu::ElemType::Float(_) => gpu::ConstantValue::Float(1.0),
                    gpu::ElemType::Int(_) => gpu::ConstantValue::Int(1),
                    gpu::ElemType::UInt(_) => gpu::ConstantValue::UInt(1),
                    gpu::ElemType::Bool => gpu::ConstantValue::Bool(true),
                };
                let div = Instruction::Div(BinaryInstruction {
                    lhs: Variable::Constant(lhs, self.compile_type(op.input.ty)),
                    rhs: input,
                    out,
                });
                let recip = Instruction::FastRecip(UnaryInstruction { input, out });

                instructions.push(self.select_fast_float(
                    elem.into(),
                    modes,
                    FastMath::AllowReciprocal
                        | FastMath::ReducedPrecision
                        | FastMath::UnsignedZero
                        | FastMath::NotInf,
                    div,
                    recip,
                ))
            }
            gpu::Arithmetic::Round(op) => {
                instructions.push(Instruction::Round(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Floor(op) => {
                instructions.push(Instruction::Floor(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Ceil(op) => {
                instructions.push(Instruction::Ceil(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Trunc(op) => {
                instructions.push(Instruction::Trunc(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Remainder(op) => {
                instructions.push(Instruction::Remainder(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::Fma(op) => instructions.push(Instruction::Fma {
                a: self.compile_variable(op.a),
                b: self.compile_variable(op.b),
                c: self.compile_variable(op.c),
                out: self.compile_variable(out),
            }),
            gpu::Arithmetic::Neg(op) => {
                instructions.push(Instruction::Neg(self.compile_unary(op, out)))
            }
            gpu::Arithmetic::Normalize(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Normalize(op),
                    Instruction::FastNormalize(op),
                ))
            }
            gpu::Arithmetic::Magnitude(op) => {
                let op = self.compile_unary(op, out);
                instructions.push(self.select_fast_float(
                    out.ty,
                    modes,
                    FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
                    Instruction::Magnitude(op),
                    Instruction::FastMagnitude(op),
                ))
            }
            gpu::Arithmetic::Dot(op) => {
                instructions.push(Instruction::Dot(self.compile_binary(op, out)))
            }
            gpu::Arithmetic::VectorSum(op) => {
                instructions.push(Instruction::VectorSum(self.compile_unary(op, out)))
            }
        };
    }

    pub(super) fn select_fast_float(
        &self,
        ty: gpu::Type,
        modes: InstructionModes,
        required_flags: EnumSet<FastMath>,
        default: Instruction<D>,
        fast: Instruction<D>,
    ) -> Instruction<D> {
        if !self.compilation_options.supports_features.fast_math
            || !matches!(ty.elem_type(), ElemType::Float(FloatKind::F32))
        {
            return default;
        }

        if modes.fp_math_mode.is_superset(required_flags) {
            fast
        } else {
            default
        }
    }

    pub(super) fn compile_comparison(
        &mut self,
        value: gpu::Comparison,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Comparison::Equal(op) => {
                instructions.push(Instruction::Equal(self.compile_binary(op, out)))
            }
            gpu::Comparison::Lower(op) => {
                instructions.push(Instruction::Lower(self.compile_binary(op, out)))
            }
            gpu::Comparison::Greater(op) => {
                instructions.push(Instruction::Greater(self.compile_binary(op, out)))
            }
            gpu::Comparison::LowerEqual(op) => {
                instructions.push(Instruction::LowerEqual(self.compile_binary(op, out)))
            }
            gpu::Comparison::GreaterEqual(op) => {
                instructions.push(Instruction::GreaterEqual(self.compile_binary(op, out)))
            }
            gpu::Comparison::NotEqual(op) => {
                instructions.push(Instruction::NotEqual(self.compile_binary(op, out)))
            }
            gpu::Comparison::IsNan(op) => {
                instructions.push(Instruction::IsNan(self.compile_unary(op, out)))
            }
            gpu::Comparison::IsInf(op) => {
                instructions.push(Instruction::IsInf(self.compile_unary(op, out)))
            }
        };
    }

    pub(super) fn compile_bitwise(
        &mut self,
        value: gpu::Bitwise,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Bitwise::BitwiseOr(op) => {
                instructions.push(Instruction::BitwiseOr(self.compile_binary(op, out)))
            }
            gpu::Bitwise::BitwiseAnd(op) => {
                instructions.push(Instruction::BitwiseAnd(self.compile_binary(op, out)))
            }
            gpu::Bitwise::BitwiseXor(op) => {
                instructions.push(Instruction::BitwiseXor(self.compile_binary(op, out)))
            }
            gpu::Bitwise::CountOnes(op) => {
                instructions.push(Instruction::CountBits(self.compile_unary(op, out)))
            }
            gpu::Bitwise::ReverseBits(op) => {
                instructions.push(Instruction::ReverseBits(self.compile_unary(op, out)))
            }
            gpu::Bitwise::ShiftLeft(op) => {
                instructions.push(Instruction::ShiftLeft(self.compile_binary(op, out)))
            }
            gpu::Bitwise::ShiftRight(op) => {
                instructions.push(Instruction::ShiftRight(self.compile_binary(op, out)))
            }
            gpu::Bitwise::BitwiseNot(op) => {
                instructions.push(Instruction::BitwiseNot(self.compile_unary(op, out)))
            }
            gpu::Bitwise::LeadingZeros(op) => {
                instructions.push(Instruction::LeadingZeros(self.compile_unary(op, out)))
            }
            gpu::Bitwise::TrailingZeros(op) => {
                instructions.push(Instruction::TrailingZeros(self.compile_unary(op, out)))
            }
            gpu::Bitwise::FindFirstSet(op) => {
                let instruction = Instruction::FindFirstSet(self.compile_unary(op, out));
                D::register_instruction_extension(&mut self.extensions, &instruction);
                instructions.push(instruction)
            }
        };
    }

    pub(super) fn compile_operator(
        &mut self,
        value: gpu::Operator,
        out: Option<gpu::Variable>,
        instructions: &mut Vec<Instruction<D>>,
    ) {
        let out = out.unwrap();
        match value {
            gpu::Operator::Index(op) | gpu::Operator::UncheckedIndex(op) => {
                instructions.push(Instruction::Index(self.compile_index(op, out)));
            }
            gpu::Operator::IndexAssign(op) | gpu::Operator::UncheckedIndexAssign(op) => {
                instructions.push(Instruction::IndexAssign(self.compile_index_assign(op, out)));
            }
            gpu::Operator::And(op) => {
                instructions.push(Instruction::And(self.compile_binary(op, out)))
            }
            gpu::Operator::Or(op) => {
                instructions.push(Instruction::Or(self.compile_binary(op, out)))
            }
            gpu::Operator::Not(op) => {
                instructions.push(Instruction::Not(self.compile_unary(op, out)))
            }
            gpu::Operator::InitVector(op) => instructions.push(Instruction::VecInit {
                inputs: op
                    .inputs
                    .into_iter()
                    .map(|it| self.compile_variable(it))
                    .collect(),
                out: self.compile_variable(out),
            }),
            gpu::Operator::CopyMemory(op) => instructions.push(Instruction::Copy {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
            }),
            gpu::Operator::CopyMemoryBulk(op) => instructions.push(Instruction::CopyBulk {
                input: self.compile_variable(op.input),
                in_index: self.compile_variable(op.in_index),
                out: self.compile_variable(out),
                out_index: self.compile_variable(op.out_index),
                len: op.len as u32,
            }),
            gpu::Operator::Select(op) => instructions.push(Instruction::Select {
                cond: self.compile_variable(op.cond),
                then: self.compile_variable(op.then),
                or_else: self.compile_variable(op.or_else),
                out: self.compile_variable(out),
            }),
            // Needs special conversion semantics
            gpu::Operator::Cast(op)
                if (is_fp4_fp6_fp8(op.input.elem_type()) || is_fp4_fp6_fp8(out.elem_type()))
                // Trivial broadcast shouldn't use special cast logic
                    && op.input.elem_type() != out.elem_type() =>
            {
                // We may need these for intermediates
                self.flags.elem_f16 = true;
                self.flags.elem_bf16 = true;
                let vec_in = op.input.ty.vector_size();
                let packing = out.storage_type().packing_factor();
                self.compile_type(op.input.ty.with_vector_size(packing));
                self.compile_type(
                    gpu::Type::scalar(gpu::ElemType::Float(FloatKind::F16))
                        .with_vector_size(vec_in),
                );
                self.compile_type(
                    gpu::Type::scalar(gpu::ElemType::Float(FloatKind::BF16))
                        .with_vector_size(vec_in),
                );
                self.compile_type(
                    gpu::Type::scalar(gpu::ElemType::Float(FloatKind::F16))
                        .with_vector_size(packing),
                );
                self.compile_type(
                    gpu::Type::scalar(gpu::ElemType::Float(FloatKind::BF16))
                        .with_vector_size(packing),
                );

                let inst = self.compile_unary(op, out);

                instructions.push(Instruction::SpecialCast(inst));
            }
            gpu::Operator::Cast(op) => {
                let op = self.compile_unary(op, out);

                if op.input.elem() == Elem::TF32 || op.out.elem() == Elem::TF32 {
                    self.flags.elem_tf32 = true;
                }

                instructions.push(Instruction::Assign(op))
            }
            gpu::Operator::Reinterpret(op) => {
                instructions.push(Instruction::Bitcast(self.compile_unary(op, out)))
            }
        };
    }

    pub(super) fn compile_binary(
        &mut self,
        value: gpu::BinaryOperator,
        out: gpu::Variable,
    ) -> BinaryInstruction<D> {
        BinaryInstruction {
            lhs: self.compile_variable(value.lhs),
            rhs: self.compile_variable(value.rhs),
            out: self.compile_variable(out),
        }
    }

    pub(super) fn compile_index_assign(
        &mut self,
        value: gpu::IndexAssignOperator,
        out: gpu::Variable,
    ) -> IndexAssignInstruction<D> {
        IndexAssignInstruction {
            index: self.compile_variable(value.index),
            value: self.compile_variable(value.value),
            vector_size: value.vector_size as u32,
            out: self.compile_variable(out),
        }
    }

    pub(super) fn compile_index(
        &mut self,
        value: gpu::IndexOperator,
        out: gpu::Variable,
    ) -> IndexInstruction<D> {
        IndexInstruction {
            list: self.compile_variable(value.list),
            index: self.compile_variable(value.index),
            vector_size: value.vector_size as u32,
            out: self.compile_variable(out),
        }
    }

    pub(super) fn compile_unary(
        &mut self,
        value: gpu::UnaryOperator,
        out: gpu::Variable,
    ) -> UnaryInstruction<D> {
        UnaryInstruction {
            input: self.compile_variable(value.input),
            out: self.compile_variable(out),
        }
    }
}
