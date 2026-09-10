use super::*;

impl<P: WgslLowering> WgslCompiler<P> {
    pub(super) fn compile_arithmetic(
        &mut self,
        value: ruda::Arithmetic,
        out: Option<ruda::Variable>,
        instructions: &mut Vec<wgsl::Instruction>,
        scope: &mut Scope,
    ) {
        let out = out.unwrap();
        match value {
            ruda::Arithmetic::Max(op) => instructions.push(wgsl::Instruction::Max {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Min(op) => instructions.push(wgsl::Instruction::Min {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Add(op) => instructions.push(wgsl::Instruction::Add {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::SaturatingAdd(_) => {
                unreachable!("Saturating add should be removed by processor");
            }
            ruda::Arithmetic::Fma(op) => instructions.push(wgsl::Instruction::Fma {
                a: self.compile_variable(op.a),
                b: self.compile_variable(op.b),
                c: self.compile_variable(op.c),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Modulo(op) => instructions.push(wgsl::Instruction::Modulo {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Sub(op) => instructions.push(wgsl::Instruction::Sub {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::SaturatingSub(_) => {
                unreachable!("Saturating sub should be removed by processor");
            }
            ruda::Arithmetic::Mul(op) => instructions.push(wgsl::Instruction::Mul {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Div(op) => instructions.push(wgsl::Instruction::Div {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Abs(op) => instructions.push(wgsl::Instruction::Abs {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Exp(op) => instructions.push(wgsl::Instruction::Exp {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Log(op) => instructions.push(wgsl::Instruction::Log {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Log1p(op) => instructions.push(wgsl::Instruction::Log1p {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Cos(op) => instructions.push(wgsl::Instruction::Cos {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Sin(op) => instructions.push(wgsl::Instruction::Sin {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Tan(op) => instructions.push(wgsl::Instruction::Tan {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Tanh(op) => instructions.push(wgsl::Instruction::Tanh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Sinh(op) => instructions.push(wgsl::Instruction::Sinh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Cosh(op) => instructions.push(wgsl::Instruction::Cosh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::ArcCos(op) => instructions.push(wgsl::Instruction::ArcCos {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::ArcSin(op) => instructions.push(wgsl::Instruction::ArcSin {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::ArcTan(op) => instructions.push(wgsl::Instruction::ArcTan {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::ArcSinh(op) => instructions.push(wgsl::Instruction::ArcSinh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::ArcCosh(op) => instructions.push(wgsl::Instruction::ArcCosh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::ArcTanh(op) => instructions.push(wgsl::Instruction::ArcTanh {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Degrees(op) => instructions.push(wgsl::Instruction::Degrees {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Radians(op) => instructions.push(wgsl::Instruction::Radians {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::ArcTan2(op) => instructions.push(wgsl::Instruction::ArcTan2 {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Powi(op) if matches!(op.lhs.elem_type(), ruda::ElemType::Float(_)) => {
                let mut scope = scope.child();
                P::expand_integer_power(&mut scope, op.lhs, op.rhs, out);
                instructions.extend(self.compile_scope(&mut scope));
            }
            ruda::Arithmetic::Powf(op) | ruda::Arithmetic::Powi(op) => {
                instructions.push(wgsl::Instruction::Powf {
                    lhs: self.compile_variable(op.lhs),
                    rhs: self.compile_variable(op.rhs),
                    out: self.compile_variable(out),
                })
            }
            ruda::Arithmetic::Hypot(op) => {
                let mut scope = scope.child();
                P::expand_hypot(&mut scope, op.lhs, op.rhs, out);
                instructions.extend(self.compile_scope(&mut scope));
            }
            ruda::Arithmetic::Rhypot(op) => {
                let mut scope = scope.child();
                P::expand_rhypot(&mut scope, op.lhs, op.rhs, out);
                instructions.extend(self.compile_scope(&mut scope));
            }

            ruda::Arithmetic::Sqrt(op) => instructions.push(wgsl::Instruction::Sqrt {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::InverseSqrt(op) => {
                instructions.push(wgsl::Instruction::InverseSqrt {
                    input: self.compile_variable(op.input),
                    out: self.compile_variable(out),
                })
            }
            ruda::Arithmetic::Round(op) => instructions.push(wgsl::Instruction::Round {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Floor(op) => instructions.push(wgsl::Instruction::Floor {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Ceil(op) => instructions.push(wgsl::Instruction::Ceil {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Trunc(op) => instructions.push(wgsl::Instruction::Trunc {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Erf(op) => {
                let mut scope = scope.child();
                P::expand_erf(&mut scope, op.input, out);
                instructions.extend(self.compile_scope(&mut scope));
            }
            ruda::Arithmetic::MulHi(op) => {
                let mut scope = scope.child();
                match self.compilation_options.supports_u64 {
                    true => P::expand_himul_64(&mut scope, op.lhs, op.rhs, out),
                    false => P::expand_himul_sim(&mut scope, op.lhs, op.rhs, out),
                }
                instructions.extend(self.compile_scope(&mut scope));
            }
            ruda::Arithmetic::Recip(op) => instructions.push(wgsl::Instruction::Recip {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Clamp(op) => instructions.push(wgsl::Instruction::Clamp {
                input: self.compile_variable(op.input),
                min_value: self.compile_variable(op.min_value),
                max_value: self.compile_variable(op.max_value),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Remainder(op) => instructions.push(wgsl::Instruction::Remainder {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Neg(op) => instructions.push(wgsl::Instruction::Negate {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Magnitude(op) => instructions.push(wgsl::Instruction::Magnitude {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Normalize(op) => instructions.push(wgsl::Instruction::Normalize {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::Dot(op) => instructions.push(wgsl::Instruction::Dot {
                lhs: self.compile_variable(op.lhs),
                rhs: self.compile_variable(op.rhs),
                out: self.compile_variable(out),
            }),
            ruda::Arithmetic::VectorSum(op) => instructions.push(wgsl::Instruction::VectorSum {
                input: self.compile_variable(op.input),
                out: self.compile_variable(out),
            }),
        }
    }
}
