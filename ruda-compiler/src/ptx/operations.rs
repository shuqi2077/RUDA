use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{
    Arithmetic, BinaryOperator, Branch, Comparison, Instruction, Metadata, Operation, Operator,
    Synchronization, Variable,
};

impl Emitter {
    pub fn instruction(&mut self, instruction: Instruction) -> Result<()> {
        let output = instruction.out;
        self.instruction_inner(instruction)?;
        if let Some(out) = output
            && matches!(out.kind, ruda_core::ir::VariableKind::LocalMut { .. }
                | ruda_core::ir::VariableKind::LocalConst { .. }
                | ruda_core::ir::VariableKind::Versioned { .. })
            && let Ok(ty) = Scalar::of(out.ty.with_vector_size(1))
            && ty.narrow()
        {
            for register in self.vector_values(out)? {
                self.normalize_integer(ty, &register);
            }
        }
        Ok(())
    }

    fn instruction_inner(&mut self, instruction: Instruction) -> Result<()> {
        self.fp8_instruction_supported(&instruction)?;
        if self.atomic_instruction(&instruction)? {
            return Ok(());
        }
        if self.vector_instruction(&instruction)? {
            return Ok(());
        }
        if self.fast_math_instruction(&instruction)? {
            return Ok(());
        }
        let out = || {
            instruction
                .out
                .ok_or_else(|| invalid("instruction requires an output"))
        };
        match instruction.operation {
            Operation::Copy(value) if matches!(value.kind, ruda_core::ir::VariableKind::Matrix { .. }) => self.matrix_copy(out()?, value),
            Operation::Copy(value) => self.copy(out()?, value),
            Operation::Arithmetic(op) => self.arithmetic(op, out()?),
            Operation::Bitwise(op) => self.bitwise(op, out()?),
            Operation::Plane(op) => self.plane(op, out()?),
            Operation::CoopMma(op) => self.matrix(op, out()?),
            Operation::Comparison(op) => self.comparison(op, out()?),
            Operation::Operator(op) => match op {
                Operator::Index(op) => {
                    Self::scalar_index(op.vector_size, op.unroll_factor)?;
                    self.memory(op.list, op.index, out()?, false, true)
                }
                Operator::UncheckedIndex(op) => {
                    Self::scalar_index(op.vector_size, op.unroll_factor)?;
                    self.memory(op.list, op.index, out()?, false, false)
                }
                Operator::IndexAssign(op) => {
                    Self::scalar_index(op.vector_size, op.unroll_factor)?;
                    self.memory(out()?, op.index, op.value, true, true)
                }
                Operator::UncheckedIndexAssign(op) => {
                    Self::scalar_index(op.vector_size, op.unroll_factor)?;
                    self.memory(out()?, op.index, op.value, true, false)
                }
                Operator::Cast(op) => self.cast(out()?, op.input),
                Operator::Reinterpret(op) => self.reinterpret(out()?, op.input),
                Operator::CopyMemory(op) => self.copy_memory(op.input, op.in_index, None, out()?, op.out_index, None, 1),
                Operator::CopyMemoryBulk(op) => self.copy_memory(op.input, op.in_index, Some(op.offset_input), out()?, op.out_index, Some(op.offset_out), op.len),
                Operator::And(op) => self.binary(out()?, op, "and", Scalar::Pred),
                Operator::Or(op) => self.binary(out()?, op, "or", Scalar::Pred),
                Operator::Not(op) => {
                    let output = out()?;
                    if Scalar::of(output.ty)? != Scalar::Pred || output.ty != op.input.ty {
                        return Err(invalid("logical not requires predicates"));
                    }
                    let input = self.value(op.input)?;
                    let dst = self.destination(output)?;
                    self.line(format!("not.pred {dst}, {input};"));
                    Ok(())
                }
                Operator::Select(op) => {
                    let output = out()?;
                    let ty = Scalar::of(output.ty)?;
                    if Scalar::of(op.cond.ty)? != Scalar::Pred
                        || op.then.ty != output.ty
                        || op.or_else.ty != output.ty
                    {
                        return Err(invalid("select operand types"));
                    }
                    let pred = self.value(op.cond)?;
                    let a = self.value(op.then)?;
                    let b = self.value(op.or_else)?;
                    let dst = self.destination(output)?;
                    if ty == Scalar::Pred {
                        let yes = self.reg(Scalar::Pred);
                        let no = self.reg(Scalar::Pred);
                        self.line(format!("and.pred {yes}, {pred}, {a};"));
                        self.line(format!("not.pred {no}, {pred};"));
                        self.line(format!("and.pred {no}, {no}, {b};"));
                        self.line(format!("or.pred {dst}, {yes}, {no};"));
                    } else {
                        self.line(format!("selp.{} {dst}, {a}, {b}, {pred};", ty.storage()));
                    }
                    Ok(())
                }
                other => Err(unsupported(format!("operator {other:?}"))),
            },
            Operation::Metadata(op) => {
                let output = out()?;
                if Scalar::of(output.ty)? != self.address {
                    return Err(invalid("metadata output must match address type"));
                }
                let value = match op {
                    Metadata::Length { var } => self.meta(var, true)?,
                    Metadata::BufferLength { var } => self.meta(var, false)?,
                    Metadata::Rank { var } => self.rank(var)?,
                    Metadata::Shape { dim, var } => self.extended_metadata(var, dim, true)?,
                    Metadata::Stride { dim, var } => self.extended_metadata(var, dim, false)?,
                };
                let dst = self.destination(output)?;
                self.line(format!("mov.{} {dst}, {value};", self.address.suffix()));
                Ok(())
            }
            Operation::Branch(branch) => self.branch(branch),
            Operation::Barrier(operation) => self.barrier(operation, instruction.out),
            Operation::Tma(operation) => self.tma(operation, instruction.out),
            Operation::NonSemantic(ruda_core::ir::NonSemantic::EnterDebugScope | ruda_core::ir::NonSemantic::ExitDebugScope) => Ok(()),
            Operation::NonSemantic(ruda_core::ir::NonSemantic::Print { format_string, args }) => {
                self.printf(&format_string, &args)
            }
            Operation::NonSemantic(ruda_core::ir::NonSemantic::Comment { content }) => {
                for line in content.lines() { self.line(format!("// {line}")); }
                Ok(())
            }
            Operation::Marker(ruda_core::ir::Marker::Free(var))
                if matches!(var.kind, ruda_core::ir::VariableKind::SharedArray { .. } | ruda_core::ir::VariableKind::Shared { .. }) =>
            {
                Ok(())
            }
            Operation::Synchronization(Synchronization::SyncRuda | Synchronization::SyncStorage) => {
                self.line("bar.sync 0;");
                Ok(())
            }
            Operation::Synchronization(Synchronization::SyncPlane) => {
                self.line("bar.warp.sync 0xffffffff;");
                Ok(())
            }
            Operation::Synchronization(Synchronization::SyncAsyncProxyShared) => {
                if self.target.sm < 90 || self.target.version < (8, 0) {
                    return Err(unsupported("async proxy fence requires SM >= 90 and PTX >= 8.0"));
                }
                self.line("fence.proxy.async.shared::cta;");
                Ok(())
            }
            other => Err(unsupported(format!("operation {other:?}"))),
        }
    }

    fn scalar_index(vector_size: usize, unroll_factor: usize) -> Result<()> {
        if vector_size > 1 || unroll_factor != 1 {
            return Err(unsupported("vectorized / unrolled array access"));
        }
        Ok(())
    }

    fn copy(&mut self, out: Variable, input: Variable) -> Result<()> {
        if out.ty != input.ty {
            return Err(invalid("copy types differ"));
        }
        if out.ty == ruda_core::ir::Type::Semantic(ruda_core::ir::SemanticType::BarrierToken) {
            let src = self.barrier_token_value(input)?;
            let dst = self.barrier_token_destination(out)?;
            self.line(format!("mov.b64 {dst}, {src};"));
            return Ok(());
        }
        let ty = Scalar::of(out.ty)?;
        let src = self.value(input)?;
        let dst = self.destination(out)?;
        self.line(format!("mov.{} {dst}, {src};", ty.storage()));
        Ok(())
    }

    fn binary(
        &mut self,
        out: Variable,
        op: BinaryOperator,
        opcode: &str,
        ty: Scalar,
    ) -> Result<()> {
        if out.ty != op.lhs.ty || out.ty != op.rhs.ty || Scalar::of(out.ty)? != ty {
            return Err(invalid(format!("{opcode} operand types differ")));
        }
        let lhs = self.value(op.lhs)?;
        let rhs = self.value(op.rhs)?;
        let dst = self.destination(out)?;
        self.line(format!("{opcode}.{} {dst}, {lhs}, {rhs};", ty.suffix()));
        Ok(())
    }

    fn arithmetic(&mut self, op: Arithmetic, out: Variable) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if ty.fp8() { return self.fp8_arithmetic(op, out, ty); }
        if let Arithmetic::Neg(op) = &op {
            return self.numeric_sign(out, op.input, false);
        }
        if let Arithmetic::Abs(op) = &op {
            return self.numeric_sign(out, op.input, true);
        }
        match &op {
            Arithmetic::Sin(op) => return self.trigonometry(out, op.input, false),
            Arithmetic::Cos(op) => return self.trigonometry(out, op.input, true),
            Arithmetic::Exp(op) => return self.exponential(out, op.input),
            Arithmetic::Log(op) => return self.logarithm(out, op.input),
            Arithmetic::Log1p(op) => return self.logarithm_one_plus(out, op.input),
            Arithmetic::Tanh(op) => return self.hyperbolic_tangent(out, op.input),
            Arithmetic::ArcTanh(op) => return self.inverse_hyperbolic_tangent(out, op.input),
            Arithmetic::ArcSinh(op) => return self.inverse_hyperbolic(out, op.input, false),
            Arithmetic::ArcCosh(op) => return self.inverse_hyperbolic(out, op.input, true),
            Arithmetic::Sinh(op) => return self.hyperbolic(out, op.input, false),
            Arithmetic::Cosh(op) => return self.hyperbolic(out, op.input, true),
            Arithmetic::Powi(op) => return self.integer_power(out, op.clone()),
            Arithmetic::Recip(op) => return self.reciprocal(out, op.input),
            Arithmetic::InverseSqrt(op) => return self.emit_square_root(out, op.input, true),
            Arithmetic::Min(op) => return self.minmax(out, op.clone(), "min"),
            Arithmetic::Max(op) => return self.minmax(out, op.clone(), "max"),
            Arithmetic::Clamp(op) => return self.clamp(out, op),
            Arithmetic::Round(op) => return self.round_float(out, op.input, "rni"),
            Arithmetic::Floor(op) => return self.round_float(out, op.input, "rmi"),
            Arithmetic::Ceil(op) => return self.round_float(out, op.input, "rpi"),
            Arithmetic::Trunc(op) => return self.round_float(out, op.input, "rzi"),
            _ => {}
        }
        if ty.half() { return self.half_arithmetic(op, out, ty); }
        if ty == Scalar::Pred {
            return Err(invalid("predicate arithmetic"));
        }
        match op {
            Arithmetic::Add(op) => {
                self.binary(out, op, if ty.float() { "add.rn" } else { "add" }, ty)
            }
            Arithmetic::Sub(op) => {
                self.binary(out, op, if ty.float() { "sub.rn" } else { "sub" }, ty)
            }
            Arithmetic::Mul(op) => {
                self.binary(out, op, if ty.float() { "mul.rn" } else { "mul.lo" }, ty)
            }
            Arithmetic::Div(op) => {
                self.binary(out, op, if ty.float() { "div.rn" } else { "div" }, ty)
            }
            Arithmetic::Remainder(op) if ty.integer() => self.binary(out, op, "rem", ty),
            Arithmetic::MulHi(op) if ty.narrow() => self.narrow_mul_hi(out, op),
            Arithmetic::MulHi(op) if ty.integer() => self.binary(out, op, "mul.hi", ty),
            Arithmetic::SaturatingAdd(op) if ty.integer() => self.saturating(out, op, false),
            Arithmetic::SaturatingSub(op) if ty.integer() => self.saturating(out, op, true),
            Arithmetic::Modulo(op) if ty.integer() && !ty.signed() => {
                self.binary(out, op, "rem", ty)
            }
            Arithmetic::Fma(op) if ty.float() => {
                if op.a.ty != out.ty || op.b.ty != out.ty || op.c.ty != out.ty {
                    return Err(invalid("FMA operand types differ"));
                }
                let a = self.value(op.a)?;
                let b = self.value(op.b)?;
                let c = self.value(op.c)?;
                let dst = self.destination(out)?;
                self.line(format!("fma.rn.{} {dst}, {a}, {b}, {c};", ty.suffix()));
                Ok(())
            }
            Arithmetic::Sqrt(op) if ty.float() => {
                if op.input.ty != out.ty {
                    return Err(invalid("sqrt operand types differ"));
                }
                let src = self.value(op.input)?;
                let dst = self.destination(out)?;
                self.line(format!("sqrt.rn.{} {dst}, {src};", ty.suffix()));
                Ok(())
            }
            other => Err(unsupported(format!("arithmetic {other:?}"))),
        }
    }

    fn comparison(&mut self, op: Comparison, out: Variable) -> Result<()> {
        if Scalar::of(out.ty)? != Scalar::Pred {
            return Err(invalid("comparison output must be a predicate"));
        }
        let (code, operands) = match op {
            Comparison::Lower(op) => ("lt", op),
            Comparison::LowerEqual(op) => ("le", op),
            Comparison::Greater(op) => ("gt", op),
            Comparison::GreaterEqual(op) => ("ge", op),
            Comparison::Equal(op) => ("eq", op),
            Comparison::NotEqual(op) => ("ne", op),
            Comparison::IsNan(op) => return self.float_classification(out, op.input, true),
            Comparison::IsInf(op) => return self.float_classification(out, op.input, false),
        };
        let lhs_ty = Scalar::of(operands.lhs.ty)?;
        let rhs_ty = Scalar::of(operands.rhs.ty)?;
        let ty = if lhs_ty == rhs_ty {
            lhs_ty
        } else if matches!((lhs_ty, rhs_ty), (Scalar::U32, Scalar::U64) | (Scalar::U64, Scalar::U32)) {
            Scalar::U64
        } else {
            return Err(invalid("comparison operand types differ"));
        };
        if ty == Scalar::Pred {
            return self.predicate_comparison(out, operands, code);
        }
        // Rust/C++ != is true for unordered (NaN) operands.
        let code = if code == "ne" && ty.float() {
            "neu"
        } else {
            code
        };
        let mut a = self.value(operands.lhs)?;
        let mut b = self.value(operands.rhs)?;
        if lhs_ty != ty {
            let wide = self.reg(ty);
            self.line(format!("cvt.u64.u32 {wide}, {a};"));
            a = wide;
        }
        if rhs_ty != ty {
            let wide = self.reg(ty);
            self.line(format!("cvt.u64.u32 {wide}, {b};"));
            b = wide;
        }
        let comparison_ty = if ty.fp8() {
            a = self.fp8_to_f32(ty, &a)?;
            b = self.fp8_to_f32(ty, &b)?;
            Scalar::F32
        } else if ty.half() {
            a = self.half_to_f32(ty, &a)?;
            b = self.half_to_f32(ty, &b)?;
            Scalar::F32
        } else { ty };
        let dst = self.destination(out)?;
        self.line(format!("setp.{code}.{} {dst}, {a}, {b};", comparison_ty.suffix()));
        Ok(())
    }

    fn cast(&mut self, out: Variable, input: Variable) -> Result<()> {
        if out.ty == input.ty {
            return self.copy(out, input);
        }
        let to = Scalar::of(out.ty)?;
        let from = Scalar::of(input.ty)?;
        if to.fp8() || from.fp8() {
            return self.fp8_cast(out, input, to, from);
        }
        if Scalar::is_tf32(out.ty) {
            return self.tf32_cast(out, input);
        }
        if to == Scalar::Pred || from == Scalar::Pred {
            return self.predicate_cast(out, input, to, from);
        }
        if matches!(to, Scalar::F32 | Scalar::F64) && (from.float() || from.integer()) {
            return self.to_float(out, input, to, from);
        }
        if to.half() && from.integer() {
            return self.integer_to_half(out, input, to, from);
        }
        if to.half() || from.half() { return self.half_cast(out, input, to, from); }
        // Float-to-integer casts require explicit overflow/rounding semantics.
        if !to.integer() || !from.integer() {
            return Err(unsupported("non-integer cast"));
        }
        let src = self.value(input)?;
        let dst = self.destination(out)?;
        self.line(format!(
            "cvt.{}.{} {dst}, {src};",
            to.suffix(),
            from.suffix()
        ));
        Ok(())
    }

    fn condition(&mut self, condition: Variable) -> Result<String> {
        if Scalar::of(condition.ty)? != Scalar::Pred {
            return Err(invalid("branch condition must be a predicate"));
        }
        self.value(condition)
    }

    fn branch(&mut self, branch: Branch) -> Result<()> {
        match branch {
            Branch::RangeLoop(branch) => self.range_loop(*branch)?,
            Branch::Switch(branch) => self.switch(*branch)?,
            Branch::If(branch) => {
                let end = self.label();
                let pred = self.condition(branch.cond)?;
                self.line(format!("@!{pred} bra {end};"));
                self.scope(branch.scope)?;
                self.line(format!("{end}:"));
            }
            Branch::IfElse(branch) => {
                let otherwise = self.label();
                let end = self.label();
                let pred = self.condition(branch.cond)?;
                self.line(format!("@!{pred} bra {otherwise};"));
                self.scope(branch.scope_if)?;
                self.line(format!("bra {end};"));
                self.line(format!("{otherwise}:"));
                self.scope(branch.scope_else)?;
                self.line(format!("{end}:"));
            }
            Branch::Loop(branch) => {
                let start = self.label();
                let end = self.label();
                self.loops.push(end.clone());
                self.line(format!("{start}:"));
                self.scope(branch.scope)?;
                self.line(format!("bra {start};"));
                self.line(format!("{end}:"));
                self.loops.pop();
            }
            Branch::Break => {
                let end = self
                    .loops
                    .last()
                    .ok_or_else(|| invalid("break outside loop or switch"))?
                    .clone();
                self.line(format!("bra {end};"));
            }
            Branch::Return => self.line("ret;"),
            other => return Err(unsupported(format!("branch {other:?}"))),
        }
        Ok(())
    }
}
