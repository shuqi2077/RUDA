use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{Arithmetic, Instruction, Operation, OperationReflect, Operator, StorageType, Type, Variable};

mod math;

impl Emitter {
    fn fp8_target(&self) -> Result<()> {
        let supported = (self.target.sm >= 90 && self.target.version >= (7, 8))
            || (self.target.sm == 89 && self.target.version >= (8, 1));
        if !supported {
            return Err(unsupported("FP8 conversion requires SM >= 90 / PTX >= 7.8 or SM 89 / PTX >= 8.1"));
        }
        Ok(())
    }

    pub fn fp8_instruction_supported(&self, instruction: &Instruction) -> Result<()> {
        let is_fp8 = |v: Variable| Scalar::memory_element(v.ty).is_ok_and(Scalar::fp8);
        let args = instruction.operation.args().unwrap_or_default();
        if !instruction.out.is_some_and(is_fp8) && !args.iter().copied().any(is_fp8) {
            return Ok(());
        }
        if instruction.out.into_iter().chain(args.iter().copied()).any(|v| {
            matches!(v.ty, Type::Scalar(StorageType::Atomic(_)) | Type::Vector(StorageType::Atomic(_), _))
        }) {
            return Err(unsupported("FP8 atomic operations"));
        }
        let supported = matches!(&instruction.operation,
            Operation::Copy(_) | Operation::Comparison(_) | Operation::Metadata(_)
            | Operation::Tma(_)
            | Operation::Barrier(ruda_core::ir::BarrierOps::TmaLoad { .. }
                | ruda_core::ir::BarrierOps::TmaLoadIm2col { .. })
            | Operation::Arithmetic(_) | Operation::Plane(_) | Operation::CoopMma(_)
            | Operation::Operator(Operator::Cast(_) | Operator::Reinterpret(_) | Operator::Select(_)
                | Operator::NativeAddress(_) | Operator::NativeLoad(_) | Operator::NativeStore(_)
                | Operator::InitVector(_) | Operator::Index(_) | Operator::UncheckedIndex(_)
                | Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_)
                | Operator::CopyMemory(_) | Operator::CopyMemoryBulk(_))
        );
        if !supported {
            return Err(unsupported(format!("FP8 operation {:?}", instruction.operation)));
        }
        Ok(())
    }

    pub fn fp8_to_f32(&mut self, ty: Scalar, source: &str) -> Result<String> {
        self.fp8_target()?;
        let packed = self.reg_b16();
        let halves = self.reg(Scalar::U32);
        let low = self.reg_b16();
        let high = self.reg_b16();
        let output = self.reg(Scalar::F32);
        self.line(format!("cvt.u16.u32 {packed}, {source};"));
        self.line(format!("cvt.rn.f16x2.{}x2 {halves}, {packed};", ty.suffix()));
        self.line(format!("mov.b32 {{{low}, {high}}}, {halves};"));
        self.line(format!("cvt.f32.f16 {output}, {low};"));
        Ok(output)
    }

    pub(super) fn f32_to_fp8(&mut self, ty: Scalar, destination: &str, source: &str) -> Result<()> {
        self.fp8_target()?;
        let packed = self.reg_b16();
        self.line(format!("cvt.rn.satfinite.{}x2.f32 {packed}, 0f00000000, {source};", ty.suffix()));
        self.line(format!("cvt.u32.u16 {destination}, {packed};"));
        self.line(format!("and.b32 {destination}, {destination}, 255;"));
        Ok(())
    }

    pub fn fp8_cast(&mut self, out: Variable, input: Variable, to: Scalar, from: Scalar) -> Result<()> {
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        let source = if from.fp8() {
            self.fp8_to_f32(from, &source)?
        } else if from.half() {
            self.half_to_f32(from, &source)?
        } else if from == Scalar::F32 {
            source
        } else if from == Scalar::Pred {
            let value = self.reg(Scalar::F32);
            self.line(format!("selp.f32 {value}, 0f3F800000, 0f00000000, {source};"));
            value
        } else if from == Scalar::F64 || from.integer() {
            let value = self.reg(Scalar::F32);
            let recovered = self.reg(from);
            let inexact = self.reg(Scalar::Pred);
            let bits = self.reg(Scalar::U32);
            self.line(format!("cvt.rz.f32.{} {value}, {source};", from.suffix()));
            if from == Scalar::F64 {
                self.line(format!("cvt.f64.f32 {recovered}, {value};"));
            } else {
                self.line(format!("cvt.rzi.{}.f32 {recovered}, {value};", from.suffix()));
            }
            self.line(format!("setp.ne.{} {inexact}, {recovered}, {source};", from.suffix()));
            self.line(format!("mov.b32 {bits}, {value};"));
            self.line(format!("@{inexact} or.b32 {bits}, {bits}, 1;"));
            self.line(format!("mov.b32 {value}, {bits};"));
            value
        } else {
            return Err(unsupported("FP8 conversion source"));
        };

        if to.fp8() {
            self.f32_to_fp8(to, &destination, &source)
        } else if Scalar::is_tf32(out.ty) {
            self.tf32_target()?;
            let bits = self.reg(Scalar::U32);
            self.line(format!("cvt.rna.tf32.f32 {bits}, {source};"));
            self.line(format!("mov.b32 {destination}, {bits};"));
            Ok(())
        } else if to == Scalar::F32 {
            self.line(format!("mov.f32 {destination}, {source};"));
            Ok(())
        } else if to == Scalar::F64 {
            self.line(format!("cvt.f64.f32 {destination}, {source};"));
            Ok(())
        } else if to.half() {
            self.half_target(to)?;
            self.line(format!("cvt.rn.{}.f32 {destination}, {source};", to.suffix()));
            Ok(())
        } else if to == Scalar::Pred {
            self.line(format!("setp.neu.f32 {destination}, {source}, 0f00000000;"));
            Ok(())
        } else {
            Err(unsupported("FP8-to-integer conversion requires explicit overflow/rounding semantics"))
        }
    }

    pub fn fp8_arithmetic(&mut self, op: Arithmetic, out: Variable, ty: Scalar) -> Result<()> {
        let (opcode, inputs): (&str, Vec<Variable>) = match op {
            Arithmetic::Add(op) => ("add.rn", vec![op.lhs, op.rhs]),
            Arithmetic::Sub(op) => ("sub.rn", vec![op.lhs, op.rhs]),
            Arithmetic::Mul(op) => ("mul.rn", vec![op.lhs, op.rhs]),
            Arithmetic::Div(op) => ("div.rn", vec![op.lhs, op.rhs]),
            Arithmetic::Fma(op) => ("fma.rn", vec![op.a, op.b, op.c]),
            Arithmetic::Neg(op) => ("neg", vec![op.input]),
            Arithmetic::Abs(op) => ("abs", vec![op.input]),
            other => return self.fp8_promoted_arithmetic(other, out, ty),
        };
        if inputs.iter().any(|input| input.ty != out.ty) {
            return Err(invalid("FP8 arithmetic operand types differ"));
        }
        let mut operands = Vec::with_capacity(inputs.len());
        for input in inputs {
            let source = self.value(input)?;
            operands.push(self.fp8_to_f32(ty, &source)?);
        }
        let result = self.reg(Scalar::F32);
        self.line(format!("{opcode}.f32 {result}, {};", operands.join(", ")));
        let destination = self.destination(out)?;
        self.f32_to_fp8(ty, &destination, &result)
    }
}
