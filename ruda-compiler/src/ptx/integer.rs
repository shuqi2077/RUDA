use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{BinaryOperator, Variable};

impl Emitter {
    pub fn normalize_integer(&mut self, ty: Scalar, register: &str) {
        if ty.narrow() {
            self.line(format!("cvt.{}.{} {register}, {register};", ty.memory_suffix(), ty.suffix()));
        }
    }

    pub fn narrow_mul_hi(&mut self, out: Variable, op: BinaryOperator) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !ty.narrow() || op.lhs.ty != out.ty || op.rhs.ty != out.ty {
            return Err(invalid("narrow mul.hi requires matching integer operands"));
        }
        let a = self.value(op.lhs)?;
        let b = self.value(op.rhs)?;
        let destination = self.destination(out)?;
        self.line(format!("mul.lo.{} {destination}, {a}, {b};", ty.suffix()));
        self.line(format!("shr.{} {destination}, {destination}, {};", ty.suffix(), ty.bytes() * 8));
        Ok(())
    }

    pub fn numeric_sign(&mut self, out: Variable, input: Variable, absolute: bool) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if out.ty != input.ty || ty == Scalar::Pred {
            return Err(invalid("abs/neg requires matching numeric operands"));
        }
        let src = self.value(input)?;
        let dst = self.destination(out)?;
        if ty.float() {
            let sign = 1_u64 << (ty.bytes() * 8 - 1);
            let (opcode, mask) = if absolute { ("and", sign - 1) } else { ("xor", sign) };
            self.line(format!("{opcode}.{} {dst}, {src}, 0x{mask:x};", ty.bits()));
        } else if absolute && ty.signed() {
            self.line(format!("abs.{} {dst}, {src};", ty.suffix()));
        } else if absolute {
            self.line(format!("mov.{} {dst}, {src};", ty.suffix()));
        } else {
            self.line(format!("sub.{} {dst}, 0, {src};", ty.suffix()));
        }
        Ok(())
    }

    pub fn saturating(&mut self, out: Variable, op: BinaryOperator, subtract: bool) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !ty.integer() || op.lhs.ty != out.ty || op.rhs.ty != out.ty {
            return Err(invalid("saturating arithmetic requires matching integer operands"));
        }
        let a = self.value(op.lhs)?;
        let b = self.value(op.rhs)?;
        let dst = self.destination(out)?;
        let result = self.reg(ty);
        let overflow = self.reg(Scalar::Pred);
        let opcode = if subtract { "sub" } else { "add" };
        if ty.narrow() {
            let width = ty.bytes() * 8;
            let (min, max) = if ty.signed() {
                (-(1_i32 << (width - 1)), (1_i32 << (width - 1)) - 1)
            } else {
                (0, (1_i32 << width) - 1)
            };
            self.line(format!("{opcode}.s32 {result}, {a}, {b};"));
            self.line(format!("max.s32 {result}, {result}, {min};"));
            self.line(format!("min.s32 {dst}, {result}, {max};"));
            return Ok(());
        }
        self.line(format!("{opcode}.{} {result}, {a}, {b};", ty.suffix()));
        let limit = if matches!(ty, Scalar::U32 | Scalar::U64) {
            if subtract {
                self.line(format!("setp.lt.{} {overflow}, {a}, {b};", ty.suffix()));
                "0".to_owned()
            } else {
                self.line(format!("setp.lt.{} {overflow}, {result}, {a};", ty.suffix()));
                if ty == Scalar::U32 { u32::MAX.to_string() } else { u64::MAX.to_string() }
            }
        } else {
            let first = self.reg(ty);
            let second = self.reg(ty);
            let negative = self.reg(Scalar::Pred);
            let limit = self.reg(ty);
            if subtract {
                self.line(format!("xor.{} {first}, {a}, {b};", ty.bits()));
                self.line(format!("xor.{} {second}, {a}, {result};", ty.bits()));
            } else {
                self.line(format!("xor.{} {first}, {result}, {a};", ty.bits()));
                self.line(format!("xor.{} {second}, {result}, {b};", ty.bits()));
            }
            self.line(format!("and.{} {first}, {first}, {second};", ty.bits()));
            self.line(format!("setp.lt.{} {overflow}, {first}, 0;", ty.suffix()));
            self.line(format!("setp.lt.{} {negative}, {a}, 0;", ty.suffix()));
            let (min, max) = if ty == Scalar::I32 {
                (i32::MIN as i64, i32::MAX as i64)
            } else {
                (i64::MIN, i64::MAX)
            };
            self.line(format!("selp.{} {limit}, {min}, {max}, {negative};", ty.suffix()));
            limit
        };
        self.line(format!("selp.{} {dst}, {limit}, {result}, {overflow};", ty.suffix()));
        Ok(())
    }
}
