use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{BinaryOperator, ConstantValue, Variable};

impl Emitter {
    pub fn integer_power(&mut self, out: Variable, op: BinaryOperator) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        let exponent_ty = Scalar::of(op.rhs.ty)?;
        if out.ty != op.lhs.ty || !ty.float() || !exponent_ty.integer() {
            return Err(invalid("integer power requires a floating base and integer exponent"));
        }
        let source = self.value(op.lhs)?;
        let exponent = self.value(op.rhs)?;
        let destination = self.destination(out)?;
        let compute_ty = if ty.half() { Scalar::F32 } else { ty };
        let magnitude_ty = if exponent_ty.bytes() == 8 { Scalar::U64 } else { Scalar::U32 };
        let n = self.reg(magnitude_ty);
        let bit = self.reg(magnitude_ty);
        let predicate = self.reg(Scalar::Pred);
        let base = if ty.half() {
            self.half_to_f32(ty, &source)?
        } else {
            let base = self.reg(compute_ty);
            self.line(format!("mov.{} {base}, {source};", compute_ty.storage()));
            base
        };
        let result = self.reg(compute_ty);
        let one = compute_ty.constant(ConstantValue::Float(1.0))?;
        let loop_start = self.label();
        let skip_multiply = self.label();
        let end = self.label();

        self.line(format!("mov.{} {n}, {exponent};", magnitude_ty.storage()));
        self.line(format!("mov.{} {result}, {one};", compute_ty.suffix()));
        if exponent_ty.signed() {
            self.line(format!("setp.lt.{} {predicate}, {exponent}, 0;", exponent_ty.suffix()));
            self.line(format!("@{predicate} sub.{} {n}, 0, {n};", magnitude_ty.suffix()));
            self.line(format!("@{predicate} div.rn.{} {base}, {one}, {base};", compute_ty.suffix()));
        }

        self.line(format!("{loop_start}:"));
        self.line(format!("setp.eq.{} {predicate}, {n}, 0;", magnitude_ty.suffix()));
        self.line(format!("@{predicate} bra {end};"));
        self.line(format!("and.{} {bit}, {n}, 1;", magnitude_ty.bits()));
        self.line(format!("setp.eq.{} {predicate}, {bit}, 0;", magnitude_ty.suffix()));
        self.line(format!("@{predicate} bra {skip_multiply};"));
        self.line(format!("mul.rn.{} {result}, {result}, {base};", compute_ty.suffix()));
        self.line(format!("{skip_multiply}:"));
        self.line(format!("shr.{} {n}, {n}, 1;", magnitude_ty.suffix()));
        self.line(format!("setp.eq.{} {predicate}, {n}, 0;", magnitude_ty.suffix()));
        self.line(format!("@{predicate} bra {end};"));
        self.line(format!("mul.rn.{} {base}, {base}, {base};", compute_ty.suffix()));
        self.line(format!("bra {loop_start};"));
        self.line(format!("{end}:"));
        if ty.half() {
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        } else {
            self.line(format!("mov.{} {destination}, {result};", ty.storage()));
        }
        Ok(())
    }
}
