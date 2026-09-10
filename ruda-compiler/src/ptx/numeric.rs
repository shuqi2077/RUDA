use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{BinaryOperator, ClampOperator, Variable};

impl Emitter {
    pub fn numeric_binary(&mut self, ty: Scalar, operation: &str, a: &str, b: &str) -> Result<String> {
        let destination = self.reg(ty);
        if ty.fp8() {
            let a = self.fp8_to_f32(ty, a)?;
            let b = self.fp8_to_f32(ty, b)?;
            let result = self.numeric_binary(Scalar::F32, operation, &a, &b)?;
            self.f32_to_fp8(ty, &destination, &result)?;
        } else if ty.half() && matches!(operation, "add" | "sub" | "mul") {
            self.half_binary(operation, ty, &destination, a, b)?;
        } else if ty == Scalar::F16 && (self.target.sm < 80 || self.target.version < (7, 0)) && matches!(operation, "min" | "max") {
            let a = self.half_to_f32(ty, a)?;
            let b = self.half_to_f32(ty, b)?;
            let result = self.reg(Scalar::F32);
            self.line(format!("{operation}.f32 {result}, {a}, {b};"));
            self.line(format!("cvt.rn.f16.f32 {destination}, {result};"));
        } else {
            self.half_target(ty)?;
            let modifier = match operation {
                "add" | "sub" | "mul" if ty.float() => ".rn",
                "mul" => ".lo",
                _ => "",
            };
            self.line(format!("{operation}{modifier}.{} {destination}, {a}, {b};", ty.suffix()));
        }
        self.normalize_integer(ty, &destination);
        Ok(destination)
    }

    pub fn minmax(&mut self, out: Variable, op: BinaryOperator, operation: &str) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if ty == Scalar::Pred || op.lhs.ty != out.ty || op.rhs.ty != out.ty {
            return Err(invalid("min/max requires matching numeric operands"));
        }
        let a = self.value(op.lhs)?;
        let b = self.value(op.rhs)?;
        let result = self.numeric_binary(ty, operation, &a, &b)?;
        let destination = self.destination(out)?;
        self.line(format!("mov.{} {destination}, {result};", ty.storage()));
        Ok(())
    }

    pub fn clamp(&mut self, out: Variable, op: &ClampOperator) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if ty == Scalar::Pred || [op.input.ty, op.min_value.ty, op.max_value.ty].iter().any(|ty| *ty != out.ty) {
            return Err(invalid("clamp requires matching numeric operands"));
        }
        let input = self.value(op.input)?;
        let minimum = self.value(op.min_value)?;
        let maximum = self.value(op.max_value)?;
        // Preserve max(minimum, min(maximum, input)), including NaN operands.
        let bounded = self.numeric_binary(ty, "min", &maximum, &input)?;
        let result = self.numeric_binary(ty, "max", &minimum, &bounded)?;
        let destination = self.destination(out)?;
        self.line(format!("mov.{} {destination}, {result};", ty.storage()));
        Ok(())
    }

    pub fn round_float(&mut self, out: Variable, input: Variable, rounding: &str) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !ty.float() || out.ty != input.ty {
            return Err(invalid("rounding requires matching floating operands"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if ty.half() {
            let widened = self.half_to_f32(ty, &source)?;
            let rounded = self.reg(Scalar::F32);
            self.line(format!("cvt.{rounding}.f32.f32 {rounded}, {widened};"));
            self.line(format!("cvt.rn.{}.f32 {destination}, {rounded};", ty.suffix()));
        } else {
            self.line(format!("cvt.{rounding}.{0}.{0} {destination}, {source};", ty.suffix()));
        }
        Ok(())
    }
}
