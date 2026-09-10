use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{Arithmetic, Variable};

pub(super) fn constant_bits(ty: Scalar, value: f64) -> u16 {
    if value.is_nan() {
        return 0x7fff;
    }
    let rounded = value as f32;
    let widened = rounded as f64;
    let mut bits = rounded.to_bits();
    if (value > 0.0 && widened > value) || (value < 0.0 && widened < value) {
        bits -= 1;
    }
    if widened != value {
        bits |= 1;
    }
    let odd = f32::from_bits(bits);
    match ty {
        Scalar::F16 => half::f16::from_f32(odd).to_bits(),
        Scalar::BF16 => half::bf16::from_f32(odd).to_bits(),
        _ => unreachable!(),
    }
}

impl Emitter {
    pub fn half_target(&self, ty: Scalar) -> Result<()> {
        if ty == Scalar::F16 && self.target.sm < 53 {
            return Err(unsupported("FP16 requires SM >= 53"));
        }
        if ty == Scalar::BF16 && (self.target.sm < 80 || self.target.version < (7, 0)) {
            return Err(unsupported("BF16 requires SM >= 80 and PTX >= 7.0"));
        }
        Ok(())
    }

    pub fn half_to_f32(&mut self, from: Scalar, source: &str) -> Result<String> {
        self.half_target(from)?;
        let result = self.reg(Scalar::F32);
        if from == Scalar::BF16 {
            self.line(format!("mov.b32 {result}, {{0, {source}}};"));
        } else {
            self.line(format!("cvt.f32.f16 {result}, {source};"));
        }
        Ok(result)
    }

    pub fn half_cast(
        &mut self,
        out: Variable,
        input: Variable,
        to: Scalar,
        from: Scalar,
    ) -> Result<()> {
        self.half_target(to)?;
        self.half_target(from)?;
        let source = self.value(input)?;
        let dst = self.destination(out)?;
        if from.half() && to == Scalar::F32 {
            let converted = self.half_to_f32(from, &source)?;
            self.line(format!("mov.f32 {dst}, {converted};"));
        } else if from == Scalar::F32 && to.half() {
            self.line(format!("cvt.rn.{}.f32 {dst}, {source};", to.suffix()));
        } else if from.half() && to.half() {
            let converted = self.half_to_f32(from, &source)?;
            self.line(format!("cvt.rn.{}.f32 {dst}, {converted};", to.suffix()));
        } else if from == Scalar::F64 && to.half() {
            let truncated = self.reg(Scalar::F32);
            let widened = self.reg(Scalar::F64);
            let bits = self.reg(Scalar::U32);
            let inexact = self.reg(Scalar::Pred);
            // Round to odd at the intermediate precision before the final RN conversion.
            self.line(format!("cvt.rz.f32.f64 {truncated}, {source};"));
            self.line(format!("cvt.f64.f32 {widened}, {truncated};"));
            self.line(format!("setp.ne.f64 {inexact}, {widened}, {source};"));
            self.line(format!("mov.b32 {bits}, {truncated};"));
            self.line(format!("@{inexact} or.b32 {bits}, {bits}, 1;"));
            self.line(format!("mov.b32 {truncated}, {bits};"));
            self.line(format!("cvt.rn.{}.f32 {dst}, {truncated};", to.suffix()));
        } else {
            return Err(unsupported("half cast requires floating-point endpoints"));
        }
        Ok(())
    }

    pub fn half_arithmetic(&mut self, op: Arithmetic, out: Variable, ty: Scalar) -> Result<()> {
        self.half_target(ty)?;
        match &op {
            Arithmetic::Sqrt(op) => return self.emit_square_root(out, op.input, false),
            Arithmetic::InverseSqrt(op) => return self.emit_square_root(out, op.input, true),
            _ => {}
        }
        let (opcode, a, b, c) = match op {
            Arithmetic::Add(op) => ("add", op.lhs, op.rhs, None),
            Arithmetic::Sub(op) => ("sub", op.lhs, op.rhs, None),
            Arithmetic::Mul(op) => ("mul", op.lhs, op.rhs, None),
            Arithmetic::Div(op) => ("div", op.lhs, op.rhs, None),
            Arithmetic::Fma(op) => ("fma", op.a, op.b, Some(op.c)),
            other => return Err(unsupported(format!("half arithmetic {other:?}"))),
        };
        if a.ty != out.ty || b.ty != out.ty || c.is_some_and(|c| c.ty != out.ty) {
            return Err(invalid("half arithmetic operand types differ"));
        }
        let a = self.value(a)?;
        let b = self.value(b)?;
        let dst = self.destination(out)?;
        if let Some(c) = c {
            let c = self.value(c)?;
            let a = self.half_to_f32(ty, &a)?;
            let b = self.half_to_f32(ty, &b)?;
            let c = self.half_to_f32(ty, &c)?;
            let intermediate = self.reg(Scalar::F32);
            self.line(format!("fma.rn.f32 {intermediate}, {a}, {b}, {c};"));
            self.line(format!("cvt.rn.{}.f32 {dst}, {intermediate};", ty.suffix()));
        } else {
            self.half_binary(opcode, ty, &dst, &a, &b)?;
        }
        Ok(())
    }

    pub fn half_binary(&mut self, opcode: &str, ty: Scalar, dst: &str, a: &str, b: &str) -> Result<()> {
        self.half_target(ty)?;
        if opcode == "div" {
            let a = self.half_to_f32(ty, a)?;
            let b = self.half_to_f32(ty, b)?;
            let quotient = self.reg(Scalar::F32);
            self.line(format!("div.rn.f32 {quotient}, {a}, {b};"));
            self.line(format!("cvt.rn.{}.f32 {dst}, {quotient};", ty.suffix()));
            return Ok(());
        }
        if ty == Scalar::F16 {
            self.line(format!("{opcode}.rn.f16 {dst}, {a}, {b};"));
        } else {
            let constant = self.reg(Scalar::BF16);
            let (bits, a, b, c) = match opcode {
                "add" => (0x3f80, a, constant.as_str(), b),
                "sub" => (0xbf80, b, constant.as_str(), a),
                "mul" => (0x8000, a, b, constant.as_str()),
                _ => unreachable!(),
            };
            self.line(format!("mov.b16 {constant}, {bits};"));
            self.line(format!("fma.rn.bf16 {dst}, {a}, {b}, {c};"));
        }
        Ok(())
    }
}
