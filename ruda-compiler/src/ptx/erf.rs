use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{ConstantValue, Variable};

impl Emitter {
    pub fn error_function(&mut self, out: Variable, input: Variable) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if out.ty != input.ty { return Err(invalid("erf operand types differ")); }
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("erf requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let source = if ty.half() { self.half_to_f32(ty, &source)? } else { source };
        let destination = self.destination(out)?;
        let result = self.reg(Scalar::F32);
        let x = self.reg(Scalar::F32);
        let t = self.reg(Scalar::F32);
        let y = self.reg(Scalar::F32);
        let z = self.reg(Scalar::F32);
        let predicate = self.reg(Scalar::Pred);
        let small = self.label();
        let signed = self.label();
        let end = self.label();
        let c = |value| Scalar::F32.constant(ConstantValue::Float(value));
        self.line(format!("abs.f32 {x}, {source};"));
        self.line(format!("setp.nan.f32 {predicate}, {source}, {source};"));
        self.line(format!("@{predicate} mov.f32 {result}, {source};"));
        self.line(format!("@{predicate} bra {end};"));
        self.line(format!("setp.ge.f32 {predicate}, {x}, {};", c(6.0)?));
        self.line(format!("@{predicate} mov.f32 {result}, {};", c(1.0)?));
        self.line(format!("@{predicate} bra {signed};"));
        self.line(format!("setp.lt.f32 {predicate}, {x}, {};", c(0.125)?));
        self.line(format!("@{predicate} bra {small};"));
        // Same Abramowitz-Stegun 7.1.26 coefficients as the Metal backend.
        self.line(format!("fma.rn.f32 {t}, {x}, {}, {};", c(0.3275911)?, c(1.0)?));
        self.line(format!("div.rn.f32 {t}, {}, {t};", c(1.0)?));
        self.line(format!("mov.f32 {y}, {};", c(1.061405429)?));
        for coefficient in [-1.453152027, 1.421413741, -0.284496736, 0.254829592] {
            self.line(format!("fma.rn.f32 {y}, {y}, {t}, {};", c(coefficient)?));
        }
        self.line(format!("mul.rn.f32 {y}, {y}, {t};"));
        self.line(format!("mul.rn.f32 {z}, {x}, {x};"));
        self.line(format!("neg.f32 {z}, {z};"));
        let e = self.reg(Scalar::F32);
        self.exponential_f32(&e, &z);
        self.line(format!("neg.f32 {y}, {y};"));
        self.line(format!("fma.rn.f32 {result}, {y}, {e}, {};", c(1.0)?));
        self.line(format!("bra {signed};"));
        // The convergent Taylor series avoids cancellation near zero.
        self.line(format!("{small}:"));
        self.line(format!("mul.rn.f32 {z}, {x}, {x};"));
        self.line(format!("mov.f32 {y}, {};", c(1.0 / 216.0)?));
        for coefficient in [-1.0 / 42.0, 1.0 / 10.0, -1.0 / 3.0, 1.0] {
            self.line(format!("fma.rn.f32 {y}, {y}, {z}, {};", c(coefficient)?));
        }
        self.line(format!("mul.rn.f32 {y}, {y}, {x};"));
        self.line(format!("mul.rn.f32 {result}, {y}, {};", c(1.1283791670955126)?));
        self.line(format!("{signed}:"));
        let bits = self.reg(Scalar::U32);
        let sign = self.reg(Scalar::U32);
        self.line(format!("mov.b32 {sign}, {source};"));
        self.line(format!("and.b32 {sign}, {sign}, 2147483648;"));
        self.line(format!("mov.b32 {bits}, {result};"));
        self.line(format!("or.b32 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b32 {result}, {bits};"));
        self.line(format!("{end}:"));
        if ty.half() { self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix())); }
        else { self.line(format!("mov.f32 {destination}, {result};")); }
        Ok(())
    }
}
