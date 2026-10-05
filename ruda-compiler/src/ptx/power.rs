use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{BinaryOperator, ConstantValue, Variable};

impl Emitter {
    pub fn floating_power(&mut self, out: Variable, op: BinaryOperator) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if out.ty != op.lhs.ty || out.ty != op.rhs.ty {
            return Err(invalid("floating power operand types differ"));
        }
        if ty != Scalar::F32 && !ty.half() {
            return Err(super::unsupported("floating power requires F32, F16 or BF16 storage"));
        }
        let x = self.value(op.lhs)?;
        let y = self.value(op.rhs)?;
        let x = if ty.half() { self.half_to_f32(ty, &x)? } else { x };
        let y = if ty.half() { self.half_to_f32(ty, &y)? } else { y };
        let destination = self.destination(out)?;
        let result = self.reg(Scalar::F32);
        let ax = self.reg(Scalar::F32);
        let ay = self.reg(Scalar::F32);
        let rounded = self.reg(Scalar::F32);
        let p = self.reg(Scalar::Pred);
        let odd = self.reg(Scalar::Pred);
        let integer = self.reg(Scalar::Pred);
        let n = self.reg(Scalar::U32);
        let bit = self.reg(Scalar::U32);
        let one = self.label();
        let nan = self.label();
        let zero = self.label();
        let infinity = self.label();
        let infinite_y = self.label();
        let signed = self.label();
        let finish = self.label();
        let general = self.label();
        let parity_done = self.label();
        let c = |value| Scalar::F32.constant(ConstantValue::Float(value));
        self.line(format!("setp.eq.f32 {p}, {y}, {};", c(0.0)?));
        self.line(format!("@{p} bra {one};"));
        self.line(format!("setp.eq.f32 {p}, {x}, {};", c(1.0)?));
        self.line(format!("@{p} bra {one};"));
        self.line(format!("setp.nan.f32 {p}, {x}, {y};"));
        self.line(format!("@{p} bra {nan};"));
        self.line(format!("abs.f32 {ax}, {x};"));
        self.line(format!("abs.f32 {ay}, {y};"));
        self.line(format!("setp.eq.f32 {p}, {ay}, 0f7f800000;"));
        self.line(format!("@{p} bra {infinite_y};"));
        self.line(format!("cvt.rzi.f32.f32 {rounded}, {y};"));
        self.line(format!("setp.eq.f32 {integer}, {rounded}, {y};"));
        self.line(format!("setp.eq.u32 {odd}, 0, 1;"));
        self.line(format!("setp.ge.f32 {p}, {ay}, {};", c(16777216.0)?));
        self.line(format!("@{p} bra {parity_done};"));
        self.line(format!("cvt.rzi.u32.f32 {n}, {ay};"));
        self.line(format!("and.b32 {bit}, {n}, 1;"));
        self.line(format!("setp.ne.u32 {odd}, {bit}, 0;"));
        self.line(format!("and.pred {odd}, {odd}, {integer};"));
        self.line(format!("{parity_done}:"));
        self.line(format!("setp.eq.f32 {p}, {ax}, {};", c(0.0)?));
        self.line(format!("@{p} bra {zero};"));
        self.line(format!("setp.eq.f32 {p}, {ax}, 0f7f800000;"));
        self.line(format!("@{p} bra {infinity};"));
        self.line(format!("setp.lt.f32 {p}, {x}, {};", c(0.0)?));
        self.line(format!("not.pred {p}, {p};"));
        self.line(format!("or.pred {p}, {p}, {integer};"));
        self.line(format!("@!{p} bra {nan};"));
        self.line(format!("@!{integer} bra {general};"));
        self.line(format!("setp.ge.f32 {p}, {ay}, {};", c(16777216.0)?));
        self.line(format!("@{p} bra {general};"));
        // Integral exponents, including the norm's square, do not go through log/exp.
        let base = self.reg(Scalar::F64);
        let product = self.reg(Scalar::F64);
        let double_one = Scalar::F64.constant(ConstantValue::Float(1.0))?;
        let loop_start = self.label();
        let skip = self.label();
        let loop_end = self.label();
        self.line(format!("cvt.f64.f32 {base}, {ax};"));
        self.line(format!("mov.f64 {product}, {double_one};"));
        self.line(format!("cvt.rzi.u32.f32 {n}, {ay};"));
        self.line(format!("setp.lt.f32 {p}, {y}, {};", c(0.0)?));
        self.line(format!("@{p} div.rn.f64 {base}, {double_one}, {base};"));
        self.line(format!("{loop_start}:"));
        self.line(format!("setp.eq.u32 {p}, {n}, 0;"));
        self.line(format!("@{p} bra {loop_end};"));
        self.line(format!("and.b32 {bit}, {n}, 1;"));
        self.line(format!("setp.eq.u32 {p}, {bit}, 0;"));
        self.line(format!("@{p} bra {skip};"));
        self.line(format!("mul.rn.f64 {product}, {product}, {base};"));
        self.line(format!("{skip}:"));
        self.line(format!("shr.u32 {n}, {n}, 1;"));
        self.line(format!("setp.eq.u32 {p}, {n}, 0;"));
        self.line(format!("@{p} bra {loop_end};"));
        self.line(format!("mul.rn.f64 {base}, {base}, {base};"));
        self.line(format!("bra {loop_start};"));
        self.line(format!("{loop_end}:"));
        self.line(format!("cvt.rn.f32.f64 {result}, {product};"));
        self.line(format!("bra {signed};"));
        self.line(format!("{general}:"));
        let logarithm = self.reg(Scalar::F32);
        let exponent = self.reg(Scalar::F32);
        self.logarithm_f32(&logarithm, &ax);
        self.line(format!("mul.rn.f32 {exponent}, {y}, {logarithm};"));
        self.exponential_f32(&result, &exponent);
        self.line(format!("bra {signed};"));
        self.line(format!("{zero}:"));
        self.line(format!("mov.f32 {result}, {};", c(0.0)?));
        self.line(format!("setp.lt.f32 {p}, {y}, {};", c(0.0)?));
        self.line(format!("@{p} mov.f32 {result}, 0f7f800000;"));
        self.line(format!("bra {signed};"));
        self.line(format!("{infinity}:"));
        self.line(format!("mov.f32 {result}, 0f7f800000;"));
        self.line(format!("setp.lt.f32 {p}, {y}, {};", c(0.0)?));
        self.line(format!("@{p} mov.f32 {result}, {};", c(0.0)?));
        self.line(format!("bra {signed};"));
        self.line(format!("{infinite_y}:"));
        self.line(format!("setp.eq.f32 {p}, {ax}, {};", c(1.0)?));
        self.line(format!("@{p} bra {one};"));
        self.line(format!("setp.gt.f32 {p}, {ax}, {};", c(1.0)?));
        self.line(format!("setp.lt.f32 {integer}, {y}, {};", c(0.0)?));
        self.line(format!("xor.pred {p}, {p}, {integer};"));
        self.line(format!("selp.f32 {result}, 0f7f800000, {}, {p};", c(0.0)?));
        self.line(format!("bra {finish};"));
        self.line(format!("{signed}:"));
        let sign = self.reg(Scalar::U32);
        let bits = self.reg(Scalar::U32);
        self.line(format!("mov.b32 {sign}, {x};"));
        self.line(format!("and.b32 {sign}, {sign}, 2147483648;"));
        self.line(format!("@!{odd} mov.u32 {sign}, 0;"));
        self.line(format!("mov.b32 {bits}, {result};"));
        self.line(format!("or.b32 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b32 {result}, {bits};"));
        self.line(format!("bra {finish};"));
        self.line(format!("{nan}:"));
        self.line(format!("mov.f32 {result}, 0f7fc00000;"));
        self.line(format!("bra {finish};"));
        self.line(format!("{one}:"));
        self.line(format!("mov.f32 {result}, {};", c(1.0)?));
        self.line(format!("{finish}:"));
        if ty.half() { self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix())); }
        else { self.line(format!("mov.f32 {destination}, {result};")); }
        Ok(())
    }

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
