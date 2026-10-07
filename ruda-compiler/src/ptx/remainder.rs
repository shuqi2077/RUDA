//! Dyadic floating remainder without quotient rounding or libdevice linkage.
use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{BinaryOperator, ConstantValue, Variable};

impl Emitter {
    pub fn floating_remainder(&mut self, out: Variable, op: BinaryOperator, floor: bool) -> Result<()> {
        let storage = Scalar::of(out.ty)?;
        if out.ty != op.lhs.ty || out.ty != op.rhs.ty {
            return Err(invalid("floating remainder operand types differ"));
        }
        if !matches!(storage, Scalar::F32 | Scalar::F64 | Scalar::F16 | Scalar::BF16) {
            return Err(unsupported("floating remainder requires F32/F64/F16/BF16 storage"));
        }
        let x = self.value(op.lhs)?;
        let y = self.value(op.rhs)?;
        let x = if storage.half() { self.half_to_f32(storage, &x)? } else { x };
        let y = if storage.half() { self.half_to_f32(storage, &y)? } else { y };
        let compute = if storage.half() { Scalar::F32 } else { storage };
        let bits = if compute == Scalar::F64 { Scalar::U64 } else { Scalar::U32 };
        let width = bits.bytes() * 8;
        let mantissa = if width == 64 { 52 } else { 23 };
        let exponent_mask = if width == 64 { 2047 } else { 255 };
        let sign_mask = 1u64 << (width - 1);
        let magnitude_mask = sign_mask - 1;
        let leading = 1u64 << mantissa;
        let infinity = (exponent_mask as u64) << mantissa;
        let quiet_nan = infinity | (leading >> 1);
        let result = self.reg(compute);
        let destination = self.destination(out)?;
        let ax = self.reg(bits);
        let ay = self.reg(bits);
        let sign = self.reg(bits);
        let temporary = self.reg(bits);
        let ex = self.reg(Scalar::I32);
        let ey = self.reg(Scalar::I32);
        let shift = self.reg(Scalar::U32);
        let predicate = self.reg(Scalar::Pred);
        let other = self.reg(Scalar::Pred);
        let nan = self.label();
        let original = self.label();
        let zero = self.label();
        let divide = self.label();
        let encode = self.label();
        let subnormal = self.label();
        let bits_done = self.label();
        let finish = self.label();
        let integer_suffix = bits.suffix();
        let bit_suffix = bits.bits();
        self.line(format!("mov.{bit_suffix} {ax}, {x};"));
        self.line(format!("mov.{bit_suffix} {ay}, {y};"));
        self.line(format!("and.{bit_suffix} {sign}, {ax}, {sign_mask};"));
        self.line(format!("and.{bit_suffix} {ax}, {ax}, {magnitude_mask};"));
        self.line(format!("and.{bit_suffix} {ay}, {ay}, {magnitude_mask};"));
        self.line(format!("setp.eq.{integer_suffix} {predicate}, {ay}, 0;"));
        self.line(format!("setp.ge.{integer_suffix} {other}, {ax}, {infinity};"));
        self.line(format!("or.pred {predicate}, {predicate}, {other};"));
        self.line(format!("setp.gt.{integer_suffix} {other}, {ay}, {infinity};"));
        self.line(format!("or.pred {predicate}, {predicate}, {other};"));
        self.line(format!("@{predicate} bra {nan};"));
        self.line(format!("setp.lt.{integer_suffix} {predicate}, {ax}, {ay};"));
        self.line(format!("@{predicate} bra {original};"));
        self.line(format!("setp.eq.{integer_suffix} {predicate}, {ax}, {ay};"));
        self.line(format!("@{predicate} bra {zero};"));

        for (value, exponent) in [(&ax, &ex), (&ay, &ey)] {
            let normal = self.label();
            let normalize = self.label();
            let normalized = self.label();
            self.line(format!("shr.{integer_suffix} {temporary}, {value}, {mantissa};"));
            self.line(format!("cvt.s32.{integer_suffix} {exponent}, {temporary};"));
            self.line(format!("setp.ne.s32 {predicate}, {exponent}, 0;"));
            self.line(format!("@{predicate} bra {normal};"));
            self.line(format!("mov.s32 {exponent}, 1;"));
            self.line(format!("{normalize}:"));
            self.line(format!("setp.ge.{integer_suffix} {predicate}, {value}, {leading};"));
            self.line(format!("@{predicate} bra {normalized};"));
            self.line(format!("shl.{bit_suffix} {value}, {value}, 1;"));
            self.line(format!("sub.s32 {exponent}, {exponent}, 1;"));
            self.line(format!("bra {normalize};"));
            self.line(format!("{normal}:"));
            self.line(format!("and.{bit_suffix} {value}, {value}, {};", leading - 1));
            self.line(format!("or.{bit_suffix} {value}, {value}, {leading};"));
            self.line(format!("{normalized}:"));
        }

        self.line(format!("{divide}:"));
        self.line(format!("setp.lt.{integer_suffix} {predicate}, {ax}, {ay};"));
        self.line(format!("@!{predicate} sub.{integer_suffix} {ax}, {ax}, {ay};"));
        self.line(format!("setp.eq.{integer_suffix} {predicate}, {ax}, 0;"));
        self.line(format!("@{predicate} bra {zero};"));
        self.line(format!("setp.le.s32 {predicate}, {ex}, {ey};"));
        self.line(format!("@{predicate} bra {encode};"));
        self.line(format!("shl.{bit_suffix} {ax}, {ax}, 1;"));
        self.line(format!("sub.s32 {ex}, {ex}, 1;"));
        self.line(format!("bra {divide};"));

        self.line(format!("{encode}:"));
        self.line(format!("setp.ge.{integer_suffix} {predicate}, {ax}, {leading};"));
        let normalized = self.label();
        self.line(format!("@{predicate} bra {normalized};"));
        self.line(format!("shl.{bit_suffix} {ax}, {ax}, 1;"));
        self.line(format!("sub.s32 {ex}, {ex}, 1;"));
        self.line(format!("bra {encode};"));
        self.line(format!("{normalized}:"));
        self.line(format!("setp.le.s32 {predicate}, {ex}, 0;"));
        self.line(format!("@{predicate} bra {subnormal};"));
        self.line(format!("sub.{integer_suffix} {ax}, {ax}, {leading};"));
        self.line(format!("cvt.{integer_suffix}.s32 {temporary}, {ex};"));
        self.line(format!("shl.{bit_suffix} {temporary}, {temporary}, {mantissa};"));
        self.line(format!("or.{bit_suffix} {ax}, {ax}, {temporary};"));
        self.line(format!("bra {bits_done};"));
        self.line(format!("{subnormal}:"));
        self.line(format!("sub.s32 {ex}, 1, {ex};"));
        self.line(format!("cvt.u32.s32 {shift}, {ex};"));
        self.line(format!("shr.{integer_suffix} {ax}, {ax}, {shift};"));
        self.line(format!("{bits_done}:"));
        self.line(format!("or.{bit_suffix} {ax}, {ax}, {sign};"));
        self.line(format!("mov.{bit_suffix} {result}, {ax};"));
        self.line(format!("bra {finish};"));
        self.line(format!("{original}:"));
        self.line(format!("mov.{} {result}, {x};", compute.suffix()));
        self.line(format!("bra {finish};"));
        self.line(format!("{zero}:"));
        self.line(format!("mov.{bit_suffix} {result}, {sign};"));
        self.line(format!("bra {finish};"));
        self.line(format!("{nan}:"));
        self.line(format!("mov.{integer_suffix} {temporary}, {quiet_nan};"));
        self.line(format!("mov.{bit_suffix} {result}, {temporary};"));
        self.line(format!("{finish}:"));
        if floor {
            let zero_value = compute.constant(ConstantValue::Float(0.0))?;
            let sign_differs = self.reg(Scalar::Pred);
            self.line(format!("setp.ne.{} {predicate}, {result}, {zero_value};", compute.suffix()));
            self.line(format!("setp.lt.{} {other}, {result}, {zero_value};", compute.suffix()));
            self.line(format!("setp.lt.{} {sign_differs}, {y}, {zero_value};", compute.suffix()));
            self.line(format!("xor.pred {sign_differs}, {sign_differs}, {other};"));
            self.line(format!("and.pred {predicate}, {predicate}, {sign_differs};"));
            self.line(format!("@{predicate} add.rn.{} {result}, {result}, {y};", compute.suffix()));
        }
        if storage.half() {
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", storage.suffix()));
        } else {
            self.line(format!("mov.{} {destination}, {result};", compute.suffix()));
        }
        Ok(())
    }
}
