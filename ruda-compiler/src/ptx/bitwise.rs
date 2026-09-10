use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{BinaryOperator, Bitwise, UnaryOperator, Variable};

impl Emitter {
    pub fn bitwise(&mut self, operation: Bitwise, out: Variable) -> Result<()> {
        match operation {
            Bitwise::BitwiseAnd(op) => self.bit_binary(out, op, "and"),
            Bitwise::BitwiseOr(op) => self.bit_binary(out, op, "or"),
            Bitwise::BitwiseXor(op) => self.bit_binary(out, op, "xor"),
            Bitwise::BitwiseNot(op) => self.bit_unary(out, op, "not"),
            Bitwise::ReverseBits(op) => self.bit_unary(out, op, "brev"),
            Bitwise::ShiftLeft(op) => self.shift(out, op, true),
            Bitwise::ShiftRight(op) => self.shift(out, op, false),
            Bitwise::CountOnes(op) => self.bit_count(out, op, "popc", false, false),
            Bitwise::LeadingZeros(op) => self.bit_count(out, op, "clz", false, false),
            Bitwise::TrailingZeros(op) => self.bit_count(out, op, "clz", true, false),
            Bitwise::FindFirstSet(op) => self.bit_count(out, op, "clz", true, true),
        }
    }

    fn bit_binary(&mut self, out: Variable, op: BinaryOperator, opcode: &str) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !(ty.integer() || ty == Scalar::Pred) || out.ty != op.lhs.ty || out.ty != op.rhs.ty {
            return Err(invalid(format!("{opcode} requires matching integer operands")));
        }
        let a = self.value(op.lhs)?;
        let b = self.value(op.rhs)?;
        let dst = self.destination(out)?;
        let suffix = if ty == Scalar::Pred { "pred" } else { ty.bits() };
        self.line(format!("{opcode}.{suffix} {dst}, {a}, {b};"));
        Ok(())
    }

    fn bit_unary(&mut self, out: Variable, op: UnaryOperator, opcode: &str) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !(ty.integer() || (ty == Scalar::Pred && opcode == "not")) || out.ty != op.input.ty {
            return Err(invalid(format!("{opcode} requires matching integer operands")));
        }
        let input = self.value(op.input)?;
        let dst = self.destination(out)?;
        let suffix = if ty == Scalar::Pred { "pred" } else { ty.bits() };
        self.line(format!("{opcode}.{suffix} {dst}, {input};"));
        if opcode == "brev" && ty.narrow() {
            self.line(format!("shr.u32 {dst}, {dst}, {};", 32 - ty.bytes() * 8));
        }
        Ok(())
    }

    fn shift(&mut self, out: Variable, op: BinaryOperator, left: bool) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        let count_ty = Scalar::of(op.rhs.ty)?;
        if !ty.integer() || !count_ty.integer() || out.ty != op.lhs.ty {
            return Err(invalid("shift requires integer operands and matching input/output types"));
        }
        let input = self.value(op.lhs)?;
        let mut count = self.value(op.rhs)?;
        if count_ty.bytes() == 8 {
            // PTX takes a U32 count and clamps oversized counts to the bit width.
            // Clamp before narrowing so a large U64 count cannot wrap to zero.
            let bounded = self.reg(Scalar::U64);
            self.line(format!("min.u64 {bounded}, {count}, {};", ty.bytes() * 8));
            count = self.reg(Scalar::U32);
            self.line(format!("cvt.u32.u64 {count}, {bounded};"));
        }
        let dst = self.destination(out)?;
        let opcode = if left { "shl" } else { "shr" };
        let suffix = if left { ty.bits() } else { ty.suffix() };
        self.line(format!("{opcode}.{suffix} {dst}, {input}, {count};"));
        Ok(())
    }

    fn bit_count(
        &mut self,
        out: Variable,
        op: UnaryOperator,
        opcode: &str,
        reverse: bool,
        one_based: bool,
    ) -> Result<()> {
        let ty = Scalar::of(op.input.ty)?;
        if !ty.integer() || Scalar::of(out.ty)? != Scalar::U32 {
            return Err(invalid("bit count requires an integer input and U32 output"));
        }
        let mut input = self.value(op.input)?;
        if ty.narrow() {
            let masked = self.reg(Scalar::U32);
            self.line(format!("and.b32 {masked}, {input}, {};", (1_u32 << (ty.bytes() * 8)) - 1));
            input = masked;
        }
        let mut counted = input.clone();
        if reverse {
            counted = self.reg(ty);
            self.line(format!("brev.{} {counted}, {input};", ty.bits()));
        }
        let dst = self.destination(out)?;
        if one_based {
            let count = self.reg(Scalar::U32);
            let nonzero = self.reg(Scalar::Pred);
            // Determine the predicate before writing dst; mutable IR may alias input.
            self.line(format!("setp.ne.{} {nonzero}, {input}, 0;", ty.suffix()));
            self.line(format!("{opcode}.{} {count}, {counted};", ty.bits()));
            self.line(format!("add.u32 {count}, {count}, 1;"));
            self.line(format!("selp.u32 {dst}, {count}, 0, {nonzero};"));
        } else {
            self.line(format!("{opcode}.{} {dst}, {counted};", ty.bits()));
            if ty.narrow() && opcode == "clz" {
                if reverse {
                    self.line(format!("min.u32 {dst}, {dst}, {};", ty.bytes() * 8));
                } else {
                    self.line(format!("sub.u32 {dst}, {dst}, {};", 32 - ty.bytes() * 8));
                }
            }
        }
        Ok(())
    }
}
