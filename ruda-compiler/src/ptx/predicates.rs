use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{BinaryOperator, Variable};

impl Emitter {
    pub(super) fn float_classification(&mut self, out: Variable, input: Variable, nan: bool) -> Result<()> {
        let ty = Scalar::of(input.ty)?;
        if !ty.float() { return Err(invalid("floating-point classification requires a float input")); }
        let source = self.value(input)?;
        let dst = self.destination(out)?;
        if matches!(ty, Scalar::F32 | Scalar::F64) {
            let property = if nan { "notanumber" } else { "infinite" };
            self.line(format!("testp.{property}.{} {dst}, {source};", ty.suffix()));
            return Ok(());
        }
        self.half_target(ty)?;
        let bits = if ty.half() {
            let bits = self.reg(Scalar::U32);
            self.line(format!("cvt.u32.u16 {bits}, {source};"));
            bits
        } else { source };
        let magnitude = self.reg(Scalar::U32);
        let (mask, infinity) = match ty {
            Scalar::F16 => (0x7fff, 0x7c00),
            Scalar::BF16 => (0x7fff, 0x7f80),
            Scalar::E5M2 => (0x7f, 0x7c),
            Scalar::E4M3 => (0x7f, 0x7f),
            _ => return Err(invalid("unsupported floating-point classification storage")),
        };
        self.line(format!("and.b32 {magnitude}, {bits}, {mask};"));
        if ty == Scalar::E4M3 && !nan {
            self.line(format!("setp.ne.u32 {dst}, 0, 0;"));
        } else {
            let comparison = if nan && ty != Scalar::E4M3 { "gt" } else { "eq" };
            self.line(format!("setp.{comparison}.u32 {dst}, {magnitude}, {infinity};"));
        }
        Ok(())
    }

    pub(super) fn predicate_comparison(&mut self, out: Variable, op: BinaryOperator, comparison: &str) -> Result<()> {
        let a = self.value(op.lhs)?;
        let b = self.value(op.rhs)?;
        let dst = self.destination(out)?;
        match comparison {
            "eq" | "ne" => {
                self.line(format!("xor.pred {dst}, {a}, {b};"));
                if comparison == "eq" { self.line(format!("not.pred {dst}, {dst};")); }
            }
            "lt" | "le" | "gt" | "ge" => {
                let (negated, other) = if matches!(comparison, "lt" | "le") { (a, b) } else { (b, a) };
                let inverse = self.reg(Scalar::Pred);
                self.line(format!("not.pred {inverse}, {negated};"));
                let operation = if matches!(comparison, "lt" | "gt") { "and" } else { "or" };
                self.line(format!("{operation}.pred {dst}, {inverse}, {other};"));
            }
            _ => return Err(invalid("unknown predicate comparison")),
        }
        Ok(())
    }
}
