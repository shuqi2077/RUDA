use super::*;

impl Emitter {
    pub(super) fn atomic_target(&self, ty: Scalar, operation: &str) -> Result<()> {
        self.half_target(ty)?;
        if matches!(ty, Scalar::I16 | Scalar::U16)
            && (self.target.sm < 70 || self.target.version < (6, 3))
        {
            return Err(unsupported("16-bit integer atomics require SM >= 70 and PTX >= 6.3"));
        }
        match operation {
            "add" => match ty {
                Scalar::F64 if self.target.sm < 60 => return Err(unsupported("F64 atomic add requires SM >= 60")),
                Scalar::F16 if self.target.sm < 70 || self.target.version < (6, 3) => return Err(unsupported("F16 atomic add requires SM >= 70 and PTX >= 6.3")),
                Scalar::Pred => return Err(invalid("predicate atomic reduction")),
                _ => {}
            },
            "and" | "or" | "xor" | "min" | "max" => {
                if !ty.integer() { return Err(unsupported("non-integer atomic bitwise/min/max operation")); }
                if ty.bytes() == 8 && self.target.sm < 32 { return Err(unsupported("64-bit atomic bitwise/min/max requires SM >= 32")); }
            }
            "cas" | "exch" => {
                if ty.bytes() == 2 && (self.target.sm < 70 || self.target.version < (6, 3)) {
                    return Err(unsupported("16-bit atomic compare/exchange requires SM >= 70 and PTX >= 6.3"));
                }
            }
            _ => return Err(invalid("unknown atomic reduction")),
        }
        Ok(())
    }

    pub(super) fn atomic_reduce(&mut self, ty: Scalar, operation: &str, pointer: &str, value: &str, destination: &str) -> Result<()> {
        self.atomic_target(ty, operation)?;
        if matches!(ty, Scalar::I16 | Scalar::U16) {
            return self.atomic_integer_update(ty, operation, pointer, value, destination);
        }
        if operation == "exch" && ty.bytes() == 2 {
            return self.atomic_half_update(ty, false, pointer, value, destination);
        }
        if operation == "add" && ty == Scalar::BF16 && (self.target.sm < 90 || self.target.version < (7, 8)) {
            return self.atomic_half_update(ty, true, pointer, value, destination);
        }
        let suffix = match operation {
            "and" | "or" | "xor" | "exch" => ty.bits(),
            "add" if ty == Scalar::I64 => "u64",
            _ => ty.suffix(),
        };
        let modifier = if ty.half() && operation == "add" { ".noftz" } else { "" };
        self.line(format!("atom.{operation}{modifier}.{suffix} {destination}, [{pointer}], {value};"));
        Ok(())
    }

    fn atomic_half_update(&mut self, ty: Scalar, add: bool, pointer: &str, value: &str, destination: &str) -> Result<()> {
        self.atomic_target(ty, "cas")?;
        let operand = self.reg(ty);
        self.line(format!("mov.b16 {operand}, {value};"));
        let desired = self.reg(ty);
        if !add { self.line(format!("mov.b16 {desired}, {operand};")); }
        let observed = self.reg(ty);
        self.line(format!("atom.cas.b16 {observed}, [{pointer}], 0, 0;"));
        let expected = self.reg(ty);
        let previous_bits = self.reg(Scalar::U32);
        let expected_bits = self.reg(Scalar::U32);
        let retry = self.reg(Scalar::Pred);
        let again = self.label();
        self.line(format!("{again}:"));
        self.line(format!("mov.b16 {expected}, {observed};"));
        if add { self.half_binary("add", ty, &desired, &operand, &expected)?; }
        self.line(format!("atom.cas.b16 {observed}, [{pointer}], {expected}, {desired};"));
        self.line(format!("cvt.u32.u16 {previous_bits}, {observed};"));
        self.line(format!("cvt.u32.u16 {expected_bits}, {expected};"));
        self.line(format!("setp.ne.u32 {retry}, {previous_bits}, {expected_bits};"));
        self.line(format!("@{retry} bra {again};"));
        self.line(format!("mov.b16 {destination}, {observed};"));
        Ok(())
    }
}
