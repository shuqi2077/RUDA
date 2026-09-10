use super::*;

impl Emitter {
    pub(super) fn atomic_integer_compare_exchange(
        &mut self,
        ty: Scalar,
        pointer: &str,
        compare: &str,
        value: &str,
        destination: &str,
    ) -> Result<()> {
        if !matches!(ty, Scalar::I16 | Scalar::U16) {
            return Err(invalid("16-bit integer compare-and-swap requires I16/U16"));
        }
        self.atomic_target(ty, "cas")?;
        let expected = self.reg_b16();
        let desired = self.reg_b16();
        let observed = self.reg_b16();
        self.line(format!("cvt.u16.u32 {expected}, {compare};"));
        self.line(format!("cvt.u16.u32 {desired}, {value};"));
        self.line(format!("atom.cas.b16 {observed}, [{pointer}], {expected}, {desired};"));
        self.line(format!("cvt.{}.{} {destination}, {observed};", ty.suffix(), ty.memory_suffix()));
        Ok(())
    }

    pub(super) fn atomic_integer_update(
        &mut self,
        ty: Scalar,
        operation: &str,
        pointer: &str,
        value: &str,
        destination: &str,
    ) -> Result<()> {
        if !matches!(ty, Scalar::I16 | Scalar::U16)
            || !matches!(operation, "add" | "min" | "max" | "and" | "or" | "xor" | "exch")
        {
            return Err(invalid("invalid 16-bit integer atomic update"));
        }
        self.atomic_target(ty, "cas")?;
        let operand = self.reg(ty);
        self.line(format!("mov.b32 {operand}, {value};"));
        let observed = self.reg_b16();
        let expected = self.reg_b16();
        let desired = self.reg_b16();
        let current = self.reg(ty);
        let updated = self.reg(ty);
        let observed_bits = self.reg(Scalar::U32);
        let expected_bits = self.reg(Scalar::U32);
        let retry = self.reg(Scalar::Pred);
        let again = self.label();

        // Read and update exactly one 16-bit element, never an adjacent word.
        self.line(format!("atom.cas.b16 {observed}, [{pointer}], 0, 0;"));
        self.line(format!("{again}:"));
        self.line(format!("mov.b16 {expected}, {observed};"));
        self.line(format!("cvt.{}.{} {current}, {expected};", ty.suffix(), ty.memory_suffix()));
        match operation {
            "exch" => self.line(format!("mov.b32 {updated}, {operand};")),
            "and" | "or" | "xor" => {
                self.line(format!("{operation}.b32 {updated}, {current}, {operand};"));
            }
            _ => self.line(format!("{operation}.{} {updated}, {current}, {operand};", ty.suffix())),
        }
        self.line(format!("cvt.u16.u32 {desired}, {updated};"));
        self.line(format!("atom.cas.b16 {observed}, [{pointer}], {expected}, {desired};"));
        self.line(format!("cvt.u32.u16 {observed_bits}, {observed};"));
        self.line(format!("cvt.u32.u16 {expected_bits}, {expected};"));
        self.line(format!("setp.ne.u32 {retry}, {observed_bits}, {expected_bits};"));
        self.line(format!("@{retry} bra {again};"));
        self.line(format!("cvt.{}.{} {destination}, {observed};", ty.suffix(), ty.memory_suffix()));
        Ok(())
    }
}
