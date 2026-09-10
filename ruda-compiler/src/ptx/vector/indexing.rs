use super::*;

impl Emitter {
    pub(super) fn vector_index(&mut self, vector: Variable, index: Variable, value: Variable, store: bool) -> Result<()> {
        let ty = Scalar::of(scalar(vector).ty)?;
        let index_ty = Scalar::of(index.ty)?;
        if !index_ty.integer() || value.ty != scalar(vector).ty {
            return Err(invalid("vector index requires an integer index and scalar element"));
        }
        let registers = if store { self.vector_destination(vector)? } else { self.vector_values(vector)? };
        let source_index = self.value(index)?;
        let saved_index = self.reg(index_ty);
        self.line(format!("mov.{} {saved_index}, {source_index};", index_ty.suffix()));
        let value_register = if store { self.value(value)? } else { self.destination(value)? };
        let saved_value = self.reg(ty);
        if store { self.line(format!("mov.{} {saved_value}, {value_register};", ty.storage())); }
        else if ty == Scalar::Pred { self.line(format!("setp.ne.u32 {saved_value}, 0, 0;")); }
        else { self.line(format!("mov.{} {saved_value}, 0;", ty.bits())); }
        for (lane, register) in registers.iter().enumerate() {
            let selected = self.reg(Scalar::Pred);
            self.line(format!("setp.eq.{} {selected}, {saved_index}, {lane};", index_ty.suffix()));
            let (destination, source) = if store { (register, &saved_value) } else { (&saved_value, register) };
            self.line(format!("@{selected} mov.{} {destination}, {source};", ty.storage()));
        }
        if !store { self.line(format!("mov.{} {value_register}, {saved_value};", ty.storage())); }
        Ok(())
    }
}
