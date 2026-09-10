use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn native_address(&mut self, array: Variable, index: Variable, output: Variable) -> Result<()> {
        if Scalar::of(output.ty)? != Scalar::U64 { return Err(invalid("native address result must be U64")); }
        let index_ty = Scalar::of(index.ty)?;
        if !matches!(index_ty, Scalar::U32 | Scalar::U64) { return Err(invalid("native element index must be unsigned")); }
        let (base, space) = self.memory_base(array, false)?;
        let bytes = Scalar::memory_element(array.ty)?.bytes().checked_mul(array.ty.vector_size())
            .ok_or_else(|| invalid("native element size overflow"))?;
        let index_value = self.value(index)?;
        let offset = self.reg(Scalar::U64);
        if index_ty == Scalar::U32 {
            self.line(format!("mul.wide.u32 {offset}, {index_value}, {bytes};"));
        } else {
            self.line(format!("mul.lo.u64 {offset}, {index_value}, {bytes};"));
        }
        let address = self.reg(Scalar::U64);
        self.line(format!("add.u64 {address}, {base}, {offset};"));
        let destination = self.destination(output)?;
        self.line(format!("cvta.{space}.u64 {destination}, {address};"));
        Ok(())
    }

    pub fn native_memory(&mut self, pointer: Variable, value: Variable, store: bool) -> Result<()> {
        if Scalar::of(pointer.ty)? != Scalar::U64 { return Err(invalid("native memory requires a U64 byte address")); }
        let base = self.value(pointer)?;
        let ty = Scalar::of(value.ty.with_vector_size(1))?;
        self.half_target(ty)?;
        let registers = if store { self.vector_values(value)? } else { self.vector_destination(value)? };
        for (index, register) in registers.iter().enumerate() {
            let address = if index == 0 { base.clone() } else {
                let address = self.reg(Scalar::U64);
                self.line(format!("add.u64 {address}, {base}, {};", index * ty.bytes()));
                address
            };
            if store {
                if ty == Scalar::Pred {
                    let byte = self.reg(Scalar::U32);
                    self.line(format!("selp.u32 {byte}, 1, 0, {register};"));
                    self.line(format!("st.u8 [{address}], {byte};"));
                } else {
                    self.line(format!("st.{} [{address}], {register};", ty.memory_suffix()));
                }
            } else if ty == Scalar::Pred {
                let byte = self.reg(Scalar::U32);
                self.line(format!("ld.u8 {byte}, [{address}];"));
                self.line(format!("setp.ne.u32 {register}, {byte}, 0;"));
            } else {
                self.line(format!("ld.{} {register}, [{address}];", ty.memory_suffix()));
            }
        }
        Ok(())
    }
}
