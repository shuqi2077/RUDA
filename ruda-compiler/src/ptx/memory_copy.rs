use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn copy_memory(
        &mut self,
        input: Variable,
        input_index: Variable,
        input_offset: Option<Variable>,
        output: Variable,
        output_index: Variable,
        output_offset: Option<Variable>,
        length: usize,
    ) -> Result<()> {
        if input.ty != output.ty || input.ty.vector_size() == 0 {
            return Err(invalid("memory copy requires matching element types and vector widths"));
        }
        let ty = Scalar::of(input.ty.with_vector_size(1))?;
        self.half_target(ty)?;
        let ty = if ty == Scalar::Pred { Scalar::U8 } else { ty };
        let stride = input.ty.vector_size().checked_mul(ty.bytes())
            .filter(|stride| *stride <= u32::MAX as usize)
            .ok_or_else(|| unsupported("memory copy stride exceeds U32"))?;
        let (source, source_space) = self.memory_base(input, false)?;
        let (destination, destination_space) = self.memory_base(output, true)?;
        let (source_index, source_type) = self.copy_index(input_index, input_offset)?;
        let (destination_index, destination_type) = self.copy_index(output_index, output_offset)?;
        if length == 0 { return Ok(()); }
        let remaining = self.reg(Scalar::U64);
        self.line(format!("mov.u64 {remaining}, {length};"));
        let again = self.label();
        self.line(format!("{again}:"));
        let source_address = self.copy_address(&source, &source_index, source_type, stride);
        let destination_address = self.copy_address(&destination, &destination_index, destination_type, stride);
        let registers = (0..input.ty.vector_size()).map(|_| self.reg(ty)).collect::<Vec<_>>();
        for (lane, register) in registers.iter().enumerate() {
            self.line(format!("ld.{source_space}.{} {register}, [{source_address}+{}];", ty.memory_suffix(), lane * ty.bytes()));
        }
        for (lane, register) in registers.iter().enumerate() {
            self.line(format!("st.{destination_space}.{} [{destination_address}+{}], {register};", ty.memory_suffix(), lane * ty.bytes()));
        }
        if length > 1 {
            self.line(format!("add.{} {source_index}, {source_index}, 1;", source_type.suffix()));
            self.line(format!("add.{} {destination_index}, {destination_index}, 1;", destination_type.suffix()));
            self.line(format!("sub.u64 {remaining}, {remaining}, 1;"));
            let more = self.reg(Scalar::Pred);
            self.line(format!("setp.ne.u64 {more}, {remaining}, 0;"));
            self.line(format!("@{more} bra {again};"));
        }
        Ok(())
    }

    fn copy_index(&mut self, index: Variable, offset: Option<Variable>) -> Result<(String, Scalar)> {
        let ty = Scalar::of(index.ty)?;
        if !matches!(ty, Scalar::U32 | Scalar::U64) { return Err(invalid("memory copy index must be unsigned")); }
        let value = self.value(index)?;
        let cursor = self.reg(ty);
        if let Some(offset) = offset {
            if offset.ty != index.ty { return Err(invalid("memory copy slice offset type mismatch")); }
            let offset = self.value(offset)?;
            self.line(format!("add.{} {cursor}, {value}, {offset};", ty.suffix()));
        } else {
            self.line(format!("mov.{} {cursor}, {value};", ty.suffix()));
        }
        Ok((cursor, ty))
    }

    fn copy_address(&mut self, base: &str, index: &str, ty: Scalar, stride: usize) -> String {
        let address = self.reg(Scalar::U64);
        let multiply = if ty == Scalar::U32 { "mul.wide.u32" } else { "mul.lo.u64" };
        self.line(format!("{multiply} {address}, {index}, {stride};"));
        self.line(format!("add.u64 {address}, {base}, {address};"));
        address
    }
}
