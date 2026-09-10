use super::*;

impl Emitter {
    pub(super) fn async_copy(
        &mut self, source: Variable, destination: Variable, source_length: Variable,
        offset_source: Variable, offset_out: Variable, copy_length: u32, checked: bool,
    ) -> Result<()> {
        if !matches!(copy_length, 4 | 8 | 16) {
            return Err(invalid("cp.async copy size must be 4, 8 or 16 bytes"));
        }
        let source_type = Scalar::memory_element(source.ty)?;
        if source_type != Scalar::memory_element(destination.ty)? {
            return Err(invalid("async copy element types differ"));
        }
        let (source_base, source_space) = self.memory_base(source, false)?;
        let (destination_base, destination_space) = self.memory_base(destination, true)?;
        if source_space != "global" || destination_space != "shared" {
            return Err(unsupported("cp.async requires global source and shared destination"));
        }
        let source_stride = source.ty.vector_size().checked_mul(source_type.bytes())
            .ok_or_else(|| invalid("async copy source stride overflow"))?;
        let destination_stride = destination.ty.vector_size().checked_mul(source_type.bytes())
            .ok_or_else(|| invalid("async copy destination stride overflow"))?;
        let source_address = self.async_address(&source_base, offset_source, source_stride)?;
        let destination_address = self.async_address(&destination_base, offset_out, destination_stride)?;
        let shared = self.reg(Scalar::U32);
        self.line(format!("cvt.u32.u64 {shared}, {destination_address};"));
        let size = if checked {
            let length_type = Scalar::of(source_length.ty)?;
            if !matches!(length_type, Scalar::U32 | Scalar::U64) {
                return Err(invalid("async copy source length must be unsigned"));
            }
            let length = self.value(source_length)?;
            let bytes = self.reg(length_type);
            self.line(format!("mul.lo.{} {bytes}, {length}, {source_stride};", length_type.suffix()));
            if length_type == Scalar::U64 {
                let size = self.reg(Scalar::U32);
                self.line(format!("cvt.u32.u64 {size}, {bytes};"));
                size
            } else { bytes }
        } else { copy_length.to_string() };
        let cache = if copy_length == 16 { "cg" } else { "ca" };
        self.line(format!("cp.async.{cache}.shared.global [{shared}], [{source_address}], {copy_length}, {size};"));
        Ok(())
    }

    pub(super) fn async_address(&mut self, base: &str, offset: Variable, stride: usize) -> Result<String> {
        let ty = Scalar::of(offset.ty)?;
        if !matches!(ty, Scalar::U32 | Scalar::U64) || stride > u32::MAX as usize {
            return Err(invalid("async copy offset must be unsigned and stride must fit U32"));
        }
        let offset = self.value(offset)?;
        let address = self.reg(Scalar::U64);
        let multiply = if ty == Scalar::U32 { "mul.wide.u32" } else { "mul.lo.u64" };
        self.line(format!("{multiply} {address}, {offset}, {stride};"));
        self.line(format!("add.u64 {address}, {base}, {address};"));
        Ok(address)
    }
}
