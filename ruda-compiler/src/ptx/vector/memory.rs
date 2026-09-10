use super::*;
use ruda_core::launch::ExecutionMode;

impl Emitter {
    pub(super) fn vector_memory(&mut self, array: Variable, index: Variable, value: Variable, vector_size: usize, unroll: usize, store: bool, checked: bool) -> Result<()> {
        let width = if vector_size == 0 { array.ty.vector_size() } else { vector_size };
        if width == 0 || unroll == 0 || value.ty.vector_size() != width || array.storage_type() != value.storage_type() {
            return Err(invalid("vector memory element type or width mismatch"));
        }
        let expanded_array = match array.kind {
            VariableKind::SharedArray { unroll_factor, .. } | VariableKind::LocalArray { unroll_factor, .. } => {
                width == array.ty.vector_size() && unroll == unroll_factor
            }
            _ => false,
        };
        let array = self.unrolled_memory_view(array, unroll)?;
        if unroll != 1 && !expanded_array && width.checked_mul(unroll) != Some(array.ty.vector_size()) {
            return Err(unsupported("vector memory unroll factor does not match storage width"));
        }
        let ty = Scalar::of(scalar(value).ty)?;
        let index_ty = Scalar::of(index.ty)?;
        if !matches!(index_ty, Scalar::U32 | Scalar::U64) { return Err(unsupported("array index must be unsigned")); }
        let (base, space) = self.memory_base(array, store)?;
        let index_reg = self.value(index)?;
        let registers = if store { self.vector_values(value)? } else { self.vector_destination(value)? };
        let end = self.label();
        if checked && self.mode != ExecutionMode::Unchecked && array.has_length() {
            let mut len = self.meta(array, false)?;
            let recast = array.ty.vector_size() != width;
            let common = if recast || index_ty == Scalar::U64 || self.address == Scalar::U64 { Scalar::U64 } else { Scalar::U32 };
            let mut offset = index_reg.clone();
            if index_ty != common {
                offset = self.reg(common);
                self.line(format!("cvt.u64.u32 {offset}, {index_reg};"));
            }
            if self.address != common {
                let wide = self.reg(common);
                self.line(format!("cvt.u64.u32 {wide}, {len};"));
                len = wide;
            }
            if recast {
                let capacity = self.reg(Scalar::U64);
                self.line(format!("mul.lo.u64 {capacity}, {len}, {};", array.ty.vector_size()));
                let count = self.reg(Scalar::U64);
                self.line(format!("div.u64 {count}, {capacity}, {width};"));
                len = count;
            }
            let oob = self.reg(Scalar::Pred);
            self.line(format!("setp.ge.{} {oob}, {offset}, {len};", common.suffix()));
            self.report_oob(array, &oob, &offset, common, &len, common, store)?;
            if !store { for register in &registers { self.memory_zero(ty, register); } }
            self.line(format!("@{oob} bra {end};"));
        }
        let stride = width.checked_mul(ty.bytes()).ok_or_else(|| invalid("vector memory stride overflow"))?;
        if stride > u32::MAX as usize { return Err(unsupported("vector memory stride exceeds U32")); }
        let address = self.reg(Scalar::U64);
        if index_ty == Scalar::U32 { self.line(format!("mul.wide.u32 {address}, {index_reg}, {stride};")); }
        else { self.line(format!("mul.lo.u64 {address}, {index_reg}, {stride};")); }
        self.line(format!("add.u64 {address}, {base}, {address};"));
        for (lane, register) in registers.iter().enumerate() {
            let offset = lane * ty.bytes();
            let element_address = format!("{address}+{offset}");
            if store { self.store_memory_scalar(space, ty, register, &element_address); }
            else { self.load_memory_scalar(space, ty, register, &element_address); }
        }
        self.line(format!("{end}:"));
        Ok(())
    }
}
