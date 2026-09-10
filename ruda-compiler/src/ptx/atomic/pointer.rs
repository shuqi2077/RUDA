use super::*;
use ruda_core::{kernel::Visibility, launch::ExecutionMode};

impl Emitter {
    pub(super) fn atomic_pointer(&self, variable: Variable) -> Result<String> {
        element(variable)?;
        self.atomic_pointers.get(&variable).cloned()
            .ok_or_else(|| invalid("atomic pointer has no indexed memory binding"))
    }

    pub(super) fn atomic_pointer_destination(&mut self, variable: Variable) -> Result<String> {
        element(variable)?;
        if !matches!(variable.kind, VariableKind::LocalConst { .. } | VariableKind::LocalMut { .. } | VariableKind::Versioned { .. }) {
            return Err(invalid("atomic pointer destination must be a local variable"));
        }
        if let Some(pointer) = self.atomic_pointers.get(&variable) { return Ok(pointer.clone()); }
        let pointer = self.reg(Scalar::U64);
        self.atomic_pointers.insert(variable, pointer.clone());
        Ok(pointer)
    }

    pub(super) fn atomic_index(&mut self, array: Variable, index: Variable, out: Variable, checked: bool) -> Result<()> {
        let ty = element(array)?;
        self.half_target(ty)?;
        if array.ty != out.ty { return Err(invalid("atomic index element type mismatch")); }
        let (base, shared) = if matches!(array.kind, VariableKind::SharedArray { .. }) {
            (self.shared_array(array)?, true)
        } else {
            let buffer = self.buffer(array)?;
            if buffer.visibility != Visibility::ReadWrite {
                return Err(invalid("atomic operations require a writable buffer"));
            }
            (format!("%buffer_{}", buffer.id), false)
        };
        let mut index_ty = Scalar::of(index.ty)?;
        if !matches!(index_ty, Scalar::U32 | Scalar::U64) { return Err(invalid("atomic array index must be unsigned")); }
        let mut index_value = self.value(index)?;
        if checked && self.mode == ExecutionMode::Validate && array.has_buffer_length() {
            let length = self.meta(array, false)?;
            let wide_index = self.validation_u64(&index_value, index_ty)?;
            let wide_length = self.validation_u64(&length, self.address)?;
            let oob = self.reg(Scalar::Pred);
            self.line(format!("setp.ge.u64 {oob}, {wide_index}, {wide_length};"));
            self.report_oob(array, &oob, &wide_index, Scalar::U64, &wide_length, Scalar::U64, false)?;
        }
        if checked && self.mode != ExecutionMode::Unchecked && array.has_buffer_length() {
            let length = self.meta(array, false)?;
            let mut last = self.reg(self.address);
            self.line(format!("sub.{} {last}, {length}, 1;", self.address.suffix()));
            if index_ty != self.address {
                let widened = self.reg(Scalar::U64);
                if index_ty == Scalar::U32 {
                    self.line(format!("cvt.u64.u32 {widened}, {index_value};"));
                    index_value = widened;
                } else {
                    self.line(format!("cvt.u64.u32 {widened}, {last};"));
                    last = widened;
                }
                index_ty = Scalar::U64;
            }
            let clamped = self.reg(index_ty);
            self.line(format!("min.{} {clamped}, {index_value}, {last};", index_ty.suffix()));
            index_value = clamped;
        }
        let offset = self.reg(Scalar::U64);
        let multiply = if index_ty == Scalar::U32 { "mul.wide.u32" } else { "mul.lo.u64" };
        self.line(format!("{multiply} {offset}, {index_value}, {};", ty.bytes()));
        self.line(format!("add.u64 {offset}, {base}, {offset};"));
        let destination = self.atomic_pointer_destination(out)?;
        if shared { self.line(format!("cvta.shared.u64 {destination}, {offset};")); }
        else { self.line(format!("mov.u64 {destination}, {offset};")); }
        Ok(())
    }
}
