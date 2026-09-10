use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::{ir::Variable, launch::ExecutionMode};

impl Emitter {
    pub(super) fn validation_u64(&mut self, value: &str, ty: Scalar) -> Result<String> {
        match ty {
            Scalar::U64 => Ok(value.to_owned()),
            Scalar::U32 => {
                let output = self.reg(Scalar::U64);
                self.line(format!("cvt.u64.u32 {output}, {value};"));
                Ok(output)
            }
            _ => Err(invalid("OOB diagnostic index and length must be unsigned")),
        }
    }

    pub(super) fn report_oob(
        &mut self, array: Variable, predicate: &str,
        index: &str, index_ty: Scalar, length: &str, length_ty: Scalar, store: bool,
    ) -> Result<()> {
        if self.mode != ExecutionMode::Validate { return Ok(()); }
        let end = self.label();
        self.line(format!("@!{predicate} bra {end};"));
        let index = self.validation_u64(index, index_ty)?;
        let length = self.validation_u64(length, length_ty)?;
        let kind = if store { "write" } else { "read" };
        let prefix = format!("[VALIDATION {}]: Encountered OOB {kind} in {array}", self.kernel_name)
            .replace('%', "%%");
        self.printf_registers(
            &format!("{prefix} at %llu, length is %llu\n"),
            vec![(Scalar::U64, index), (Scalar::U64, length)],
        )?;
        self.line(format!("{end}:"));
        Ok(())
    }
}
