use super::*;

impl Emitter {
    fn matrix_register_capacity(&self, variable: Variable, count: usize) -> Result<Scalar> {
        let ty = Scalar::of(variable.ty.with_vector_size(1))?;
        self.half_target(ty)?;
        if !ty.half() && !ty.fp8() && !matches!(ty, Scalar::I8 | Scalar::U8 | Scalar::U32 | Scalar::I32 | Scalar::F32 | Scalar::F64) {
            return Err(unsupported("manual matrix registers require FP8, I8/U8, half, 32-bit or F64 elements"));
        }
        let length = match variable.kind {
            VariableKind::LocalArray { length, unroll_factor, .. } => length.checked_mul(unroll_factor),
            VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } | VariableKind::Versioned { .. } => Some(1),
            _ => return Err(unsupported("manual matrix registers must be local vectors or arrays")),
        };
        let bytes = length.and_then(|length| length.checked_mul(variable.ty.vector_size())).and_then(|length| length.checked_mul(ty.bytes()))
            .ok_or_else(|| invalid("manual matrix register capacity overflow"))?;
        let required = count.checked_mul(ty.bytes().max(4)).ok_or_else(|| invalid("manual matrix register count overflow"))?;
        if bytes < required { return Err(invalid("manual matrix register container is too small")); }
        Ok(ty)
    }

    pub(super) fn matrix_read_registers(&mut self, variable: Variable, count: usize) -> Result<Vec<String>> {
        let ty = self.matrix_register_capacity(variable, count)?;
        let register_type = if matches!(ty, Scalar::F32 | Scalar::F64) && !Scalar::is_tf32(variable.ty) { ty } else { Scalar::U32 };
        let registers = (0..count).map(|_| if register_type == Scalar::U32 {
            self.reg_b32()
        } else { self.reg(register_type) }).collect::<Vec<_>>();
        if matches!(variable.kind, VariableKind::LocalArray { .. }) {
            let (base, space) = self.memory_base(variable, false)?;
            for (index, register) in registers.iter().enumerate() {
                if ty.half() {
                    let low = self.reg(ty);
                    let high = self.reg(ty);
                    self.line(format!("ld.{space}.b16 {low}, [{base}+{}];", index * 4));
                    self.line(format!("ld.{space}.b16 {high}, [{base}+{}];", index * 4 + 2));
                    self.line(format!("mov.b32 {register}, {{{low}, {high}}};"));
                } else if ty.fp8() || matches!(ty, Scalar::I8 | Scalar::U8) {
                    let mut bytes = Vec::with_capacity(4);
                    for byte in 0..4 {
                        let value = self.reg(Scalar::U32);
                        self.line(format!("ld.{space}.u8 {value}, [{base}+{}];", index * 4 + byte));
                        bytes.push(value);
                    }
                    self.matrix_pack_bytes(register, &bytes);
                } else if ty == Scalar::F64 {
                    self.line(format!("ld.{space}.f64 {register}, [{base}+{}];", index * 8));
                } else { self.line(format!("ld.{space}.b32 {register}, [{base}+{}];", index * 4)); }
            }
        } else {
            let values = self.vector_values(variable)?;
            for (index, register) in registers.iter().enumerate() {
                if ty.half() { self.line(format!("mov.b32 {register}, {{{}, {}}};", values[index * 2], values[index * 2 + 1])); }
                else if ty.fp8() || matches!(ty, Scalar::I8 | Scalar::U8) { self.matrix_pack_bytes(register, &values[index * 4..index * 4 + 4]); }
                else if ty == Scalar::F64 { self.line(format!("mov.f64 {register}, {};", values[index])); }
                else { self.line(format!("mov.b32 {register}, {};", values[index])); }
            }
        }
        Ok(registers)
    }

    pub(super) fn matrix_write_registers(&mut self, variable: Variable, registers: &[String]) -> Result<()> {
        let ty = self.matrix_register_capacity(variable, registers.len())?;
        if matches!(variable.kind, VariableKind::LocalArray { .. }) {
            let (base, space) = self.memory_base(variable, true)?;
            for (index, register) in registers.iter().enumerate() {
                if ty.half() {
                    let low = self.reg(ty);
                    let high = self.reg(ty);
                    self.line(format!("mov.b32 {{{low}, {high}}}, {register};"));
                    self.line(format!("st.{space}.b16 [{base}+{}], {low};", index * 4));
                    self.line(format!("st.{space}.b16 [{base}+{}], {high};", index * 4 + 2));
                } else if ty.fp8() || matches!(ty, Scalar::I8 | Scalar::U8) {
                    for byte in 0..4 {
                        let value = self.reg(Scalar::U32);
                        self.line(format!("bfe.u32 {value}, {register}, {}, 8;", byte * 8));
                        self.line(format!("st.{space}.u8 [{base}+{}], {value};", index * 4 + byte));
                    }
                } else if ty == Scalar::F64 {
                    self.line(format!("st.{space}.f64 [{base}+{}], {register};", index * 8));
                } else { self.line(format!("st.{space}.b32 [{base}+{}], {register};", index * 4)); }
            }
        } else {
            let values = self.vector_destination(variable)?;
            for (index, register) in registers.iter().enumerate() {
                if ty.half() { self.line(format!("mov.b32 {{{}, {}}}, {register};", values[index * 2], values[index * 2 + 1])); }
                else if ty.fp8() || matches!(ty, Scalar::I8 | Scalar::U8) {
                    let suffix = if ty.fp8() { "u32" } else { ty.suffix() };
                    for byte in 0..4 {
                        self.line(format!("bfe.{suffix} {}, {register}, {}, 8;", values[index * 4 + byte], byte * 8));
                    }
                }
                else if ty == Scalar::F64 { self.line(format!("mov.f64 {}, {register};", values[index])); }
                else { self.line(format!("mov.b32 {}, {register};", values[index])); }
            }
        }
        Ok(())
    }

    fn matrix_pack_bytes(&mut self, destination: &str, values: &[String]) {
        self.line(format!("and.b32 {destination}, {}, 255;", values[0]));
        let byte = self.reg(Scalar::U32);
        for (index, value) in values.iter().enumerate().skip(1) {
            self.line(format!("and.b32 {byte}, {value}, 255;"));
            self.line(format!("shl.b32 {byte}, {byte}, {};", index * 8));
            self.line(format!("or.b32 {destination}, {destination}, {byte};"));
        }
    }
}
