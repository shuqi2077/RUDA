use super::*;

impl Emitter {
    pub(super) fn matrix_volta(&mut self, out: Variable, a: Variable, b: Variable, c: Variable) -> Result<()> {
        if self.target.sm < 70 || self.target.version < (6, 4) {
            return Err(unsupported("FP16 m8n8k4 MMA requires SM >= 70 and PTX >= 6.4"));
        }
        let a_ty = Scalar::of(a.ty.with_vector_size(1))?;
        let b_ty = Scalar::of(b.ty.with_vector_size(1))?;
        let c_ty = Scalar::of(c.ty.with_vector_size(1))?;
        let d_ty = Scalar::of(out.ty.with_vector_size(1))?;
        if a_ty != Scalar::F16 || b_ty != Scalar::F16
            || !matches!((c_ty, d_ty), (Scalar::F16, Scalar::F16 | Scalar::F32) | (Scalar::F32, Scalar::F32))
            || Scalar::is_tf32(c.ty) || Scalar::is_tf32(out.ty)
        {
            return Err(unsupported("FP16 m8n8k4 MMA accumulation type combination"));
        }
        let a = self.matrix_read_registers(a, 2)?;
        let b = self.matrix_read_registers(b, 2)?;
        let c = self.matrix_read_registers(c, if c_ty == Scalar::F16 { 4 } else { 8 })?;
        let d_count = if d_ty == Scalar::F16 { 4 } else { 8 };
        let d_register_ty = if d_ty == Scalar::F16 { Scalar::U32 } else { Scalar::F32 };
        let d = (0..d_count).map(|_| if d_register_ty == Scalar::U32 {
            self.reg_b32()
        } else { self.reg(d_register_ty) }).collect::<Vec<_>>();
        self.line(format!("mma.sync.aligned.m8n8k4.row.col.{}.f16.f16.{} {{{}}}, {{{}}}, {{{}}}, {{{}}};",
            d_ty.suffix(), c_ty.suffix(), d.join(", "), a.join(", "), b.join(", "), c.join(", ")));
        self.matrix_write_registers(out, &d)
    }

    pub(super) fn matrix_volta_index(
        &mut self, out: Variable, lane: Variable, index: Variable,
        ident: MatrixIdent, ty: Scalar, row: bool,
    ) -> Result<()> {
        let lane = self.value(lane)?;
        let index = self.value(index)?;
        let result = if ident == MatrixIdent::Accumulator && ty == Scalar::F32 {
            if row {
                let thread = self.matrix_index_op("and.b32", &lane, "1");
                let element = self.matrix_index_op("and.b32", &index, "2");
                let low = self.matrix_index_op("add.u32", &thread, &element);
                let high = self.matrix_index_op("shr.u32", &lane, "2");
                let high = self.matrix_index_op("and.b32", &high, "4");
                self.matrix_index_op("add.u32", &low, &high)
            } else {
                let block = self.matrix_index_op("and.b32", &index, "4");
                let thread = self.matrix_index_op("and.b32", &lane, "2");
                let element = self.matrix_index_op("and.b32", &index, "1");
                let low = self.matrix_index_op("add.u32", &thread, &element);
                self.matrix_index_op("add.u32", &block, &low)
            }
        } else if matches!((ident, row), (MatrixIdent::A | MatrixIdent::Accumulator, true) | (MatrixIdent::B, false)) {
            let low = self.matrix_index_op("and.b32", &lane, "3");
            let high = self.matrix_index_op("shr.u32", &lane, "2");
            let high = self.matrix_index_op("and.b32", &high, "4");
            self.matrix_index_op("add.u32", &low, &high)
        } else { index };
        let destination = self.destination(out)?;
        self.line(format!("mov.u32 {destination}, {result};"));
        Ok(())
    }
}
