use super::*;

impl Emitter {
    pub(super) fn matrix_execute(&mut self, out: Variable, a: Variable, b: Variable, c: Variable) -> Result<()> {
        let (a_matrix, a_ty, a) = self.matrix_fragment(a)?;
        let (b_matrix, b_ty, b) = self.matrix_fragment(b)?;
        let (c_matrix, c_ty, c) = self.matrix_fragment(c)?;
        let (d_matrix, d_ty, d) = self.matrix_fragment(out)?;
        if a_matrix.ident != MatrixIdent::A || b_matrix.ident != MatrixIdent::B || c_matrix.ident != MatrixIdent::Accumulator || d_matrix.ident != MatrixIdent::Accumulator ||
            [b_matrix, c_matrix, d_matrix].iter().any(|matrix| !same_shape(a_matrix, *matrix)) {
            return Err(invalid("WMMA operand roles or shapes differ"));
        }
        let a_layout = layout(a_matrix.layout)?;
        let b_layout = layout(b_matrix.layout)?;
        let types = match (a_ty, b_ty, c_ty, d_ty) {
            (Scalar::F32, Scalar::F32, Scalar::F32, Scalar::F32)
                if Scalar::is_tf32(Type::new(a_matrix.storage)) && Scalar::is_tf32(Type::new(b_matrix.storage)) => "f32.tf32.tf32.f32".into(),
            (Scalar::F16, Scalar::F16, Scalar::F16 | Scalar::F32, Scalar::F16 | Scalar::F32) => format!("{}.{}", d_ty.suffix(), c_ty.suffix()),
            (Scalar::BF16, Scalar::BF16, Scalar::F32, Scalar::F32) => "f32.bf16.bf16.f32".into(),
            (Scalar::F64, Scalar::F64, Scalar::F64, Scalar::F64) => "rn.f64.f64.f64.f64".into(),
            (Scalar::I8, Scalar::I8, Scalar::I32, Scalar::I32) |
            (Scalar::U8, Scalar::U8, Scalar::I32, Scalar::I32) => format!("s32.{}.{}.s32", a_ty.memory_suffix(), b_ty.memory_suffix()),
            _ => return Err(unsupported("WMMA multiplication/accumulation type combination")),
        };
        self.line(format!("wmma.mma.sync.aligned.{a_layout}.{b_layout}.{}.{types} {{{}}}, {{{}}}, {{{}}}, {{{}}};", shape(a_matrix), d.join(", "), a.join(", "), b.join(", "), c.join(", ")));
        Ok(())
    }
}
