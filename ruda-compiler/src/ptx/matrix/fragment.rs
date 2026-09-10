use super::*;

impl Emitter {
    pub fn matrix_copy(&mut self, out: Variable, input: Variable) -> Result<()> {
        let (source_matrix, ty, source) = self.matrix_fragment(input)?;
        let (destination_matrix, _, destination) = self.matrix_fragment(out)?;
        if source_matrix != destination_matrix { return Err(invalid("matrix copy descriptor mismatch")); }
        for (destination, source) in destination.iter().zip(source) {
            self.line(format!("mov.{} {destination}, {source};", if ty.half() || ty.narrow() || Scalar::is_tf32(input.ty) { "b32" } else { ty.storage() }));
        }
        Ok(())
    }

    pub(super) fn matrix_fragment(&mut self, variable: Variable) -> Result<(Matrix, Scalar, Vec<String>)> {
        let matrix = descriptor(variable)?;
        let ty = Scalar::of(Type::new(matrix.storage))?;
        if self.target.sm < 70 || self.target.version < (6, 3) {
            return Err(unsupported("aligned WMMA requires SM >= 70 and PTX >= 6.3"));
        }
        self.half_target(ty)?;
        if ty.integer() && self.target.sm < 72 {
            return Err(unsupported("integer WMMA requires SM >= 72 and PTX >= 6.3"));
        }
        let tf32 = Scalar::is_tf32(variable.ty);
        let tf32_shape = (matrix.m, matrix.n, matrix.k) == (16, 16, 8);
        if tf32_shape { self.tf32_target()?; }
        if tf32 && (!tf32_shape || matrix.ident == MatrixIdent::Accumulator) {
            return Err(unsupported("TF32 WMMA requires m16n16k8 multiplicands"));
        }
        let double = ty == Scalar::F64 && (matrix.m, matrix.n, matrix.k) == (8, 8, 4);
        if double && (self.target.sm < 80 || self.target.version < (7, 0)) {
            return Err(unsupported("F64 WMMA requires SM >= 80 and PTX >= 7.0"));
        }
        if !double && !tf32_shape && !matches!((matrix.m, matrix.n, matrix.k), (16, 16, 16) | (8, 32, 16) | (32, 8, 16)) {
            return Err(unsupported(format!("WMMA fragment shape {}", shape(matrix))));
        }
        let (register_type, count) = match (matrix.ident, ty) {
            (MatrixIdent::A | MatrixIdent::B, Scalar::F32) if tf32 => (Scalar::U32, 4),
            (_, _) if tf32_shape && !(matrix.ident == MatrixIdent::Accumulator && ty == Scalar::F32) => return Err(unsupported("m16n16k8 requires TF32 multiplicands and FP32 accumulators")),
            (MatrixIdent::A | MatrixIdent::B, Scalar::F64) if double => (Scalar::F64, 1),
            (MatrixIdent::Accumulator, Scalar::F64) if double => (Scalar::F64, 2),
            (MatrixIdent::A | MatrixIdent::B, Scalar::F16) => (Scalar::U32, 8),
            (MatrixIdent::A | MatrixIdent::B, Scalar::BF16) => {
                let elements = if matrix.ident == MatrixIdent::A { matrix.m * matrix.k } else { matrix.k * matrix.n };
                (Scalar::U32, elements / 64)
            }
            (MatrixIdent::A | MatrixIdent::B, Scalar::I8 | Scalar::U8) => {
                let elements = if matrix.ident == MatrixIdent::A { matrix.m * matrix.k } else { matrix.k * matrix.n };
                (Scalar::U32, elements / 128)
            }
            (MatrixIdent::Accumulator, Scalar::I32) => (Scalar::I32, 8),
            (MatrixIdent::Accumulator, Scalar::F32) => (Scalar::F32, 8),
            (MatrixIdent::Accumulator, Scalar::F16 | Scalar::BF16) => (Scalar::U32, 4),
            _ => return Err(unsupported("WMMA fragment element/role combination")),
        };
        if let Some(registers) = self.matrix_fragments.get(&variable) {
            return Ok((matrix, ty, registers.clone()));
        }
        let registers = (0..count).map(|_| if register_type == Scalar::U32 {
            self.reg_b32()
        } else { self.reg(register_type) }).collect::<Vec<_>>();
        self.matrix_fragments.insert(variable, registers.clone());
        Ok((matrix, ty, registers))
    }

    pub(super) fn matrix_fill(&mut self, out: Variable, value: Variable) -> Result<()> {
        let (_, ty, registers) = self.matrix_fragment(out)?;
        if Scalar::of(value.ty)? != ty { return Err(invalid("matrix fill element type mismatch")); }
        let value = self.value(value)?;
        let (source, storage) = if ty.half() {
            let packed = self.reg(Scalar::U32);
            self.line(format!("mov.b32 {packed}, {{{value}, {value}}};"));
            (packed, "b32")
        } else if matches!(ty, Scalar::I8 | Scalar::U8) {
            let packed = self.reg(Scalar::U32);
            self.line(format!("and.b32 {packed}, {value}, 255;"));
            self.line(format!("mul.lo.u32 {packed}, {packed}, 16843009;"));
            (packed, "b32")
        } else { (value, if Scalar::is_tf32(out.ty) { "b32" } else { ty.storage() }) };
        for register in registers { self.line(format!("mov.{storage} {register}, {source};")); }
        Ok(())
    }

    pub(super) fn matrix_cast(&mut self, out: Variable, input: Variable) -> Result<()> {
        let (source_matrix, from, source) = self.matrix_fragment(input)?;
        let (destination_matrix, to, destination) = self.matrix_fragment(out)?;
        if !same_shape(source_matrix, destination_matrix) || source_matrix.ident != MatrixIdent::Accumulator || destination_matrix.ident != MatrixIdent::Accumulator {
            return Err(unsupported("matrix casts require matching accumulator shapes"));
        }
        if from == to {
            for (destination, source) in destination.iter().zip(source) {
                self.line(format!("mov.{} {destination}, {source};", if to.half() { "b32" } else { to.storage() }));
            }
            return Ok(());
        }
        if from.integer() || to.integer() {
            return Err(unsupported("WMMA integer accumulator casts require identical element types"));
        }
        let mut elements = Vec::new();
        for source in source {
            if from.half() {
                let low = self.reg(from);
                let high = self.reg(from);
                self.line(format!("mov.b32 {{{low}, {high}}}, {source};"));
                elements.push(self.half_to_f32(from, &low)?);
                elements.push(self.half_to_f32(from, &high)?);
            } else { elements.push(source); }
        }
        if to.half() {
            if elements.len() != destination.len() * 2 { return Err(invalid("matrix cast fragment size mismatch")); }
            for (destination, pair) in destination.iter().zip(elements.chunks_exact(2)) {
                let low = self.reg(to);
                let high = self.reg(to);
                self.line(format!("cvt.rn.{}.f32 {low}, {};", to.suffix(), pair[0]));
                self.line(format!("cvt.rn.{}.f32 {high}, {};", to.suffix(), pair[1]));
                self.line(format!("mov.b32 {destination}, {{{low}, {high}}};"));
            }
        } else {
            if elements.len() != destination.len() { return Err(invalid("matrix cast fragment size mismatch")); }
            for (destination, source) in destination.iter().zip(elements) {
                self.line(format!("mov.f32 {destination}, {source};"));
            }
        }
        Ok(())
    }
}
