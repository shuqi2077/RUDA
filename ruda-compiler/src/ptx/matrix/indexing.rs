use super::*;

impl Emitter {
    pub(super) fn matrix_index(&mut self, out: Variable, lane: Variable, index: Variable, matrix: Matrix, row: bool) -> Result<()> {
        if [out.ty, lane.ty, index.ty].iter().any(|ty| Scalar::of(*ty).ok() != Some(Scalar::U32)) {
            return Err(invalid("manual matrix coordinates require U32 inputs and output"));
        }
        let ty = Scalar::of(Type::new(matrix.storage))?;
        if (matrix.m, matrix.n, matrix.k) == (8, 8, 4)
            && (ty == Scalar::F16 || matrix.ident == MatrixIdent::Accumulator
                && ty == Scalar::F32 && !Scalar::is_tf32(Type::new(matrix.storage)))
        {
            return self.matrix_volta_index(out, lane, index, matrix.ident, ty, row);
        }
        let tf32 = Scalar::is_tf32(Type::new(matrix.storage));
        let integer = matches!((matrix.ident, ty), (MatrixIdent::A | MatrixIdent::B, Scalar::I8 | Scalar::U8) | (MatrixIdent::Accumulator, Scalar::I32));
        let fp8 = ty.fp8() && matrix.ident != MatrixIdent::Accumulator;
        let double = ty == Scalar::F64;
        let fp8_accumulator = matrix.ident == MatrixIdent::Accumulator
            && matches!(ty, Scalar::F16 | Scalar::F32)
            && (matrix.m, matrix.n, matrix.k) == (16, 8, 32);
        let supported_shape = if integer { matches!((matrix.m, matrix.n, matrix.k), (8, 8, 16) | (16, 8, 16) | (16, 8, 32)) }
            else if double { matches!((matrix.m, matrix.n, matrix.k), (8, 8, 4) | (16, 8, 4) | (16, 8, 8) | (16, 8, 16)) }
            else if fp8 { matches!((matrix.m, matrix.n, matrix.k), (16, 8, 16) | (16, 8, 32)) }
            else if fp8_accumulator { true }
            else if tf32 { matches!((matrix.m, matrix.n, matrix.k), (16, 8, 4) | (16, 8, 8)) }
            else { matches!((matrix.m, matrix.n, matrix.k), (16, 8, 8) | (16, 8, 16)) || matrix.ident == MatrixIdent::Accumulator && (matrix.m, matrix.n, matrix.k) == (16, 8, 4) };
        let supported_element = integer || fp8 || double || matches!((matrix.ident, ty), (MatrixIdent::A | MatrixIdent::B, Scalar::F16 | Scalar::BF16) | (MatrixIdent::Accumulator, Scalar::F16 | Scalar::F32)) || tf32 && matrix.ident != MatrixIdent::Accumulator;
        if !supported_shape || !supported_element {
            return Err(unsupported("manual matrix coordinate shape or element type"));
        }
        let elements = (4 / ty.bytes()).max(1).to_string();
        let lane = self.value(lane)?;
        let index = self.value(index)?;
        let result = match (matrix.ident, row) {
            (MatrixIdent::A | MatrixIdent::Accumulator, true) if matrix.m == 8 => self.matrix_index_op("div.u32", &lane, "4"),
            (MatrixIdent::A, true) => {
                let group = self.matrix_index_op("div.u32", &lane, "4");
                let register = self.matrix_index_op("div.u32", &index, &elements);
                let odd = self.matrix_index_op("and.b32", &register, "1");
                let offset = self.matrix_index_op("mul.lo.u32", &odd, "8");
                self.matrix_index_op("add.u32", &group, &offset)
            }
            (MatrixIdent::B, false) => self.matrix_index_op("div.u32", &lane, "4"),
            (MatrixIdent::A, false) | (MatrixIdent::B, true) if double => {
                let thread = self.matrix_index_op("rem.u32", &lane, "4");
                let register = if matrix.ident == MatrixIdent::A {
                    self.matrix_index_op("div.u32", &index, "2")
                } else { index.clone() };
                let offset = self.matrix_index_op("mul.lo.u32", &register, "4");
                self.matrix_index_op("add.u32", &thread, &offset)
            }
            (MatrixIdent::A, false) | (MatrixIdent::B, true) => {
                let thread = self.matrix_index_op("rem.u32", &lane, "4");
                let base = self.matrix_index_op("mul.lo.u32", &thread, &elements);
                let part = self.matrix_index_op("rem.u32", &index, &elements);
                let offset = self.matrix_index_op("add.u32", &base, &part);
                let register = if matrix.ident == MatrixIdent::A {
                    let register = self.matrix_index_op("div.u32", &index, &(8 / ty.bytes()).to_string());
                    self.matrix_index_op("and.b32", &register, "1")
                } else { self.matrix_index_op("div.u32", &index, &elements) };
                let register_offset = self.matrix_index_op("mul.lo.u32", &register, &(16 / ty.bytes()).to_string());
                self.matrix_index_op("add.u32", &offset, &register_offset)
            }
            (MatrixIdent::Accumulator, true) => {
                let group = self.matrix_index_op("div.u32", &lane, "4");
                let scaled = self.matrix_index_op("shl.b32", &index, "2");
                let offset = self.matrix_index_op("and.b32", &scaled, "8");
                self.matrix_index_op("add.u32", &group, &offset)
            }
            (MatrixIdent::Accumulator, false) => {
                let thread = self.matrix_index_op("rem.u32", &lane, "4");
                let base = self.matrix_index_op("mul.lo.u32", &thread, "2");
                let part = self.matrix_index_op("rem.u32", &index, "2");
                self.matrix_index_op("add.u32", &base, &part)
            }
        };
        let destination = self.destination(out)?;
        self.line(format!("mov.u32 {destination}, {result};"));
        Ok(())
    }

    pub(super) fn matrix_index_op(&mut self, opcode: &str, a: &str, b: &str) -> String {
        let result = self.reg(Scalar::U32);
        self.line(format!("{opcode} {result}, {a}, {b};"));
        result
    }
}
