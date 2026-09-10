use super::*;

impl Emitter {
    pub(super) fn matrix_transfer(&mut self, registers: Variable, buffer: Variable, offset: Variable, width: Option<usize>, factor: usize, transpose: bool, store: bool) -> Result<()> {
        let (sm, version) = if store { (90, (7, 8)) } else { (75, (6, 5)) };
        if self.target.sm < sm || self.target.version < version {
            return Err(unsupported(format!("{}matrix requires SM >= {sm} and PTX >= {}.{}", if store { "st" } else { "ld" }, version.0, version.1)));
        }
        if !matches!(factor, 1 | 2 | 4) { return Err(invalid("matrix transfer factor must be 1, 2 or 4")); }
        let register_ty = Scalar::of(registers.ty.with_vector_size(1))?;
        let buffer_ty = Scalar::of(buffer.ty.with_vector_size(1))?;
        let byte_elements = (register_ty.fp8() || matches!(register_ty, Scalar::I8 | Scalar::U8))
            && (buffer_ty.fp8() || matches!(buffer_ty, Scalar::I8 | Scalar::U8));
        if !byte_elements && !(register_ty.half() && buffer_ty.half()) {
            return Err(unsupported("matrix transfer requires matching 8-bit or 16-bit element widths"));
        }
        if byte_elements && transpose {
            return Err(unsupported("byte matrix transfer does not yet support element-wise transpose"));
        }
        let (address, space) = self.matrix_address(buffer, offset, width, store)?;
        if space != "shared" { return Err(invalid("ldmatrix/stmatrix requires shared memory")); }
        let shared_address = self.reg(Scalar::U32);
        self.line(format!("cvt.u32.u64 {shared_address}, {address};"));
        let transposed = if transpose { ".trans" } else { "" };
        if store {
            let source = self.matrix_read_registers(registers, factor)?;
            self.line(format!("stmatrix.sync.aligned.m8n8.x{factor}{transposed}.shared.b16 [{shared_address}], {{{}}};", source.join(", ")));
        } else {
            let destination = (0..factor).map(|_| self.reg_b32()).collect::<Vec<_>>();
            self.line(format!("ldmatrix.sync.aligned.m8n8.x{factor}{transposed}.shared.b16 {{{}}}, [{shared_address}];", destination.join(", ")));
            self.matrix_write_registers(registers, &destination)?;
        }
        Ok(())
    }

    pub(super) fn matrix_manual(&mut self, out: Variable, matrix: Matrix, a: Variable, b: Variable, c: Variable) -> Result<()> {
        if (matrix.m, matrix.n, matrix.k) == (8, 8, 4)
            && Scalar::of(a.ty.with_vector_size(1))? == Scalar::F16
        {
            return self.matrix_volta(out, a, b, c);
        }
        let a_ty = Scalar::of(a.ty.with_vector_size(1))?;
        let b_ty = Scalar::of(b.ty.with_vector_size(1))?;
        let c_ty = Scalar::of(c.ty.with_vector_size(1))?;
        let d_ty = Scalar::of(out.ty.with_vector_size(1))?;
        let tf32 = Scalar::is_tf32(a.ty) && Scalar::is_tf32(b.ty);
        let integer = matches!(a_ty, Scalar::I8 | Scalar::U8) && matches!(b_ty, Scalar::I8 | Scalar::U8);
        let fp8 = a_ty.fp8() && b_ty.fp8();
        let double = a_ty == Scalar::F64 && b_ty == Scalar::F64;
        let supported_types = if integer { c_ty == Scalar::I32 && d_ty == Scalar::I32 }
            else if double { c_ty == Scalar::F64 && d_ty == Scalar::F64 }
            else if fp8 { matches!(c_ty, Scalar::F16 | Scalar::F32) && d_ty == c_ty
                && !Scalar::is_tf32(c.ty) && !Scalar::is_tf32(out.ty) }
            else { a_ty == b_ty && !Scalar::is_tf32(c.ty) && !Scalar::is_tf32(out.ty) && match a_ty {
                Scalar::F16 => matches!(c_ty, Scalar::F16 | Scalar::F32) && d_ty == c_ty,
                Scalar::BF16 => c_ty == Scalar::F32 && d_ty == Scalar::F32,
                Scalar::F32 if tf32 => c_ty == Scalar::F32 && d_ty == Scalar::F32,
                _ => false,
            }};
        let supported_shape = if integer { matches!((matrix.m, matrix.n, matrix.k), (8, 8, 16) | (16, 8, 16) | (16, 8, 32)) }
            else if double { matches!((matrix.m, matrix.n, matrix.k), (8, 8, 4) | (16, 8, 4) | (16, 8, 8) | (16, 8, 16)) }
            else if fp8 { matches!((matrix.m, matrix.n, matrix.k), (16, 8, 16) | (16, 8, 32)) }
            else if tf32 { matches!((matrix.m, matrix.n, matrix.k), (16, 8, 4) | (16, 8, 8)) }
            else { matches!((matrix.m, matrix.n, matrix.k), (16, 8, 8) | (16, 8, 16)) };
        if !supported_types || !supported_shape {
            return Err(unsupported("manual MMA shape or element combination"));
        }
        let (sm, version) = if double {
            if matrix.m == 8 { (80, (7, 0)) } else { (90, (7, 8)) }
        } else if fp8 {
            (89, if matrix.k == 16 || c_ty == Scalar::F16 { (8, 7) } else { (8, 4) })
        } else if integer {
            match (matrix.m, matrix.k) {
                (8, 16) => (75, (6, 5)),
                (16, 16) => (75, (7, 0)),
                _ => (80, (7, 0)),
            }
        } else if a_ty == Scalar::F16 && matrix.k == 8 { (75, (6, 5)) } else { (80, (7, 0)) };
        if self.target.sm < sm || self.target.version < version {
            return Err(unsupported(format!("manual MMA requires SM >= {sm} and PTX >= {}.{}", version.0, version.1)));
        }
        let register_bytes = if double { 8 } else { 4 };
        let a = self.matrix_read_registers(a, matrix.m * matrix.k / 32 * a_ty.bytes() / register_bytes)?;
        let b = self.matrix_read_registers(b, matrix.k * matrix.n / 32 * b_ty.bytes() / register_bytes)?;
        let count = matrix.m * matrix.n / 32 * c_ty.bytes() / register_bytes;
        let c = self.matrix_read_registers(c, count)?;
        let register_type = if matches!(d_ty, Scalar::F32 | Scalar::F64 | Scalar::I32) { d_ty } else { Scalar::U32 };
        let d = (0..count).map(|_| if register_type == Scalar::U32 {
            self.reg_b32()
        } else { self.reg(register_type) }).collect::<Vec<_>>();
        let a_element = if tf32 { "tf32" } else if integer { a_ty.memory_suffix() } else { a_ty.suffix() };
        let b_element = if tf32 { "tf32" } else if integer { b_ty.memory_suffix() } else { b_ty.suffix() };
        self.line(format!("mma.sync.aligned.{}.row.col.{}.{a_element}.{b_element}.{} {{{}}}, {{{}}}, {{{}}}, {{{}}};", shape(matrix), d_ty.suffix(), c_ty.suffix(), d.join(", "), a.join(", "), b.join(", "), c.join(", ")));
        self.matrix_write_registers(out, &d)
    }
}
