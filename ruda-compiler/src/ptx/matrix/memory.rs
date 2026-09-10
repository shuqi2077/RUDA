use super::*;

impl Emitter {
    pub(super) fn matrix_address(&mut self, array: Variable, offset: Variable, width: Option<usize>, store: bool) -> Result<(String, &'static str)> {
        let ty = Scalar::of(array.ty.with_vector_size(1))?;
        let width = width.unwrap_or(array.ty.vector_size());
        let bytes = width.checked_mul(ty.bytes()).filter(|bytes| *bytes > 0 && *bytes <= u32::MAX as usize)
            .ok_or_else(|| invalid("matrix pointer element size overflow"))?;
        let (base, space) = self.memory_base(array, store)?;
        if !matches!(space, "global" | "shared") { return Err(unsupported("matrix memory must be global or shared")); }
        let offset_ty = Scalar::of(offset.ty)?;
        if !matches!(offset_ty, Scalar::U32 | Scalar::U64) { return Err(invalid("matrix offset must be unsigned")); }
        let offset = self.value(offset)?;
        let address = self.reg(Scalar::U64);
        let multiply = if offset_ty == Scalar::U32 { "mul.wide.u32" } else { "mul.lo.u64" };
        self.line(format!("{multiply} {address}, {offset}, {bytes};"));
        self.line(format!("add.u64 {address}, {base}, {address};"));
        Ok((address, space))
    }

    fn matrix_stride(&mut self, stride: Variable) -> Result<String> {
        let ty = Scalar::of(stride.ty)?;
        if !matches!(ty, Scalar::U32 | Scalar::U64) { return Err(invalid("matrix stride must be unsigned")); }
        let source = self.value(stride)?;
        if ty == Scalar::U32 { return Ok(source); }
        let stride = self.reg(Scalar::U32);
        self.line(format!("cvt.u32.u64 {stride}, {source};"));
        Ok(stride)
    }

    pub(super) fn matrix_load(&mut self, out: Variable, input: Variable, offset: Variable, stride: Variable, explicit_layout: Option<MatrixLayout>) -> Result<()> {
        let (matrix, ty, registers) = self.matrix_fragment(out)?;
        let tf32 = Scalar::is_tf32(out.ty);
        if input.storage_type() != matrix.storage && !(tf32 && Scalar::of(input.ty.with_vector_size(1))? == Scalar::F32) {
            return Err(invalid("matrix load storage type mismatch"));
        }
        let order = if matrix.layout != MatrixLayout::Undefined { matrix.layout }
            else { explicit_layout.unwrap_or(MatrixLayout::Undefined) };
        let order = layout(order)?;
        let ident = match matrix.ident { MatrixIdent::A => "a", MatrixIdent::B => "b", MatrixIdent::Accumulator => "c" };
        let (address, space) = self.matrix_address(input, offset, None, false)?;
        let stride = self.matrix_stride(stride)?;
        let element = if tf32 { "tf32" } else if matrix.ident == MatrixIdent::Accumulator && ty == Scalar::BF16 { "f16" }
            else if matches!(ty, Scalar::I8 | Scalar::U8) { ty.memory_suffix() } else { ty.suffix() };
        self.line(format!("wmma.load.{ident}.sync.aligned.{order}.{}.{space}.{element} {{{}}}, [{address}], {stride};", shape(matrix), registers.join(", ")));
        Ok(())
    }

    pub(super) fn matrix_store(&mut self, output: Variable, input: Variable, offset: Variable, stride: Variable, order: MatrixLayout) -> Result<()> {
        let (matrix, ty, registers) = self.matrix_fragment(input)?;
        if matrix.ident != MatrixIdent::Accumulator || output.storage_type() != matrix.storage {
            return Err(invalid("matrix store requires an accumulator and matching storage"));
        }
        let order = layout(order)?;
        let (address, space) = self.matrix_address(output, offset, None, true)?;
        let stride = self.matrix_stride(stride)?;
        let element = if ty == Scalar::BF16 { "f16" } else { ty.suffix() };
        self.line(format!("wmma.store.d.sync.aligned.{order}.{}.{space}.{element} [{address}], {{{}}}, {stride};", shape(matrix), registers.join(", ")));
        Ok(())
    }
}
