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
        if self.register_array_memory(array, index, value, width, store)? { return Ok(()); }
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
        let element_bytes = array.ty.vector_size().checked_mul(ty.bytes()).unwrap_or(0);
        let alignment = match array.kind {
            VariableKind::SharedArray { alignment, .. } => alignment.unwrap_or(element_bytes),
            VariableKind::ConstantArray { .. } => element_bytes.checked_next_power_of_two().unwrap_or(0),
            VariableKind::GlobalInputArray(_) | VariableKind::GlobalOutputArray(_) if self.assume_aligned => stride,
            _ => 0,
        };
        let aligned = alignment >= stride && alignment % stride == 0;
        if !matches!(array.kind, VariableKind::LocalArray { .. })
            && self.packed_integer_memory(space, ty, &registers, &address, &base, store, aligned)
        {
            self.line(format!("{end}:"));
            return Ok(());
        }
        if !matches!(array.kind, VariableKind::LocalArray { .. })
            && self.packed_vector_memory(space, ty, &registers, &address, &base, store, aligned)
        {
            self.line(format!("{end}:"));
            return Ok(());
        }
        let vectorized = space == "global" && width.is_power_of_two()
            && ((ty.half() && width >= 2) || (ty.bytes() >= 4 && width >= 2));
        if vectorized {
            let scalar_path = self.label();
            let bytes = stride.min(self.vector_memory_bytes(space));
            if !aligned {
                let misaligned = self.vector_misaligned(space, &base, &address, stride, bytes);
                self.line(format!("@{misaligned} bra.uni {scalar_path};"));
            }
            let elements = bytes / ty.bytes();
            for (chunk, values) in registers.chunks(elements).enumerate() {
                let offset = chunk * bytes;
                let packed = if ty.half() {
                    (0..values.len()/2).map(|_| self.reg_b32()).collect::<Vec<_>>()
                } else { values.to_vec() };
                if ty.half() && store {
                    for (pair, register) in values.chunks_exact(2).zip(&packed) {
                        self.line(format!("mov.b32 {register}, {{{}, {}}};", pair[0], pair[1]));
                    }
                }
                let suffix = if ty.half() { "b32" } else { ty.memory_suffix() };
                let (vector, operands) = if packed.len() == 1 {
                    (String::new(), packed[0].clone())
                } else { (format!("v{}.", packed.len()), format!("{{{}}}", packed.join(", "))) };
                if store { self.line(format!("st.global.{vector}{suffix} [{address}+{offset}], {operands};")); }
                else { self.line(format!("ld.global.{vector}{suffix} {operands}, [{address}+{offset}];")); }
                if ty.half() && !store {
                    for (pair, register) in values.chunks_exact(2).zip(&packed) {
                        self.line(format!("mov.b32 {{{}, {}}}, {register};", pair[0], pair[1]));
                    }
                }
            }
            if aligned {
                self.line(format!("{end}:"));
                return Ok(());
            }
            self.line(format!("bra {end};"));
            self.line(format!("{scalar_path}:"));
        }
        for (lane, register) in registers.iter().enumerate() {
            let offset = lane * ty.bytes();
            let element_address = format!("{address}+{offset}");
            if store { self.store_memory_scalar(space, ty, register, &element_address); }
            else { self.load_memory_scalar(space, ty, register, &element_address); }
        }
        self.line(format!("{end}:"));
        Ok(())
    }

    /// For base + i * stride, address % bytes == base % bytes when bytes
    /// divides stride. Only apply this to stable global buffer parameters.
    fn vector_misaligned(&mut self, space: &str, base: &str, address: &str, stride: usize, bytes: usize) -> String {
        debug_assert!(bytes.is_power_of_two() && stride.is_multiple_of(bytes));
        if space == "global" {
            let key = (base.to_owned(), bytes);
            if let Some(p) = self.base_alignment.get(&key) { return p.clone(); }
            let bits = self.reg(Scalar::U64);
            let p = self.reg(Scalar::Pred);
            self.prologue += &format!("    and.b64 {bits}, {base}, {};\n    setp.ne.u64 {p}, {bits}, 0;\n", bytes - 1);
            self.base_alignment.insert(key, p.clone());
            p
        } else {
            let bits = self.reg(Scalar::U64);
            let p = self.reg(Scalar::Pred);
            self.line(format!("and.b64 {bits}, {address}, {};", bytes - 1));
            self.line(format!("setp.ne.u64 {p}, {bits}, 0;"));
            p
        }
    }

    fn vector_memory_bytes(&self, space: &str) -> usize {
        if space == "global" && self.target.sm >= 100 && self.target.version >= (8, 8) { 32 } else { 16 }
    }

    fn packed_integer_memory(&mut self, space: &str, ty: Scalar, registers: &[String], address: &str, base: &str, store: bool, aligned: bool) -> bool {
        if !(ty.narrow() || ty.fp8()) { return false; }
        let lane_bits = ty.bytes() * 8;
        let lanes = 32 / lane_bits;
        let size = registers.len() * ty.bytes();
        if size < 4 || !size.is_power_of_two() { return false; }
        let bytes = size.min(self.vector_memory_bytes(space));
        let fallback = if aligned { None } else { Some((self.label(), self.label())) };
        if let Some((scalar_path, _)) = &fallback {
            let unaligned = self.vector_misaligned(space, base, address, size, bytes);
            let branch = if space == "global" { "bra.uni" } else { "bra" };
            self.line(format!("@{unaligned} {branch} {scalar_path};"));
        }
        for (chunk, values) in registers.chunks(bytes / ty.bytes()).enumerate() {
            let words = (0..values.len() / lanes).map(|_| self.reg_b32()).collect::<Vec<_>>();
            if store {
                for (group, word) in values.chunks_exact(lanes).zip(&words) {
                    self.line(format!("mov.b32 {word}, 0;"));
                    for (lane, value) in group.iter().enumerate() {
                        self.line(format!("bfi.b32 {word}, {value}, {word}, {}, {lane_bits};", lane * lane_bits));
                    }
                }
            }
            let vector = if words.len() == 1 { String::new() } else { format!("v{}.", words.len()) };
            let operands = if words.len() == 1 { words[0].clone() } else { format!("{{{}}}", words.join(", ")) };
            let offset = chunk * bytes;
            if store { self.line(format!("st.{space}.{vector}b32 [{address}+{offset}], {operands};")); }
            else {
                self.line(format!("ld.{space}.{vector}b32 {operands}, [{address}+{offset}];"));
                let extension = if ty.signed() { "s32" } else { "u32" };
                for (group, word) in values.chunks_exact(lanes).zip(&words) {
                    for (lane, value) in group.iter().enumerate() {
                        self.line(format!("bfe.{extension} {value}, {word}, {}, {lane_bits};", lane * lane_bits));
                    }
                }
            }
        }
        if let Some((scalar_path, done)) = fallback {
            self.line(format!("bra {done};"));
            self.line(format!("{scalar_path}:"));
            for (lane, register) in registers.iter().enumerate() {
                let element_address = format!("{address}+{}", lane * ty.bytes());
                if store { self.store_memory_scalar(space, ty, register, &element_address); }
                else { self.load_memory_scalar(space, ty, register, &element_address); }
            }
            self.line(format!("{done}:"));
        }
        true
    }

    fn packed_vector_memory(&mut self, space: &str, ty: Scalar, registers: &[String], address: &str, base: &str, store: bool, aligned: bool) -> bool {
        let half = matches!(ty, Scalar::F16 | Scalar::BF16);
        let width = registers.len();
        let wide = self.vector_memory_bytes(space) == 32;
        if !(half && (matches!(width, 2 | 4 | 8) || wide && width == 16)
            || ty == Scalar::F32 && (matches!(width, 2 | 4) || wide && width == 8)) {
            return false;
        }
        let alignment = width * ty.bytes();
        let fallback = if aligned { None } else { Some((self.label(), self.label())) };
        if let Some((scalar_path, _)) = &fallback {
            let unaligned = self.vector_misaligned(space, base, address, alignment, alignment);
            let branch = if space == "global" { "bra.uni" } else { "bra" };
            self.line(format!("@{unaligned} {branch} {scalar_path};"));
        }
        let packed = if half { (0..width / 2).map(|_| self.reg_b32()).collect::<Vec<_>>() } else { registers.to_vec() };
        if half && store {
            for (i, value) in packed.iter().enumerate() {
                self.line(format!("mov.b32 {value}, {{{}, {}}};", registers[i * 2], registers[i * 2 + 1]));
            }
        }
        let suffix = if half { "b32" } else { "f32" };
        let vector = if packed.len() == 1 { String::new() } else { format!("v{}.", packed.len()) };
        let values = if packed.len() == 1 { packed[0].clone() } else { format!("{{{}}}", packed.join(", ")) };
        if store { self.line(format!("st.{space}.{vector}{suffix} [{address}], {values};")); }
        else { self.line(format!("ld.{space}.{vector}{suffix} {values}, [{address}];")); }
        if half && !store {
            for (i, value) in packed.iter().enumerate() {
                self.line(format!("mov.b32 {{{}, {}}}, {value};", registers[i * 2], registers[i * 2 + 1]));
            }
        }
        if let Some((scalar_path, done)) = fallback {
            self.line(format!("bra {done};"));
            self.line(format!("{scalar_path}:"));
            for (lane, register) in registers.iter().enumerate() {
                let element_address = format!("{address}+{}", lane * ty.bytes());
                if store { self.store_memory_scalar(space, ty, register, &element_address); }
                else { self.load_memory_scalar(space, ty, register, &element_address); }
            }
            self.line(format!("{done}:"));
        }
        true
    }
}
