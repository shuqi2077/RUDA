use super::{PtxKernel, PtxTarget, Result, invalid, types::Scalar, unsupported};
use ruda_core::{
    arguments::{Info, Metadata},
    ir::{Builtin, ConstantValue, OperationReflect, Scope, StorageType, Type, UIntKind, Variable, VariableKind},
    kernel::{KernelArg, KernelDefinition},
    launch::ExecutionMode,
};
use std::collections::HashMap;

pub(super) struct Emitter {
    pub body: String,
    pub prologue: String,
    pub declarations: String,
    pub registers: HashMap<Variable, String>,
    pub vector_registers: HashMap<Variable, Vec<String>>,
    pub atomic_pointers: HashMap<Variable, String>,
    pub matrix_fragments: HashMap<Variable, Vec<String>>,
    pub local_arrays: HashMap<u32, Variable>,
    pub constant_arrays: HashMap<u32, (Variable, Vec<Variable>)>,
    pub module_declarations: String,
    pub expf_table_symbol: String,
    pub expf_table_base: Option<String>,
    pub logf_table_symbol: String,
    pub logf_table_base: Option<String>,
    pub trig_table_base: Option<String>,
    pub source_files: HashMap<String, u32>,
    pub current_source: Option<(u32, u32, u32)>,
    pub printf_declared: bool,
    pub kernel_name: String,
    next_id: usize,
    pub loops: Vec<String>,
    pub mode: ExecutionMode,
    pub address: Scalar,
    pub info: Info,
    pub buffers: Vec<KernelArg>,
    pub tensor_maps: Vec<KernelArg>,
    pub shared_arrays: HashMap<u32, Variable>,
    pub unit_barriers: Vec<(Variable, String)>,
    pub shared_declaration: String,
    pub target: PtxTarget,
    pub plane_width: u32,
}

impl Emitter {
    pub fn compile(
        mut kernel: KernelDefinition,
        target: PtxTarget,
        mode: ExecutionMode,
        address: StorageType,
    ) -> Result<PtxKernel> {
        let errors = kernel.body.pop_errors();
        if !errors.is_empty() {
            return Err(invalid(errors.join("\n")));
        }
        let cluster_directive = super::cluster::directive(kernel.options.cluster_dim, target)?;
        if !kernel.tensor_maps.is_empty() && (target.sm < 90 || target.version < (8, 0)) {
            return Err(unsupported("tensor maps require SM >= 90 and PTX >= 8.0"));
        }
        if kernel.options.debug_symbols {
            return Err(unsupported("debug symbols"));
        }
        if target.version.0 < 6 || target.version.1 > 9 || target.sm < 30 {
            return Err(invalid(
                "PTX >= 6.0 and SM >= 30 required; target must match the device",
            ));
        }
        let name = &kernel.options.kernel_name;
        if name.is_empty()
            || !name
                .bytes()
                .enumerate()
                .all(|(i, c)| c.is_ascii_alphabetic() || c == b'_' || (i > 0 && c.is_ascii_digit()))
        {
            return Err(invalid("entrypoint must be an ASCII identifier"));
        }
        let dim = kernel.ruda_dim;
        if dim.x == 0
            || dim.y == 0
            || dim.z == 0
            || u128::from(dim.x) * u128::from(dim.y) * u128::from(dim.z) > 1024
        {
            return Err(invalid("invalid block dimensions"));
        }
        if !matches!(
            address,
            StorageType::Scalar(ruda_core::ir::ElemType::UInt(UIntKind::U32 | UIntKind::U64))
        ) {
            return Err(unsupported("address index type must be u32 or u64"));
        }
        let address_scalar = Scalar::of(Type::new(address))?;
        let mut buffers = kernel.tensor_maps.iter().chain(&kernel.buffers).cloned().collect::<Vec<_>>();
        buffers.sort_by_key(|buffer| buffer.id);
        for (position, buffer) in buffers.iter().enumerate() {
            if buffer.id != position as u32 {
                return Err(invalid("buffer and tensor map IDs must form the positional metadata ABI"));
            }
        }
        let extended = buffers
            .iter()
            .filter(|b| b.has_extended_meta)
            .count() as u32;
        let info = Info::new(
            &kernel.scalars,
            Metadata::new(buffers.len() as u32, extended),
            address,
        );
        let mut emitter = Self {
            body: String::new(),
            prologue: String::new(),
            declarations: String::new(),
            registers: HashMap::new(),
            vector_registers: HashMap::new(),
            atomic_pointers: HashMap::new(),
            matrix_fragments: HashMap::new(),
            local_arrays: HashMap::new(),
            constant_arrays: HashMap::new(),
            module_declarations: String::new(),
            expf_table_symbol: format!("{name}__ruda_expf_table"),
            expf_table_base: None,
            logf_table_symbol: format!("{name}__ruda_logf_table"),
            logf_table_base: None,
            trig_table_base: None,
            source_files: HashMap::new(),
            current_source: None,
            printf_declared: false,
            kernel_name: name.clone(),
            next_id: 0,
            loops: Vec::new(),
            mode,
            address: address_scalar,
            info,
            buffers,
            tensor_maps: kernel.tensor_maps.clone(),
            shared_arrays: HashMap::new(),
            unit_barriers: Vec::new(),
            shared_declaration: String::new(),
            target,
            plane_width: (dim.x * dim.y * dim.z).min(32),
        };
        let mut params = Vec::new();
        for map in &kernel.tensor_maps {
            params.push(format!(".param .align 64 .b8 tensor_map_{}[128]", map.id));
            emitter.declarations += &format!("    .reg .u64 %tensor_map_{};\n", map.id);
            emitter.prologue += &format!("    cvta.param.u64 %tensor_map_{0}, tensor_map_{0};\n", map.id);
        }
        for buffer in kernel.tensor_maps.iter().chain(&kernel.buffers) {
            emitter.half_target(Scalar::memory_element(buffer.ty)?)?;
            params.push(format!(".param .u64 buffer_{}", buffer.id));
            emitter.declarations += &format!("    .reg .u64 %buffer_{};\n", buffer.id);
            emitter.prologue +=
                &format!("    ld.param.u64 %buffer_{0}, [buffer_{0}];\n", buffer.id);
        }
        let dynamic_metadata_index = emitter
            .info
            .has_dynamic_meta
            .then_some(emitter.buffers.len());
        if dynamic_metadata_index.is_some() {
            params.push(".param .u64 dynamic_meta".into());
            emitter.declarations += "    .reg .u64 %dynamic_meta;\n";
            emitter.prologue += "    ld.param.u64 %dynamic_meta, [dynamic_meta];\n";
        }
        if emitter.info.has_info() {
            params.push(format!(
                ".param .align 8 .b8 info[{}]",
                emitter.info.dynamic_meta_offset
            ));
        }
        for scalar in &kernel.scalars {
            emitter.half_target(Scalar::of(Type::new(scalar.ty))?)?;
        }
        emitter.scope(kernel.body.clone())?;
        let shared_memory_bytes = emitter.allocate_shared(kernel.body, dim)?;
        let source = format!(
            ".version {}.{}\n.target sm_{}\n.address_size 64\n{}{}\n.visible .entry {}(\n    {}\n)\n.reqntid {}, {}, {}\n{cluster_directive}{{\n{}{}{}    ret;\n}}\n",
            target.version.0,
            target.version.1,
            target.sm,
            emitter.shared_declaration,
            emitter.module_declarations,
            name,
            params.join(",\n    "),
            dim.x,
            dim.y,
            dim.z,
            emitter.declarations,
            emitter.prologue,
            emitter.body,
        );
        Ok(PtxKernel {
            source,
            entrypoint: name.clone(),
            ruda_dim: dim,
            shared_memory_bytes,
            dynamic_metadata_index,
        })
    }

    pub fn reg(&mut self, ty: Scalar) -> String {
        self.reg_with_storage(ty.storage())
    }

    pub fn reg_b16(&mut self) -> String {
        self.reg_with_storage("b16")
    }

    pub fn reg_b32(&mut self) -> String {
        self.reg_with_storage("b32")
    }

    fn reg_with_storage(&mut self, storage: &str) -> String {
        let name = format!("%r{}", self.next_id);
        self.next_id += 1;
        self.declarations += &format!("    .reg .{storage} {name};\n");
        name
    }

    pub fn label(&mut self) -> String {
        let name = format!("L{}", self.next_id);
        self.next_id += 1;
        name
    }

    pub fn line(&mut self, line: impl AsRef<str>) {
        self.body += &format!("    {}\n", line.as_ref());
    }

    pub fn destination(&mut self, var: Variable) -> Result<String> {
        if !matches!(
            var.kind,
            VariableKind::LocalMut { .. }
                | VariableKind::LocalConst { .. }
                | VariableKind::Versioned { .. }
        ) {
            return Err(invalid(format!(
                "invalid register destination {:?}",
                var.kind
            )));
        }
        self.value(var)
    }

    pub fn value(&mut self, var: Variable) -> Result<String> {
        let ty = Scalar::of(var.ty)?;
        self.half_target(ty)?;
        if let VariableKind::Constant(value) = var.kind {
            if ty.half() {
                if let Some(reg) = self.registers.get(&var) { return Ok(reg.clone()); }
                let constant = ty.constant(value)?;
                let reg = self.reg(ty);
                self.prologue += &format!("    mov.b16 {reg}, {constant};\n");
                self.registers.insert(var, reg.clone());
                return Ok(reg);
            }
            if let ConstantValue::Bool(value) = value {
                if ty != Scalar::Pred {
                    return Err(invalid("boolean constant type"));
                }
                if let Some(reg) = self.registers.get(&var) {
                    return Ok(reg.clone());
                }
                let reg = self.reg(Scalar::Pred);
                self.prologue += &format!("    setp.eq.u32 {reg}, 0, {};\n", u32::from(!value));
                self.registers.insert(var, reg.clone());
                return Ok(reg);
            }
            return ty.constant(value);
        }
        if let Some(reg) = self.registers.get(&var) {
            return Ok(reg.clone());
        }
        let reg = self.reg(ty);
        match var.kind {
            VariableKind::LocalMut { .. }
            | VariableKind::LocalConst { .. }
            | VariableKind::Versioned { .. } => {}
            VariableKind::GlobalScalar(id) => {
                let field = self
                    .info
                    .scalars
                    .iter()
                    .find(|f| f.ty == var.storage_type())
                    .ok_or_else(|| invalid("scalar argument group missing"))?;
                if id as usize >= field.size {
                    return Err(invalid("scalar argument index out of range"));
                }
                let offset = field.offset + id as usize * ty.bytes();
                if ty == Scalar::Pred {
                    let byte = self.reg(Scalar::U32);
                    self.prologue += &format!(
                        "    ld.param.u8 {byte}, [info+{offset}];\n    setp.ne.u32 {reg}, {byte}, 0;\n"
                    );
                } else {
                    self.prologue += &format!(
                        "    ld.param.{} {reg}, [info+{offset}];\n", ty.memory_suffix()
                    );
                }
            }
            VariableKind::Builtin(builtin) => {
                if ty != Scalar::U32 && ty != Scalar::U64 {
                    return Err(invalid("builtin type must be u32/u64"));
                }
                let src = self.builtin(builtin, ty)?;
                self.prologue += &format!("    mov.{} {reg}, {src};\n", ty.suffix());
            }
            _ => return Err(unsupported(format!("variable {:?}", var.kind))),
        }
        self.registers.insert(var, reg.clone());
        Ok(reg)
    }

    pub(super) fn special(&mut self, source: &str, ty: Scalar) -> String {
        let reg = self.reg(ty);
        let op = if ty == Scalar::U32 {
            "mov.u32"
        } else {
            "cvt.u64.u32"
        };
        self.prologue += &format!("    {op} {reg}, {source};\n");
        reg
    }

    fn builtin(&mut self, builtin: Builtin, ty: Scalar) -> Result<String> {
        if let Some(value) = self.cluster_builtin(builtin, ty) {
            return Ok(value);
        }
        let direct = match builtin {
            Builtin::UnitPosX => Some("%tid.x"),
            Builtin::UnitPosY => Some("%tid.y"),
            Builtin::UnitPosZ => Some("%tid.z"),
            Builtin::RudaPosX => Some("%ctaid.x"),
            Builtin::RudaPosY => Some("%ctaid.y"),
            Builtin::RudaPosZ => Some("%ctaid.z"),
            Builtin::RudaDimX => Some("%ntid.x"),
            Builtin::RudaDimY => Some("%ntid.y"),
            Builtin::RudaDimZ => Some("%ntid.z"),
            Builtin::RudaCountX => Some("%nctaid.x"),
            Builtin::RudaCountY => Some("%nctaid.y"),
            Builtin::RudaCountZ => Some("%nctaid.z"),
            Builtin::PlaneDim => Some("32"),
            Builtin::UnitPosPlane => Some("%laneid"),
            _ => None,
        };
        if let Some(value) = direct {
            return Ok(self.special(value, ty));
        }
        let out = self.reg(ty);
        let suffix = ty.suffix();
        match builtin {
            Builtin::AbsolutePosX | Builtin::AbsolutePosY | Builtin::AbsolutePosZ => {
                let axis = match builtin {
                    Builtin::AbsolutePosX => "x",
                    Builtin::AbsolutePosY => "y",
                    _ => "z",
                };
                let block = self.special(&format!("%ctaid.{axis}"), ty);
                let size = self.special(&format!("%ntid.{axis}"), ty);
                let unit = self.special(&format!("%tid.{axis}"), ty);
                self.prologue += &format!("    mad.lo.{suffix} {out}, {block}, {size}, {unit};\n");
            }
            Builtin::UnitPos | Builtin::RudaPos => {
                let (pos, size) = if builtin == Builtin::UnitPos {
                    ("tid", "ntid")
                } else {
                    ("ctaid", "nctaid")
                };
                let x = self.special(&format!("%{pos}.x"), ty);
                let y = self.special(&format!("%{pos}.y"), ty);
                let z = self.special(&format!("%{pos}.z"), ty);
                let sx = self.special(&format!("%{size}.x"), ty);
                let sy = self.special(&format!("%{size}.y"), ty);
                self.prologue += &format!(
                    "    mad.lo.{suffix} {out}, {z}, {sy}, {y};\n    mad.lo.{suffix} {out}, {out}, {sx}, {x};\n"
                );
            }
            Builtin::RudaDim | Builtin::RudaCount => {
                let size = if builtin == Builtin::RudaDim {
                    "ntid"
                } else {
                    "nctaid"
                };
                let x = self.special(&format!("%{size}.x"), ty);
                let y = self.special(&format!("%{size}.y"), ty);
                let z = self.special(&format!("%{size}.z"), ty);
                self.prologue += &format!(
                    "    mul.lo.{suffix} {out}, {x}, {y};\n    mul.lo.{suffix} {out}, {out}, {z};\n"
                );
            }
            Builtin::AbsolutePos => {
                let x = self.builtin(Builtin::AbsolutePosX, ty)?;
                let y = self.builtin(Builtin::AbsolutePosY, ty)?;
                let z = self.builtin(Builtin::AbsolutePosZ, ty)?;
                let nx = self.special("%nctaid.x", ty);
                let ny = self.special("%nctaid.y", ty);
                let dx = self.special("%ntid.x", ty);
                let dy = self.special("%ntid.y", ty);
                let width = self.reg(ty);
                let height = self.reg(ty);
                self.prologue += &format!(
                    "    mul.lo.{suffix} {width}, {nx}, {dx};\n    mul.lo.{suffix} {height}, {ny}, {dy};\n    mad.lo.{suffix} {out}, {z}, {height}, {y};\n    mad.lo.{suffix} {out}, {out}, {width}, {x};\n"
                );
            }
            Builtin::PlanePos => {
                let unit = self.builtin(Builtin::UnitPos, ty)?;
                self.prologue += &format!("    shr.{suffix} {out}, {unit}, 5;\n");
            }
            Builtin::PlaneCount => {
                let units = self.builtin(Builtin::RudaDim, ty)?;
                let remainder = self.reg(ty);
                let predicate = self.reg(Scalar::Pred);
                self.prologue += &format!(
                    "    and.{bits} {remainder}, {units}, 31;\n    setp.ne.{suffix} {predicate}, {remainder}, 0;\n    selp.{suffix} {remainder}, 1, 0, {predicate};\n    shr.{suffix} {out}, {units}, 5;\n    add.{suffix} {out}, {out}, {remainder};\n",
                    bits = ty.bits(),
                );
            }
            _ => return Err(unsupported(format!("builtin {builtin:?}"))),
        }
        Ok(out)
    }

    pub fn buffer(&self, var: Variable) -> Result<&KernelArg> {
        let id = match var.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => id,
            _ => return Err(unsupported("non-global array access")),
        };
        let buffer = self
            .buffers
            .get(id as usize)
            .ok_or_else(|| invalid("buffer index out of range"))?;
        if buffer.ty != var.ty {
            return Err(invalid("buffer access type does not match argument"));
        }
        Ok(buffer)
    }

    pub fn meta(&mut self, var: Variable, logical: bool) -> Result<String> {
        if logical && let VariableKind::SharedArray { length, .. } = var.kind {
            self.shared_array(var)?;
            return Ok(length.to_string());
        }
        if let VariableKind::LocalArray { length, .. } | VariableKind::ConstantArray { length, .. } = var.kind {
            self.memory_base(var, false)?;
            return Ok(length.to_string());
        }
        let id = self.buffer(var)?.id;
        let index = if logical {
            self.info.metadata.len_index(id)
        } else {
            self.info.metadata.buffer_len_index(id)
        };
        self.static_metadata(index)
    }

    pub fn static_metadata(&mut self, index: u32) -> Result<String> {
        let offset = self
            .info
            .sized_meta
            .ok_or_else(|| invalid("missing metadata ABI"))?
            .offset
            + index as usize * self.address.bytes();
        let reg = self.reg(self.address);
        self.line(format!(
            "ld.param.{} {reg}, [info+{offset}];",
            self.address.suffix()
        ));
        Ok(reg)
    }

    pub fn memory(
        &mut self,
        array: Variable,
        index: Variable,
        value: Variable,
        store: bool,
        checked: bool,
    ) -> Result<()> {
        let (base, space) = self.memory_base(array, store)?;
        let ty = Scalar::of(array.ty)?;
        if value.ty != array.ty {
            return Err(invalid("load/store element type mismatch"));
        }
        let index_ty = Scalar::of(index.ty)?;
        if !matches!(index_ty, Scalar::U32 | Scalar::U64) {
            return Err(unsupported("array index must be unsigned"));
        }
        let index_reg = self.value(index)?;
        let register = if store {
            self.value(value)?
        } else {
            self.destination(value)?
        };
        let end = self.label();
        if checked && self.mode != ExecutionMode::Unchecked && array.has_length() {
            let len = self.meta(array, false)?;
            let common = if index_ty == self.address {
                index_reg.clone()
            } else {
                let wide = self.reg(Scalar::U64);
                if index_ty == Scalar::U32 {
                    self.line(format!("cvt.u64.u32 {wide}, {index_reg};"));
                } else {
                    self.line(format!("cvt.u64.u32 {wide}, {len};"));
                }
                let pred = self.reg(Scalar::Pred);
                let (a, b) = if index_ty == Scalar::U32 {
                    (wide, len.clone())
                } else {
                    (index_reg.clone(), wide)
                };
                self.line(format!("setp.ge.u64 {pred}, {a}, {b};"));
                self.report_oob(array, &pred, &a, Scalar::U64, &b, Scalar::U64, store)?;
                if !store {
                    self.memory_zero(ty, &register);
                }
                self.line(format!("@{pred} bra {end};"));
                String::new()
            };
            if !common.is_empty() {
                let pred = self.reg(Scalar::Pred);
                self.line(format!(
                    "setp.ge.{} {pred}, {common}, {len};",
                    self.address.suffix()
                ));
                self.report_oob(array, &pred, &common, self.address, &len, self.address, store)?;
                if !store {
                    self.memory_zero(ty, &register);
                }
                self.line(format!("@{pred} bra {end};"));
            }
        }
        let offset = self.reg(Scalar::U64);
        if index_ty == Scalar::U32 {
            self.line(format!(
                "mul.wide.u32 {offset}, {index_reg}, {};",
                ty.bytes()
            ));
        } else {
            self.line(format!("mul.lo.u64 {offset}, {index_reg}, {};", ty.bytes()));
        }
        self.line(format!("add.u64 {offset}, {base}, {offset};"));
        if store {
            self.store_memory_scalar(space, ty, &register, &offset);
        } else {
            self.load_memory_scalar(space, ty, &register, &offset);
        }
        self.line(format!("{end}:"));
        Ok(())
    }

    pub fn scope(&mut self, mut scope: Scope) -> Result<()> {
        for (array, values) in scope.const_arrays.drain(..) {
            self.constant_array(array, values)?;
        }
        let processed = scope.process(std::iter::empty::<&dyn ruda_core::ir::Processor>());
        for instruction in processed.instructions {
            let source = instruction.source_loc.clone();
            let opcode = instruction.operation.op_code();
            self.source_location(source.as_ref())?;
            let result = self.instruction(instruction);
            result.map_err(|error| super::diagnostics::instruction_context(error, opcode, source.as_ref()))?;
        }
        Ok(())
    }
}
