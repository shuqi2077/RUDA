use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn printf(&mut self, format: &str, args: &[Variable]) -> Result<()> {
        let args = args.iter().map(|&arg| self.printf_argument(arg)).collect::<Result<Vec<_>>>()?;
        self.printf_registers(format, args)
    }

    pub(super) fn printf_registers(&mut self, format: &str, args: Vec<(Scalar, String)>) -> Result<()> {
        let mut packed = Vec::with_capacity(args.len());
        let mut bytes = 0usize;
        let mut alignment = 1usize;
        for (ty, register) in args {
            let align = ty.bytes();
            alignment = alignment.max(align);
            let offset = bytes.checked_add(align - 1)
                .map(|value| value & !(align - 1))
                .ok_or_else(|| invalid("printf argument alignment overflow"))?;
            bytes = offset.checked_add(ty.bytes())
                .ok_or_else(|| invalid("printf argument buffer overflow"))?;
            packed.push((offset, ty, register));
        }
        if !self.printf_declared {
            self.module_declarations += ".extern .func (.param .s32 ruda_printf_status) vprintf (.param .b64 ruda_printf_format, .param .b64 ruda_printf_args);\n";
            self.printf_declared = true;
        }
        let id = self.label();
        let format_symbol = format!("$ruda_printf_format_{id}");
        let mut data: Vec<String> = format.bytes().map(|byte| byte.to_string()).collect();
        data.push("0".into());
        self.module_declarations += &format!(
            ".global .align 1 .b8 {format_symbol}[{}] = {{{}}};\n", data.len(), data.join(", ")
        );
        let format_pointer = self.reg(Scalar::U64);
        self.line(format!("mov.u64 {format_pointer}, {format_symbol};"));
        self.line(format!("cvta.global.u64 {format_pointer}, {format_pointer};"));
        let argument_pointer = self.reg(Scalar::U64);
        if packed.is_empty() {
            self.line(format!("mov.u64 {argument_pointer}, 0;"));
        } else {
            let size = bytes.checked_add(alignment - 1)
                .map(|value| value & !(alignment - 1))
                .ok_or_else(|| invalid("printf buffer alignment overflow"))?;
            let symbol = format!("$ruda_printf_args_{id}");
            self.declarations += &format!("    .local .align {alignment} .b8 {symbol}[{size}];\n");
            let local = self.reg(Scalar::U64);
            self.line(format!("mov.u64 {local}, {symbol};"));
            for (offset, ty, register) in packed {
                self.line(format!("st.local.{} [{local}+{offset}], {register};", ty.memory_suffix()));
            }
            self.line(format!("cvta.local.u64 {argument_pointer}, {local};"));
        }
        self.line("{");
        self.line(".param .b64 printf_format;");
        self.line(".param .b64 printf_args;");
        self.line(".param .s32 printf_status;");
        self.line(format!("st.param.b64 [printf_format], {format_pointer};"));
        self.line(format!("st.param.b64 [printf_args], {argument_pointer};"));
        self.line("call.uni (printf_status), vprintf, (printf_format, printf_args);");
        self.line("}");
        Ok(())
    }

    fn printf_argument(&mut self, arg: Variable) -> Result<(Scalar, String)> {
        let ty = Scalar::of(arg.ty)?;
        let promoted = match ty {
            Scalar::Pred | Scalar::I8 | Scalar::U8 | Scalar::I16 | Scalar::U16 => Scalar::I32,
            Scalar::F32 => Scalar::F64,
            Scalar::I32 | Scalar::U32 | Scalar::I64 | Scalar::U64 | Scalar::F64 => ty,
            _ => return Err(unsupported("printf requires scalar integers, bool, F32 or F64; cast packed or half floats explicitly")),
        };
        let source = self.value(arg)?;
        let register = self.reg(promoted);
        match ty {
            Scalar::Pred => self.line(format!("selp.s32 {register}, 1, 0, {source};")),
            Scalar::F32 => {
                let value = self.reg(Scalar::F32);
                self.line(format!("mov.f32 {value}, {source};"));
                self.line(format!("cvt.f64.f32 {register}, {value};"));
            }
            _ => self.line(format!("mov.{} {register}, {source};", promoted.bits())),
        }
        Ok((promoted, register))
    }
}
