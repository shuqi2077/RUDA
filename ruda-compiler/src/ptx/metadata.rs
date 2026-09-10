use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    fn extended_position(&self, variable: Variable) -> Result<u32> {
        let buffer = self.buffer(variable)?;
        if !buffer.has_extended_meta {
            return Err(invalid("rank/shape/stride requires a tensor argument"));
        }
        Ok(self.buffers[..buffer.id as usize]
            .iter()
            .filter(|arg| arg.has_extended_meta)
            .count() as u32)
    }

    pub fn rank(&mut self, variable: Variable) -> Result<String> {
        let position = self.extended_position(variable)?;
        self.static_metadata(self.info.metadata.rank_index(position))
    }

    pub fn extended_metadata(
        &mut self,
        variable: Variable,
        dim: Variable,
        shape: bool,
    ) -> Result<String> {
        let position = self.extended_position(variable)?;
        let field = if shape {
            self.info.metadata.shape_offset_index(position)
        } else {
            self.info.metadata.stride_offset_index(position)
        };
        let dimension_type = Scalar::of(dim.ty)?;
        if !matches!(dimension_type, Scalar::U32 | Scalar::U64) {
            return Err(unsupported("metadata dimension must be unsigned"));
        }
        let mut offset = self.static_metadata(field)?;
        let mut dimension = self.value(dim)?;
        let index_type = if self.address == Scalar::U64 || dimension_type == Scalar::U64 {
            Scalar::U64
        } else {
            Scalar::U32
        };
        if index_type != self.address {
            let wide = self.reg(Scalar::U64);
            self.line(format!("cvt.u64.u32 {wide}, {offset};"));
            offset = wide;
        }
        if index_type != dimension_type {
            let wide = self.reg(Scalar::U64);
            self.line(format!("cvt.u64.u32 {wide}, {dimension};"));
            dimension = wide;
        }
        let index = self.reg(index_type);
        self.line(format!(
            "add.{} {index}, {offset}, {dimension};",
            index_type.suffix()
        ));
        let pointer = self.reg(Scalar::U64);
        let multiply = if index_type == Scalar::U32 {
            "mul.wide.u32"
        } else {
            "mul.lo.u64"
        };
        self.line(format!(
            "{multiply} {pointer}, {index}, {};",
            self.address.bytes()
        ));
        self.line(format!("add.u64 {pointer}, %dynamic_meta, {pointer};"));
        let result = self.reg(self.address);
        self.line(format!(
            "ld.global.{} {result}, [{pointer}];",
            self.address.suffix()
        ));
        Ok(result)
    }
}
