use super::*;
use ruda_core::ir::BinaryOperator;

impl Emitter {
    pub(super) fn plane_shuffle(&mut self, op: BinaryOperator, out: Variable, mode: &str) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        let index_ty = Scalar::of(op.rhs.ty)?;
        if out.ty != op.lhs.ty || !index_ty.integer() {
            return Err(invalid("plane shuffle requires matching value types and an integer lane"));
        }
        let input = self.value(op.lhs)?;
        let mut index = self.value(op.rhs)?;
        if index_ty.bytes() == 8 {
            let narrowed = self.reg(Scalar::U32);
            self.line(format!("cvt.u32.{} {narrowed}, {index};", index_ty.suffix()));
            index = narrowed;
        }
        let mask = self.plane_active_mask();
        let result = self.shuffle_register_masked(ty, &input, &index, mode, &mask)?;
        let destination = self.destination(out)?;
        self.line(format!("mov.{} {destination}, {result};", ty.storage()));
        Ok(())
    }

    pub(super) fn shuffle_register(&mut self, ty: Scalar, input: &str, index: &str, mode: &str) -> Result<String> {
        self.shuffle_register_masked(ty, input, index, mode, "0xffffffff")
    }

    pub(super) fn shuffle_register_masked(&mut self, ty: Scalar, input: &str, index: &str, mode: &str, mask: &str) -> Result<String> {
        let result = self.reg(ty);
        let clamp = if mode == "up" { 0 } else { 31 };
        if ty == Scalar::Pred {
            let bits = self.reg(Scalar::U32);
            self.line(format!("selp.u32 {bits}, 1, 0, {input};"));
            self.line(format!("shfl.sync.{mode}.b32 {bits}, {bits}, {index}, {clamp}, {mask};"));
            self.line(format!("setp.ne.u32 {result}, {bits}, 0;"));
        } else if ty.bytes() == 8 {
            let low = self.reg(Scalar::U32);
            let high = self.reg(Scalar::U32);
            self.line(format!("mov.b64 {{{low}, {high}}}, {input};"));
            self.line(format!("shfl.sync.{mode}.b32 {low}, {low}, {index}, {clamp}, {mask};"));
            self.line(format!("shfl.sync.{mode}.b32 {high}, {high}, {index}, {clamp}, {mask};"));
            self.line(format!("mov.b64 {result}, {{{low}, {high}}};"));
        } else if ty.half() {
            self.half_target(ty)?;
            let bits = self.reg(Scalar::U32);
            self.line(format!("cvt.u32.u16 {bits}, {input};"));
            self.line(format!("shfl.sync.{mode}.b32 {bits}, {bits}, {index}, {clamp}, {mask};"));
            self.line(format!("cvt.u16.u32 {result}, {bits};"));
        } else {
            self.line(format!("shfl.sync.{mode}.b32 {result}, {input}, {index}, {clamp}, {mask};"));
        }
        Ok(result)
    }
}
