use super::{emit::Emitter, types::Scalar};

impl Emitter {
    pub fn memory_zero(&mut self, ty: Scalar, register: &str) {
        if ty == Scalar::Pred {
            self.line(format!("setp.ne.u32 {register}, 0, 0;"));
        } else {
            self.line(format!("mov.{} {register}, 0;", ty.bits()));
        }
    }

    pub fn load_memory_scalar(&mut self, space: &str, ty: Scalar, register: &str, address: &str) {
        if ty == Scalar::Pred {
            let byte = self.reg(Scalar::U32);
            self.line(format!("ld.{space}.u8 {byte}, [{address}];"));
            self.line(format!("setp.ne.u32 {register}, {byte}, 0;"));
        } else {
            self.line(format!("ld.{space}.{} {register}, [{address}];", ty.memory_suffix()));
        }
    }

    pub fn store_memory_scalar(&mut self, space: &str, ty: Scalar, register: &str, address: &str) {
        if ty == Scalar::Pred {
            let byte = self.reg(Scalar::U32);
            self.line(format!("selp.u32 {byte}, 1, 0, {register};"));
            self.line(format!("st.{space}.u8 [{address}], {byte};"));
        } else {
            self.line(format!("st.{space}.{} [{address}], {register};", ty.memory_suffix()));
        }
    }
}
